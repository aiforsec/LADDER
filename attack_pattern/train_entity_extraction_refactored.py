import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import torch.multiprocessing
from tqdm import tqdm

from argparser import parse_entity_recognition_arguments
from dataset import EntityRecognitionDataset as Dataset
from models import EntityRecognition
from config import *


class EntityClassification:
    def __init__(self, args):
        torch.multiprocessing.set_sharing_strategy('file_system')  # https://github.com/pytorch/pytorch/issues/11201

        self.args = args

        # for reproducibility
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)

        # tokenizer
        self.tokenizer = MODELS[self.args.pretrained_model][1].from_pretrained(self.args.pretrained_model)
        self.token_style = MODELS[self.args.pretrained_model][3]
        sequence_len = self.args.sequence_length

        train_set = Dataset(os.path.join(self.args.dataset, 'train.txt'), tokenizer=self.tokenizer, sequence_len=sequence_len,
                            token_style =self.token_style)
        val_set = Dataset(os.path.join(self.args.dataset, 'dev.txt'), tokenizer=self.tokenizer, sequence_len=sequence_len,
                          token_style=self.token_style)
        test_set_1 = Dataset(os.path.join(self.args.dataset, 'test.txt'), tokenizer=self.tokenizer, sequence_len=sequence_len,
                             token_style=self.token_style)
        test_set = [test_set_1]

        # Data Loaders
        train_data_loader_params = {
            'batch_size': self.args.batch_size,
            'shuffle': True,
            'num_workers': 1
        }
        test_data_loader_params = {
            'batch_size': self.args.batch_size,
            'shuffle': False,
            'num_workers': 1
        }

        self.train_loader = torch.utils.data.DataLoader(train_set, **train_data_loader_params)
        self.val_loader = torch.utils.data.DataLoader(val_set, **test_data_loader_params)
        self.test_loaders = [torch.utils.data.DataLoader(x, **test_data_loader_params) for x in test_set]

        # logs
        os.makedirs(self.args.save_path, exist_ok=True)
        self.model_save_path = os.path.join(self.args.save_path, 'weights.pt')
        self.log_path = os.path.join(self.args.save_path, self.args.name + '_logs.txt')

        # Model
        self.device = torch.device('cuda' if (self.args.cuda and torch.cuda.is_available()) else 'cpu')

        self.entity_extraction = EntityRecognition(self.args.pretrained_model, freeze_bert=self.args.freeze_bert,
                                              lstm_dim=self.args.lstm_dim)
        self.entity_extraction.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.entity_extraction.parameters(), lr=self.args.lr, weight_decay=self.args.decay)

    def validate(self, data_loader):
        """
        :return: validation accuracy, validation loss
        """
        num_iteration = 0
        self.entity_extraction.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for x, y, att in tqdm(data_loader, desc='eval'):
                x, y, att = x.to(self.device), y.to(self.device), att.to(self.device)
                y_predict = self.entity_extraction(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                loss = self.criterion(y_predict, y)
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
                val_loss += loss.item()
                num_iteration += 1
                att = att.view(-1)
                # subtract 2 for start and end sequence tokens
                correct += torch.sum(att * (y_predict == y).long()).item() - 2 * x.shape[0]
                total += torch.sum(att).item() - 2 * x.shape[0]
        return correct / total, val_loss / num_iteration

    def test(self,data_loader):
        """
        :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
        """
        num_iteration = 0
        self.entity_extraction.eval()
        # +1 for overall result
        tp = np.zeros(1 + len(entity_mapping), dtype=np.int)
        fp = np.zeros(1 + len(entity_mapping), dtype=np.int)
        fn = np.zeros(1 + len(entity_mapping), dtype=np.int)
        cm = np.zeros((len(entity_mapping), len(entity_mapping)), dtype=np.int)
        correct = 0
        total = 0
        pred_str = ''
        with torch.no_grad():
            for x, y, att in tqdm(data_loader, desc='test'):
                x, y, att = x.to(self.device), y.to(self.device), att.to(self.device)

                y_predict = self.entity_extraction(x, att)

                for ii in range(y.shape[0]):
                    for jj in range(len(y[ii])):
                        if x[ii][jj].item() == TOKEN_IDX[self.token_style]['PAD']:
                            break
                        pred_str += self.tokenizer.convert_ids_to_tokens(x[ii][jj].item()) + '\t'
                        pred_str += str(reverse_entity_mapping[y[ii][jj].item()]) + '\t'
                        pred_str += str(reverse_entity_mapping[torch.argmax(y_predict[ii][jj]).item()])
                        pred_str += '\n'
                    pred_str += '[SENT_BREAK]\n'

                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
                num_iteration += 1

                att = att.view(-1)
                x = x.view(-1)

                correct += torch.sum(att * (y_predict == y).long()).item()
                total += torch.sum(att).item()

                sos_token = TOKEN_IDX[self.token_style]['START_SEQ']
                eos_token = TOKEN_IDX[self.token_style]['END_SEQ']

                for i in range(y.shape[0]):

                    input_token = x[i].item()
                    if att[i] == 0 or input_token == sos_token or input_token == eos_token:
                        # do not count these as they are trivially no-entity tokens
                        continue
                    cor = y[i]
                    prd = y_predict[i]
                    if cor == prd:
                        tp[cor] += 1
                    else:
                        fn[cor] += 1
                        fp[prd] += 1
                    cm[cor][prd] += 1
        # ignore first index which is for no entity
        tp[-1] = np.sum(tp[1:])
        fp[-1] = np.sum(fp[1:])
        fn[-1] = np.sum(fn[1:])
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        with open(os.path.join(self.args.save_path, 'prediction.txt'), 'w') as f:
            f.write(pred_str)

        return precision, recall, f1, correct / total, cm

    def train(self):
        with open(self.log_path, 'a') as f:
            f.write(str(self.args) + '\n')
        best_val_acc = 0
        for epoch in range(self.args.epoch):
            train_loss = 0.0
            train_iteration = 0
            correct = 0
            total = 0
            self.entity_extraction.train()
            for x, y, att in tqdm(self.train_loader, desc='train'):
                x, y, att = x.to(self.device), y.to(self.device), att.to(self.device)

                y_predict = self.entity_extraction(x, att)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y = y.view(-1)
                loss = self.criterion(y_predict, y)
                y_predict = torch.argmax(y_predict, dim=1).view(-1)

                att = att.view(-1)
                correct += torch.sum(att * (y_predict == y).long()).item() - 2 * x.shape[0]

                self.optimizer.zero_grad()
                train_loss += loss.item()
                train_iteration += 1
                loss.backward()

                if self.args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.entity_extraction.parameters(), self.args.gradient_clip)
                self.optimizer.step()

                total += torch.sum(att).item() - 2 * x.shape[0]

            train_loss /= train_iteration
            log = 'epoch: {}, Train loss: {}, Train accuracy: {}'.format(epoch, train_loss, correct / total)
            with open(self.log_path, 'a') as f:
                f.write(log + '\n')
            print(log)

            val_acc, val_loss = self.validate(self.val_loader)
            log = 'epoch: {}, Val loss: {}, Val accuracy: {}'.format(epoch, val_loss, val_acc)
            with open(self.log_path, 'a') as f:
                f.write(log + '\n')
            print(log)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.entity_extraction.state_dict(), self.model_save_path)

        print('Best validation Acc:', best_val_acc)
        self.entity_extraction.load_state_dict(torch.load(self.model_save_path))
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        for loader in self.test_loaders:
            precision, recall, f1, accuracy, cm = self.test(loader)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1_list)
            accuracy_list.append(accuracy)

            log = 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + 'F1 score: ' + str(f1) + \
                  '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix' + str(cm) + '\n'
            print(log)
            with open(self.log_path, 'a') as f:
                f.write(log)
            log_text = ''
            for i in range(1, 2):
                log_text += str(precision[i] * 100) + ' ' + str(recall[i] * 100) + ' ' + str(f1[i] * 100) + ' '
            with open(self.log_path, 'a') as f:
                f.write(log_text[:-1] + '\n\n')

        return accuracy_list, f1_list, precision_list, recall_list



def train():
    cls = EntityClassification(parse_entity_recognition_arguments())
    cls.train()


if __name__ == '__main__':
    train()
