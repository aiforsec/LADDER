import csv
import os
import random

import numpy as np
import sklearn
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from models import SentenceClassificationBERT, SentenceClassificationRoBERTa
from dataset import SentenceClassificationDatasetBERT as Dataset
from utils import get_task_config
from argparser import parse_bert_sentence_classification_arguments as parse_args
from config import MODELS
from transformers import *


class Classification:
    def __init__(self, args):
        self.args = args

        self.use_cuda = args.cuda and torch.cuda.is_available()

        self.tokenizer = MODELS[args.model][1]

        # for reproducibility
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.config = get_task_config(args.task)
        self.num_classes = self.config.num_class

        # data loaders
        train_dataset = Dataset(os.path.join(args.dataset, 'train.csv'), self.config.sequence_len, args.model)
        val_dataset = Dataset(os.path.join(args.dataset, 'dev.csv'), self.config.sequence_len, args.model)
        test_dataset = Dataset(os.path.join(args.dataset, 'test.csv'), self.config.sequence_len, args.model)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # model
        if MODELS[args.model][3] == 'bert':
            self.model = SentenceClassificationBERT(pretrained_model=args.model, num_class=self.config.num_class,
                                                    fine_tune=args.fine_tune)
        elif MODELS[args.model][3] == 'roberta':
            self.model = SentenceClassificationRoBERTa(pretrained_model=args.model, num_class=self.config.num_class,
                                                    fine_tune=args.fine_tune)
        else:
            raise ValueError('Unknown model')

        self.device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
        self.model.to(self.device)

        # logs
        os.makedirs(args.save_path, exist_ok=True)
        self.model_save_path = os.path.join(args.save_path, 'weights.pt')
        self.log_path = os.path.join(args.save_path, args.task + '_logs.csv')
        print(str(args))
        with open(self.log_path, 'a') as f:
            f.write(str(args) + '\n')
        with open(self.log_path, 'a', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['mode', 'epoch', 'step', 'loss', 'acc'])

        # optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.decay)

        # for early stopping
        self.best_val_acc = -1
        self.early_stop = False
        self.val_patience = 0  # successive iteration when validation acc did not improve

        self.iteration_number = 0

    def test(self, loader):
        self.model.eval()
        test_loss = 0
        total = 0
        correct = 0

        y_true_all = np.zeros((0,), dtype=int)
        y_pred_all = np.zeros((0,), dtype=int)

        model_predictions = []

        with torch.no_grad():
            for x, att, y, text in loader:
                x, y, att = x.to(self.device), y.to(self.device), att.to(self.device)
                y_pred = self.model(x, att)
                loss = self.criterion(y_pred, y)

                test_loss += loss.item() * y.shape[0]
                total += y.shape[0]
                correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()

                y_true_all = np.concatenate([y_true_all, y.cpu().detach().numpy()])
                y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                y_pred_all = np.concatenate([y_pred_all, y_pred])

                for ii in range(x.shape[0]):
                    decoded = text[ii]
                    model_predictions.append([decoded, int(y[ii]), int(y_pred[ii])])

        import json
        with open(os.path.join(self.args.save_path, 'predictions.json'), 'w') as fw:
            json.dump(model_predictions, fw)

        avg_loss = test_loss / total
        acc = correct / total
        return avg_loss, acc, y_true_all, y_pred_all

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        total = 0
        correct = 0
        for x, att, y, _ in self.train_loader:
            x, y, att = x.to(self.device), y.to(self.device), att.to(self.device)
            y_pred = self.model(x, att)
            loss = self.criterion(y_pred, y)

            train_loss += loss.item() * y.shape[0]
            total += y.shape[0]
            correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # eval
            self.iteration_number += 1
            if self.iteration_number % self.config.eval_interval == 0:
                avg_loss = train_loss / total
                train_acc = correct / total
                print('Train loss: {}, Train acc: {}'.format(avg_loss, train_acc))
                train_loss = 0
                total = 0
                correct = 0

                val_loss, val_acc, _, _ = self.test(self.val_loader)
                print('Val loss: {}, Val Acc: {}'.format(val_loss, val_acc))
                if val_acc > self.best_val_acc:
                    torch.save(self.model.state_dict(), self.model_save_path)
                    self.best_val_acc = val_acc
                    self.val_patience = 0
                else:
                    self.val_patience += 1
                    if self.val_patience == self.config.patience:
                        self.early_stop = True
                        return
                with open(self.log_path, 'a', newline='') as out:
                    writer = csv.writer(out)
                    writer.writerow(['train', epoch, self.iteration_number, avg_loss, train_acc])
                    writer.writerow(['val', epoch, self.iteration_number, val_loss, val_acc])
                self.model.train()

    def run(self):
        for epoch in range(self.args.epoch):
            print('------------------------------------- Epoch {} -------------------------------------'.format(epoch))
            self.train(epoch)
            if self.early_stop:
                break
        print('Training complete!')
        print('Best Validation Acc: ', self.best_val_acc)

        self.model.load_state_dict(torch.load(self.model_save_path))
        train_loss, train_acc, _, _ = self.test(self.train_loader)
        val_loss, val_acc, _, _ = self.test(self.val_loader)
        test_loss, test_acc, y_true, y_pred = self.test(self.test_loader)

        with open(self.log_path, 'a', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['train', -1, -1, train_loss, train_acc])
            writer.writerow(['val', -1, -1, val_loss, val_acc])
            writer.writerow(['test', -1, -1, test_loss, test_acc])

        print('Train loss: {}, Train Acc: {}'.format(train_loss, train_acc))
        print('Val loss: {}, Val Acc: {}'.format(val_loss, val_acc))
        print('Test loss: {}, Test Acc: {}'.format(test_loss, test_acc))

        pr, re, f1, _ = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')
        print('Test Pr: {}, Test Re: {}, Test F1: {}'.format(pr, re, f1))
        with open(self.log_path, 'a', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['test(acc/pr/re/f1)', test_acc, pr, re, f1])
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

        # write final results
        result_file = os.path.join(self.args.save_path, self.args.task + '_result.txt')
        with open(result_file, 'a') as f:
            f.write(str(self.args) + '\n')
            f.write('Test Acc: {}, Test Pr: {}, Test Re: {}, Test F1: {}\n'.format(test_acc, pr, re, f1))
            f.write(str(cm) + '\n')

        return test_acc, f1, y_true, y_pred


def run():
    cls = Classification(parse_args())
    cls.run()


if __name__ == '__main__':
    run()
