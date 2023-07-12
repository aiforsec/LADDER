import sys
from networks import RelationClassificationBERT, RelationClassificationRoBERTa
from transformers import AdamW
from transformers import AutoTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import os
import time, json
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, classification_report
from collections import Counter

# ------------------------init parameters----------------------------

CUDA = "0"
DATASET = '150'  # dataset selection tacred,SemEval
NUM_LABELS = 11  # TACRED:42, SemEval:19
MAX_LENGTH = 512
BATCH_SIZE = 8
LR = 1e-5  # TODO: argparse
EPS = 1e-8
EPOCHS = 10
seed_val = 1
MODEL = 'bert-base-uncased'


os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
device = torch.device("cuda")


import functools
print = functools.partial(print, flush=True)


# ------------------------functions----------------------------
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    non_zero_idx = (labels_flat != 0)
    return np.sum(pred_flat[non_zero_idx] == labels_flat[non_zero_idx]) / len(labels_flat[non_zero_idx])


# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# cited: https://github.com/INK-USC/DualRE/blob/master/utils/scorer.py#L26
def score(key, prediction, verbose=True, no_relation=-1):
    key = key.astype(int)
    prediction = prediction.astype(int)
    if NUM_LABELS == 11:
      print(classification_report(key, prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], target_names=['N/A', 'isA', 'targets', 'uses', 'hasAuthor', 'has', 'variantOf', 'hasAlias', 'indicates', 'discoveredIn', 'exploits'], digits=4))
    else:
        raise ValueError('Number of labels is not correct!')

    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == no_relation and guess == no_relation:
            pass
        elif gold == no_relation and guess != no_relation:
            guessed_by_relation[guess] += 1
        elif gold != no_relation and guess == no_relation:
            gold_by_relation[gold] += 1
        elif gold != no_relation and guess != no_relation:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("SET NO_RELATION ID: ", no_relation)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

# ------------------------prepare sentences----------------------------


# Tokenize all of the sentences and map the tokens to their word IDs.
def pre_processing(sentence_train, sentence_train_label):

    e1_id = -1
    e2_id = -1
    if MODEL in ['bert-base-uncased', 'bert-large-uncased']:
        e1_id = 2487
        e2_id = 2475
    elif MODEL in ['roberta-base', 'roberta-large']:
        e1_id = 134
        e2_id = 176
    elif MODEL in ['xlm-roberta-base', 'xlm-roberta-large']:
        e1_id = 418
        e2_id = 304
    else:
        raise ValueError('Unknown Model')

    input_ids = []
    attention_masks = []
    labels = []
    e1_pos = []
    e2_pos = []

    # Load tokenizer.
    print('Loading BERT tokenizer...')
    # TODO: tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # pre-processing sentences to BERT pattern
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_LENGTH,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        try:
            # Find e1(id:2487) and e2(id:2475) position
            pos1 = (encoded_dict['input_ids'] == e1_id).nonzero()[0][1].item()
            pos2 = (encoded_dict['input_ids'] == e2_id).nonzero()[0][1].item()
            e1_pos.append(pos1)
            e2_pos.append(pos2)
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(sentence_train_label[i])
        except:
            pass

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels, device='cuda')
    e1_pos = torch.tensor(e1_pos, device='cuda')
    e2_pos = torch.tensor(e2_pos, device='cuda')
    w = torch.ones(len(e1_pos), device='cuda')

    # Combine the training inputs into a TensorDataset.
    train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, w)

    return train_dataset


# ------------------------training----------------------------
def main(argv=None):

    print('-'*50)
    print('Model:', MODEL)
    print('LR:', LR)

    # Load the dataset.
    sentence_train = json.load(open('data/' + DATASET + '/train_sentence.json', 'r'))
    sentence_train_label = json.load(open('data/' + DATASET + '/train_label_id.json', 'r'))
    train_dataset = pre_processing(sentence_train, sentence_train_label)

    sentence_val = json.load(open('data/' + DATASET + '/test_sentence.json', 'r'))
    sentence_val_label = json.load(open('data/' + DATASET + '/test_label_id.json', 'r'))
    val_dataset = pre_processing(sentence_val, sentence_val_label)
    print('val dataset size:', len(val_dataset))

    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=BATCH_SIZE  # Trains with this batch size.
    )

    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=BATCH_SIZE  # Evaluate with this batch size.
    )

    # Load models
    if 'roberta' in MODEL:
        modelf1 = RelationClassificationRoBERTa.from_pretrained(
            MODEL,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=NUM_LABELS,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
    else:
        print('trying to ')
        modelf1 = RelationClassificationBERT.from_pretrained(
            MODEL,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=NUM_LABELS,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

    modelf1 = nn.DataParallel(modelf1)
    modelf1.to(device)

    optimizer1 = AdamW(modelf1.parameters(),
                       lr=LR,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                       eps=EPS  # args.adam_epsilon  - default is 1e-8.
                       )

    # Set the seed value all over the place to make this reproducible.
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # validation accuracy, and timings.
    training_stats = []
    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # ========================================
    #               train
    # ========================================

    print("training start...")
    cnt = 0

    # TODO:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    best_f1_score = 0

    for epoch in range(EPOCHS):
        print('--- Epoch {:} / {:} ---'.format(epoch + 1, EPOCHS))
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        # Put the model into training mode
        modelf1.train()

        for step, batch in enumerate(train_dataloader):
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_e1_pos = batch[3].to(device)
            b_e2_pos = batch[4].to(device)
            b_w = batch[5].to(device)

            modelf1.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch)
            loss, logits, _ = modelf1(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels,
                                      e1_pos=b_e1_pos,
                                      e2_pos=b_e2_pos,
                                      w=b_w)

            # Accumulate the training loss over all of the batches
            total_train_loss += loss.sum().item()
            # Perform a backward pass to calculate the gradients.
            loss.sum().backward()
            # Clip the norm of the gradients to 1.0 to prevent exploding gradients problem.
            torch.nn.utils.clip_grad_norm_(modelf1.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient
            optimizer1.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        print("Average training loss: {0:.4f}".format(avg_train_loss))

        print("validation start...")
        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        modelf1.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        all_prediction = np.array([])
        all_ground_truth = np.array([])
        model_predictions = []

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_e1_pos = batch[3].to(device)
            b_e2_pos = batch[4].to(device)
            b_w = batch[5].to(device)

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                (loss, logits, _) = modelf1(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,
                                            labels=b_labels,
                                            e1_pos=b_e1_pos,
                                            e2_pos=b_e2_pos,
                                            w=b_w)

            # Accumulate the validation loss.
            total_eval_loss += loss.sum().item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)

            for ii in range(b_input_ids.shape[0]):
                decoded = tokenizer.decode(b_input_ids[ii], skip_special_tokens=False)
                model_predictions.append([decoded, int(labels_flat[ii]), int(pred_flat[ii])])

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        _, _, f1_score_this = score(all_ground_truth, all_prediction)
        if f1_score_this > best_f1_score:
            best_f1_score = f1_score_this
            torch.save(modelf1.state_dict(), DATASET + '_best_1.pt')

            with open(MODEL + '_' + str(LR) +'_test_predictions.json', 'w') as fw:
                json.dump(model_predictions, fw)

    print("final validation start...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
    modelf1.eval()
    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    all_prediction = np.array([])
    all_ground_truth = np.array([])

    # Evaluate data for one epoch
    model_predictions = []
    for batch in validation_dataloader:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)

        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)
        b_w = batch[5].to(device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            (loss, logits, _) = modelf1(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        e1_pos=b_e1_pos,
                                        e2_pos=b_e2_pos,
                                        w=b_w)

        # Accumulate the validation loss.
        total_eval_loss += loss.sum().item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
        all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)

        for ii in range(b_input_ids.shape[0]):
            decoded = tokenizer.decode(b_input_ids[ii], skip_special_tokens=False)
            model_predictions.append([decoded, int(labels_flat[ii]), int(pred_flat[ii])])

    print('*' * 20)
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    non_zero_idx = (all_ground_truth != 0)
    validation_acc = np.sum(all_prediction[non_zero_idx] == all_ground_truth[non_zero_idx]) / len(
        all_ground_truth[non_zero_idx])
    validation_f1_score = f1_score(all_ground_truth[non_zero_idx], all_prediction[non_zero_idx], average="micro")

    _, _, f1_score_epoch = score(all_ground_truth, all_prediction)

    # print('*' * 20)
    # with open(MODEL + '_final_predictions.json', 'w') as fw:
    #     json.dump(model_predictions, fw)


if __name__ == "__main__":
    main()