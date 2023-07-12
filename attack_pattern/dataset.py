import torch
import spacy
from torchtext.vocab import vocab
import numpy as np
import pandas as pd

from config import *


def parse_entity_data(file_path, tokenizer, sequence_len, token_style):
    """

    :param file_path: text file path that contains tokens and entity separated by space in lines,
    blank line separates two sentences
    :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
    :param sequence_len: maximum length of each sequence
    :param token_style: For getting index of special tokens in config.TOKEN_IDX
    :return: list of [tokens_index, entity_index, attention_masks], each having sequence_len
    """
    data_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        idx = 0
        # loop until end of the entire text
        while idx < len(lines):
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            y = [0]

            # loop until we have required sequence length
            # -1 because we will have a special end of sequence token at the end
            while len(x) < sequence_len - 1 and idx < len(lines):
                # blank line separates sentences
                if len(lines[idx]) == 0:
                    idx += 1
                    break
                #
                word, entity = lines[idx].split(' ')
                tokens = tokenizer.tokenize(word)
                # if taking these tokens exceeds sequence length we finish current sequence with padding
                # then start next sequence from this token
                if len(tokens) + len(x) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y.append(entity_mapping[entity])
                    if len(tokens) > 0:
                        x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x.append(TOKEN_IDX[token_style]['UNK'])
                    y.append(entity_mapping[entity])
                    idx += 1
            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            y.append(0)
            if len(x) < sequence_len:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
                y = y + [0 for _ in range(sequence_len - len(y))]
            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
            data_items.append([x, y, attn_mask])
    return data_items


class EntityRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, files, tokenizer, sequence_len, token_style):
        """

        :param files: single file or list of text files containing tokens and punctuations separated by tab in lines
        :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        """
        if isinstance(files, list):
            self.data = []
            for file in files:
                self.data += parse_entity_data(file, tokenizer, sequence_len, token_style)
        else:
            self.data = parse_entity_data(files, tokenizer, sequence_len, token_style)
        self.sequence_len = sequence_len
        self.token_style = token_style

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]

        x = torch.tensor(x)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        return x, y, attn_mask


class SentenceClassificationDatasetBERT(torch.utils.data.Dataset):
    def __init__(self, file_path, sequence_len, bert_model, num_class=2, balance=False):
        df = pd.read_csv(file_path, sep='\t')
        _data = []
        for _, row in df.iterrows():
            _data.append([row['text'], row['label']])
        if balance:
            self.data = self.balance(_data, num_class)
        else:
            self.data = _data
        tokenizer = MODELS[bert_model][1]
        self.tokenizer = tokenizer.from_pretrained(bert_model)
        self.sequence_len = sequence_len
        token_style = MODELS[bert_model][3]
        self.start_token = TOKENS[token_style]['START_SEQ']
        self.end_token = TOKENS[token_style]['END_SEQ']
        self.pad_token = TOKENS[token_style]['PAD']
        self.pad_idx = TOKEN_IDX[token_style]['PAD']

    @staticmethod
    def balance(data, num_class):
        # get count
        count = {}
        for x in data:
            label = x[1]
            if label not in count:
                count[label] = 0
            count[label] += 1

        # minimum count
        min_count = 99999999
        for _, v in count.items():
            min_count = min(min_count, v)

        # filter
        random.shuffle(data)
        new_data = []
        count_rem = [min_count] * num_class
        for x in data:
            label = x[1]
            if count_rem[label] > 0:
                new_data.append(x)
            count_rem[label] -= 1

        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index][0]
        label = self.data[index][1]
        tokens_text = self.tokenizer.tokenize(text)
        tokens = [self.start_token] + tokens_text + [self.end_token]
        if len(tokens) < self.sequence_len:
            tokens = tokens + [self.pad_token for _ in range(self.sequence_len - len(tokens))]
        else:
            tokens = tokens[:self.sequence_len - 1] + [self.end_token]

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != self.pad_idx).long()
        return tokens_ids_tensor, attn_mask, label, text
