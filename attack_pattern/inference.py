import torch
import nltk
import os

from argparser import parse_inference_arguments as parse_args
from models import EntityRecognition, SentenceClassificationBERT, SentenceClassificationRoBERTa
from config import *

def remove_consec_newline(s):
    ret = s[0]
    for x in s[1:]:
        if not (x == ret[-1] and ret[-1]=='\n'):
            ret += x
    return ret

def extract_sentences(text):
    text = remove_consec_newline(text)
    text = text.replace('\t', ' ')
    text = text.replace("\'", "'")
    sents_nltk = nltk.sent_tokenize(text)
    sents = []
    for x in sents_nltk:
        sents += x.split('\n')
    return sents


def classify_sent(sent, model, tokenizer, token_style, sequence_len, device):
    start_token = TOKENS[token_style]['START_SEQ']
    end_token = TOKENS[token_style]['END_SEQ']
    pad_token = TOKENS[token_style]['PAD']
    pad_idx = TOKEN_IDX[token_style]['PAD']

    tokens_text = tokenizer.tokenize(sent)
    tokens = [start_token] + tokens_text + [end_token]

    if len(tokens) < sequence_len:
        tokens = tokens + [pad_token for _ in range(sequence_len - len(tokens))]
    else:
        tokens = tokens[:sequence_len - 1] + [end_token]

    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    x = torch.tensor(tokens_ids).reshape(1, -1).reshape(1, -1)
    att = (x != pad_idx).long()

    x, att = x.to(device), att.to(device)

    with torch.no_grad():
        y_pred = model(x, att)
        return torch.argmax(y_pred).item()


def extract_entities(sent, model, tokenizer, token_style, sequence_len, device):
    words_original_case = nltk.word_tokenize(sent)
    words = [x.lower() for x in words_original_case]
    token_to_word_mapping = {}

    word_pos = 0
    x = [TOKEN_IDX[token_style]['START_SEQ']]
    while word_pos < len(words):
        tokens = tokenizer.tokenize(words[word_pos])

        if len(tokens) + len(x) >= sequence_len:
            break
        else:
            for i in range(len(tokens) - 1):
                x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
            x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
            token_to_word_mapping[len(x) - 1] = words_original_case[word_pos]
            word_pos += 1
    x.append(TOKEN_IDX[token_style]['END_SEQ'])
    if len(x) < sequence_len:
        x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
    attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

    x = torch.tensor(x).reshape(1, -1)
    attn_mask = torch.tensor(attn_mask).reshape(1, -1)
    x, attn_mask = x.to(device), attn_mask.to(device)

    ret = ''
    cur = ''
    cur_word_count = 0
    with torch.no_grad():
        y_pred = model(x, attn_mask)
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        x = x.view(-1)
        for i in range(y_pred.shape[0]):
            if x[i].item() == TOKEN_IDX[token_style]['PAD']:
                break

            token_pred = torch.argmax(y_pred[i]).item()

            # print(tokenizer.convert_ids_to_tokens(x[i].item()), token_pred)

            if i in token_to_word_mapping:
                if token_pred == entity_mapping['ATK']:
                    cur += token_to_word_mapping[i] + ' '
                    cur_word_count += 1
                else:
                    if len(cur) > 0 and cur_word_count >= 2 and is_valid_step(cur):
                        ret += cur[:-1] + '\n'
                        cur = ''
                        cur_word_count = 0
                    else:
                        cur = ''
                        cur_word_count = 0
        if len(cur) > 0 and cur_word_count >= 2:
            ret += cur[:-1] + '\n'
    return ret


def is_valid_step(text):
    verb_codes = {
        'VB',  # Verb, base form
        'VBD',  # Verb, past tense
        'VBG',  # Verb, gerund or present participle
        'VBN',  # Verb, past participle
        'VBP',  # Verb, non-3rd person singular present
        'VBZ',  # Verb, 3rd person singular present
    }
    pos = nltk.pos_tag(nltk.word_tokenize(text))
    for x in pos:
        if x[1] in verb_codes:
            return True
    return False


def infer():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    args = parse_args()

    args.entity_extraction_model = 'roberta-large'
    args.sentence_classification_model = 'roberta-large'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # print(device)
    
    entity_model = EntityRecognition(args.entity_extraction_model).to(device)
    entity_model.load_state_dict(torch.load(args.entity_extraction_weight, map_location=device))


    if MODELS[args.sentence_classification_model][3] == 'bert':
        sentence_model = SentenceClassificationBERT(args.sentence_classification_model, num_class=2).to(device)
        sentence_model.load_state_dict(torch.load(args.sentence_classification_weight, map_location=device))
    elif MODELS[args.sentence_classification_model][3] == 'roberta':
        sentence_model = SentenceClassificationRoBERTa(args.sentence_classification_model, num_class=2).to(device)
        sentence_model.load_state_dict(torch.load(args.sentence_classification_weight, map_location=device))
    else:
        raise ValueError('Unknown sentence classification model')


    tokenizer_sen = MODELS[args.sentence_classification_model][1]
    token_style_sen = MODELS[args.sentence_classification_model][3]
    tokenizer_sen = tokenizer_sen.from_pretrained(args.sentence_classification_model)
    sequence_len_sen = args.sequence_length_sentence

    tokenizer_ent = MODELS[args.entity_extraction_model][1]
    token_style_ent = MODELS[args.entity_extraction_model][3]
    tokenizer_ent = tokenizer_ent.from_pretrained(args.entity_extraction_model)
    sequence_len_ent = args.sequence_length_entity

    files = os.listdir(args.input_doc)

    for fname in files:
        with open(os.path.join(args.input_doc, fname), 'r', encoding='utf-8') as f:
            text = f.read()       

        sents = extract_sentences(text)
        result = ''
        for x in sents:
            # class 1: attack pattern sentence
            if classify_sent(x, sentence_model, tokenizer_sen, token_style_sen, sequence_len_sen, device):
                ex = extract_entities(x, entity_model, tokenizer_ent, token_style_ent, sequence_len_ent, device)
                result += ex
        with open(os.path.join(args.save_path, fname), 'w', encoding='utf-8') as f:
            f.write(result)

if __name__ == '__main__':
    infer()
