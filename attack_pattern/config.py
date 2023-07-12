from transformers import AutoTokenizer, AutoModel

# special tokens indices in different models available in transformers
TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}


TOKENS = {
    'bert': {
        'START_SEQ': '[CLS]',
        'PAD': '[PAD]',
        'END_SEQ': '[SEP]',
        'UNK': '[UNK]'
    },
    'roberta': {
        'START_SEQ': '<s>',
        'PAD': '<pad>',
        'END_SEQ': '</s>',
        'UNK': '<unk>'
    },
}

# 'O' -> No entity
entity_mapping = {'O': 0, 'ATK': 1}
reverse_entity_mapping = {v: k for k, v in entity_mapping.items()}

# pretrained model name: (model class, model tokenizer, output dimension, token style)
MODELS = {
    'bert-base-uncased': (AutoModel, AutoTokenizer, 768, 'bert'),
    'bert-large-uncased': (AutoModel, AutoTokenizer, 1024, 'bert'),
    'roberta-base': (AutoModel, AutoTokenizer, 768, 'roberta'),
    'roberta-large': (AutoModel, AutoTokenizer, 1024, 'roberta'),
    'xlm-roberta-base': (AutoModel, AutoTokenizer, 768, 'roberta'),
    'xlm-roberta-large': (AutoModel, AutoTokenizer, 1024, 'roberta'),   
}
