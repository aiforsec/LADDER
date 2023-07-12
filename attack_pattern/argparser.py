import argparse


def parse_entity_recognition_arguments():
    parser = argparse.ArgumentParser(description='Entity recognition')
    parser.add_argument('--name', default='entity_recognition', type=str, help='name of run')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, help='pretrained language model')
    parser.add_argument('--freeze-bert', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Freeze BERT layers or not')
    parser.add_argument('--lstm-dim', default=-1, type=int,
                        help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
    parser.add_argument('--dataset', default='data/entity_extraction', type=str,
                        help='data directory containing train/dev/test files')
    parser.add_argument('--sequence-length', default=256, type=int,
                        help='sequence length to use when preparing dataset (default 256)')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='weight decay (default: 0)')
    parser.add_argument('--gradient-clip', default=-1, type=float, help='gradient clipping (default: -1 i.e., none)')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 32)')
    parser.add_argument('--epoch', default=30, type=int, help='total epochs (default: 30)')
    parser.add_argument('--save-path', default='out/entity_extraction', type=str, help='model and log save directory')

    args = parser.parse_args()
    return args


def parse_bert_sentence_classification_arguments():
    parser = argparse.ArgumentParser(description='Sentence classification')
    parser.add_argument('--task', default='atk-pattern', type=str, help='Task name')
    parser.add_argument('--dataset', default='data/sentence_classification', type=str,
                        help='data directory containing train/dev/test files')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--decay', default=0., type=float, help='weight decay')
    parser.add_argument('--model', default="bert-base-uncased", type=str, help='pretrained BERT model name')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 64)')
    parser.add_argument('--epoch', default=10, type=int, help='total epochs (default: 200)')
    parser.add_argument('--fine-tune', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to fine-tune embedding or not')
    parser.add_argument('--save-path', default='out/sentence_classification', type=str, help='output log/result directory')
    args = parser.parse_args()
    return args


def parse_inference_arguments():
    parser = argparse.ArgumentParser(description='Inference on document')
    parser.add_argument('--entity-extraction-model', default="bert-base-uncased", type=str,
                        help='transformer model for entity extraction')
    parser.add_argument('--sequence-length-entity', default=256, type=int,
                        help='sequence length for model')
    parser.add_argument('--sentence-classification-model', default="bert-base-uncased", type=str,
                        help='transformer model for sentence classification')
    parser.add_argument('--sequence-length-sentence', default=256, type=int,
                        help='sequence length for model')
    parser.add_argument('--entity-extraction-weight', default="out/entity_extraction/weights.pt", type=str,
                        help='saved weight for entity extraction model')
    parser.add_argument('--sentence-classification-weight', default="out/sentence_classification/weights.pt", type=str,
                        help='saved weight for sentence classification model')
    parser.add_argument('--input-doc', default="samples/1.txt", type=str, help='input document path')
    parser.add_argument('--save-path', default='samples/1_predict.txt', type=str, help='path for output document')
    args = parser.parse_args()
    return args


def parse_word_embedding_sentence_classification_args():
    parser = argparse.ArgumentParser(description='Sentence classification using word embedding')
    parser.add_argument('--task', default='atk-pattern', type=str, help='Task name')
    parser.add_argument('--w2v-file', default=None, type=str, help='word embedding file')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--decay', default=0., type=float, help='weight decay')
    parser.add_argument('--model', default="TextCNN", type=str, help='model type (default: TextCNN)')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--batch-size', default=50, type=int, help='batch size (default: 128)')
    parser.add_argument('--epoch', default=50, type=int, help='total epochs (default: 200)')
    parser.add_argument('--fine-tune', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to fine-tune embedding or not')
    parser.add_argument('--save-path', default='out/sentence_classification', type=str, help='output directory')
    parser.add_argument('--num-runs', default=3, type=int, help='number of runs')
    args = parser.parse_args()
    return args

