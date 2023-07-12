import cyner
import os

model = cyner.CyNER(transformer_model='xlm-roberta-base', use_heuristic=True, flair_model=None, spacy_model=None, dictionary=None, priority='DHTFS')


model.create_brat_annotation('sample_input.txt', 'sample_output.txt')
