# LADDER

This repository contains implementation for the paper "Looking Beyond IoCs: Automatically Extracting Attack Patterns from External CTI" accepted at RAID, 2023


### How to Run

#### Attack Pattern Extraction
Change directory to `attack_pattern`  
Run `pip install -r requirements.txt` to install the dependencies in a virtual environment

To train the sentence classification model: `python finetune_sentence_classification.py --save-path=logs/sentence_classification`  
Edit the `params_dict` values in the python file to fine tune other combination of hyperparameters.

To train the entity extraction model: `python finetune_entity_extraction.py --save-path=logs/entity_extraction`  
Edit the `params_dict` values in the python file to fine tune other combination of hyperparameters.


#### Named Entity Recognition
Change directory to `ner`  
Run `pip install -r requirements.txt` to install the dependencies in a virtual environment

To train NER model run: `python train_ner.py`
cfgEdit `cfg` values in the python file to fine tune other combination of hyperparameters.

`infer.py` shows examples on how to create annotation in BRAT format with trained model.


#### Relation Extraction
Change directory to `relation_extraction`  
Run `pip install -r requirements.txt` to install the dependencies in a virtual environment

To train Relation Extraction model run: `python train_supervised.py`  

### Demo
Open the notebook in `notebooks/attack-pattern-extraction.ipynb` in Google colab for demo on attack pattern extraction and mapping from CTI texts.   
Open the notebook in `notebooks/malware-similarity.ipynb` in Google colab for malware similarity analysis from all triples
 

### Known Issues
- [ ] Demo notebooks not working proprely due to trnasformer version issue