import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_bert import *
from transformers.modeling_roberta import *

import math

class BertForSequenceClassificationUserDefined(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.classifier_2 = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()
        self.output_emebedding = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, e1_pos=None, e2_pos=None, w=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        e_pos_outputs = []
        sequence_output = outputs[0]
        for i in range(0, len(e1_pos)):
            e1_pos_output_i = sequence_output[i, e1_pos[i].item(), :]
            e2_pos_output_i = sequence_output[i, e2_pos[i].item(), :]
            e_pos_output_i = torch.cat((e1_pos_output_i, e2_pos_output_i), dim=0)
            e_pos_outputs.append(e_pos_output_i)
        e_pos_output = torch.stack(e_pos_outputs)
        self.output_emebedding = e_pos_output #e1&e2 cancat output

        e_pos_output = self.dropout(e_pos_output)
        hidden = self.classifier(e_pos_output)
        logits = self.classifier_2(hidden)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = 0
                for i in range(len(w)):
                    loss += math.exp(w[i]-1) * loss_fct(logits[i].view(-1, self.num_labels), labels[i].view(-1))
                    #loss += w[i] * loss_fct(logits[i].view(-1, self.num_labels), labels[i].view(-1))
                loss = loss / len(w)
            outputs = (loss, ) + outputs + (self.output_emebedding,)

        return outputs  # (loss), logits, (hidden_states), (attentions), (self.output_emebedding)


# f_theta1
class RelationClassificationBERT(BertForSequenceClassificationUserDefined):
    def __init__(self, config):
        super().__init__(config)


# g_theta2
class LabelGeneration(BertForSequenceClassificationUserDefined):
    def __init__(self, config):
        super().__init__(config)








class RobertaForSequenceClassificationUserDefined(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"
    def __init__(self, config):
        print('Initing')
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = RobertaModel(config)
        print('roberta config')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.classifier_2 = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()
        self.output_emebedding = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, e1_pos=None, e2_pos=None, w=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        e_pos_outputs = []
        sequence_output = outputs[0]
        for i in range(0, len(e1_pos)):
            e1_pos_output_i = sequence_output[i, e1_pos[i].item(), :]
            e2_pos_output_i = sequence_output[i, e2_pos[i].item(), :]
            e_pos_output_i = torch.cat((e1_pos_output_i, e2_pos_output_i), dim=0)
            e_pos_outputs.append(e_pos_output_i)
        e_pos_output = torch.stack(e_pos_outputs)
        self.output_emebedding = e_pos_output #e1&e2 cancat output

        e_pos_output = self.dropout(e_pos_output)
        hidden = self.classifier(e_pos_output)
        logits = self.classifier_2(hidden)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = 0
                for i in range(len(w)):
                    loss += math.exp(w[i]-1) * loss_fct(logits[i].view(-1, self.num_labels), labels[i].view(-1))
                    #loss += w[i] * loss_fct(logits[i].view(-1, self.num_labels), labels[i].view(-1))
                loss = loss / len(w)
            outputs = (loss, ) + outputs + (self.output_emebedding,)

        return outputs  # (loss), logits, (hidden_states), (attentions), (self.output_emebedding)


# f_theta1
class RelationClassificationRoBERTa(RobertaForSequenceClassificationUserDefined):
    def __init__(self, config):
        super().__init__(config)
