import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
# Use torch's AdamW: transformers.AdamW is deprecated and was removed in newer versions.
from torch.optim import AdamW

class ScenarioModel(nn.Module):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size

    # task1: add necessary class variables as you wish.
    
    # task2: initilize the dropout and classify layers
    self.dropout = nn.Dropout(...)
    self.classify = Classifier(...)
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    # task1: get a pretrained model of 'bert-base-uncased'
    self.encoder = BertModel.from_pretrained(...)
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

  def forward(self, inputs, targets):
    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the output of the dropout layer to the Classifier which is provided for you.
    """
  
class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit


class CustomModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size):
    super().__init__(args, tokenizer, target_size)
    
    # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model

class SupConModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)

    # task1: initialize a linear head layer
 
  def forward(self, inputs, targets):

    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the normalized output of the dropout layer to the linear head layer; return the embedding.

    NOTE: the shared run_eval scores accuracy by argmax over class logits, so the model must
    be able to produce classification logits at eval time (see the note in supcon_train). How
    you reconcile that with this contrastive forward is your design choice.
    """
