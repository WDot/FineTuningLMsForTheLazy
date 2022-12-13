#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
import numpy as np
import os
import os.path
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
from lmdata import LMData

DATA_PATH = 'example.txt'
MODEL = 'gpt2-medium'
EXP_NAME = 'lmexample1'
K=256

def tokenize_all_data(textPath,model_name,kernel_size):
    with open(textPath) as x: text = x.read()
    tokens = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for i in range(LMData.get_length(text,kernel_size,kernel_size)):
        cur_text = LMData.get_independent_window(text,i,kernel_size,kernel_size)
        cur_tokens = tokenizer(cur_text)
        tokens.extend(cur_tokens['input_ids'])
    tokens = np.array(tokens)
    return tokens

tokens = tokenize_all_data(DATA_PATH,MODEL,K)
np.savez_compressed('tokens.npz',tokens=tokens)