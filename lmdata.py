#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
#import h5py
import numpy as np
from torch.utils.data import Dataset
#from keras.utils import to_categorical
import torch
import PIL
import json

import warnings
class LMData(Dataset):
    def __init__(self,tokens,kernel_size=256,stride=128):
        self.kernel_size = kernel_size
        self.stride = stride
        self.tokens = tokens
        
    @staticmethod
    def get_independent_window(iterable,i,kernel_size,stride):
        return iterable[stride*i:(stride*i+kernel_size)]
        
    @staticmethod
    def get_length(iterable,kernel_size,stride):
        return ((len(iterable) - kernel_size)// stride)
        
    
    def __getitem__(self,i):
        cur_tokens = LMData.get_independent_window(self.tokens,i,self.kernel_size+1,self.stride)
        X = cur_tokens[:self.kernel_size]
        Y = cur_tokens[1:(self.kernel_size+1)]
        return(X,Y)
    
    def __len__(self):
        #Usually + 1, but we want to generate two sequences every time!
        return LMData.get_length(self.tokens,self.kernel_size+1,self.stride)