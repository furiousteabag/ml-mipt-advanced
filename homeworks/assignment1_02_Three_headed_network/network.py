
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        
        self.title_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.title_conv =  nn.Conv1d(in_channels=hid_size, out_channels=hid_size * 2, kernel_size=3)
        self.title_relu = nn.ReLU()
        self.title_pool = nn.AdaptiveMaxPool1d(1)
        self.title_flatten = nn.Flatten()
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_conv =  nn.Conv1d(in_channels=hid_size, out_channels=hid_size * 2, kernel_size=3)
        self.full_relu = nn.ReLU()
        self.full_pool = nn.AdaptiveMaxPool1d(1)
        self.full_flatten = nn.Flatten()
        
        self.cat_linear = nn.Linear(in_features=n_cat_features, out_features=hid_size * 2)
        self.cat_relu = nn.ReLU()
        
        self.linear = nn.Linear(in_features=3 * hid_size * 2, out_features=1)
        

    def forward(self, whole_input):
        
        input1, input2, input3 = whole_input
        
        title = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_conv(title)
        title = self.title_relu(title)
        title = self.title_pool(title)
        title = self.title_flatten(title)
        
        full = self.full_emb(input2).permute((0, 2, 1))
        full = self.full_conv(full)
        full = self.full_relu(full)
        full = self.full_pool(full)
        full = self.full_flatten(full)
        
        category = self.cat_linear(input3)
        category = self.cat_relu(category)
        
        concatenated = torch.cat([title, full, category], dim=1)
        
        out = self.linear(concatenated)
        
        return out

