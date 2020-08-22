#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """Highway Network for cnn based encoder
       using single gate based on Sigmoid and 
       ReLu activation
    """

    def __init__(self, embed_size):
        """
        @param embed_size(int): Embedding size for words
        """
        super(Highway, self).__init__()
        self.proj_layer = nn.Linear(embed_size, embed_size, bias=True)
        self.gate_layer = nn.Linear(embed_size, embed_size, bias=True)


    def forward(self, X_conv_out: torch.Tensor) -> torch.Tensor:
        """
        @param X_conv_out input(Tensor): shape (max_sentence_length, batch_size, embed_size)
        @param X_highway output(Tensor): shape (max_sentence_length, batch_size, embed_size)
        """
        x_proj = F.relu(self.proj_layer(X_conv_out))
        x_gate = torch.sigmoid(self.gate_layer(X_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * X_conv_out
        return x_highway
   


