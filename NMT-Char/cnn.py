#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, 
                 char_embed_size:int,
                 word_embed_size:int,
                 kernel_size:int = 5):
        super(CNN, self).__init__()

        self.conv_layer = nn.Conv1d(in_channels=char_embed_size, out_channels=word_embed_size, kernel_size=kernel_size)
        # word_embed_size is equal to number of filters used is same as number of channels in the output.

    def forward(self, X_reshaped: torch.Tensor) -> torch.Tensor:
        """
        @param X_reshaped: a tensor with shape(batch_size, char_embed_size, max_word_length)
                           reshape done since pytorch applies convolutions only to last dimension
        @return X_out : a tensor of shape char_embedding_size
            """ 
        X_conv = self.conv_layer(X_reshaped) # shape: (batch_size, char_embed_size, max_word_length-kernel_size + 1)
        # Max word length is getting converted to word_embed_size
        X_out = torch.max(torch.relu(X_conv), dim=2) # we are using maxpooling through time
        return  X_out[0]
        


