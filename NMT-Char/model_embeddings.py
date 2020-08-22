#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.char_embed_size = 50
        self.embed_size = embed_size
        pad_token_idx = vocab.char2id['<pad>']
        self.char_embedding = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=pad_token_idx)
        self.cnn=CNN(char_embed_size=self.char_embed_size, word_embed_size=embed_size, kernel_size=5)
        self.highway = Highway(embed_size=embed_size)
        self.dropout = nn.Dropout(0.3)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code
        ### YOUR CODE HERE for part 1f
        X_word_embed_list = []
        # for every sentence length
        for X_padded in input_tensor:
            X_embed = self.char_embedding(X_padded)
            #print(X_embed.shape)
            X_reshaped = X_embed.permute(0,2,1)
            #print(X_reshaped.shape)
            X_conv_out = self.cnn(X_reshaped)
            X_highway = self.highway(X_conv_out)
            X_word_embed = self.dropout(X_highway)
            X_word_embed_list.append(X_word_embed)

        X_word_emb_out = torch.stack(X_word_embed_list)
        return X_word_emb_out

   
        ### END YOUR CODE
