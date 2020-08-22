#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, 
                                        padding_idx=target_vocab.char2id['<pad>'])
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        self.target_vocab = target_vocab



    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        #print('input shape', input.shape)
        input_embed = self.decoderCharEmb(input) # shape: (length, batch, self.hidden_size)
        #print('input_embed shape', input_embed.shape)
        outputs, (h_n, c_n) = self.charDecoder(input_embed, dec_hidden) # shape: (length, batch, hidden_size)
        s_t = self.char_output_projection(outputs) # shape: (length, batch, vocab_size)
        #print('s_t shape', s_t.shape)
        return s_t, (h_n, c_n)


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        #print('Char sequence shape', char_sequence.shape)
        s_t, (h_n, c_n) = self.forward(char_sequence[:-1], dec_hidden)
        #print('shape s_t', s_t.shape)
        #print('h_n shape', h_n.shape)
        #print('c_n shape', c_n.shape)
        loss_func = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        lsfn_input = s_t.reshape(-1, len(self.target_vocab.char2id))
        lsfn_target = char_sequence[1:].contiguous().view(-1)
        loss = loss_func(lsfn_input, lsfn_target)
        return loss

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        h_prev, c_prev = initialStates
        batch_size = initialStates[0].shape[1]
        output_words_tensor = []
        decodedWords = ['{'] * batch_size
        start_index = self.target_vocab.start_of_word
        end_index = self.target_vocab.end_of_word
        curr_char_tensor = torch.tensor([start_index] * batch_size, device=device)
        #print(curr_char_tensor.shape)
        for _ in range(max_length):
            s_t, (h_new, c_new) = self.forward(curr_char_tensor.unsqueeze(0), (h_prev,c_prev)) #shape of s_t (1,b,v)
            #s_t, (h_t, c_t) = self.forward(curr_char_tensor, (h_t,c_t)) #shape of s_t (1,b,v)
            #print(s_t.shape)
            score = self.char_output_projection(h_new.squeeze(0)) # (b,v)
            probabilities = torch.softmax(score, dim=1)
            #curr_char_tensor = s_t.argmax(2) #shape (1,b)
            curr_char_tensor = torch.argmax(probabilities, dim=1)
            curr_char_tensor_t = curr_char_tensor 
            #print(curr_char_tensor_t)
            for idx in range(batch_size):
                decodedWords[idx] += self.target_vocab.id2char[curr_char_tensor_t[idx].item()]
                #print(self.target_vocab.id2char[curr_char_tensor_t[idx].item()])
            h_prev = h_new
            c_prev = c_new
            #print(decodedWords)
        for idx in range(batch_size):
            decodedWords[idx] = decodedWords[idx][1:] # removing start character
            decodedWords[idx] = decodedWords[idx].split('}')[0] # removing the end character

        return decodedWords 
          
