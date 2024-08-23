import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
from torch import nn
import torch

import os
import math
import re
from itertools import groupby
import pandas as pd


class GreedyCharacterDecoder(nn.Module):
    ''' Greedy CTC decoder - Argmax logits and remove duplicates. '''
    def __init__(self):
        super(GreedyCharacterDecoder, self).__init__()

    def forward(self, x):
        #print(x.shape)
        indices = torch.argmax(x, dim=-1)
        uncollapsed_indices = indices
        indices = torch.unique_consecutive(indices, dim=-1)
        return indices.tolist(), uncollapsed_indices.tolist()
        
class CTCGreedyCharacterDecoder(nn.Module):
    ''' Greedy CTC decoder - Argmax logits and remove duplicates. '''
    def __init__(self):
        super(CTCGreedyCharacterDecoder, self).__init__()

    def forward(self, x, input_lengths):
        #print(x.shape)
        #indices = torch.argmax(x, dim=-1)
        ctc_collapsed_list = []
        ctc_uncollapsed_list = []
        for i in range(x.shape[0]):
            unpadded_input = x[i,:input_lengths[i],:]
            #print(correct_input.shape)
            argmax_preds = torch.argmax(unpadded_input, dim=-1)
            #print(correct_indices.shape)
            ctc_uncollapsed_list.append(argmax_preds.tolist())
            ctc_collapsed_preds = torch.unique_consecutive(argmax_preds)
            ctc_collapsed_list.append(ctc_collapsed_preds.tolist())
        return ctc_collapsed_list, ctc_uncollapsed_list

def int_to_text_with_blank(labels, blank, char_list):
    ''' Map integer sequence to text string '''
    string = []
    # using groupby() + list comprehension
    # removing consecutive duplicates
    for i in labels:
        if i == blank:# blank char
            string.append("*")
            continue
        else:
            string.append(char_list[i])
    return ''.join(string)

def int_to_text(labels, blank, char_list):
    ''' Map integer sequence to text string without blanks'''
    string = []
    for i in labels:
        if i == blank: # blank char
            #string.append('*') #TRIAL FOR CEM
            continue
        else:
            string.append(char_list[i])
    return ''.join(string)

    
def load_char_dict(checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        raise 'Checkpoint does not exist'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint['char_dict']

def load_model_inference(encoder, decoder, checkpoint_path, device):
    ''' Load model for inference '''
    if not os.path.exists(checkpoint_path):
        raise 'Checkpoint does not exist'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return checkpoint['epoch'], checkpoint['valid_loss'], checkpoint["wer"]
