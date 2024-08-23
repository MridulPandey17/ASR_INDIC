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

DEVICE = 'cuda:0'

class TransformerLrScheduler():
    '''
    Transformer LR scheduler from "Attention is all you need." https://arxiv.org/abs/1706.03762
    multiplier and warmup_steps taken from conformer paper: https://arxiv.org/abs/2005.08100
    '''
    def __init__(self, optimizer, d_model, warmup_steps, multiplier=5):
        self._optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.n_steps = 0
        self.multiplier = multiplier

    def step(self):
        self.n_steps += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        return self.multiplier * (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * (self.warmup_steps ** (-1.5)))
    
def model_size(model, name):
  #  ''' Print model size in num_params and MB'''
    param_size = 0
    num_params = 0
    for param in model.parameters():
        num_params += param.nelement()
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        num_params += buffer.nelement()
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'{name} - num_params: {round(num_params / 1000000, 2)}M,  size: {round(size_all_mb, 2)}MB')
    
class MyDataset(Dataset):
    """
    The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
    """
    def __init__(self, data_frame, char_dict, transform=None):
        self.data_frame = data_frame
        #self.root_dir = root_dir
        self.transform = transform
        #self.max_transcript_len = N
        self.char_dict = char_dict
    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_path = self.data_frame.iloc[idx, 0]
        waveform, sample_rate = torchaudio.load(file_path, normalize = True)
        if sample_rate!=16000:
            waveform = transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        mel_spec = self.transform(waveform) # (batch, n_mels, time)
        mel_spec = mel_spec.squeeze(0).transpose(0,1)  # (time, n_mels)
        mel_spec_len = ((mel_spec.shape[0] - 1) // 2 - 1) // 2
        
        transcript = self.data_frame.iloc[idx,1]
        transcript = str(transcript)
        input_transcript_list = []
        for char in transcript:
            input_transcript_list.append(self.char_dict[char])
        
        label_tensor = torch.LongTensor(input_transcript_list)
        label_len = len(input_transcript_list)
    
        return mel_spec, label_tensor, mel_spec_len, label_len, transcript
    
def collate_batch(batch):
    
    input_list, label_list, input_len_list, label_len_list, references = [], [], [], [], []

    for (_input, _label, _input_len, _label_len, _transcript) in batch:
        input_list.append(_input)
        label_list.append(_label)
        input_len_list.append(_input_len)
        label_len_list.append(_label_len)
        references.append(_transcript)
        #print(_input.shape, _label.shape, _input_len, _label_len)
    input_tens = nn.utils.rnn.pad_sequence(input_list, batch_first=True)
    labels_tens = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    
    
    mask = torch.ones(input_tens.shape[0], input_tens.shape[1], input_tens.shape[1])
    for i, l in enumerate(input_len_list):
        mask[i, :, :l] = 0
    return input_tens, labels_tens, input_len_list, label_len_list, references, mask.bool()

#class GreedyCharacterDecoder(nn.Module):
#    ''' Greedy CTC decoder - Argmax logits and remove duplicates. '''
#    def __init__(self):
#        super(GreedyCharacterDecoder, self).__init__()

#    def forward(self, x):
        #print(x.shape)
#        indices = torch.argmax(x, dim=-1)
#        indices = torch.unique_consecutive(indices, dim=-1)
#        return indices.tolist()
    
class CTCGreedyCharacterDecoder(nn.Module):
    ''' Greedy CTC decoder - Argmax logits and remove duplicates. '''
    def __init__(self):
        super(CTCGreedyCharacterDecoder, self).__init__()

    def forward(self, x, input_lengths):
        #print(x.shape)
        #indices = torch.argmax(x, dim=-1)
        indices_list = []
        uncollapsed_list = []
        for i in range(x.shape[0]):
            correct_input = x[i,:input_lengths[i],:]
            #print(correct_input.shape)
            correct_indices = torch.argmax(correct_input, dim=-1)
            #print(correct_indices.shape)
            uncollapsed_list.append(correct_indices.tolist())
            indices_list.append(torch.unique_consecutive(correct_indices).tolist())
        return indices_list, uncollapsed_list

def int_to_text(labels, blank, char_list):
    ''' Map integer sequence to text string '''
    string = []
    # using groupby() + list comprehension
    # removing consecutive duplicates
    labels = [i[0] for i in groupby(labels)]
    for i in labels:
        if i == blank: # blank char
            continue
        else:
            string.append(char_list[i])
    return ''.join(string)


def load_checkpoint(encoder, decoder, optimizer, scheduler, checkpoint_path, device=DEVICE):
    ''' Load model checkpoint '''
    if not os.path.exists(checkpoint_path):
        raise 'Checkpoint does not exist'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    scheduler.n_steps = checkpoint['scheduler_n_steps']
    scheduler.multiplier = checkpoint['scheduler_multiplier']
    scheduler.warmup_steps = checkpoint['scheduler_warmup_steps']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['valid_loss'], checkpoint['wer']

def save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch, checkpoint_path, wer, cer, char_dict):
    ''' Save model checkpoint '''
    torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'wer': wer, 
            'cer': cer,
            'scheduler_n_steps': scheduler.n_steps,
            'scheduler_multiplier': scheduler.multiplier,
            'scheduler_warmup_steps': scheduler.warmup_steps,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'char_dict': char_dict
            }, checkpoint_path)
    
def add_model_noise(model, std=0.0001, gpu=True):
#   '''
#     Add variational noise to model weights: https://ieeexplore.ieee.org/abstract/document/548170
#     STD may need some fine tuning...
#   '''
    with torch.no_grad():
        for param in model.parameters():
            if gpu:
                param.add_(torch.randn(param.size()).cuda() * std)
            else:
                param.add_(torch.randn(param.size()).cuda() * std)
    
def load_char_dict(checkpoint_path, device):
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

def load_checkpoint_finetune(encoder, decoder, checkpoint_path, device=DEVICE):
    ''' Load model checkpoint '''
    if not os.path.exists(checkpoint_path):
        raise 'Checkpoint does not exist'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    #scheduler.n_steps = checkpoint['scheduler_n_steps']
    #scheduler.multiplier = checkpoint['scheduler_multiplier']
    #scheduler.warmup_steps = checkpoint['scheduler_warmup_steps']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['valid_loss'], checkpoint['wer']
