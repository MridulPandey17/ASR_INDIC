#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#FILL IN @ SYMBOLS

# ML Libraries
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate
from torch.cuda.amp import autocast, GradScaler


# Support Libraries
import math
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
import gc
from itertools import groupby

# Python Scripts
from conf_model import ConformerEncoder, LSTMDecoder
from conf_utils import *

total_devices = torch.cuda.device_count()
print(f"Available GPU Devices : {total_devices}")
DEVICES = [i for i in range(0,total_devices)]
print(f"Using devices : {DEVICES}")
DEVICE = f"cuda:{DEVICES[0]}"

TRAIN_BS = 400 # need to set. prev - 800
TEST_BS = 204 
EPOCHS = 1000
NUM_WORKERS = 4*len(DEVICES)
start_epoch = 0
best_wer = float('inf')
ACCUM_ITER = 1

train_tsv = "/nlsasfs/home/nltm-st/akankss/thish/datasets/madasr23/bhojpuri/train_bh.tsv"
dev_tsv = "/nlsasfs/home/nltm-st/akankss/thish/datasets/madasr23/bhojpuri/dev_bh.tsv"

metadata_train = pd.read_csv(train_tsv, sep = '\t', header = None)
metadata_dev = pd.read_csv(dev_tsv, sep = '\t', header = None)

metadata_train = metadata_train[metadata_train[2]<=16].reset_index(drop=True)
metadata_train = metadata_train[metadata_train[2]>=2].reset_index(drop=True)


metadata_train = metadata_train.sample(frac=1).reset_index(drop=True)



metadata_dev = metadata_dev[metadata_dev[2]<=16].reset_index(drop=True)
metadata_dev = metadata_dev[metadata_dev[2]>=2].reset_index(drop=True)


metadata_dev = metadata_dev.sample(frac=1).reset_index(drop=True)


metadata_train[0] = "/nlsasfs/home/nltm-st/akankss/thish/datasets/madasr23/bhojpuri/train/" + metadata_train[0]

metadata_dev[0] = "/nlsasfs/home/nltm-st/akankss/thish/datasets/madasr23/bhojpuri/dev/" + metadata_dev[0]

print(metadata_dev)
print(metadata_train)

# In[ ]:


feat_dict = {
        "sample_rate":16000,
        "n_mels":80, #as per reference
    }
# Removing masks for Hindi data
#time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)]
#train_transform = nn.Sequential(
#    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160), #80 filter banks, 25ms window size, 10ms hop
#    torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
#    *time_masks,
#  )

train_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)
validation_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)


# In[ ]:


char_set = [' ', '.', 'ँ', 'ं', 'ः', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ऑ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल', 'व', 'श', 'ष', 'स', 'ह', '़', 'ऽ', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॅ', 'े', 'ै', 'ॉ', 'ो', 'ौ', '्', 'क़', 'ख़', 'ज़', 'ड़', 'ढ़', 'फ़']
char_list = sorted(char_set)
num_embeddings = len(char_list)
print(num_embeddings)
char_ids = [int(i) for i in range(num_embeddings)]
char_dict = dict(zip(char_list,char_ids))
print(f"Character mappings : {char_dict}")


# In[ ]:




trainset = MyDataset(metadata_train, char_dict, train_transform)
devset = MyDataset(metadata_dev, char_dict, validation_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = TRAIN_BS, shuffle = True, collate_fn = collate_batch, drop_last = True, num_workers = NUM_WORKERS, pin_memory = True)
devloader = torch.utils.data.DataLoader(devset, batch_size = TEST_BS, shuffle = True, collate_fn = collate_batch, drop_last = True, num_workers = NUM_WORKERS, pin_memory = True)

# In[ ]:


encoder_params = {
    "d_input": 80,
    "d_model": 144,
    "num_layers": 16,
    "conv_kernel_size": 32,
    "dropout": 0.1,
    "num_heads": 4
}

decoder_params = {
    "d_encoder": 144,
    "d_decoder": 320,
    "num_layers": 1,
    "num_classes":len(char_dict)+1
}


# In[ ]:


encoder = ConformerEncoder(
                      d_input=encoder_params['d_input'],
                      d_model=encoder_params['d_model'],
                      num_layers=encoder_params['num_layers'],
                      conv_kernel_size=encoder_params['conv_kernel_size'], 
                      dropout=encoder_params['dropout'],
                      num_heads=encoder_params['num_heads']
                    )
  
decoder = LSTMDecoder(
                  d_encoder=decoder_params['d_encoder'], 
                  d_decoder=decoder_params['d_decoder'], 
                  num_layers=decoder_params['num_layers'],
                  num_classes= decoder_params['num_classes'])


# In[ ]:


encoder = encoder.to(DEVICE)
decoder = decoder.to(DEVICE)


# In[ ]:


#char_decoder =  GreedyCharacterDecoder().eval()
ctc_char_decoder = CTCGreedyCharacterDecoder().eval()
criterion = nn.CTCLoss(blank=len(char_list), zero_infinity=True)
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-5, betas=(.9, .98), eps = 1e-09, weight_decay=1e-4)
scheduler = TransformerLrScheduler(optimizer, encoder_params['d_model'], 10000)




# In[ ]:


model_size(encoder, 'Encoder')
model_size(decoder, 'Decoder')


# In[ ]:


torch.cuda.set_device(DEVICE)
criterion = criterion.to(DEVICE)
#char_decoder = char_decoder.to(DEVICE)
ctc_char_decoder = ctc_char_decoder.to(DEVICE)
torch.cuda.empty_cache()


checkpoint_load_path = "/nlsasfs/home/nltm-st/akankss/thish/expts/asru_thish/bh/ckpt_bh_all/ckpt_bh_best_wer.pt"

if(os.path.exists(checkpoint_load_path)):
    start_epoch, best_loss, best_wer = load_checkpoint(encoder, decoder, optimizer, scheduler, checkpoint_load_path)
    print(f'Resuming training from checkpoint starting at epoch {start_epoch}.')
    print(f'Current model WER : {best_wer}%')


# In[ ]:    


gc.collect()


# In[ ]:


encoder_parallel = nn.DataParallel(encoder, device_ids=DEVICES)
decoder_parallel = nn.DataParallel(decoder, device_ids=DEVICES)


# In[ ]:


def train(encoder, decoder, ctc_char_decoder, optimizer, scheduler, criterion, grad_scaler, train_loader, device): 
    wer = WordErrorRate()
    cer = CharErrorRate()
    accum_iter = ACCUM_ITER #ADDED  
    encoder.train()
    decoder.train()
    avg_loss = 0
    avg_wer = 0
    avg_cer = 0
    batch_count = 0
    for batch in tqdm(train_loader):
        batch_count += 1
        scheduler.step()
        gc.collect()
        spectrograms, labels, input_lengths_list, label_lengths, references, mask = batch 

        spectrograms = spectrograms.squeeze(1).to(device)
        labels = labels.to(device)
        input_lengths = torch.tensor(input_lengths_list).to(device)
        label_lengths = torch.tensor(label_lengths).to(device)
        mask = mask.to(device)
    
        outputs, _ = encoder(spectrograms, mask)
        outputs, _ = decoder(outputs)
        loss = criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)
        #loss.backward()
        loss = loss / accum_iter #ADDED
        

        grad_scaler.scale(loss).backward()
        #if (i+1) % args.accumulate_iters == 0:
        #use for small datasets
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
        if ((batch_count) % accum_iter == 0) or (batch_count == len(train_loader)):#ADDED
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
        #avg_loss.update(loss.detach().item())
        

#         optimizer.step()
#         optimizer.zero_grad()
        
        avg_loss += loss.detach().item()
        inds, _ = ctc_char_decoder(outputs.detach(), input_lengths_list)
        #inds = char_decoder(outputs.detach())
        # print("train shape:",inds.shape)
        predictions = []
        for sample in inds:
            #print(sample.shape)
            predictions.append(int_to_text(sample, len(char_list), char_list))
        avg_wer += wer(predictions, references) * 100
        avg_cer += cer(predictions, references) * 100

    avg_loss = avg_loss/batch_count
    avg_wer = avg_wer/batch_count
    avg_cer = avg_cer/batch_count
    print(f'Avg WER: {avg_wer}%, Avg Loss: {avg_loss}')  
    for i in range(5):
        print('Prediction: ', predictions[i])
        print('Reference: ', references[i])
    
    # Print metrics and predictions 
    del spectrograms, labels, input_lengths, label_lengths, references, outputs, inds, predictions
    return avg_wer, avg_cer, avg_loss
    
def validate(encoder, decoder, ctc_char_decoder, criterion, test_loader, device):
    ''' Evaluate model on test dataset. '''

    wer = WordErrorRate()
    cer = CharErrorRate()
        
    avg_loss = 0
    avg_wer = 0
    batch_count = 0
    avg_cer = 0
    
    encoder.eval()
    decoder.eval()
    for batch in tqdm(test_loader):
        gc.collect()
        batch_count += 1
        spectrograms, labels, input_lengths_list, label_lengths, references, mask = batch 
  
    # Move to GPU
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        input_lengths = torch.tensor(input_lengths_list).to(device)
        label_lengths = torch.tensor(label_lengths).to(device)
        mask = mask.to(device)

        with torch.no_grad():
            outputs, _ = encoder(spectrograms, mask)
            outputs, _ = decoder(outputs)
            loss = criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)
            avg_loss += loss.item()

#            inds = char_decoder(outputs.detach())
            inds, _ = ctc_char_decoder(outputs.detach(), input_lengths_list)
            # print("validation shape:", inds.shape)
            predictions = []
            for sample in inds:
                predictions.append(int_to_text(sample, len(char_list), char_list))

            avg_wer += wer(predictions, references) * 100
            avg_cer += cer(predictions, references) * 100
    print(".............................TEST PREDICTIONS...........................")
    for i in range(5):
        print('Prediction: ', predictions[i])
        print('Reference: ', references[i])
    print("************************************************************************")
    
    return avg_wer/batch_count, avg_cer/batch_count, loss/batch_count 


# In[ ]:


checkpoint_save_path = "/nlsasfs/home/nltm-st/akankss/thish/expts/asru_thish/bh/ckpt_bh_all/ckpt_bh_curent_epoch.pt"
best_wer_save_path = "/nlsasfs/home/nltm-st/akankss/thish/expts/asru_thish/bh/ckpt_bh_all/ckpt_bh_best_wer.pt"
best_loss_save_path = "/nlsasfs/home/nltm-st/akankss/thish/expts/asru_thish/bh/ckpt_bh_all/ckpt_bh_best_loss.pt"


# In[ ]:


best_loss = float('inf')

optimizer.zero_grad()

use_amp = True
variational_noise_std = 0.0001

# Mixed Precision Setup
if use_amp:
    print('Using Mixed Precision')
grad_scaler = GradScaler(enabled=use_amp)

optimizer.zero_grad()
for epoch in range(start_epoch, EPOCHS):
    print(f"Epoch : {epoch+1}/{EPOCHS}")
    torch.cuda.empty_cache()
    
    #variational noise for regularization
    #add_model_noise(encoder, std=variational_noise_std, gpu=True)
    #add_model_noise(decoder, std=variational_noise_std, gpu=True)

    
    # Train/Validation loops
    wer, cer, loss = train(encoder_parallel, decoder_parallel, ctc_char_decoder, optimizer, scheduler, criterion, grad_scaler, trainloader, DEVICE) 
    #wer, cer, loss = train(encoder, decoder, char_decoder, optimizer, scheduler, criterion, grad_scaler, trainloader, DEVICE) 
    valid_wer, valid_cer, valid_loss = validate(encoder, decoder, ctc_char_decoder, criterion, devloader, DEVICE)
    print(f'Epoch {epoch} - Valid WER: {valid_wer}%, Valid CER: {valid_cer}%, Valid Loss: {valid_loss}, Train WER: {wer}%, Train CER: {cer}%, Train Loss: {loss}')  
    
    # Save best model
    if valid_loss <= best_loss:
        print('Validation loss improved, saving best model.')
        best_loss = valid_loss
        save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch+1, best_loss_save_path, valid_wer, valid_cer, char_dict)
    
    if epoch%5==0:
        print(f'Saving checkpoint at epoch:{epoch+1}')
        save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch+1, checkpoint_save_path, valid_wer, valid_cer, char_dict)
    
    if valid_wer <= best_wer:
        print(f"Validation WER improved, saving best model.")
        best_wer = valid_wer
        save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch+1, best_wer_save_path, valid_wer, valid_cer, char_dict)

