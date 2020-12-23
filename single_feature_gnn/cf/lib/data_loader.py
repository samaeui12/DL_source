# -*- coding:utf-8 -*-

import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader, Dataset

class TmapDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, input_data, target_data):
        'Initialization'
        self.input_data = input_data 
        self.target_data = target_data

  def __len__(self):
        'Denotes the total number of samples'
        
        return self.input_data.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        

        # Load data and get label
        X = self.input_data[index,:,:,:]
        y = self.target_data[index,:,:]

        return X, y

def data_loader(all_data, batch_size, shuffle = True):
    '''

   
    : config_dict : data를 가져오기위한 configuration file
    : all_data : dict 형태의 모든 데이터 (data_prepare.py 의 return 값)
    : param DEVICE :
    : param batch_size: int
    
    
    return:
    
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''
        
    train_x = all_data['train']['x']
    train_target = all_data['train']['target'] 
    
    print(train_x.shape)
    print(train_target.shape)
    
    val_x = all_data['val']['x']
    val_target = all_data['val']['target']

    test_x = all_data['test']['x']
    test_target = all_data['test']['target']

    mean = all_data['stats']['_mean'][0]  # (1, 1, 3, 1)
    std = all_data['stats']['_std'][0]  # (1, 1, 3, 1)
    
    
    train_dataset = TmapDataset(train_x, train_target)
    val_dataset = TmapDataset(val_x, val_target)
    test_dataset = TmapDataset(test_x, test_target)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, mean, std