#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
#from model.ASTGCN_embedded import make_model

from lib.data_loader import data_loader
from lib.utils import *
from lib.config_parser import *
from lib.prepareData import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import  TensorDataset, DataLoader, Dataset

import pathlib
import json


parser = argparse.ArgumentParser()


## conffile_name
parser.add_argument("--config", default='Tmap.conf', type=str,
                    help="configuration file name")

## model name
parser.add_argument("--model", default='ASTGCN', type=str,
                    help="configuration file name")

## log folder를 지울지 말지 결정하는 bool
parser.add_argument("--force", default = False, type = bool,
                    help = "configuration file name")
args = parser.parse_args()

force = args.force

## conf_file dir
base_conf_dir = './configurations'

config_path = os.path.join(base_conf_dir,args.config)

config = configparser.ConfigParser()


print('Read configuration file: %s' % (args.config))
print('Running model is %s' % (args.model))

config_dict = EnvManager.back_bone(config_path, args.config, args.force)


## config dict
log_config = config_dict['log']
data_config = config_dict['data']
model_back_bone = config_dict['model']
training_config = config_dict['training']


## data_config ##

''' data setttng 
    
    -- file directory
    -- data type
    -- data_setting

'''


data_dir = data_config['dir']['data_dir']
adj_mat = data_config['dir']['adjmat_dir']
feature_type = data_config['type']['feature_type']
adj_type = data_config['type']['adj_type']

if feature_type == 'M':
    from model.ASTGCN_embedded import make_model
else:
    from moodel.ASTGCN import make_model

    

num_of_weeks = data_config['setting']['num_of_weeks']
num_of_days = data_config['setting']['num_of_days']
num_of_hours = data_config['setting']['num_of_hours']
num_for_predict = data_config['setting']['num_for_predict']
points_per_hour = data_config['setting']['points_per_hour']

## training_config ##


''' training setttng 
    
    -- cpu / single_gpu / multi-gpu/
    -- learning_rate_base
    -- optimizer
    -- batchsize
    -- epoch
    -- learning_rate_scheduler
'''

Device_type = training_config['Device_info']['Device_type']
GPU_COUNT = training_config['Device_info']['GPU_COUNT']
gpu_number = training_config['Device_info']['ctx']

optimizer = training_config['Training_parameter']['optimizer']
lr_scheduler = training_config['Training_parameter']['lr_scheduler']
loss = training_config['Training_parameter']['loss']
lr = training_config['Training_parameter']['learning_rate']
epochs = training_config['Training_parameter']['epochs']
batch_size = training_config['Training_parameter']['batch_size']
  
sw = SummaryWriter(log_dir=log_config['summary_log_dir'], flush_secs=5)


def train_epoch(train_loader, batch_size, global_step, epochs):
    """
    Trains one epoch with the given data.
    
    :param training_loader: train_data_loader -- iterator fir (train_week, day, hour, target)
    :param batch_size: Batch size to use during training.
    :global_step : iterations 
    :return: Average loss for this epoch.
    
    """
    
    epoch_training_losses = []
    
    train_loader_len = len(train_loader)
    print('len', train_loader_len)
    for  train_week, train_day, train_hour, train_target  in (train_loader):
         
        
        optimizer.zero_grad()
        
        train_week = train_week.float().type(torch.FloatTensor).to(DEVICE)
        train_day = train_day.float().type(torch.FloatTensor).to(DEVICE)
        train_hour = train_hour.float().type(torch.FloatTensor).to(DEVICE)
        train_target = train_target.float().type(torch.FloatTensor).to(DEVICE)
        out = net([train_week,train_day, train_hour])        
        loss = loss_criterion(out, train_target)
        
        if global_step % 100 == 0:
            
           print('training batch %s / %s, loss: %.2f' % ((global_step)/(epochs+1) + 1, train_loader_len, loss.item()))

        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        
        mean_loss = sum(epoch_training_losses)/len(epoch_training_losses)
        
        sw.add_scalar(tag='batch_training_loss',
              scalar_value = mean_loss,
              global_step=global_step)
        
        global_step += 1
        
    return sum(epoch_training_losses)/len(epoch_training_losses), global_step
    
    


if __name__ == '__main__':
    
    num_of_vertices = data_config['setting']['num_of_vertices']
    adj_mx, distance_mx = get_adjacency_matrix(adj_mat, num_of_vertices, id_filename = None)
    
    if Device_type == 'GPU':
    
        print('gpu',GPU_COUNT)

        if GPU_COUNT == 1:
            DEVICE = torch.device('cuda:%d' % (gpu_number))
            #DEVICE = torch.device('cuda:0')
            print("CUDA:", DEVICE)

            if adj_type == 'D':

                net = make_model(model_back_bone, distance_mx)

            else:

                net = make_model(model_back_bone, adj_mx)

            net.to(DEVICE)
            
    loss_criterion = nn.MSELoss()
    
    assert optimizer in ['SGD', 'Adam', 'RMSprop'], '%s optimizer can not be found' % (optimizer)
    assert lr_scheduler in ['Step_wise', 'exponential_decay', 'Cos_lr'],  '%s lr scheduler can not be found' % (lr_scheduler)
    
    print('%s optimizer, %s lr_scheduler is selcted' %(optimizer, lr_scheduler))  
    
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=0.9)
                 
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr = lr) 
                                     
    elif optimizer =='RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr = lr)                                  
    
    if lr_scheduler =='Step_wise':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.7, last_epoch = -1)
                                     
    elif lr_scheduler =='exponential_decay':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95, last_epoch = -1)
    
    all_data = read_and_generate_dataset(data_dir, feature_type, num_of_weeks, num_of_days, num_of_hours, num_for_predict,\
                                         points_per_hour=points_per_hour, save=True)
    
    
    train_loader, val_loader, test_loader, stats = data_loader(all_data, batch_size, shuffle = True)
    
    week_stats = stats['week']
    day_stats = stats['day']
    hour_stats = stats['hours']
    
    training_losses = []
    val_losses = []
    test_losses = []
    
    global_step = 1 
    best_epoch = 0
    best_val_loss = np.inf
     
    
    for i in range(epochs):
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print('epoch: {:3d}, lr={:.6f}'.format(epochs, lr))
        scheduler.step()
               
        train_loss, global_step = train_epoch(train_loader, batch_size=batch_size, global_step = global_step, epochs = i)
                                
        print('%d epoch_training loss %f' % (i, train_loss))
                
        val_loss = compute_val_loss_astgcn(net, val_loader, loss_criterion, sw, i, DEVICE)
        
        print('%d epoch val_loss is : %f' %(i, val_loss))
        
        sw.add_scalar(tag='epoch: %d validation_loss' %(i),
              scalar_value = val_loss,
              global_step = i)
                
        if val_loss < best_val_loss:
            
            file_name = '{}_epoch_best_params.pt'.format(i)
   
            if os.path.exists(log_config['best_params_dir'] ):
                shutil.rmtree(log_config['best_params_dir'] )
                print('Params folder {} exists! and removing'.format(log_config['best_params_dir'] ))
            pathlib.Path(log_config['best_params_dir']).mkdir(parents = True, exist_ok=True)
            
            best_val_loss = val_loss
            best_epoch = i
            
            torch.save({
                
                'epoch': i,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            
            }, log_config['best_params_dir'] +'/' + file_name)

