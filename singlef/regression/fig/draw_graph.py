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
from model.ASTGCN_test import make_model

from lib.data_loader import data_loader
from lib.utils import *
from lib.config_parser import Yml_manager
from lib.prepareData2 import *
from lib.metrics import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import  TensorDataset, DataLoader, Dataset

import pathlib
import json


parser = argparse.ArgumentParser()

parser.add_argument("--config", default = 'Tmap.conf', type = str,
                    help = "configuration file name")

parser.add_argument("--model", default = 'ASTGCN', type = str,
                    help = "configuration file name")


parser.add_argument("--force", default = False, type = bool,
                    help = "configuration file name")

args = parser.parse_args()

force = args.force

base_conf_dir = './configurations'
config_path = os.path.join(base_conf_dir,args.config)

config = configparser.ConfigParser()


print('Read configuration file: %s' % (args.config))
print('Running model is %s' % (args.model))


yml_key = ['data_file', 'model_file', 'training_file','log_file']
config_obj = Yml_manager(yml_key, config_path, force)
config_obj.parsing_yml()
config_obj.processing_yml()
config_dict = config_obj.result


## config dict

log_config = config_dict['log']
data_config = config_dict['data']
training_config = config_dict['training']
model_back_bone = config_dict['model_back_bone']




## data_config ##

''' data setttng 
    
    -- file directory
    -- data type
    -- data_setting

'''


prod_data = data_config['dir']['prod_data']
real_data = data_config['dir']['real_data']
adj_mat = data_config['dir']['adj_filename']
adj_type = data_config['type']['adj_type']



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





def extract_metric(prediction, target,  mode = 'TEST' ):
    
    num_of_vertices = prediction.shape[1]
          
    for i in  [1,3,5]:

        total_mae = mean_absolute_error(prediction[:, :, i], target[:, :, i])
        total_rmse = mean_squared_error(prediction[:, :, i], target[:, :, i]) ** 0.5
        total_mape = masked_mape_np(prediction[:, :, i], target[:, :, i], 0)

        print('%s mode %d minutes total MAE: %.2f' % (mode, (i+1)*5, total_mae))
        print('%s mode %d minutes total RMSE: %.2f' % (mode, (i+1)*5, total_rmse))
        print('%s mode %d minutes total MAPE: %.2f' % (mode, (i+1)*5, total_mape))


    for i in range(num_of_vertices):

        for j in [1,3,5]:

            print('tsdlinkid : %s, predict %s points' % (i, j))


            mae = mean_absolute_error(prediction[:, i, :j].reshape(prediction[:, i, :j].shape[0], -1).flatten(),
                                      target[:, i, :j].reshape(target[:, i, :j].shape[0], -1 ).flatten())


            rmse = mean_squared_error(prediction[:, i, :j].reshape(prediction[:, i, :j].shape[0],-1).flatten(), 
                                      target[:, i, :j].reshape(target[:, i, :j].shape[0],-1).flatten()) ** 0.5


            mape = masked_mape_np(prediction[:, i, :j].reshape(prediction[:, i, :j].shape[0],-1).flatten(), 
                                 target[:, i, :j].reshape(target[:, i, :j].shape[0],-1).flatten(), 0)

            print('%s mode %s _tsd  %d minutes MAE: %.2f' % (mode, i, (j+1)*5, mae))
            print('%s mode %s _tsd  %d minutes RMSE: %.2f' % (mode, i,(j+1)*5, rmse))
            print('%s mode %s _tsd  %d minutes MAPE: %.2f' % (mode, i,(j+1)*5, mape))

            



    
    
    
   
def return_parameter_file(path):
    param_file = os.listdir(path)
    assert len(param_file) ==1, ' Error many param file exists'
    return os.path.join(path, param_file[0])




def test(test_loader, batch_size, PATH):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    
    test_loss = []
    
#     checkpoint = torch.load(PATH)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     net.eval()
    
    
#     for param_tensor in net.state_dict():
#         print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        
    with torch.no_grad():
        
        prediction = []
        target = []
        loss_list = []
        
        for test_week, test_day, test_hour, test_target in test_loader:
            
            print('aaaaaaaaaa')
            print(test_week.shape, test_day.shape, test_hour.shape)

            test_week = test_week.float().type(torch.FloatTensor).to(DEVICE)
            test_day = test_day.float().type(torch.FloatTensor).to(DEVICE)
            test_hour = test_hour.float().type(torch.FloatTensor).to(DEVICE)
            test_target = test_target.float().type(torch.FloatTensor).to(DEVICE)
            
            sw.add_graph(net,(test_week,test_day,test_hour),verbose = True)
            
            outputs = net(test_week,test_day, test_hour)
            loss = loss_criterion(outputs, test_target)  

            
            prediction.append(outputs.detach().cpu().numpy())
            loss_list.append(loss.detach().cpu().numpy())
            target.append(test_target.detach().cpu().numpy())
        
        prediction = np.concatenate(prediction, 0)  # (test_size,  # of nodes , prediction_size)
        target = np.concatenate(target,0)
       
        
        return prediction, target, loss_list
    
    
    
    
    
if __name__ == '__main__':
    
    num_of_vertices = data_config['setting']['num_of_vertices']
    adj_mx, distance_mx = get_adjacency_matrix(adj_mat, num_of_vertices, id_filename = None)
    
    DEVICE = torch.device('cpu')

    if adj_type == 'D':

        net = make_model(model_back_bone, distance_mx)

    else:

        net = make_model(model_back_bone, adj_mx)

        net.to(DEVICE)
    
    
    all_data = read_and_generate_dataset(prod_data, real_data, num_of_weeks, num_of_days, num_of_hours, num_for_predict,\
                                         points_per_hour=points_per_hour, save=True)  
    
    loss_criterion = nn.MSELoss()
    
    param_file = return_parameter_file(log_config['best_params_dir'])
    

    train_loader, val_loader, test_loader, stats = data_loader(all_data, batch_size, shuffle = True)
    prediction, target, loss = test(test_loader, batch_size, param_file)
    extract_metric(prediction, target)
    
    

    
    np.savez_compressed(
            os.path.normpath(log_config['prediction_base_dir'] +'/'+ 'prediction.npz'),
            prediction=prediction,
            ground_truth=all_data['test']['target']
        )
    


    
    




























