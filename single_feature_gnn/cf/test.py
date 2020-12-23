import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from lib.metrics import cal_accuracy, cal_confusion_matrix

from model.ASTGCN import make_model

from lib.data_loader import data_loader
from lib.utils import *
from lib.config_parser import Yml_manager
from lib.prepareData_cf import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import  TensorDataset, DataLoader, Dataset

import pathlib
import json


parser = argparse.ArgumentParser()

parser.add_argument("--config", default='Tmap.conf', type=str,
                    help="configuration file name")

parser.add_argument("--model", default='ASTGCN', type=str,
                    help="configuration file name")


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


full_data = data_config['dir']['data']
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

loss = training_config['Training_parameter']['loss']
batch_size = training_config['Training_parameter']['batch_size']
    
sw = SummaryWriter(log_dir=log_config['summary_log_dir'], flush_secs=5)





def extract_metric(prediction, target,  mode = 'TEST' ):
    
    num_of_vertices = prediction.shape[1]
    
    
    totalaccuracy, accuracy_dict = cal_accuracy(prediction[:, :, :], target[:, :, :])
    conf_dict = cal_confusion_matrix(target[:, :, :], prediction[:, :, :])     
    
    
    for i in range(num_of_vertices):

        ith_road_accur = accuracy_dict[i]


        print('%s mode %s _tsd accuracy: %.2f' % (mode, i , ith_road_accur))
        print('%s mode %s _tsd recall:' % (mode,i))
        print(conf_dict[i])
    
   
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
    
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    
    
    for param_tensor in net.state_dict():
        
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        
    with torch.no_grad():
        
        outputs_list = []
        softmax_output_list = []
        class_output_list = []
        
        testdata = []
        target = []
        loss_list = []
        
        tt= 0
        
        for test_week, test_day, test_hour, test_target in test_loader:

            
            
            test_week = test_week.float().type(torch.FloatTensor).to(DEVICE)
            test_day = test_day.float().type(torch.FloatTensor).to(DEVICE)
            test_hour = test_hour.float().type(torch.FloatTensor).to(DEVICE)
            test_target = test_target.float().type(torch.FloatTensor).to(DEVICE)
            outputs = net([test_week,test_day, test_hour])
            softmax_output = torch.nn.functional.softmax(outputs,dim=2) 
            class_ouput = torch.argmax(outputs,dim=2)
            
           
            road_loss = []
            
            for i in range(10):
                
                road_loss.append(loss_criterion(outputs[:,i,:], test_target[:,i,:]))
                
            loss = sum(road_loss) / len(road_loss)
            
            

            
            
            
            
            #m = nn.Softmax(dim=2)
            #soft_output = m(input)
            #print('output shape', soft_output.shape)
            
            #loss = loss_criterion(outputs, test_target)  

            
            outputs_list.append(outputs.detach().cpu().numpy())
            softmax_output_list.append(softmax_output.detach().cpu().numpy())
            class_output_list.append(class_ouput.detach().cpu().numpy())
            
            
            loss_list.append(loss.detach().cpu().numpy())
            target.append(test_target.detach().cpu().numpy())
            testdata.append(test_hour.detach().cpu().numpy())
        
        outputs_list = np.concatenate(outputs_list, 0)  # (test_size,  # of nodes , prediction_size)
        softmax_output_list = np.concatenate(softmax_output_list, 0)
        class_output_list = np.concatenate(class_output_list, 0)
        testdata = np.concatenate(testdata,0)
        target = np.concatenate(target,0)
       
        
        return testdata, outputs_list, softmax_output_list, class_output_list, target, loss_list
    
    
    
    
    
if __name__ == '__main__':
    
    num_of_vertices = data_config['setting']['num_of_vertices']
    adj_mx, distance_mx = get_adjacency_matrix(adj_mat, num_of_vertices, id_filename = None)
    
    DEVICE = torch.device('cpu')

    if adj_type == 'D':

        net = make_model(model_back_bone, distance_mx)

    else:

        net = make_model(model_back_bone, adj_mx)

        net.to(DEVICE)
    
    
    all_data = read_and_generate_dataset(full_data, num_of_weeks, num_of_days, num_of_hours, num_for_predict,\
                                         points_per_hour=points_per_hour, save=True) 
    
    
    
    loss_criterion = nn.MSELoss()
    
    param_file = return_parameter_file(log_config['best_params_dir'])
    
    _, _, test_loader, stats = data_loader(all_data, batch_size, shuffle = True)
    time_index = all_data['test']['time_index']
    testdata, output, softmax_output, class_output, target, loss = test(test_loader, batch_size, param_file)
    test_loss = sum(loss)
    extract_metric(output, target)
    
    

    
    np.savez_compressed(
            os.path.normpath(log_config['prediction_base_dir'] +'/'+ 'prediction.npz'),
            hourdata = testdata,
            output=output,
            softmax_output = softmax_output,
            class_output =class_output,
            ground_truth=all_data['test']['target'].squeeze(),
            time_index = time_index,
            loss= test_loss
        )
    


    
    




























