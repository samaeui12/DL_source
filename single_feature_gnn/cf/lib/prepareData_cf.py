import os
import numpy as np
import argparse
import configparser
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

def normalization(train, val, test, con_f_index = [0], normalize_type = 'z_score'):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    conf_f_index : index of continuous variable as tuple , if None: no categorical value
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    
    
    def Z_score_normalize(x):
        for i in range(x.shape[1]):
            x[:,i,:,:] = (x[:,i,:,:] - mean[i])/std[i]
        return x
    
    def min_max_normalize(x):
        
        for i in range(x.shape[1]):
            x[:,i,:,:] = (x[:,i,:,:] - min_val[i])/(max_val[i] - min_val[i])
            x[:,i,:,:] = x[:,i,:,:] * 2. - 1.
        return x
    
    if not con_f_index:
        mean = train.mean(axis=(0,1,3), keepdims=True)
        std = train.std(axis=(0,1,3), keepdims=True)
        max_val = train.max(axis=(0,1,3), keepdims=True)
        min_val = train.min(axis=(0,1,3), keepdims=True)
#         print('mean.shape:',mean.shape)
#         print('std.shape:',std.shape)

        if normalize_type =='z_socre':
            
            train_norm = Z_score_normalize(train)
            val_norm = Z_score_normalize(val)
            test_norm = Z_score_normalize(test)
            
        elif normalize_type =='min_max':
            
            train_norm = min_max_normalize(train)
            val_norm = min_max_normalize(val)
            test_norm = min_max_normalize(test)
        return {'_mean': mean, '_std': std, '-max' : max_val, '-min' : min_val}, train_norm, val_norm, test_norm
    
    else:
        mean = []
        std = []
        max_val = []
        min_val = []
        
        for i in range(train.shape[1]):
            mean.append(train[:,[i], con_f_index,:].mean(axis = (0,2)))
            std.append(train[:,[i], con_f_index,:].mean(axis = (0,2)))
            max_val.append(train[:,[i], con_f_index,:].max(axis = (0,2)))
            min_val.append(train[:,[i], con_f_index,:].min(axis = (0,2)))
        
        
        if normalize_type =='z_socre':
            
            train[:,:, con_f_index,:] = Z_score_normalize(train[:,:, con_f_index,:])
            val[:,:, con_f_index,:] = Z_score_normalize(val[:,:, con_f_index,:])
            test[:,:, con_f_index,:] = Z_score_normalize(test[:,:, con_f_index,:])
            
        elif normalize_type =='min_max':
            
            train[:,:, con_f_index,:] = min_max_normalize(train[:,:, con_f_index,:])
            val[:,:, con_f_index,:] = min_max_normalize(val[:,:, con_f_index,:])
            test[:,:, con_f_index,:] = min_max_normalize(test[:,:, con_f_index,:])
            
        return {'_mean': mean, '_std': std, '_max' : max_val, '_min' : min_val}, train, val, test
        


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns : binary Matrix와 distance based adj 가 전부 return 되는 형태 (km단위로 환산)
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:
            
            ## node idx 기준으로 데이터를 정렬하여 adj matrix를 만들고 싶을 때 
            
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance / 1000
            return A, distaneA
        
        ## 그냥 distance_df_filename 쓰여져 있는 순서대로 adj matrix를 만들고 싶을 때 
        
        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance / 1000
                    
            return A, distaneA


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
        

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + points_per_hour
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, data_sequence2, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 
    num_for_predict: int, the number of points will be predicted for each sample           
    points_per_hour: int, default 12, number of points per hour
    
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None
    
    
    def extract_imbalance_weight():
        
    
        target = np.zeros((10, 1))
        # 클래스가 3개라고 가정
        count = [0 , 0, 0]
        
        ## data_sequence2 shape -> (10,137664)
        for i in range( data_sequence2[:, label_start_idx: label_start_idx + num_for_predict].shape[0]):
             
            if 4.0 in data_sequence2[i, label_start_idx: label_start_idx + num_for_predict]:
                target[i,:] = 2
                
                #count +=1
                count[2] += 1

            elif 2.0 in data_sequence2[label_start_idx: label_start_idx + num_for_predict]:

                target[i,:] = 1
                count[1] += 1
                
            else:
                target[i,:] = 0
                count[0] +=1
        
        return target, count 
    
    target, count = extract_imbalance_weight()
    
    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour) 
        
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None
        
        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)
        
        #print('day_sample', day_sample.shape)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None
            
        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)
    

    return week_sample, day_sample, hour_sample, target, count



def read_and_generate_dataset(graph_signal_matrix_filename, 
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12,
                              save=False
                              ):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    # [time 길이 , 도로 수 , feature 수]
    
    
    full_data = np.load(graph_signal_matrix_filename)
    
    prod_data = full_data['data'][:,:,0:1]
    print(prod_data.shape)
    
    length_dict = full_data['length_dict']
    congestion_class = full_data['label']
    all_samples = []
    count_list = []
    num_nodes = prod_data.shape[1]

    for idx in range(prod_data.shape[0]):
        sample = get_sample_indices(prod_data, congestion_class, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) or (sample[1] is None) or (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target, count = sample

        
        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]
        
        
        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)
            
        if num_of_days > 0:
            
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)


        target = np.expand_dims(target, axis=0)  # (1,N,T)
        sample.append(target)
        
        count = np.expand_dims(np.array([count]), axis=0)  # (1,1)        
        sample.append(count)        


        ## idx : sample index 
        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)        
        sample.append(time_sample)
        
        all_samples.append(sample) 
                

    #：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)
    
    
    training_set = [np.concatenate(i, axis=0)
                  for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    

    
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]
    

    train_week, train_day, train_hour, train_target, train_count, train_time_index = training_set
    val_week, val_day, val_hour, val_target,val_count, vali_time_index = validation_set
    test_week, test_day, test_hour, test_target, test_count, test_time_index = testing_set
    

    
    def make_weights_for_balanced_classes(train_data_count, num_nodes):  
        
        ## assume 3 class
        sum_count = [0, 0, 0]
        
        for i in range(len(sum_count)):
            
            
            sum_count[i] = train_data_count[:,0, i ].sum()
            #sum([count[i] for count in train_data_count])
        
        weight_per_class = [0] * len(sum_count)
        #print('sum_count', sum_count)
        for i in range(len(sum_count)):
            
            if sum_count[i] !=0:
                
                weight_per_class[i] = sum(sum_count) / sum_count[i]
                
            else:
                weight_per_class[i] = 0
                
            
#        print('weigt_per_classsssss',weight_per_class)
#         print('sum_count', sum_count)
            
#         count = [0] * num_nodes
#         print('aaaaa', train_data_count.shape)
#         for i in range(train_data_count.shape[0]):
            
#             count[train_data_count[i,0]-1] += 1    
            
#         weight_per_class = [0.] * num_nodes
        
#         N = float(sum(count))    
        
#         for i in range(num_nodes):                                                   
#             weight_per_class[i] = N/float(count[i])
        
        #print('weight_per_class',weight_per_class)
            
        weight_per_data = [0] * train_data_count.shape[0]  
        
        
        
        for i in range(train_data_count.shape[0]):
            
            #if train_data_count[i, 0, 2] == 0 and train_data_count[:, 0, 1] == 0:
            
            for j in range(train_data_count.shape[2]):
                
                weight_per_data[i] += (weight_per_class[j] * train_data_count[i,:, j])[0]
                #print('eeeeeeeeee',(weight_per_class[j] * train_data_count[i,:, j])[0])
            
            #print('wwwwwww',weight_per_data)
            
            #weight[i] = weight_per_class[0]               
            #weight[i] = weight_per_class[train_data_count[i,0]-1]      
                                            
        return weight_per_data              
    
    sampling_weight = make_weights_for_balanced_classes(train_count,num_nodes)
    print('sampling_weihgt', sampling_weight[0])
    
        
    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day,
                                                  val_day,
                                                  test_day)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour)
    
    
    
    #(stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'hours': train_recent_norm,
            'weight': sampling_weight,
            'target': train_target,
            'time_index': train_time_index
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'hours': val_recent_norm,
            'target': val_target,
            'time_index': vali_time_index
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'hours': test_recent_norm,
            'target': test_target,
            'time_index' : test_time_index
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'hours': recent_stats
        }
    }
    
    
    
    return all_data





class TmapDataset(torch.utils.data.Dataset):

    def __init__(self, input):
        
        'Initialization'

        #self.input = input  
        
        self.week_input = input['week']
        self.day_input =  input['day']
        self.hour_input = input['hours']
        self.target_input = input['target']                       
    
    def __len__(self):
        
        'Denotes the total number of samples'

        return self.hour_input.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample


        # Load data and get label
        week = self.week_input[index,:,:,:]
        day = self.day_input[index,:,:,:]        
        hour = self.hour_input[index,:,:,:]               
        target = self.target_input[index,:,:]

        return week, day, hour, target
    

def data_loader(all_data, batch_size, shuffle = True):
    '''

   
    :all_data : dict 형태의 모든 데이터 (data_prepare.py 의 return 값)
    :param DEVICE :
    :param batch_size: int
    
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''
    
         
    train_x = all_data['train']
    
    val_x = all_data['val']

    test_x = all_data['test']

    stats = all_data['stats']
    
    
    samples_weight = train_x['weight']
    samples_weight = torch.DoubleTensor(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement = True)
    
    train_dataset = TmapDataset(train_x)
    val_dataset = TmapDataset(val_x)
    test_dataset = TmapDataset(test_x)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)#, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)#, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)#, num_workers=10)

    return train_loader, val_loader, test_loader, stats

