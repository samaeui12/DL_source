# -*- coding:utf-8 -*-
import yaml
import configparser
import pathlib
import os
import json
import shutil

config = configparser.ConfigParser()

 
class Yml_manager():
    
    def __init__(self, key_list, config_path, force = False):
        
        self.config_path = config_path
        self.config = self.config_path.split('/')[-1]
        self.key_list = key_list
        self.force = force
        self.yml_dict = {}
        
    def parsing_yml(self):
        
        config.read(self.config_path)
        
        for key in self.key_list: 
        
            file = config['File'][key]
            #print(file)
        
            with open(file) as f:

                self.yml_dict[key.split('_')[0]] = yaml.load(f, Loader=yaml.FullLoader)
        
    def processing_yml(self):
        
        log_dict = self.yml_dict['log']
        data_dict = self.yml_dict['data']
        training_dict = self.yml_dict['training']
        model_dict = self.yml_dict['model']
        self.result = {}
        
        def make_dir(dir_dict):
            if self.force:
                for value in log_dict.values():
                    print(value)
                    if os.path.exists(value):
                        shutil.rmtree(value)
                        print('Params folder {} exists! and removing'.format(value))
                    pathlib.Path(value).mkdir(parents = True, exist_ok=True)
            else:
                for value in log_dict.values():
                    if os.path.exists(value):
                        continue
                    else:
                        pathlib.Path(value).mkdir(parents = True, exist_ok=True)
                        
                
        def make_log_dict():
            
            log_dict['summary_log_dir'] += '/' + self.config
            #log_dict['summary_root_dir'] = log_dict['summary_log_dir']
            #log_dict['summary_log_dir'] = log_dict['summary_log_dir'] +'/' + 'log'
            log_dict['params_dir']  = log_dict['summary_log_dir'] + '/' + 'params'
            log_dict['best_params_dir']  = log_dict['summary_log_dir'] + '/' + 'best_params'
            log_dict['prediction_base_dir']  = log_dict['summary_log_dir'] + '/' + 'prediction_dir'    
            
            make_dir(log_dict)
            
            self.result['log'] = log_dict
            
        
        def processing_data_config():
            
            feature_type = data_dict['type']['feature_type']
            adj_type = data_dict['type']['adj_type']
            
            full_data = data_dict['dir']['data']
            adj_filename = data_dict['dir']['adj_filename']
            
            data_dict['dir'] = {}          
            data_dict['dir']['data'] = full_data
            data_dict['dir']['adj_filename'] = adj_filename
            data_dict['type']['adj_type'] = 'D'
            
            self.result['data'] = data_dict
        
        def saving_json():
            
            file_name = log_dict['summary_log_dir'] + '/'+'model_summary'
            tmp_dict = {}
            
            tmp_dict['data'] = data_dict
            tmp_dict['model'] = model_dict
            tmp_dict['training'] = training_dict
            
            with open(file_name, 'w') as fp:
                json.dump(tmp_dict, fp)
                            
        def training_info():
            
            self.result['training'] = training_dict
        
        def make_model_backbone():
            
            if model_dict['structure']['num_layer'] ==2:
                
                model_back_bone = {

                    'common':
                    {    
                
                        'nb_input' : 3 ,
                        'nb_block' : model_dict['structure']['num_layer'],
                        'K' : model_dict['etc_hyperparam']['K'],
                        'batch_size' : training_dict['Training_parameter']['batch_size'],
                        'embedded_input': 5,
                        'embedded_output': 3


                    },

                     'layer_1': 

                    {   

                        'in_channels' : model_dict['structure']['layer1_input'],
                        'nb_chev_filter' : model_dict['structure']['layer1_nb_chev_filter'],
                        'nb_time_filter' : model_dict['structure']['layer1_nb_time_filter'],
                        'time_strides' : model_dict['structure']['layer1_time_strides'],
                        'num_for_predict' : data_dict['setting']['num_for_predict'],
                        'len_input' : data_dict['setting']['len_input'],
                        'num_of_vertices' : data_dict['setting']['num_of_vertices']
                    },

                     'layer_2': \

                    {

                        'in_channels' : model_dict['structure']['layer1_nb_chev_filter'] ,
                        'nb_chev_filter' : model_dict['structure']['layer2_nb_chev_filter'],
                        'nb_time_filter' : model_dict['structure']['layer2_nb_time_filter'],
                        'time_strides' : model_dict['structure']['layer2_time_strides'],
                        'num_for_predict' : data_dict['setting']['num_for_predict'],
                        'len_input' : data_dict['setting']['len_input'],
                        'num_of_vertices' : data_dict['setting']['num_of_vertices']
                     }
                }
                
            
            elif model_dict['structure']['num_layer'] == 3:
                    
                model_back_bone = {

                    'common':
                    {    

                        'nb_input' : 3 ,
                        'nb_block' : model_dict['structure']['num_layer'],
                        'K' : model_dict['etc_hyperparam']['K'],
                        'batch_size' : training_dict['Training_parameter']['batch_size']


                    },

                     'layer_1': 

                    {   

                        'in_channels' : model_dict['structure']['layer1_input'],
                        'nb_chev_filter' : model_dict['structure']['layer1_nb_chev_filter'],
                        'nb_time_filter' : model_dict['structure']['layer1_nb_time_filter'],
                        'time_strides' : model_dict['structure']['layer1_time_strides'],
                        'num_for_predict' : data_dict['setting']['num_for_predict'],
                        'len_input' : data_dict['setting']['len_input'],
                        'num_of_vertices' : data_dict['setting']['num_of_vertices']
                    },

                     'layer_2': \

                    {

                        'in_channels' : model_dict['structure']['layer1_nb_chev_filter'] ,
                        'nb_chev_filter' : model_dict['structure']['layer2_nb_chev_filter'],
                        'nb_time_filter' : model_dict['structure']['layer2_nb_time_filter'],
                        'time_strides' : model_dict['structure']['layer2_time_strides'],
                        'num_for_predict' : data_dict['setting']['num_for_predict'],
                        'len_input' : data_dict['setting']['len_input'],
                        'num_of_vertices' : data_dict['setting']['num_of_vertices']
                     },

                     'layer_3': \

                    {

                        'in_channels' : model_dict['structure']['layer2_nb_chev_filter'] ,
                        'nb_chev_filter' : model_dict['structure']['layer3_nb_chev_filter'],
                        'nb_time_filter' : model_dict['structure']['layer3_nb_time_filter'],
                        'time_strides' : model_dict['structure']['layer3_time_strides'],
                        'num_for_predict' : data_dict['setting']['num_for_predict'],
                        'len_input' : data_dict['setting']['len_input'],
                        'num_of_vertices' : data_dict['setting']['num_of_vertices']
                     }

                }  
            
            self.result['model_back_bone'] = model_back_bone
            
        
        make_log_dict()
        processing_data_config()         
        saving_json()
        make_model_backbone()
        training_info()
        
        
        
            
            
            
        
        
            
            
            
          
              
                
        
        



    
    
    
    