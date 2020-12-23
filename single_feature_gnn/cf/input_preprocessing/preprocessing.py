import pandas as pd
import numpy as np
import os
import csv
import copy
from datetime import timedelta
import matplotlib.pyplot as plt
import pathlib
from operator import itemgetter
from itertools import *




class Tr_dataset_manger():
    
    """
        class 설명 : 데이터 셋 만드는 코드
        
        
        argument 설명 :
        
            header_flags : 읽어드리는 csv file에 header 여부 
        
            multi_feature_flags : 
    
    """
    
    
    def __init__(self, data_path, file_name, header_flags = False, multi_feature_flags = True, load_cate = 'express'):
        
            
        
        self.data_path = data_path
        
        self.traffic_data_path = os.path.join(data_path, file_name)        
        self.header_flags = header_flags
        self.column_dict = None
        self.data = None
        self.multi_feature_flags = multi_feature_flags
                    
        self.info = {

            'start_tm': '201902090000',
            'end_tm' :  '202005312359'
        }

        self.column_dict = {

            'col_1' : 'periodtime_1m',
            'col_2' : 'periodtime_5m',
            'col_3' : 'tsdlinkid',
            'col_4' : 'nexttsdlinkid',
            'col_5' : 'representlength',
            'col_6' : 'roadclass',
            'col_7' : 'linkclass',
            'col_8' : 'congestionclass',
            'col_9' : 'real_ts',
            'col_10' : 'real_tt',
            'col_11' : 'real_pc',
            'col_12' : 'real_con',
            'col_13' : 'real_na_code',
            'col_14' : 'prod_ts',
            'col_15' : 'prod_tt',
            'col_16' : 'prod_pc',
            'col_17' : 'prod_con',
            'col_18' : 'prod_ret',
            'col_19' : 'prod_na_code',
            'col_20' : 'pat_ts',
            'col_21' : 'pat_tt',
            'col_22' : 'pat_accum_pc',
            'col_23' : 'pat_con',
            'col_24' : 'patuseflag',
            'col_25' : 'pat_na_code',
            'col_26' : 'periodtype',
            'col_27' : 'tm',
            'col_28' : 'dt'
        }
            
        
    def read_file(self):
        
        if self.header_flags ==False:
            
            data = pd.read_csv(self.traffic_data_path, names = list(self.column_dict.values()))
            
        else:
            
            data = pd.read_csv(self.traffic_data_path)
        
        return data
    
    def time_indexing(self, data):
        
        data.index = pd.to_datetime(data['periodtime_1m'].astype('str'), format='%Y%m%d%H%M')
        
        return data
        
    def sorting_df (self, data, col_name  = ['periodtime_1m']):
        
        return data.sort_values(by= col_name)
    
    
    def fill_missing_value(self, data, freq = '1T'):
        
        try:
            
            start_tm = str(min(data.periodtime_1m.values))
            end_tm =  str(max(data.periodtime_1m.values))
            
        except AttributeError:
            
            print(' no {} in data_column'.format('periodtime_1m'))
        
        def check_missing() :
            
            if len(pd.date_range(start_tm , end_tm , freq= freq)) == len(data.index):
                
                return True
            
            else:
                
                return False
            
        if check_missing() == True:
            
            return data
        
        else:
        
            full_time_data = pd.DataFrame(index=pd.date_range(start_tm , end_tm , freq='1T'), columns=['tmp'])
            filled_data = data.combine_first(full_time_data) 
            
            return filled_data
    
    def interpolation(self, data, freq = '1T'):
        
        if not isinstance(data.index, pd.DatetimeIndex):
            
            data = self.time_indexing(data)
            
        data.interpolate(method = 'values',inplace = True)
        
        return data
        
                
    def resampling(self, data, sampling_time = '5T'):
        
        try:
            
            start_tm = str(min(data.periodtime_1m.values))
            end_tm =  str(max(data.periodtime_1m.values))
            
        except AttributeError:
            
            print(' no {} in data_column'.format('periodtime_1m'))
                
        if not isinstance(data.index, pd.DatetimeIndex):
            
            data = self.time_indexing(data)
            
        return data.loc[pd.date_range(self.info['start_tm'], self.info['end_tm'] , freq = sampling_time)]
        
    def moving_average(self, data, minutes = 3):
         
        col_name= 'prod_MA' + '_' + str(minutes)
        
        data.loc[:, col_name] = data.prod_tt.iloc[:].rolling(window=3,min_periods=1).mean()
        
        return data
    

    def convert_speed_to_time(self,data):
        
        data.loc[:,'real_length'] = data['real_tt']*data['real_ts']/3.6
        col_tt = data.loc[: , ["prod_tt_0",'prod_tt_1',"prod_tt_2"]]
        col_ts = data.loc[: ,[ "prod_ts_0",'prod_ts_1',"prod_ts_2"]]
        data['past_prod_ts_mean'] =  col_ts.mean(axis=1)
        data['past_prod_tt_mean_1'] =  col_tt.mean(axis=1)
        data['past_prod_tt_mean_2'] = (data['real_length'] / data['past_prod_ts_mean'])*3.6
        return data
    
    def checkNa(self,data):
        
        if (data['real_tt'].isnull().values.any()) or (data['prod_tt'].isnull().values.any()):
            print(np.where(np.asanyarray(np.isnan(data))))
            return True
        
        else:
            
            return False
            
        ## interpol -> resample 이후 na 가 있는지 확인하는 로직 채워 넣기 (prod, real부분 )        
            
    def save_df(self, df, folder, file_name, filetype):
        
        if isinstance(df, pd.DataFrame):
            
            if filetype == 'csv':
                
                df.to_csv(os.path.join(folder, 'preprocessed_' + file_name), index = False) 
        
        elif isinstance(df, dict):
            
            for key in df.keys():
                
                if isinstance(df[key], pd.DataFrame):
                    
                    df[key].to_csv(os.path.join(folder, 'preprocessed_' + str(key)), index = False) 
                    
                else:
                    
                    print('datatype is not dataframe')
                    return False    
        else:
                    
            print('datatype is not dataframe')
            return False    
            
        
    def preprocessing_data(self):
        ## 수정 중  return 결과물이 npz 파일로 return 되게끔 수정 필요 
        
        '''
        result_prod = np.array([])
        result_real  = np.array([]) 
        result_con = np.array([])
        '''
        data = self.read_file()
        self.preprocessed_data_dict = {}
        
        idx_dict = pd.read_csv(os.path.join(self.data_path,'tsd_mapping.csv'))[['idx','tsdlink_id']].set_index('idx')\
        ['tsdlink_id'].to_dict()
        
        for i, key in enumerate(idx_dict.keys()):

            tsd = idx_dict[key]
            file_name = 'preprocessed_' + str(tsd) +'_' + '.csv'

            if data.dtypes['tsdlinkid'] != 'int':

                data['tsdlinkid'] = data['tsdlinkid'].astype('int')

            tmp_data = copy.deepcopy(data.loc[(data.tsdlinkid == tsd)])
            tmp_data = self.sorting_df(tmp_data)
            tmp_data = self.fill_missing_value(tmp_data)
            tmp_data = self.interpolation(tmp_data)
            tmp_data = self.resampling(tmp_data).loc['2019-01-02 00:00:00':]
            tmp_data = self.moving_average(tmp_data)
            tmp_data['row_idx'] = np.arange(len(tmp_data))
            tmp_data['row_idx'] = tmp_data['row_idx'].astype('int')
            
            tmp_data['lagged_real_con'] = tmp_data.real_con.shift(-1)
            tmp_data['lagged2_real_con'] = tmp_data.real_con.shift(-2)
            
            if self.checkNa(tmp_data):

                print('{} : NA exists'.format(key))
                
                return False
           
            else:
                
                self.preprocessed_data_dict[tsd] = tmp_data

                
                
def make_negative_data_set(index_list, jam_threshold = 4):
    
    def cluster_index(index_list):

        result = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(index_list), lambda x: x[0]-x[1])]

        return result
    
    clustered_jam = [event for event in cluster_index(index_list) if len(event) > jam_threshold]
    clustered_jam = [val for sublist in clustered_jam for val in sublist]

    return clustered_jam
    
                
def extract_jam_data(data):
    
    index_list = []   
    
    if isinstance(data, dict): 
        
        for key in data.keys():
            
            if isinstance(data[key], pd.DataFrame):
                          
                data[key]['start_jam'] = np.where((data[key].real_con == 1.0) & (data[key].lagged_real_con != 1.0), 1, 0)  
                data[key]['ing_jam'] = np.where((data[key].real_con != 1.0) & (data[key].lagged_real_con != 1.0), 1, 0)
                data[key]['end_jam'] = np.where((data[key].real_con == 4.0) & (data[key].lagged_real_con != 4.0), 1, 0)
                
                index_list.append(data[key][data[key].start_jam ==1].row_idx.to_list())
                index_list.append(data[key][data[key].ing_jam ==1].row_idx.to_list())
                index_list.append(data[key][data[key].end_jam ==1].row_idx.to_list())
                #index_list = [y for x in index_list for y in x]
                #index_set = list(dict.fromkeys(index_list))
                #index_list.append(data[data.start_jam ==1].row_idx.to_list())
                
            else:
                print('wrong type')
                return False
    
    else:
        
        if isinstance(data, pd.DataFrame):
            
            data['start_jam'] = np.where((data.real_con == 1.0) & (data.lagged_real_con != 1.0), 1, 0)
            data['ing_jam'] = np.where((data.real_con != 1.0) & (data.lagged_real_con != 1.0), 1, 0)
            data['end_jam'] = np.where((data.real_con == 4.0) & (data.lagged_real_con != 4.0), 1, 0)
            
            index_list.append(data[data.start_jam ==1].row_idx.to_list())
            index_list.append(data[data.ing_jam ==1].row_idx.to_list())
            index_list.append(data[data.end_jam ==1].row_idx.to_list())               
            #index_list = [y for x in index_list for y in x]
            #index_set = list(dict.fromkeys(index_list))
            #index_list.append(data[data.start_jam == 1].row_idx.to_list())
            
        else:
            
            print('wrong type')
            return False
    index_list = [y for x in index_list for y in x]
    index_set = list(dict.fromkeys(index_list))
    return index_set

def write_csv(folder_path, data):
    
    file = open(os.path.join(folder_path,'jam_index.csv'), 'w+', newline ='') 
    with file:     
        write = csv.writer(file) 
        write.writerow(data)        
    
    

                                                
def main():
     
    raw_data_path = '../data/raw_data/'
    file_name = 'traffic_express.csv'
    data_manger = Tr_dataset_manger(data_path = raw_data_path , file_name = file_name, multi_feature_flags = True,\
                                    load_cate = 'express')
    data_manger.preprocessing_data()
    index_set = extract_jam_data(data_manger.preprocessed_data_dict)
    index_set = make_negative_data_set(index_set)
    write_csv('../data/raw_data',index_set)

if __name__ =="__main__":
    main()
                
            
        
 
        
        

                






        
