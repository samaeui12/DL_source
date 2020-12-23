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

def save_df(df : pd.DataFrame, folder : str, file_name : str, filetype : str) -> None:
    
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
                
        print('df wrong type argument')
        return False  


def write_csv(folder_path : str, data : pd.DataFrame) -> None:
    
    file = open(os.path.join(folder_path,'jam_index.csv'), 'w+', newline ='') 
    with file:     
        write = csv.writer(file) 
        write.writerow(data)   

class Preprocessor:

    @staticmethod
    def to_datetime(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data['periodtime_1m'].astype('str'), format='%Y%m%d%H%M')
        return data
    @staticmethod
    def sorting_df(data : pd.DataFrame, col_name  = ['periodtime_1m']) -> pd.DataFrame :
        
        return data.sort_values(by= col_name)
    @staticmethod
    def fill_missing_index(data : pd.DataFrame, start_tm: str, end_tm: str, freq = '1T') -> pd.DataFrame :
       
        try:
            
            start_index = str(min(data.periodtime_1m.values))
            end_index =  str(max(data.periodtime_1m.values))
            
        except AttributeError:
            
            print(' no {} in data_column'.format('periodtime_1m'))

        if not isinstance(data.index, pd.DatetimeIndex):
           
            data.index = pd.to_datetime(data['periodtime_1m'].astype('str'), format='%Y%m%d%H%M')
            
        if len(pd.date_range(start_index, end_index , freq= freq)) == len(data.index):
                
            return data

        else:
                            
            full_time_data = pd.DataFrame(index=pd.date_range(start_index , end_index , freq='1T'), columns=['tmp'])
            data = data.combine_first(full_time_data)
            
            return data.loc[start_tm:end_tm]

    @staticmethod
    def linear_interpolation( data, freq = '1T'):
        data.interpolate(method = 'values',inplace = True)
        data.bfill(inplace =True)
        data.ffill(inplace =True)
        if Preprocessor.checkNa(data):

            raise ValueError('Na exist')

        else:
            return data

    @staticmethod
    def resampling(data : pd.DataFrame, start_tm, end_tm, sampling_time = '5T') -> pd.DataFrame:
        
        try:
            
            start_tm = str(min(data.periodtime_1m.values))
            end_tm =  str(max(data.periodtime_1m.values))
            
        except AttributeError:
            
            print(' no {} in data_column'.format('periodtime_1m'))
                
        if not isinstance(data.index, pd.DatetimeIndex):
            
            data = Preprocessor.to_datetime(data)
            
        return data.loc[pd.date_range(start_tm, end_tm , freq = sampling_time)]
    @staticmethod
    def moving_average(data : pd.DataFrame, minutes = 3):
         
        col_name= 'prod_MA' + '_' + str(minutes)
        
        data.loc[:, col_name] = data.prod_tt.iloc[:].rolling(window = minutes, min_periods=1).mean()
        
        return data
    @staticmethod
    def convert_speed_to_time(data:pd.DataFrame) -> pd.DataFrame:
        
        data.loc[:,'real_length'] = data['real_tt']*data['real_ts']/3.6
        col_tt = data.loc[: , ["prod_tt_0",'prod_tt_1',"prod_tt_2"]]
        col_ts = data.loc[: ,[ "prod_ts_0",'prod_ts_1',"prod_ts_2"]]
        data['past_prod_ts_mean'] =  col_ts.mean(axis=1)
        data['past_prod_tt_mean_1'] =  col_tt.mean(axis=1)
        data['past_prod_tt_mean_2'] = (data['real_length'] / data['past_prod_ts_mean'])*3.6
        return data
    @staticmethod
    def checkNa(data:pd.DataFrame) -> bool:
        if (data['real_tt'].isnull().values.any()) or (data['prod_tt'].isnull().values.any()):
            print(np.where(np.asanyarray(np.isnan(data))))
            
            return True
        
        else:
            
            return False
            





class Tr_dataset_manger():
    
    """
        class 설명 : 데이터 셋 만드는 코드
        
        argument 설명 :
        
            header_flags : 읽어드리는 csv file에 header 여부 
        
            multi_feature_flags {
                
                True > featrue 를 travel_speed + congestion (prod)
                False > feature를 travel_speed 단일 피쳐
                
                }

            load_cate : 'express' > 고속도로 데이터
    """
    
    
    def __init__(self, data_path, file_name, header_flags = False, multi_feature_flags = True, load_cate = 'express'):
        
            
        
        self.data_path = data_path
        self.traffic_data_path = os.path.join(data_path, file_name)        
        self.header_flags = header_flags
        self.column_dict = None
        self.data = None
        self.multi_feature_flags = multi_feature_flags

        ## data 기간 정해주는 부분 
        self.info = {

            'start_tm': '201902090000',
            'end_tm' :  '202005312359'
        }
        # csv파일에 헤더가 없을 경우 만들어주는 부분 (header_flags argument와 연동)
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
        
    def preprocessing_data(self):

        ## 수정 중  return 결과물이 npz 파일로 return 되게끔 수정 필요 
        
        result_prod = np.array([])
        result_real  = np.array([]) 
        result_prod_con = np.array([])
        result_real_con = np.array([])
        result_jam_flags = np.array([])
        
        
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
            """ 

            추후 PREPROCESSOR 클래스 매소드의 일부는 데이더 추출 과정에서 big-query나 spark / hive로 작업해오는게 더 나음

            """
            tmp_data = Preprocessor.sorting_df(tmp_data)
            tmp_data = Preprocessor.fill_missing_index(tmp_data, self.info['start_tm'], self.info['end_tm'])
            tmp_data = Preprocessor.linear_interpolation(tmp_data)
            tmp_data = Preprocessor.resampling(tmp_data, self.info['start_tm'], self.info['end_tm'])
            tmp_data = Preprocessor.moving_average(tmp_data)

            tmp_data['row_idx'] = np.arange(len(tmp_data))
            tmp_data['row_idx'] = tmp_data['row_idx'].astype('int')
            
            tmp_data['lagged_real_con'] = tmp_data.real_con.shift(-1)
            tmp_data['lagged2_real_con'] = tmp_data.real_con.shift(-2)
            tmp_data['jam_flags'] = np.where((tmp_data.real_con != 1.0) | (tmp_data.lagged_real_con != 1.0)\
                                              | (tmp_data.lagged2_real_con != 1.0), 1, 0)
            
            
            
            if Preprocessor.checkNa(tmp_data):
                raise ValueError('NA exists')

            #     print('{} : NA exists'.format(key))
            #     print()
            #     return tmp_data
            #     #return False
           
            else:
                
                self.preprocessed_data_dict[tsd] = tmp_data

                prod_data = tmp_data.prod_tt.to_numpy()
                real_data = tmp_data.real_tt.to_numpy()
                prod_con = tmp_data.prod_con.to_numpy()
                real_con = tmp_data.real_con.to_numpy()
                jam_flags = tmp_data.jam_flags.to_numpy()
                
            if i == 0:

                result_prod = prod_data
                result_real = real_data
                result_prod_con = prod_con
                result_real_con = real_con
                result_jam_flags = jam_flags

            else:

                try:

                    result_prod = np.vstack((result_prod, prod_data))
                    result_real = np.vstack((result_real, real_data))
                    result_prod_con = np.vstack((result_prod_con, prod_con))
                    result_real_con =  np.vstack((result_real_con, real_con))
                    result_jam_flags = np.vstack((result_jam_flags, jam_flags))
                    
                except TypeError:        
                        
                    print('Type error in key _{}'.format(key))

                except ValueError:
                    
                    print('Value error in key _{}'.format(key))

        
        prod_speed = np.expand_dims(result_prod.T, axis = 2)
        prod_con = np.expand_dims(result_prod_con.T, axis = 2)
        real_con = np.expand_dims(result_prod_con.T, axis = 2)
        real_speed = np.expand_dims(result_real.T, axis = 2)  
        real_jam_flags = np.expand_dims(result_jam_flags.T, axis = 2) 
        np.savez('../data/Tmap/{}'.format('data_final.npz'), prod_speed = prod_speed, real_speed = real_speed \
                , prod_congestion = prod_con, real_congestion = real_con, jam_flags = real_jam_flags )

        #prod_result = np.dstack((result_prod, result_con)).T.transpose(1,2,0)
        #real_result = np.expand_dims(result_real.T, axis = 2)
                
                
def make_negative_data_set(index_list, jam_threshold = 4):
    
    def cluster_index(index_list):

        result = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(index_list), lambda x: x[0]-x[1])]

        return result
    
    clustered_jam = [event for event in cluster_index(index_list) if len(event) > jam_threshold]
    clustered_jam = [val for sublist in clustered_jam for val in sublist]

    return clustered_jam
    
                
def extract_jam_data(data):
    
    ## 수정해야할 부분 list에 해당 index들을 모두 넣어버리지 말고 정체인 부분의 roadkey를 기억해놔야함 
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

            
        else:
            
            print('wrong type')
            return False

    index_list = [y for x in index_list for y in x]
    index_set = list(dict.fromkeys(index_list))
    return index_set

     
    
    

                                                
def main():
     
    raw_data_path = '../data/raw_data/'
    file_name = 'traffic_express.csv'
    data_manger = Tr_dataset_manger(data_path = raw_data_path , file_name = file_name, multi_feature_flags = True,\
                                    load_cate = 'express')
    data = data_manger.preprocessing_data()
    index_set = extract_jam_data(data_manger.preprocessed_data_dict)
    index_set = make_negative_data_set(index_set)
    write_csv('../data/Tmap',index_set)

if __name__ =="__main__":
    main()
                

        
 
        
        

                




        
