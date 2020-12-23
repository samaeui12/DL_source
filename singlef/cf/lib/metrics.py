
import numpy as np
import torch
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def cal_accuracy(output, true_label):
    """
    net_output : (#_batch, # node, # class)
    true_label : (#_batch, # node, 1)
    
    """
    predicted_label = np.argmax(output, axis=2)

    
    Total_correct_num = 0
    
    accuracy_list = {}
    
    for i in range(output.shape[1]):
            
        correct_num = np.sum(predicted_label[:,i] == true_label[:,i,0])
        
        accuracy_list[i] = (correct_num / (output.shape[0]))
        
        Total_correct_num += correct_num
        
    total_accuracy = 100 * (Total_correct_num / (output.shape[0] * output.shape[1]))
           
    return total_accuracy, accuracy_list

def cal_confusion_matrix(y_true, y_pred):
    
    y_pred = np.argmax(y_pred, axis=2)
    
    conf_matrix_dict = {}
    
    for i in range(y_pred.shape[1]):
        
        conf_matrix_dict[i] = confusion_matrix(y_true[:,i,0], y_pred[:,i])
    
    return conf_matrix_dict

    
# def print_metric(conf_matrix):
#     for i in range(conf_matirx.keys()):
#         print('confmat_shape',conf_matrix[i].shape)
        
#         recall =  /(conf_matrix[i][0,0]+conf_matrix[i][1,0])
    
    

def masked_mape_np(y_pred, y_true, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        #mask = mask.float()
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def mean_absolute_error(y_pred, y_true):
    '''
    mean absolute error

    Parameters
    ----------
    y_true, y_pred: np.ndarray, shape is (batch_size, num_of_features)

    Returns
    ----------
    np.float64

    '''

    return np.abs(y_true - y_pred).mean()


def mean_squared_error(y_pred, y_true):
    '''
    mean squared error

    Parameters
    ----------
    y_true, y_pred: np.ndarray, shape is (batch_size, num_of_features)

    Returns
    ----------
    np.float64

    '''
    return ((y_true - y_pred) ** 2).mean()