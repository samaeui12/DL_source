# -*- coding:utf-8 -*-

import numpy as np


def masked_mape_np(y_pred, y_true, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
         
        zero_mask = np.not_equal(y_true,0)
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