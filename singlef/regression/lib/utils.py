import os
import numpy as np
import torch
import torch.utils.data
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np, mean_absolute_error, mean_squared_error 
from scipy.sparse.linalg import eigs
from .prepareData import read_and_generate_dataset
from torch.utils.tensorboard import SummaryWriter


       
        
        
def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def re_normalization(x, mean, std):
    
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x




def extract_metric(prediction, target, sw, epoch, mode ):
    
    num_of_vertices = prediction.shape[1]
    print('prediction',prediction.shape)
          
    for i in  [2,4,6]:
        
        if i ==6:
            
            
            total_mae = mean_absolute_error(prediction[:, :, :], target[:, :, :])
            total_rmse = mean_squared_error(prediction[:, :, :], target[:, :, :]) ** 0.5
            total_mape = masked_mape_np(prediction[:, :, :], target[:, :, :], 0)
            
        else:
            

            total_mae = mean_absolute_error(prediction[:, :, i], target[:, :, i])
            total_rmse = mean_squared_error(prediction[:, :, i], target[:, :, i]) ** 0.5
            total_mape = masked_mape_np(prediction[:, :, i], target[:, :, i], 0)

        print('%s mode %d minutes total MAE: %.2f' % (mode, i*5, total_mae))
        print('%s mode %d minutes total RMSE: %.2f' % (mode, i*5, total_rmse))
        print('%s mode %d minutes total MAPE: %.2f' % (mode, i*5, total_mape))

        sw.add_scalar(tag='%s mode Total MAE_%s_points' % (mode, i),
                      scalar_value = total_mae,
                      global_step=epoch)
        sw.add_scalar(tag='%s mode Total RMSE_%s_points' % (mode, i),
                      scalar_value=total_rmse,
                      global_step=epoch)
        sw.add_scalar(tag='%s mode Total MAPE_%s_points' % (mode, i),
                      scalar_value=total_mape,
                      global_step=epoch)

    for i in range(num_of_vertices):

        for j in [2,4,6]:
            
            print('current epoch: %s, tsdlinkid : %s, predict %s points' % (epoch, i,j))
            
            if j==6:
                
                mae = mean_absolute_error(prediction[:, i, :].reshape(prediction[:, i, :].shape[0], -1).flatten(),
                          target[:, i, :].reshape(target[:, i, :].shape[0], -1 ).flatten())


                rmse = mean_squared_error(prediction[:, i, :].reshape(prediction[:, i, :].shape[0],-1).flatten(), 
                                          target[:, i, :].reshape(target[:, i, :].shape[0],-1).flatten()) ** 0.5


                mape = masked_mape_np(prediction[:, i, :].reshape(prediction[:, i, :].shape[0],-1).flatten(), 
                                     target[:, i, :].reshape(target[:, i, :].shape[0],-1).flatten(), 0)
                
            
            else:

           
                mae = mean_absolute_error(prediction[:, i, :j].reshape(prediction[:, i, :j].shape[0], -1).flatten(),
                                          target[:, i, :j].reshape(target[:, i, :j].shape[0], -1 ).flatten())


                rmse = mean_squared_error(prediction[:, i, :j].reshape(prediction[:, i, :j].shape[0],-1).flatten(), 
                                          target[:, i, :j].reshape(target[:, i, :j].shape[0],-1).flatten()) ** 0.5


                mape = masked_mape_np(prediction[:, i, :j].reshape(prediction[:, i, :j].shape[0],-1).flatten(), 
                                     target[:, i, :j].reshape(target[:, i, :j].shape[0],-1).flatten(), 0)
                

            print('%s mode %d minutes %s _tsd mae: %.2f' % (mode, j*5, i, mae))
            print('%s mode %d minutes %s _tsd rmse: %.2f' % (mode, j*5, i, rmse))
            print('%s mode %d minutes %s _tsd MAPE: %.2f' % (mode, j*5, i, mape))


            sw.add_scalar(tag='%s mode %s _tsdlinkid MAE_%s_points' % (mode, i, j),
                          scalar_value = mae,
                          global_step = epoch)
            sw.add_scalar(tag='%s mode %s _tsdlinkid RMSE_%s_points' % (mode, i, j),
                          scalar_value = rmse,
                          global_step = epoch)
            sw.add_scalar(tag='%s mode %s _tsdlinkid MAPE_%s_points' % (mode, i, j),
                          scalar_value = mape,
                          global_step = epoch)
        
        
       
            
        



def compute_val_loss_astgcn(net, val_loader, criterion, sw, epoch, DEVICE, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''
    net.train(False)  # ensure dropout layers are in evaluation mode
    with torch.no_grad():
        val_loader_length = len(val_loader)  # nb of batch
        prediction = []
        target = []
        loss_tmp = []
        batch_index = 1
        for val_week, val_day, val_hour, val_target in val_loader:

            
            val_week = val_week.float().type(torch.FloatTensor).to(DEVICE)
            val_day = val_day.float().type(torch.FloatTensor).to(DEVICE)
            val_hour = val_hour.float().type(torch.FloatTensor).to(DEVICE)
            val_target = val_target.float().type(torch.FloatTensor).to(DEVICE)            
            outputs = net([val_week, val_day, val_hour])
            loss = criterion(outputs, val_target)
            
            prediction.append(outputs.detach().cpu().numpy())
            target.append(val_target.detach().cpu().numpy())
            loss_tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break
            batch_index += 1
        validation_loss = sum(loss_tmp) / len(loss_tmp)
        print('%d epochs val_loss is :' % validation_loss)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    prediction = np.concatenate(prediction, 0)  # (batch, T’, 1)
    target = np.concatenate(target,0)
    extract_metric(prediction, target, sw, epoch, mode = 'Validation')
    return validation_loss



    
def evaluate_on_test_astgcn(net, test_loader, criterion , sw, epoch, DEVICE):
    '''
    for rnn, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.

    :param net: model
    :param test_loader: torch.utils.data.utils.DataLoader
    :param test_target_tensor: torch.tensor (B, N_nodes, T_output, out_feature)=(B, N_nodes, T_output, 1)
    :param sw:
    :param epoch: int, current epoch
    :param _mean: (1, 1, 3(features), 1)
    :param _std: (1, 1, 3(features), 1)
    '''

    net.train(False)  # ensure dropout layers are in test mode
    batch_index = 1 
    with torch.no_grad():

        test_loader_length = len(test_loader)
        
        prediction = []  # batch_output
        target = []
        
        for test_week, test_day, test_hour, test_target in test_loader:
            
            test_week = test_week.float().type(torch.FloatTensor).to(DEVICE)
            test_day = test_day.float().type(torch.FloatTensor).to(DEVICE)
            test_hour = test_hour.float().type(torch.FloatTensor).to(DEVICE)
            test_target = test_target.float().type(torch.FloatTensor).to(DEVICE)
           
            outputs = net([test_week,test_day, test_hour])
            
            loss = criterion(outputs, test_target)  
                        
            prediction.append(outputs.detach().cpu().numpy())
            target.append(test_target.detach().cpu().numpy())

            if batch_index % 100 == 0:
                
                print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
                
            batch_index += 1
            
        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
        target = np.concatenate(target,0)
        extract_metric(prediction, target, sw, epoch, mode = 'Test')
        
        
        
        

                






        
