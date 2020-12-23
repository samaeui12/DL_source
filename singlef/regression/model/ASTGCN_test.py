# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial



class EmbeddingBlock(nn.Module):
    
    def __init__(self, input_dim , output_dim ): 
        super(EmbeddingBlock,self).__init__()
        self.embedd_layer = nn.Embedding(num_embeddings = input_dim, 
                               embedding_dim = output_dim)
            
 
    def forward(self, X):
        assert X.shape[3] != 1, 'no categorical embedding needed'
            
        
        catego_X = X[:,:,1,:].long()
        #print('cate',catego_X)

        embedded_X = self.embedd_layer(catego_X)
        return torch.cat((X[:,:,0,:].unsqueeze(3), embedded_X),3).permute(0,1,3,2).contiguous()


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, model_topology):
        
        super(Spatial_Attention_layer, self).__init__()
        self.model_topology = model_topology
        self.W1 = nn.Parameter(torch.FloatTensor(model_topology['len_input']))
        self.W2 = nn.Parameter(torch.FloatTensor(model_topology['in_channels'], model_topology['len_input']))
        self.W3 = nn.Parameter(torch.FloatTensor(model_topology['in_channels']))
        self.bs = nn.Parameter(torch.FloatTensor(1, model_topology['num_of_vertices'], model_topology['num_of_vertices']))
        self.Vs = nn.Parameter(torch.FloatTensor(model_topology['num_of_vertices'], model_topology['num_of_vertices']))


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, model_topology):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        #self.K = self.register_buffer('K', torch.tensor(model_topology['K']))
        self.K = model_topology['K']
        
        self.cheb_polynomials = []
        for i in range(self.K):
            
            self.register_buffer('cheb_polynomials_' + str(i), model_topology['cheb_polynomials'][i])
                        
        self.in_channels = model_topology['in_channels']
        self.out_channels = model_topology['nb_chev_filter']
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(model_topology['in_channels'],\
                                                                      model_topology['nb_chev_filter'])) for _ in\
                                                                      range(model_topology['K'])])

    def params(self, i):
        
        return self.__getattr__('cheb_polynomials_'+str(i))
    
    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
        batch_size, num_of_vertices, in_channels, len_input = x.shape

        outputs = []

        for time_step in range(len_input):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            #output = torch.zeros(batch_size, num_of_vertices, self.out_channels)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.params(k)  # (N,N)
                
                
                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 
                
                if k == 0:
                    
                    output = rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
                    
                else:
                    output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    
    def __init__(self, model_topology):
        
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(model_topology['num_of_vertices']))
        self.U2 = nn.Parameter(torch.FloatTensor(model_topology['in_channels'], model_topology['num_of_vertices']))
        self.U3 = nn.Parameter(torch.FloatTensor(model_topology['in_channels']))
        self.be = nn.Parameter(torch.FloatTensor(1, model_topology['len_input'], model_topology['len_input']))
        self.Ve = nn.Parameter(torch.FloatTensor(model_topology['len_input'], model_topology['len_input']))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        
        _, num_of_vertices, num_of_features, len_input = x.shape
        
        
        #print('xxx',x.permute(0,3,2,1).shape)
        v = torch.matmul(x.permute(0, 3, 2, 1), self.U1)
        
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, model_topology):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = model_topology['K']
        self.cheb_polynomials = model_topology['cheb_polynomials']
        self.in_channels = model_topology['in_channels']
        self.out_channels = model_topology['nb_chev_filter']
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(model_topology['in_channels'], \
                                                                      model_topology['nb_chev_filter'])) for _ in \
                                       range(model_topology['K'])])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, len_input = x.shape

        outputs = []

        for time_step in range(len_input):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):

    def __init__(self, K, model_topology):
        
        super(ASTGCN_block, self).__init__()
        self.K = K
        self.model_topology = model_topology
        self.model_topology['K'] = K
        self.TAt = Temporal_Attention_layer(self.model_topology)
        self.SAt = Spatial_Attention_layer(self.model_topology)
        self.cheb_conv_SAt = cheb_conv_withSAt(self.model_topology)
        self.time_conv = nn.Conv2d(self.model_topology['nb_chev_filter'], self.model_topology['nb_time_filter'], \
                                   kernel_size=(1, 3), stride=(1, self.model_topology['time_strides']), padding=(0, 1))
        
        self.residual_conv = nn.Conv2d(self.model_topology['in_channels'], self.model_topology['nb_time_filter'], \
                                       kernel_size=(1, 1), stride=(1, self.model_topology['time_strides']))

        self.ln = nn.LayerNorm(self.model_topology['nb_time_filter'])  

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, len_input = x.shape

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, len_input),\
                             temporal_At).reshape(batch_size, num_of_vertices, num_of_features, len_input)

        # SAt
        spatial_At = self.SAt(x_TAt)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3).contiguous())  # (b,N,F,T)->(b,F,N,T) (1,3) ->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3).contiguous())  # (b,N,F,T)->(b,F,N,T) (1,1) -> (b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual


class ASTGCN_submodule(nn.Module):

    def __init__(self, K, model_topology):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(ASTGCN_submodule, self).__init__()
        
        
        self.K = K 
        
#         self.embbeded =  EmbeddingBlock(model_topology['common']['embedded_input'] ,\
#                                         model_topology['common']['embedded_output'] )
        
        self.final_layer_topology = model_topology['layer_' + str(model_topology['common']['nb_block'])]
        
        self.nb_block = model_topology['common']['nb_block']
        
        self.BlockList = nn.ModuleList([ASTGCN_block(K, model_topology['layer_1'])])
        
        self.BlockList.extend([ASTGCN_block(K, model_topology['layer_' + str(i+2)]) for i in\
                               range(model_topology['common']['nb_block']-1)])
                
        self.final_conv = nn.Conv2d(int(self.final_layer_topology['len_input']/ self.final_layer_topology['time_strides']), \
                                    self.final_layer_topology['num_for_predict']\
                                    , kernel_size=(1, self.final_layer_topology['nb_time_filter']))
        

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        
        
        #x = self.embbeded(x)
        
        
        
        for i, block in enumerate(self.BlockList):
            
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output

class ASTGCN(nn.Module):
    
    def __init__(self, K, model_topology):
        
        super(ASTGCN, self).__init__()
        
        self.K = K
        self.embbeded =  EmbeddingBlock(model_topology['common']['embedded_input'] ,\
                                        model_topology['common']['embedded_output'] )
        self.model_topology = model_topology
        self.nb_input = model_topology['common']['nb_input']
        self.submodules = nn.ModuleList([ASTGCN_submodule(self.K, model_topology)])
        self.submodules.extend([ASTGCN_submodule(self.K, model_topology) for _ in range(self.nb_input-1)])

            
    def forward(self, x1,x2,x3):
        x_list = [x1,x2,x3]
        
        if len(x_list) != len(self.submodules):
            raise ValueError("num of submodule not equals to length of the input list")
        
        for i in x_list:
            print(int(i.shape[1]))
        
        num_of_vertices_set = {int(i.shape[1]) for i in x_list}
        print('num_of_vertices_set',num_of_vertices_set)
        
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! "
                             "Check if your input data have same size"
                             "at axis 1.")              
        batch_size_set = {int(i.shape[0]) for i in x_list}
        
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")
            
        embedded_x_list = [self.embbeded(x_list[idx]) for idx in range(len(x_list))]
        
#         for i in range(len(embedded_x_list)):
#             print('%d : embedded_data shape is:' %i)
#             print(embedded_x_list[i].shape)                         
        
        submodule_outputs = [self.submodules[idx](embedded_x_list[idx])
                             for idx in range(len(embedded_x_list))]
        
        #print('output',sum(submodule_outputs).shape)
        return sum(submodule_outputs)
                
    
def make_model(model_topology, adj_mx):
    '''
    model_topology : dict for architecture hyper parameter

    '''
   
    L_tilde = scaled_Laplacian(adj_mx)
    
    K = model_topology['common']['K']
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor) for i in cheb_polynomial(L_tilde, K )]
                                                                                       
    
    for i in range(model_topology['common']['nb_block']):
        key = 'layer_'+ str(i+1)
        model_topology[key]['cheb_polynomials'] = cheb_polynomials

    
    model = ASTGCN(K, model_topology)
    for p in model.parameters():

        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model