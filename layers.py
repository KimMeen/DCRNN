# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:00:11 2020

@author: Ming Jin
"""

import torch
import torch.nn as nn
import numpy as np
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionGraphConv(nn.Module):
    """
    Diffusion graph convolution layer (Eq.2 and 3)
    
    adj_mx: Adjacency matrix (np.ndarray)
    nodes: Number of nodes
    input_dim: P dim in the paper
    output_dim: Q dim in the paper
    filter_type: Transition matrix generation
    activation: Activation in Eq. 3
    
    ** Notice: 
        DO NOT place .to(device) on weight and bias manually otherwise they will not be trained
    """
    def __init__(self, 
                 adj_mx, 
                 diffusion_steps,
                 nodes,
                 input_dim, 
                 output_dim, 
                 filter_type="dual_random_walk", 
                 activation=None):
        
        super().__init__()
        
        trans_matrices = []
        self.trans_matrices = []
        
        if filter_type == "laplacian":
            trans_matrices.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            trans_matrices.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            trans_matrices.append(utils.calculate_random_walk_matrix(adj_mx))
            trans_matrices.append(utils.calculate_random_walk_matrix(adj_mx.T))
        else:
            trans_matrices.append(utils.calculate_scaled_laplacian(adj_mx))
            
        for trans_mx in trans_matrices:
            self.trans_matrices.append(utils.build_sparse_matrix(trans_mx).to(device))  # to PyTorch sparse tensor
        
        self.diff_steps = diffusion_steps  # K is diffusion_steps
        self.num_matrices = len(self.trans_matrices) * diffusion_steps + 1  # K*2 is num_matrices; Don't forget to add for x itself
        self.num_nodes = nodes
        self.input_size = input_dim  # P is input_dim
        self.output_size = output_dim  # Q is output_dim
        self.activation = activation
        
        self.weight = nn.Parameter(torch.FloatTensor(self.input_size*self.num_matrices, self.output_size))  # input_size*num_matrices = Q*P*K*2
        self.bias = nn.Parameter(torch.FloatTensor(self.output_size))
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Reinitialize learnable parameters
        """
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.bias.data, val=0.0)
    
    def forward(self, inputs):
        """        
        inputs: [batch_size, num_nodes * input_dim]  ** Notice that we have concatenated X with the state
        outputs: [batch_size, num_nodes * output_dim]
        
        """
        batch_size = inputs.shape[0]
        inputs_and_state = torch.reshape(inputs, (batch_size, self.num_nodes, self.input_size))  # reshape to [batch, nodes, inp_dim]
        
        x = inputs_and_state  # [batch, nodes, inp_dim]
        x0 = torch.transpose(x, dim0=0, dim1=1)
        x0 = torch.transpose(x0, dim0=1, dim1=2)  # [nodes, inp_dim, batch]
        x0 = torch.reshape(x0, shape=[self.num_nodes, self.input_size * batch_size])  # [nodes, inp_dim * batch]
        x = torch.unsqueeze(x0, dim=0)  # [1, nodes, inp_dim * batch]
        
        if self.diff_steps == 0:
            pass
        else:
            for trans_mx in self.trans_matrices:
                x1 = torch.sparse.mm(trans_mx, x0)
                x = torch.cat([x, torch.unsqueeze(x1, dim=0)], dim=0)
                for k in range(2, self.diff_steps + 1):
                    x2 = 2 * torch.sparse.mm(trans_mx, x1) - x0  # why?
                    x = torch.cat([x, torch.unsqueeze(x2, dim=0)], dim=0)
                    x1, x0 = x2, x1
        
        # before reshape: [num_matrices, nodes, inp_dim * batch]
        x = torch.reshape(x, shape=[self.num_matrices, self.num_nodes, self.input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)  # [batch_size, num_nodes, input_size, order]
        x = torch.reshape(x, shape=[batch_size * self.num_nodes, self.input_size * self.num_matrices])
        x = torch.add(torch.matmul(x, self.weight), self.bias)  # (batch_size * self._num_nodes, output_size)
        
        if self.activation is not None:
            return self.activation(torch.reshape(x, [batch_size, self.num_nodes * self.output_size]))
        else:
            return torch.reshape(x, [batch_size, self.num_nodes * self.output_size])
               
    
class DCGRUCell(nn.Module):                
    """
    Diffusion Convolutional Gated Recurrent Unit (Sec. 2.3)
    
    adj_mx: Adjacency matrix (np.ndarray)
    nodes: Number of nodes
    input_dim: P dim in the paper
    hid_dim: Hidden dim of the RNN
    filter_type: Transition matrix generation
    activation: Activation in Eq. 3
    
    """
    def __init__(self, 
                 adj_mx, 
                 diffusion_steps,
                 nodes,
                 input_dim,
                 hid_dim,
                 filter_type="dual_random_walk",
                 activation=torch.tanh):
        
        super().__init__()                
        
        self.adj_mx = adj_mx
        self.diff_steps = diffusion_steps
        self.num_nodes = nodes
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.filter_type = filter_type
        self.activation = activation
            
        self.diff_conv_gate = DiffusionGraphConv(self.adj_mx, 
                                                 self.diff_steps,
                                                 self.num_nodes,
                                                 self.input_dim + self.hid_dim,
                                                 2 * self.hid_dim,
                                                 self.filter_type)
        
        self.diff_conv_candidate = DiffusionGraphConv(self.adj_mx, 
                                                      self.diff_steps,
                                                      self.num_nodes,
                                                      self.input_dim + self.hid_dim,
                                                      self.hid_dim,
                                                      self.filter_type)
    
    def forward(self, inputs, state):
        """
        inputs: [batch, nodes*input_dim]
        state: [batch, nodes*hid_dim]
        
        outputs: [batch, nodes*hid_dim]
        new_state: [batch, nodes*hid_dim]
        """
        # concat inputs and state
        i = torch.reshape(inputs, (inputs.shape[0], self.num_nodes, self.input_dim))
        s = torch.reshape(state, (state.shape[0], self.num_nodes, self.hid_dim))
        inp_with_state = torch.reshape(torch.cat([i, s], dim=2), (inputs.shape[0], -1))  # [batch, nodes*(inp_dim+hid_dim)] 
        # convs in r and u 
        value = torch.sigmoid(self.diff_conv_gate(inp_with_state))
        value = torch.reshape(value, (-1, self.num_nodes, 2 * self.hid_dim))
        r, u = torch.split(value, split_size_or_sections=self.hid_dim, dim=-1)  # r and u: [batch, nodes, hid_dim]
        r = torch.reshape(r, (-1, self.num_nodes * self.hid_dim))
        u = torch.reshape(u, (-1, self.num_nodes * self.hid_dim))
        # calculate ct
        r_dot_state = r * state  # [batch, nodes*hid_dim]
        rs = torch.reshape(r_dot_state, (r_dot_state.shape[0], self.num_nodes, self.hid_dim))
        inp_with_rstate = torch.reshape(torch.cat([i, rs], dim=2), (inputs.shape[0], -1))   # [batch, nodes*(inp_dim+hid_dim)]
        c = self.diff_conv_candidate(inp_with_rstate)  # [batch, nodes*hid_dim]
        if self.activation is not None:
            c = self.activation(c)
        # calculate output & new state
        outputs = new_state = u * state + (1 - u) * c  # [batch, nodes*hid_dim]
        
        return outputs, new_state