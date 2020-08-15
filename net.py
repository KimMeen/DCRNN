# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:11:54 2020

@author: Ming Jin
"""

from layers import DCGRUCell

import torch
import torch.nn as nn
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNEncoder(nn.Module):
    
    def __init__(self, 
                 adj_mx, 
                 diffusion_steps,
                 nodes,
                 input_dim,
                 hid_dim,
                 rnn_layers,
                 filter_type="dual_random_walk"):
        
        super().__init__()
        
        self.adj_mx = adj_mx
        self.diff_steps = diffusion_steps
        self.num_nodes = nodes
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_layers = rnn_layers
        self.filter_type = filter_type
        
        self.encoder_cells = nn.ModuleList()
        # first rnn layer
        self.encoder_cells.append(DCGRUCell(self.adj_mx, self.diff_steps, 
                                            self.num_nodes, self.input_dim,
                                            self.hid_dim, self.filter_type))
        # second layer and so on
        for _ in range(1, self.num_layers):
            self.encoder_cells.append(DCGRUCell(self.adj_mx, self.diff_steps, 
                                               self.num_nodes, self.hid_dim,
                                               self.hid_dim, self.filter_type))
            
    def forward(self, inputs, init_state):
        """
        inputs: [seq_length, batch, nodes, input_dim]
        init_state: [num_layers, batch, nodes*hid_dim]
        
        context: [num_layers, batch, nodes*hid_dim]
        outputs: i.e. current_inputs with shape [seq_length, batch, nodes*hid_dim]
        
        """
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))
        
        current_inputs = inputs
        context = []
        for layer_i in range(self.num_layers):
            hidden_state = init_state[layer_i]
            inner_output = []
            for t in range(seq_length):
                output, hidden_state = self.encoder_cells[layer_i](current_inputs[t, ...], hidden_state)
                inner_output.append(output)
            context.append(hidden_state)
            current_inputs = torch.stack(inner_output, dim=0).to(device)
        
        return context, current_inputs


class DCRNNDecoder(nn.Module):
    
    def __init__(self, 
                 adj_mx, 
                 diffusion_steps,
                 nodes,
                 input_dim,
                 hid_dim,
                 output_dim,
                 rnn_layers,
                 filter_type="dual_random_walk"):
        
        super().__init__()  
         
        self.adj_mx = adj_mx
        self.diff_steps = diffusion_steps
        self.num_nodes = nodes
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_layers = rnn_layers
        self.filter_type = filter_type
        
        self.decoder_cells = nn.ModuleList()
        self.decoder_cells.append(DCGRUCell(self.adj_mx, self.diff_steps, 
                                            self.num_nodes, self.input_dim,
                                            self.hid_dim, self.filter_type))
        
        # second layer and so on, except for the last layer
        for _ in range(1, self.num_layers):
            self.decoder_cells.append(DCGRUCell(self.adj_mx, self.diff_steps, 
                                                self.num_nodes, self.hid_dim,
                                                self.hid_dim, self.filter_type))
            
       # final linear mapping
        self.FCN = nn.Linear(self.hid_dim, self.output_dim)
        
    def forward(self, inputs, init_state, teaching_force_ratio):
        """
        inputs: ground truth with shape [horizon + 1, batch, nodes, input_dim]
        init_state: [num_layers, batch, nodes*hid_dim]
        
        outputs: [horizon, batch, nodes*output_dim]
        """
        horizon_1 = inputs.shape[0]   # horizon + 1
        batch_size = inputs.shape[1]
        inputs = torch.reshape(inputs, (horizon_1, batch_size, -1))
        
        # placeholder
        outputs = torch.zeros(horizon_1, batch_size, self.num_nodes*self.output_dim).to(device)
        
        current_input = inputs[0]  # the first element is GO token
        for t in range(1, horizon_1):
            propagate_hidden_state = []
            for layer_i in range(self.num_layers):
                hidden_state = init_state[layer_i]
                output, hidden_state = self.decoder_cells[layer_i](current_input, hidden_state)
                current_input = output
                propagate_hidden_state.append(hidden_state)
            init_state = torch.stack(propagate_hidden_state, dim=0)  # [num_layers, batch, nodes*hid_dim]
            # collect otuput
            output = torch.reshape(output, (-1, self.hid_dim))   # [batch*nodes, hid_dim]
            output = torch.reshape(self.FCN(output), (batch_size, self.num_nodes*self.output_dim))  # [batch, nodes*output_dim]
            outputs[t] = output
            teacher_force = random.random() < teaching_force_ratio  # a bool value
            current_input = (inputs[t] if teacher_force else output)
        
        return outputs

    
class DCRNNModel(nn.Module):
    
    def __init__(self, 
                 adj_mx,
                 diffusion_steps,
                 nodes,
                 batch_size,
                 enc_input_dim,
                 dec_input_dim,
                 hid_dim,
                 output_dim,
                 rnn_layers,
                 filter_type="dual_random_walk"):
        
        super().__init__()
        
        self.adj_mx = adj_mx
        self.diff_steps = diffusion_steps
        self.num_nodes = nodes
        self.batch_size = batch_size
        self.enc_input_dim = enc_input_dim
        self.dec_input_dim = dec_input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_layers = rnn_layers
        self.filter_type = filter_type
        
        self.go_token = torch.zeros(1, self.batch_size, self.num_nodes * self.output_dim, 1).to(device)
        
        self.encoder = DCRNNEncoder(self.adj_mx, self.diff_steps, 
                                    self.num_nodes, self.enc_input_dim,
                                    self.hid_dim, self.num_layers, self.filter_type)
        
        self.decoder = DCRNNDecoder(self.adj_mx, self.diff_steps, 
                                    self.num_nodes, self.dec_input_dim,
                                    self.hid_dim, self.output_dim,
                                    self.num_layers, self.filter_type)
        
    def forward(self, source, target, teaching_force_ratio):
        """
        source: historical observations [batch, seq_len, nodes, enc_input_dim]
        target: sequence to predict [batch, horizon, nodes, dec_input_dim]
        
        Theoretically, enc_input_dim should equals to dec_input_dim (e.g., Speed)
        but enc_input_dim in this case will be 2, and dec_input_dim will be 1
        cuz enc_input_dim has encoded time_in_day as extra features, which is not in dec_input_dim
        
        dec_input_dim should equals to output_dim, which is 1 corresponding to the speed attribute
        """
        source = torch.transpose(source, dim0=0, dim1=1)  # [seq_len, batch, nodes, enc_input_dim]
        target = torch.transpose(target[..., :self.output_dim], dim0=0, dim1=1)  # [horizon, batch, nodes, dec_input_dim]
        target = torch.cat([self.go_token, target], dim=0)  # [horizon + 1, batch, nodes, dec_input_dim]
        
        # initial hidden state for the encoder
        init_hidden_state = []
        for i in range(self.num_layers):
            init_hidden_state.append(torch.zeros(self.batch_size, self.num_nodes * self.hid_dim))
        init_hidden_state = torch.stack(init_hidden_state, dim=0).to(device)  # [num_layers, batch, nodes*hid_dim]
        
        context, _ = self.encoder(source, init_hidden_state)  # [num_layers, batch, nodes*hid_dim]
        outputs = self.decoder(target, context, teaching_force_ratio=teaching_force_ratio)
        
        # the elements of the first time step of the outputs are all zeros.
        return outputs[1:, :, :]  # [horizon, batch_size, num_nodes*output_dim]