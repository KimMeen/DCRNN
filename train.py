# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:27:03 2020

@author: Ming Jin
"""

import math
import tqdm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import utils
from net import DCRNNModel

# import sys
# sys.path.append("./xlwang_version")
# from dcrnn_model import DCRNNModel

"""
Hyperparameters

"""
batch_size = 64
enc_input_dim = 2
dec_input_dim = 1
hidden_dim = 64
output_dim = 1
diffusion_steps = 2
num_nodes = 207
rnn_layers = 2
seq_length = 12
horizon = 12
cl_decay_steps = 2000  # decrease teaching force ratio in global steps
filter_type = "dual_random_walk"

epochs = 100
lr = 0.01
weight_decay = 0.0
epsilon = 1.0e-3
amsgard = True
lr_decay_ratio = 0.1
lr_decay_steps = [20, 30, 40, 50]
max_grad_norm = 5

checkpoints = './checkpoints/dcrnn.pt'
sensor_ids = './data/METR-LA/graph_sensor_ids.txt'
sensor_distance = './data/METR-LA/distances_la_2012.csv'
recording='data/processed/METR-LA'

"""
Dataset

"""
# read sensor IDs
with open(sensor_ids) as f:
    sensor_ids = f.read().strip().split(',')

# read sensor distance
distance_df = pd.read_csv(sensor_distance, dtype={'from': 'str', 'to': 'str'})

# build adj matrix based on equation (10)
adj_mx = utils.get_adjacency_matrix(distance_df, sensor_ids)

data = utils.load_dataset(dataset_dir=recording, batch_size=batch_size, test_batch_size=batch_size)
train_data_loader = data['train_loader']
val_data_loader = data['val_loader']
test_data_loader = data['test_loader']
standard_scaler = data['scaler']

"""
Init model

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DCRNNModel(adj_mx, 
                    diffusion_steps, 
                    num_nodes, 
                    batch_size, 
                    enc_input_dim, 
                    dec_input_dim, 
                    hidden_dim, 
                    output_dim,
                    rnn_layers,
                    filter_type).to(device)

# model = DCRNNModel(adj_mx, 
#                     batch_size, 
#                     enc_input_dim, 
#                     dec_input_dim, 
#                     diffusion_steps, 
#                     num_nodes, 
#                     rnn_layers, 
#                     hidden_dim,
#                     horizon,
#                     output_dim,
#                     filter_type).to(device)

optimizer = torch.optim.Adam(model.parameters(), 
                             lr=lr, eps=epsilon, 
                             weight_decay=weight_decay, 
                             amsgard=amsgard)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=lr_decay_steps, 
                                                    gamma=lr_decay_ratio)

"""
DCRNN Training
"""
def compute_mae_loss(y_true, y_predicted, standard_scaler):
    y_true = standard_scaler.inverse_transform(y_true)
    y_predicted = standard_scaler.inverse_transform(y_predicted)
    return utils.masked_mae_loss(y_predicted, y_true, null_val=0.0)

def eval_metrics(y_true_np, y_predicted_np, standard_scaler):
    metrics = np.zeros(3)
    y_true_np = standard_scaler.inverse_transform(y_true_np)
    y_predicted_np = standard_scaler.inverse_transform(y_predicted_np)
    mae = utils.masked_mae_np(y_predicted_np, y_true_np, null_val=0.0)
    mape = utils.masked_mape_np(y_predicted_np, y_true_np, null_val=0.0)
    rmse = utils.masked_rmse_np(y_predicted_np, y_true_np, null_val=0.0)
    metrics[0] += mae
    metrics[1] += mape
    metrics[2] += rmse
    return metrics
    
# some pre-calculated properties
num_train_iteration_per_epoch = math.ceil(data['x_train'].shape[0] / batch_size)
num_val_iteration_per_epoch = math.ceil(data['x_val'].shape[0] / batch_size)
num_test_iteration_per_epoch = math.ceil(data['x_test'].shape[0] / batch_size)

# start training
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Total number of trainable parameters:", params)
print("Initialization complete. Start training... ==>", epochs, "epochs with", num_train_iteration_per_epoch, "batches per epoch.")

for epoch in range(1, epochs + 1):
    
    model.train()
    
    train_iterator = train_data_loader.get_iterator()
    val_iterator = val_data_loader.get_iterator()
    total_loss = 0.0
    total_metrics = np.zeros(3)  # Three matrics: MAE, MAPE, RMSE
    total_val_metrics = np.zeros(3)
    
    for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_iterator)):
        
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        y_true  = y[..., :output_dim]  # delete time encoding to form as label
        # x:[batch, seq_len, nodes, enc_input_dim]
        # y:[batch, horizon, nodes, output_dim + 1]
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # compute teaching force ratio: decrease this gradually to 0
        global_steps = (epoch - 1) * num_train_iteration_per_epoch + batch_idx
        teaching_force_ratio = cl_decay_steps / (cl_decay_steps + math.exp(global_steps / cl_decay_steps))
        
        # feedforward
        y_hat = model(x, y, teaching_force_ratio)  # [horizon, batch, nodes*output_dim]
        y_hat = torch.transpose(torch.reshape(y_hat, (horizon, batch_size, num_nodes, output_dim)), 0, 1)  # [batch, horizon, nodes, output_dim]
        
        # back propagation
        loss = compute_mae_loss(y_true, y_hat.cpu(), standard_scaler)
        loss.backward()
        
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
                
        # training statistics
        total_loss += loss.item()
        t_metrics = eval_metrics(y_true.numpy(), y_hat.detach().cpu().numpy(), standard_scaler)
        total_metrics += t_metrics
        # print('Batch_idx {:03d} | TF {:.4f} | Train MAE {:.5f} | Train MAPE {:.5f} | Train RMSE {:.5f}'.format(
        #     batch_idx, teaching_force_ratio, loss.item(), t_metrics[1], t_metrics[2]))

    # validation after each epoch
    model.eval()
    with torch.no_grad():
        for _, (val_x, val_y) in enumerate(tqdm.tqdm(val_iterator)):
            val_x = torch.FloatTensor(val_x)
            val_y = torch.FloatTensor(val_y)
            val_y_true  = val_y[..., :output_dim]  # delete time encoding to form as label
            # val_x:[batch, seq_len, nodes, enc_input_dim]
            # val_y:[batch, horizon, nodes, output_dim + 1]
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_y_hat = model(val_x, val_y, 0)
            val_y_hat = torch.transpose(torch.reshape(val_y_hat, (horizon, batch_size, num_nodes, output_dim)), 0, 1)  # [batch, horizon, nodes, output_dim]
            total_val_metrics += eval_metrics(val_y_true.numpy(), val_y_hat.detach().cpu().numpy(), standard_scaler)
            
    # learning rate scheduling
    lr_scheduler.step()
    
    # GPU mem usage
    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
    
    # save model every epoch
    torch.save(model.state_dict(), checkpoints)
    
    # logging
    val_metrics = (total_val_metrics / num_val_iteration_per_epoch).tolist()
    print('Epoch {:03d} | lr {:.6f} |Train loss {:.5f} | Val MAE {:.5f} | Val MAPE {:.5f} | Val RMSE {:.5f}| GPU {:.1f} MiB'.format(
        epoch, optimizer.param_groups[0]['lr'], total_loss / num_train_iteration_per_epoch, val_metrics[0], val_metrics[1], val_metrics[2], gpu_mem_alloc))

print("Training complete.")


"""
DCRNN Testing
"""

print("\nmodel testing...")
test_iterator = test_data_loader.get_iterator()
total_test_metrics = np.zeros(3)
model.eval()
with torch.no_grad():
    for _, (test_x, test_y) in enumerate(tqdm.tqdm(test_iterator)):
        test_x = torch.FloatTensor(test_x)
        test_y = torch.FloatTensor(test_y)
        test_y_true  = test_y[..., :output_dim]  # delete time encoding to form as label
        # test_x:[batch, seq_len, nodes, enc_input_dim]
        # test_y:[batch, horizon, nodes, output_dim + 1]
        test_x, test_y = test_x.to(device), test_y.to(device)
        test_y_hat = model(test_x, test_y, 0)
        test_y_hat = torch.transpose(torch.reshape(test_y_hat, (horizon, batch_size, num_nodes, output_dim)), 0, 1)  # [batch, horizon, nodes, output_dim]
        total_test_metrics += eval_metrics(test_y_true.numpy(), test_y_hat.detach().cpu().numpy(), standard_scaler)
        
test_metrics = (total_test_metrics / num_test_iteration_per_epoch).tolist()
print('Test MAE {:.5f} | Test MAPE {:.5f} | Test RMSE {:.5f}'.format(test_metrics[0], test_metrics[1], test_metrics[2]))   