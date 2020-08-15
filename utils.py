# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:01:56 2020

@author: Ming Jin
"""
import os
import torch
import tqdm
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

        
############################GRAPH RELATED FUNCTIONS########################

def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    
    """
    Construct adjacency matrix by using distances between sensors
    Eq. (10) in STGCN paper
    
    ----------------------- 
    distance_df : DataFrame
        Sensor distances, data frame with three columns: [from, to, distance]
    sensor_ids : List
        List of sensor ids
    normalized_k : Int
        Entries that become lower than normalized_k after normalization are set to zero for sparsity
    
    -------------
    adj_mx: Numpy array
        Adjacency matrix
        
    """
    # init dist_mx
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    
    # builds sensor id to index map
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
    
    # fills cells in the dist_mx with distances
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()  # get all valid distances
    std = distances.std()
    # variation of equation (10) in the paper
    adj_mx = np.exp(-np.square(dist_mx / std))
    # make the adjacent matrix symmetric by taking the max
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # sets entries that lower than a threshold, i.e., k, to zero for sparsity
    # normalized_k is epsilon in equation (10) to control the sparsity 
    adj_mx[adj_mx < normalized_k] = 0
    
    return adj_mx


def calculate_normalized_laplacian(adj):
    
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    D = diag(A 1)
    
    Calculate normalized laplacian L by using given adj_mx
    Normalized laplacian is similar to the negative random walk transition matrix
    (See Appendix B in the paper)
    
    ----------------------- 
    adj_mx: Numpy array
        Adjacency matrix
    
    -------------
    normalized_laplacian: ?
        Adjacency matrix

    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian  # I - D^-1/2 A D^-1/2


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    
    """   
    Calculate transition matrix by using provided adj_mx
    Random walk transition matrix is similar to the negative normalized laplacian
    (See Appendix B in the paper)
    
    ----------------------- 
    adj_mx: Numpy array
        Adjacency matrix
    lambda_max: Int
        I dont know the meaning of it
    undirected: Bool
        Transform a directed/undirected adj_mx to a undirected one. 
        Default value is true because normalized laplacian is for undirected graph 
    -------------
    L: Numpy array
        Transition matrix calculated by normalized laplacian

    """
    if undirected:
        # transform directed graph to a undirected one (make it symmetric)
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    # calculate normalized laplacian based on given adj_mx
    L = calculate_normalized_laplacian(adj_mx)
    # I dont know the meaning of lambda_max here...
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    # identity matrix
    I = sp.identity(M, format='csr', dtype=L.dtype)
    # D^-1 W = - (I - D^-1/2 A D^-1/2)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def calculate_random_walk_matrix(adj_mx):
    
    """   
    Calculate random walk transition matrix directly
    This method is for both directed and undirected graphs
    
    ----------------------- 
    adj_mx: Numpy array
        Adjacency matrix
    -------------
    random_walk_mx: Numpy array
        Transition matrix calculated by D^-1 W

    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.float_power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()  # D^-1 * A
    return random_walk_mx


def build_sparse_matrix(trans_mx):
    
    """   
    Transform a transition matrix to PyTorch sparse matrix
    
    ----------------------- 
    trans_mx: Numpy array
        Transition matrix
    -------------
    Output: Sparse tensor
        Transition matrix

    """
    shape = trans_mx.shape
    i = torch.LongTensor(np.vstack((trans_mx.row, trans_mx.col)).astype(int))
    v = torch.FloatTensor(trans_mx.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

##############################################################################

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
  

def load_dataset(dataset_dir, batch_size, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()
    
##############################################################################

def masked_mae_loss(preds, labels, null_val=np.nan):
    """
    mask[i] = 1.0 if labels[i] != null_val
    
    Calculate the loss based on the part where labels != null_val
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = torch.ne(labels, null_val)
    mask = mask.to(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)  # fill nan to zeros
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)  # fill nan to zeros
    return torch.mean(loss)

def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)
    
def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)
    
def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)
    
def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))
