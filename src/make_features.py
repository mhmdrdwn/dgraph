#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:34:26 2025

@author: mohamedr
"""


from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import DataLoader

import torch

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from scipy.stats import pearsonr

from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm

import numpy as np

from sklearn.model_selection import train_test_split
from mne_connectivity import spectral_connectivity_time


def gc(x1, x2):
    X = np.vstack([x1, x2]).T
    gc = grangercausalitytests(X, [2], addconst=True, verbose=False)[2][0]['ssr_ftest'][1]
    return gc
    
     


def calc_conn(data, method):
    min_freq = 5
    max_freq = 45
    freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 2 + 1))
    nfreqs = len(freqs)
    n_channels = 19
    n_epochs = data.shape[0]
    fs = 100
    conns = np.zeros((n_epochs, n_channels, n_channels, nfreqs))
    conn = spectral_connectivity_time(data, method=method, mode="cwt_morlet", 
                                      sfreq=fs, freqs=freqs, n_cycles=5,
                                      verbose=False, fmin=min_freq, fmax=max_freq)
    
    conns = conn.get_data(output="dense")

    conns = np.mean(conns, -1)
    for epoch_conn_idx in range(n_epochs):
        conns[epoch_conn_idx, :, :] = np.maximum(conns[epoch_conn_idx, :, :], 
                                                 conns[epoch_conn_idx, :, :].transpose())
    return conns


def gen_graphs(X, conn_metric):
    #eegs(snapshots, bands, timpoints)
    n_epochs, n_channels, ntimes = X.shape 
    
    conn = np.zeros((n_channels, n_channels, n_epochs))
    if conn_metric in ["plv", "coh", "wpli", "ciplv"]: 
        for i in range(n_epochs):
            c = calc_conn(X, conn_metric)
            conn[:, :, i] = c
        return conn

    elif conn_metric == "gc":
        for i in range(n_epochs):
            for ch_i in range(n_channels):
                for ch_j in range(n_channels):
                    c = gc(X[ch_i], X[ch_j])
                    conn[ch_i, ch_j, i] = c
        return conn
        
    
    """    
    for i in range(num_nodes):
        c1 = []
        for j in range(num_nodes):
            if cal_conn == "pearson":
                conn = pearsonr(eegs[i], eegs[j])[0]
            elif cal_conn == "cc":
                conn = calculate_cc(eegs, i, j)
            elif cal_conn == "plv": 
                #conn = hilphase(eegs[i], eegs[j])
                conn = calc_conn(eegs, cal_conn)
            elif cal_conn == "pli":
                conn = pli(eegs, i, j)
            elif cal_conn == "gc": 
                conn = gc(eegs[i], eegs[j])
            elif cal_conn == "mi": 
                conn = sp.mutual_Info(eegs[i], eegs[j])
            elif cal_conn == "con-entropy": 
                conn = sp.entropy_cond(eegs[i],eegs[j])
            elif cal_conn == "cross-entropy": 
                conn = sp.entropy_cross(eegs[i],eegs[j])
            elif cal_conn == "kld-entropy": 
                conn = sp.entropy_kld(eegs[i],eegs[j])
            elif cal_conn == "joint-entropy": 
                conn = sp.entropy_joint(eegs[i],eegs[j])
            c1.append(conn)
        c.append(c1)
    """
    #return c


def gen_adj(X, y, method):
    print("Gen features")
    graphs = []
    
    n_samples, n_epochs, n_channels, ntimes = X.shape 
    
    graphs = np.zeros((n_samples, n_epochs, n_channels, n_channels))
    print("calculating connectivity")
    for i in tqdm(range(X.shape[0])):
        if method in ["plv", "coh", "wpli", "ciplv"]: #plv, pli, coh From mne framework
            g = calc_conn(X[i], method=method)
            g = np.array(g).squeeze()
            graphs[i, :, :, :] = g
        elif method == "gc": #Granger Causality
            g = np.zeros((n_epochs, n_channels, n_channels))
            for epoch_idx in range(n_epochs):
                for ch_i in range(n_channels):
                    for ch_j in range(n_channels):
                        c = gc(X[i, epoch_idx, ch_i, :], X[i, epoch_idx, ch_j, :])
                        g[epoch_idx, ch_i, ch_j] = c
            graphs[i, :, :, :] = g
        elif method == "pc": #pearson correlation
            g = np.zeros((n_epochs, n_channels, n_channels))
            for epoch_idx in range(n_epochs):
                for ch_i in range(n_channels):
                    for ch_j in range(n_channels):
                        c = pearsonr(X[i, epoch_idx, ch_i, :], X[i, epoch_idx, ch_j, :])[0]
                        g[epoch_idx, ch_i, ch_j] = c
            graphs[i, :, :, :] = g
            
    return X, graphs, y



def train_test(train_X, test_X, train_y, test_y, method, use_test_windows=False):
    print("read_data")
        
    train_X, train_graphs, train_y = gen_adj(train_X, train_y, method=method)
    test_X, test_graphs, test_y = gen_adj(test_X, test_y, method=method)
    
    return train_X, train_graphs, train_y, test_X, test_graphs, test_y
    
    
    


def build_pyg_dl(x, a, y, time_points, device):
    """Convert features and adjacency to PyTorch Geometric Dataloader"""
    a = torch.from_numpy(a)
    a = a + 1e-10 
    edge_attr = []
    
    for edge_time_idx in range(time_points):
        Af = a[edge_time_idx, :, :]
        Af.fill_diagonal_(1)
        edge_index, attrf = dense_to_sparse(Af)
        edge_attr.append(attrf)
    
    edge_attr = torch.stack(edge_attr)
    edge_attr = torch.moveaxis(edge_attr, 0, 1).to(device)
    edge_index = edge_index.to(device)
    x = torch.from_numpy(x).to(device)
    y = torch.tensor([y], dtype=torch.float).to(device)
    data = Data(x=x, edge_index=edge_index, 
                edge_attr=edge_attr, 
                y=y)
    return data

        
def fill_diag(x):
    num_channels = x.shape[1]
    num_bands = x.shape[0]
    x_diag = np.zeros((num_bands, num_channels, num_channels))
    for band_idx, band in enumerate(x):
        for idx, i in enumerate(band):
            x_diag[band_idx, idx, idx] = 0
            if idx == 0:
                x_diag[band_idx, idx, 1:] = i
            elif idx > 0 and idx < num_channels-1:
                x_diag[band_idx, idx, idx+1:] = i[idx:]
                x_diag[band_idx, idx, :idx] = i[:idx]
            elif idx == num_channels-1:
                x_diag[band_idx, idx, :-1] = i
    return x_diag


def loaders(train_X, train_graphs, train_y, test_X, test_graphs, test_y, device, batch_size, num_windows):
    #ohe
    ohe = OneHotEncoder()
    train_y_ohe = ohe.fit_transform(train_y).toarray()
    test_y_ohe = ohe.transform(test_y).toarray()
    
    # build pyg dataloader
    train_dataset = [build_pyg_dl(x, g, y, num_windows, device) for x, g, y in zip(train_X, train_graphs, train_y_ohe)]
    test_dataset = [build_pyg_dl(x, g, y, num_windows, device) for x, g, y in zip(test_X, test_graphs, test_y_ohe)]
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter



def norm_adj(train_graphs, test_graphs):
    for i in range(train_graphs.shape[0]):
        for j in range(train_graphs.shape[1]):
            min_ = (train_graphs[i, j, :, :]).min()
            max_ = (train_graphs[i, j, :, :]).max()
            train_graphs[i, j, :,  :] = (train_graphs[i, j, :,  :] - min_)/(max_ - min_)
                
    for i in range(test_graphs.shape[0]):
        for j in range(test_graphs.shape[1]):
            min_ = (test_graphs[i, j, :, :]).min()
            max_ = (test_graphs[i, j, :, :]).max()
            test_graphs[i, j, :, :] = (test_graphs[i, j, :,  :] - min_)/(max_ - min_)
            
    return train_graphs, test_graphs


def train_val(files, device, num_epochs=200):
    all_train_losses = []
    all_val_losses = []

    train_subset_files, val_files = train_test_split(files, test_size=0.2, random_state=2025)
    train_iter, val_iter = train_test(train_files=train_subset_files, 
                                      test_files=val_files,
                                      num_windows=100,
                                      cal_conn="pearson",
                                      use_test_windows=False,
                                      device = device)
    
    model = EEGModel(num_nodes=19, node_features=51, num_classes=2, num_windows=100, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    print("Training model")
    
    for epoch in range(num_epochs):
        losses = 0
        for idx, (X, A, y) in enumerate(tqdm(train_iter)):
            optimizer.zero_grad()
            out = model(X, A)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses += loss.item()*X.shape[0]
        losses = losses/len(train_iter.dataset)
        print("Epoch ", epoch+1, ":")
        print("train loss", losses)
        all_train_losses.append(losses)
        
        losses = 0
        for idx, (X, A, y) in enumerate(val_iter):
            out = model(X, A)
            loss = criterion(out, y)
            losses += loss.item()*X.shape[0]
        losses = losses/len(val_iter.dataset)
        print("val loss", losses)
        all_val_losses.append(losses)

    return all_train_losses, all_val_losses