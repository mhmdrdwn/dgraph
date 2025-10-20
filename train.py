#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:34:26 2025

@author: mohamedr
"""


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import torch
import numpy as np


def print_acc(model, data_iter):
    outs= []
    ys = []
    
    model.eval()
    with torch.no_grad():
        for batch in data_iter:
            x = batch.x.float()
            device = x.device
            idx = batch.edge_index.long().to(device)
            attr = batch.edge_attr.float().to(device)
            y = batch.y.to(device)
            batch = batch.batch.to(device)
            y = torch.argmax(y, -1)
            out = model(x, idx, attr, batch)
            outs.extend(out.cpu().detach().numpy())
            ys.extend(y.cpu().detach().numpy())
    
    outs = np.array(outs)
    ys = np.array(ys)
    outs = np.argmax(outs, -1)

    metrics = [accuracy_score(outs, ys), f1_score(outs, ys), 
               precision_score(outs, ys), recall_score(outs, ys)]
    return metrics

    
def train_model(model, num_epochs, data_iter, val_iter=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    highest_acc = 0 
    saved_model = None
    model.train()
    for epoch in tqdm(range(num_epochs)): 
        model.train()
        losses = 0
        for _, batch in enumerate(data_iter):
            x = batch.x.float()
            device = x.device
            idx = batch.edge_index.long().to(device)
            attr = batch.edge_attr.float().to(device)
            y = batch.y.to(device)
            batch = batch.batch.to(device)
            y = torch.argmax(y, -1)
            optimizer.zero_grad()
            out = model(x, idx, attr, batch)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses += loss.item() 
        val_acc = print_acc(model, val_iter)
        
        if val_acc[0] > highest_acc:
            saved_model = model

    return saved_model               