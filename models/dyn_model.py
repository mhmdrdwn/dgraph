#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 07:34:26 2025

@author: mohamedr
"""


import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric_temporal.nn.recurrent import DCRNN, GConvGRU, GConvLSTM, GCLSTM
from models.operators.A3TGCN import A3TGCN
from models.operators.TGCN import TGCN


class GraphTemporal(nn.Module):
    def __init__(self, num_ch, num_t, op):
        super(GraphTemporal, self).__init__()
        self.linear = nn.Linear(32, 2)
        self.BN2 = nn.BatchNorm1d(56)
        self.BN1 = nn.BatchNorm1d(19)
        self.relu = nn.ReLU()

        # CNN feature extractors
        self.conv1 = nn.Conv2d(num_ch, num_ch, (3, 3), stride=(1, 2),
                               padding=(1, 0), dilation=(1, 3))
        self.conv2 = nn.Conv2d(num_ch, num_ch, (3, 5), stride=(1, 2),
                               padding=(1, 1), dilation=(1, 3))
        self.conv3 = nn.Conv2d(num_ch, num_ch, (3, 10), stride=(1, 2),
                               padding=(1, 3), dilation=(1, 3))
        self.maxpool = nn.MaxPool2d((1, 5))

        # Choose temporal graph operator
        if op == "GCLSTM":
            self.GraphOp = GCLSTM(in_channels=56, out_channels=32, K=2)
        elif op == "GConvLSTM":
            self.GraphOp = GConvLSTM(in_channels=56, out_channels=32, K=2)
        elif op == "A3TGCN":
            self.GraphOp = A3TGCN(in_channels=56, out_channels=32, periods=num_t)
        elif op == "TGCN":
            self.GraphOp = TGCN(in_channels=56, out_channels=32)
        elif op == "DCRNN":
            self.GraphOp = DCRNN(in_channels=56, out_channels=32, K=2)
        elif op == "GConvGRU":
            self.GraphOp = GConvGRU(in_channels=56, out_channels=32, K=2)
        else:
            raise ValueError(f"Unknown temporal operator: {op}")

        self.num_ch = num_ch
        self.num_t = num_t
        self.op = op

    def forward(self, x, idx, attr, batch=None):
        device = next(self.parameters()).device
        x, idx, attr = x.to(device), idx.to(device), attr.to(device)
        if batch is not None:
            batch = batch.to(device)

        batch_size = int(x.shape[0] / self.num_ch)
        x = x.reshape(batch_size, self.num_ch, x.shape[-2], x.shape[-1])

        # Temporal CNN feature extraction
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.maxpool(x)

        # Prepare for GNN input
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1)
        x = self.BN2(x)
        x = x.permute(0, 2, 1)

        # Temporal Graph Operator
        if self.op == "A3TGCN":
            HS = self.GraphOp(x, idx, attr)
        else:
            HS = self._temporal_loop(x, idx, attr)

        out = self.relu(HS)
        out = global_mean_pool(out, batch)
        out = self.linear(out)
        return out

    @torch.jit.ignore
    def _temporal_loop(self, x, idx, attr):
        """Runs time steps sequentially (no Python overhead from outside calls)."""
        HS = None
        for t_idx in range(self.num_t):
            attr_t = attr[:, t_idx]
            x_t = x[:, t_idx]
            if t_idx == 0:
                HS = self.GraphOp(x_t, idx, attr_t)
            else:
                HS = self.GraphOp(x_t, idx, attr_t, HS)
            if isinstance(HS, tuple):
                HS = HS[0]
        return HS
