#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/7 9:38
# @Version : 1.0
# @File    : SelfAttention.py

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

class SelfAttention(nn.Module):
    """"""

    def __init__(self, n_h, d_model, d_k, d_v, device, dropout=0.5):
        super(SelfAttention, self).__init__()
        """"""
        self.n_h = n_h
        self.d_k = d_k
        self.d_v = d_v
        self.device = device
        self.w_qs = nn.Linear(d_model, n_h * d_k)
        self.w_ks = nn.Linear(d_model, n_h * d_k)
        self.w_vs = nn.Linear(d_model, n_h * d_v)
        self.fc = nn.Linear(n_h * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor, batch):
        """
        attention propagation
        :param batch: graph batch
        :param q: dim is c x n x d_model
        :param k: dim is c x n x d_model
        :param v: dim is c x n x d_model
        :return:
        """
        d_k, d_v, d_h = self.d_k, self.d_v, self.n_h
        sz_c, d, sz_b = q.size(0), q.size(2), len(torch.unique(batch))

        # *******************************to dense batch******************************
        # for c in range(sz_c):
        #     if c == 0:
        #         batch_data =










class ScaledDotProductAttention(nn.Module):
    """"""

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        """"""
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
