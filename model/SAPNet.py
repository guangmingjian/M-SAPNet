#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 9:23
# @Version : 1.0
# @File    : SAPNet.py
import torch.nn.functional as F
from torch import nn

from model.layers.CNNLayers import LeNet1
from model.layers.MultiGCNLayers import MultiGCNLayers
from model.layers.SMULayer import SMULayer
from model.layers.VLRLayers import VLRLayers
from utils import tools
from model.layers.SimilarMeasure import SimilarMeasure


class SAPNet(nn.Module):
    """"""

    def __init__(self, net_params):
        super(SAPNet, self).__init__()
        """"""
        # *******************************def params******************************
        in_dim = net_params['in_dim']
        out_dim = net_params['out_dim']
        self.device = net_params["device"]
        d_kv = 32

        gcn_nums = net_params['gcn_num']
        dropout = net_params['dropout']
        gcn_droupt = net_params['gcn_droupt']
        att_droupt = net_params['att_droupt']
        graph_norm = net_params['graph_norm']
        sz_c = net_params['sz_c']
        if net_params["g_name"] != "GAT":
            gcn_h_dim = net_params["h_dim"] // sz_c
        else:
            gcn_h_dim = net_params["h_dim"]
        g_name = net_params["g_name"]
        s_l_nums = net_params["s_l_nums"]
        self.alpha = net_params["alpha"]
        self.smu_flag = net_params["SMUFlag"]
        # self.gcn_out = net_params["gcn_out"]

        # *******************************def models******************************
        self.fea_embed = nn.Sequential(nn.Linear(in_dim, gcn_h_dim), nn.Linear(gcn_h_dim, gcn_h_dim))
        self.mgl = MultiGCNLayers(gcn_h_dim, gcn_h_dim, gcn_h_dim, sz_c, gcn_nums,
                                  drop_rate=gcn_droupt, device=self.device, g_name=g_name, g_norm=graph_norm)
        self.loss1 = SimilarMeasure("mean")
        if self.smu_flag:
            self.smus = nn.ModuleList([SMULayer(s_l_nums, gcn_h_dim, gcn_h_dim) for _ in range(sz_c)])
        self.vlr = VLRLayers(d_kv, self.device, gcn_h_dim, att_droupt)
        self.cnn_net = LeNet1(size_c=sz_c, num_class=out_dim, dropout=dropout)

    def reset_parameters(self):
        self.fea_embed.apply(tools.weight_reset)
        if self.smu_flag:
            for m in self.smus:
                m.reset_parameters()
        all_res = [self.cnn_net, self.vlr, self.mgl]
        for res in all_res:
            if res != None:
                res.reset_parameters()

    def forward(self, data, edge_index, batch):
        # *******************************feature embedding******************************
        data = self.fea_embed(data)
        # *******************************multi-channel encoder******************************
        if self.smu_flag:
            z, batches = self.mgl(data, edge_index, batch, self.smus)
        else:
            z, batches = self.mgl(data, edge_index, batch)
        self.div_loss = self.loss1.correlation_loss(z)
        # *******************************CNN Decoder******************************
        batch_data = self.vlr(z, batch)  # VLR
        batch_data = self.cnn_net(batch_data)  # CPB
        return F.log_softmax(batch_data, dim=-1)

    def loss(self, y_pre, y_true):
        return self.alpha * self.div_loss + (1 - self.alpha) * F.nll_loss(y_pre, y_true)
        # return self.alpha * self.div_loss + F.nll_loss(y_pre, y_true)
