#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 9:58
# @Version : 1.0
# @File    : SMULayer.py
import torch
from torch import nn
from utils import tools

class SMULayer(nn.Module):
    """"""

    def __init__(self, l_nums, h_dim, d_in):
        super(SMULayer, self).__init__()
        """"""
        layers = []
        for l in range(l_nums):
            if l == 0:
                layers.extend([nn.Linear(d_in * 2, h_dim), nn.ReLU()])
            elif l == l_nums - 1:
                layers.extend([nn.Linear(h_dim, d_in), nn.Tanh()])
            else:
                layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU()])
        # print(layers)
        self.smu = nn.Sequential(*layers)

    def reset_parameters(self):
        self.smu.apply(tools.weight_reset)

    def forward(self, h_l, s_l):
        z = torch.cat([h_l, s_l], dim=-1)
        z = self.smu(z)
        nh_l = (1 - z) * h_l + z * s_l
        return nh_l


if __name__ == '__main__':
    tools.set_seed(12234)
    x = torch.randn([5, 3])
    y = torch.randn([5, 3])
    model = SMULayer(5, 16, 3)
    model.reset_parameters()
    print(model)
    print(model(x, y))
