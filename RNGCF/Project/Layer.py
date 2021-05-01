# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:32:29 2020

@author: book
"""
import torch

import torch.nn as nn


class NgcfConv(torch.nn.Module):

    def __init__(self, in_features, out_features, dropout=0.1):
        super(NgcfConv, self).__init__()
        self.W1 = nn.Linear(in_features, out_features)
        self.W2 = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, mat, node_feat):
        agg_feat = torch.sparse.mm(mat, node_feat)
        part1 = self.act(self.W1(agg_feat))
        part2 = self.act(self.W2(agg_feat * node_feat))
        out = part1 + part2
        out = self.dropout(out)
        return out


class LgcnConv(torch.nn.Module):

    def __init__(self):
        super(LgcnConv, self).__init__()

    def forward(self, mat, feat):
        return torch.sparse.mm(mat, feat)


class SgcnConv(torch.nn.Module):
    def __init__(self, dim, dropout):
        super(SgcnConv, self).__init__()

    def forward(self, adj, feat):
        feat = torch.sparse.mm(adj, feat)
        return feat


class SimpleHighway(torch.nn.Module):

    def __init__(self, dim, dropout):
        super(SimpleHighway, self).__init__()
        self.H = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )

    def forward(self, feat, interesting):
        gate = self.H(interesting) + 1
        feat = gate * feat + feat
        return feat
