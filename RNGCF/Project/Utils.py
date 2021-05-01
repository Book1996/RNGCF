"""
======================
@author:Book
@time:2020/11/24:23:02
======================
"""
import numpy as np
import torch

dtypes = {'index': torch.int64, 'weights': torch.float32}


def norm(edge_index: torch.Tensor, weight: torch.Tensor, num_nodes: int, mode: str):
    assert mode in ['bi', 'right', 'left'], 'bad norm mode'
    row, col = edge_index[0], edge_index[1]
    deg = torch.tensor(np.zeros(num_nodes), dtype=dtypes['weights'], device=weight.device)
    deg = deg.scatter_add(0, col, weight)

    if mode == 'bi':
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        weight = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    elif mode == 'right':
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        weight = weight * deg_inv_sqrt[col]
    else:
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        weight = deg_inv_sqrt[row] * weight
    return edge_index, weight


def add_self_loop(edge_index: torch.Tensor, weight: torch.Tensor, num_nodes: int):
    loop_index = torch.arange(0, num_nodes, dtype=dtypes['index'], device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    loop_weight = torch.ones(num_nodes, dtype=dtypes['weights'], device=edge_index.device)
    loop_weight = loop_weight*weight[0]
    edge_index = torch.cat([edge_index, loop_index], -1)
    weight = torch.cat([weight, loop_weight], -1)
    return edge_index, weight


def toTensor(edge_index, weight):
    weight = torch.tensor(weight, dtype=dtypes['weights'])
    edge_index = torch.tensor(edge_index, dtype=dtypes['index'])
    return edge_index, weight
