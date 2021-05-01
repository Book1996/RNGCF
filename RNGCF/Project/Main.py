# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:11:29 2020

@author: book
"""
import argparse
import pickle
import sys
import torch
from Project.Session import Session
from torch.utils.data import DataLoader
from Project.Dataset import TrainDataset, TestDataset
from Project.Models import NGCF, LGCN, PGCF
import torch.optim as opti
import os
import numpy as np
import torch.multiprocessing as mp
import Project.Utils as Utils
import csv

sys.path.append('..')
cores = mp.cpu_count() // 16


def collate(batch):
    batch = torch.LongTensor(batch)
    return batch


def test_collate(batch):
    batch = np.array(batch, dtype=object)
    user = torch.LongTensor(batch[:, 0].astype(np.int64))
    train_mask = torch.Tensor(batch[:, 1].tolist())
    test = batch[:, 2].tolist()
    tot = batch[:, 3]
    return [user, train_mask, test, tot]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1', type=str)
    # parser.add_argument('--dataset_name', default='gowalla', type=str)
    # parser.add_argument('--dataset_name', default='yelp2018', type=str)
    parser.add_argument('--dataset_name', default='amazon-book', type=str)
    parser.add_argument('--batch_size', default=2048*4, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--n_layers', default=3, type=int)
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--dropout_node', default=0.2, type=float)
    parser.add_argument('--dropout_mess', default=0.5, type=float)
    parser.add_argument('--decay', default='1e-5', type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--save', default='True', type=str)
    parser.add_argument('--load', default='False', type=str)
    parser.add_argument('--runID', default=9, type=int)
    args = parser.parse_args()
    # build parameter
    dataset_name = args.dataset_name
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if dataset_name == 'gowalla':
        os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    if dataset_name == 'yelp2018':
        os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    if dataset_name == 'amazon-book':
        os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    verbose = 1000
    batch_size = args.batch_size
    num_epochs = args.num_epochs + 1
    n_layers = args.n_layers
    dim = args.dim
    dropout_node = args.dropout_node
    dropout_mess = args.dropout_mess
    decay = args.decay
    lr = args.lr
    save = eval(args.save)
    load = eval(args.load)
    exp_name = 'PGCF_performance_s'
    seed = 2020
    # build data and setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "PGCF"
    path = 'dim' + str(dim) + '_dp' + str(dropout_node)+ '_dp' + str(dropout_mess) + '_decay' + str(
        decay) + '_' + dataset_name + "_" + model_name + '_l' + str(
        n_layers) + '_' + exp_name + 'parameter.pkl'
    print(path)
    with open('data/processed/pre_' + dataset_name + '.pkl', 'rb') as f:
        train_U2I = pickle.load(f)
        test_U2I = pickle.load(f)
        indices = pickle.load(f)
        weight = pickle.load(f)
        num_item, num_user = pickle.load(f)
    num_nodes = num_item + num_user
    print(num_item, num_user, len(test_U2I) + len(train_U2I))
    # graph definition
    indices, weight = Utils.toTensor(indices, weight)
    adj = torch.sparse_coo_tensor(indices, weight, torch.Size([num_nodes, num_nodes])).to(device)
    adj = adj.coalesce()
    # set seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # build train_set
    train_set = [i for i in range(num_user)] * verbose
    # dataset definition
    train_data = TrainDataset(train_set, train_U2I, num_item)
    test_data = TestDataset(test_U2I, train_U2I, num_item)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=cores, collate_fn=test_collate)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=cores,
                              collate_fn=collate, shuffle=True)

    # model definition
    model = PGCF(adj, num_item, num_user, n_layers, dim, decay, dropout_node, dropout_mess).to(device)
    if load:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    optimizer = opti.Adam(model.parameters(), lr=lr)
    # session definition
    session = Session(model, device)
    # do epochs
    max_recall = 0
    with open('log/'+dataset_name+'/'+exp_name + 'log.csv', 'w+', newline='') as csvFile:
        csv_writer = csv.writer(csvFile)
        csv_writer.writerow(['epoch', 'loss', 'recall', 'ndcg'])
        for e in range(num_epochs):
            loss = session.train(train_loader, optimizer)
            print("epoch: {}, loss: {}".format(e, loss))
            ret = session.test(test_loader)
            print(ret)
            row = [e, loss, ret['recall'], ret['ndcg']]
            csv_writer.writerow(row)
            if max_recall < ret['recall'] and save:
                max_recall = ret['recall']
                torch.save(model.state_dict(), path)
