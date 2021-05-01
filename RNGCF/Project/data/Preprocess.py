# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:40:58 2020

@author: book
"""
from os.path import join
import pickle
import numpy as np


def read_raw(path):
    with open(path) as file:
        User, Item, User2Item = [], [], {}
        for line in file.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                uid = int(line[0])
                try:
                    items = [int(i) for i in line[1:]]
                except BaseException as e:
                    print(uid, e)
                    continue
                else:
                    User2Item[uid] = items
                    User.extend([uid] * len(items))
                    Item.extend(items)
    return np.array(User), np.array(Item), User2Item


if __name__ == '__main__':
    # set
    root = 'raw'
    dataset_name = 'gowalla'
    # dataset_name = 'yelp2018'
    # dataset_name = 'amazon-book'
    # read
    train_path = join(root, dataset_name, 'train.txt')
    test_path = join(root, dataset_name, 'test.txt')
    train_U, train_I, train_U2I = read_raw(train_path)
    test_U, test_I, test_U2I = read_raw(test_path)
    num_user = len(np.unique(np.concatenate([train_U, test_U])))
    num_item = len(np.unique(np.concatenate([train_I, test_I])))
    # build Graph
    row = np.concatenate([train_U, train_I + num_user])
    col = np.concatenate([train_I + num_user, train_U])
    edge_weight = np.ones_like(row).tolist()
    edge_indies = np.stack([row, col]).tolist()
    # print information
    print('item_num = %d, user_num = %d, sample = %d' % (num_user, num_item, len(row)))
    # save
    with open('processed/pre_' + dataset_name + '.pkl', 'wb') as f:
        pickle.dump(train_U2I, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_U2I, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(edge_indies, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(edge_weight, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump([num_item, num_user], f, pickle.HIGHEST_PROTOCOL)
