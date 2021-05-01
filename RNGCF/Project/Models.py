# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:32:29 2020

@author: book
"""

from Project.Layer import NgcfConv, LgcnConv, SgcnConv, SimpleHighway
from torch import nn
import torch
import torch.nn.functional as F
import Project.Utils as Utils


class NGCF(torch.nn.Module):
    def __init__(self, adj, num_item, num_user, n_layers, dim, dropout_mess, decay, dropout_mode=0.1):
        super(NGCF, self).__init__()
        assert len(dim) == n_layers + 1, "error"
        self.layer_stack = nn.ModuleList([
            NgcfConv(dim[i], dim[i + 1], dropout_mess[i])
            for i in range(n_layers)
        ])
        self.embedding = nn.Embedding(num_item + num_user, dim[0])
        self.loss_function = nn.LogSigmoid()
        self.num_item = num_item
        self.num_user = num_user
        self.decay = decay
        self.adj = adj
        self.dropout = nn.Dropout(dropout_mode)
        # init
        nn.init.xavier_uniform_(self.embedding.weight)

    def dropout_sparse(self):
        i = self.adj.indices()
        v = self.adj.values()
        v = self.dropout(v)
        adj = torch.sparse_coo_tensor(i, v, self.adj.shape).to(self.adj.device)
        adj = adj.coalesce()
        return adj

    def loss(self, batch_samples):
        user_index, pos_item_index, neg_item_index = batch_samples[:, 0], batch_samples[:, 1], batch_samples[:, 2]
        user_weight, item_weight = self.forward()
        batch_size = user_index.shape[0]
        user_emb = F.embedding(user_index, user_weight)  # [B,T]
        pos_item_emb = F.embedding(pos_item_index, item_weight)  # [B,T]
        neg_item_emb = F.embedding(neg_item_index, item_weight)  # [B,T]
        pos_scores = torch.sum(user_emb * pos_item_emb, 1)  # [B,T]
        neg_scores = torch.sum(user_emb * neg_item_emb, 1)  # [B,T]
        loss = -self.loss_function(pos_scores - neg_scores).mean()
        regularizer = (torch.sum(user_emb ** 2)
                       + torch.sum(pos_item_emb ** 2)
                       + torch.sum(neg_item_emb ** 2)) / 2
        emb_loss = self.decay * regularizer / batch_size
        batch_loss = emb_loss + loss
        return batch_loss

    def predict(self):
        user_weight, item_weight = self.forward()
        return user_weight, item_weight

    def forward(self):
        node_feat = self.embedding.weight
        mat = self.dropout_sparse()
        all_embeddings = [node_feat]
        for conv in self.layer_stack:
            node_feat = conv(mat, node_feat)  # [B,T]
            all_embeddings += [F.normalize(node_feat)]
        embedding_weight = torch.cat(all_embeddings, -1)  # [node_num,n*T]
        user_weight, item_weight = torch.split(embedding_weight, [self.num_user, self.num_item], dim=0)
        return user_weight, item_weight


class LGCN(torch.nn.Module):

    def __init__(self, adj, num_item, num_user, n_layers, dim, decay, dropout, dropout_mess):
        super(LGCN, self).__init__()
        self.layer_stack = nn.ModuleList([
            LgcnConv()
            for i in range(n_layers)
        ])
        self.embedding = nn.Embedding(num_item + num_user, dim)
        self.loss_function = nn.LogSigmoid()
        self.num_item = num_item
        self.num_user = num_user
        self.decay = decay
        self.adj = adj
        self.num_nodes = self.num_item + self.num_user

        # init
        nn.init.normal_(self.embedding.weight, std=0.01)

    def get_adj(self):
        i = self.adj.indices()
        v = self.adj.values()
        i, v = Utils.norm(i, v, self.num_nodes, 'bi')
        adj = torch.sparse_coo_tensor(i, v, self.adj.shape).to(self.adj.device)
        adj = adj.coalesce()
        return adj

    def loss(self, batch_samples):
        user_index, pos_item_index, neg_item_index = batch_samples[:, 0], batch_samples[:, 1], batch_samples[:, 2]
        user_weight, item_weight = self.forward()
        user_weight0, item_weight0 = torch.split(self.embedding.weight, [self.num_user, self.num_item], dim=0)
        batch_size = user_index.shape[0]

        user_emb = F.embedding(user_index, user_weight)  # [B,T]
        pos_item_emb = F.embedding(pos_item_index, item_weight)  # [B,T]
        neg_item_emb = F.embedding(neg_item_index, item_weight)  # [B,T]

        user_emb0 = F.embedding(user_index, user_weight0)  # [B,T]
        pos_item_emb0 = F.embedding(pos_item_index, item_weight0)  # [B,T]
        neg_item_emb0 = F.embedding(neg_item_index, item_weight0)  # [B,T]

        pos_scores = torch.sum(user_emb * pos_item_emb, 1)  # [B,T]
        neg_scores = torch.sum(user_emb * neg_item_emb, 1)  # [B,T]
        bpr_loss = -self.loss_function(pos_scores - neg_scores).mean()
        regularizer = (torch.sum(user_emb0 ** 2)
                       + torch.sum(pos_item_emb0 ** 2)
                       + torch.sum(neg_item_emb0 ** 2)) / 2
        rl_loss = self.decay * regularizer / batch_size
        loss = bpr_loss + rl_loss

        return loss, bpr_loss

    def predict(self):
        user_weight, item_weight = self.forward()

        # i = self.adj.indices()
        # row, col = i[0, :], i[1, :]
        # feat = torch.cat([user_weight, item_weight], dim=0).detach()
        # feat_i, feat_j = feat[row], feat[col]
        # src = F.cosine_similarity(feat_i, feat_j)
        # print(src.mean())
        return user_weight, item_weight

    def forward(self):
        node_feat = self.embedding.weight
        mat = self.get_adj()
        all_embeddings = [node_feat]
        for conv in self.layer_stack:
            node_feat = conv(mat, node_feat)  # [B,T]
            all_embeddings += [node_feat]
        embedding_weight = torch.stack(all_embeddings, 1)  # [node_num,n*T]
        embedding_weight = torch.mean(embedding_weight, dim=1)
        user_weight, item_weight = torch.split(embedding_weight, [self.num_user, self.num_item], dim=0)
        return user_weight, item_weight


class PGCF(torch.nn.Module):

    def __init__(self, adj, num_item, num_user, n_layers, dim, decay, dropout, dropout_mess):
        super(PGCF, self).__init__()
        self.SimpleHighway = SimpleHighway(dim, dropout_mess)
        self.LgcnConvs = nn.ModuleList([
            SgcnConv(dim, dropout_mess) for _ in range(n_layers)
        ])
        self.E = nn.Embedding(num_item + num_user, dim)
        self.loss_function = nn.LogSigmoid()
        self.num_item = num_item
        self.num_user = num_user
        self.decay = decay
        self.adj = adj
        self.dim = dim
        self.dp = nn.Dropout(dropout)
        self.num_nodes = self.num_item + self.num_user
        self.gru = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True)
        nn.init.normal_(self.E.weight, std=0.01)

    def dropout_node(self):
        i = self.adj.indices()
        v = self.adj.values()
        v = v[:v.shape[0] // 2]
        v = self.dp(v)
        v = v.repeat(2)
        i, v = Utils.add_self_loop(i, v, self.num_nodes)
        i, v = Utils.norm(i, v, self.num_nodes, 'bi')
        adj = torch.sparse_coo_tensor(i, v, self.adj.shape).to(self.adj.device)
        adj = adj.coalesce()
        return adj

    def fuse(self, index, all_feat, feat):
        emb = feat[index, :]  # [B,T]
        all_emb = all_feat[index, :, :]
        _, context = self.gru(self.dp(all_emb))
        context = context.squeeze()
        emb = self.SimpleHighway(emb, context)  # [B,T]
        return emb

    def loss(self, batch_samples):
        all_feat, feat = self.forward()
        user_feat, item_feat = torch.split(feat, [self.num_user, self.num_item], dim=0)
        user_all_feat, item_all_feat = torch.split(all_feat, [self.num_user, self.num_item], dim=0)
        user_weight0, item_weight0 = torch.split(self.E.weight, [self.num_user, self.num_item], dim=0)
        user_index, pos_item_index, neg_item_index = batch_samples[:, 0], batch_samples[:, 1], batch_samples[:, 2]
        batch_size = user_index.shape[0]

        user_emb = self.fuse(user_index, user_all_feat, user_feat)
        pos_item_emb = self.fuse(pos_item_index, item_all_feat, item_feat)
        neg_item_emb = self.fuse(neg_item_index, item_all_feat, item_feat)

        pos_scores = torch.sum(user_emb * pos_item_emb, 1)  # [B,T]
        neg_scores = torch.sum(user_emb * neg_item_emb, 1)  # [B,T]
        bpr_loss = -self.loss_function(pos_scores - neg_scores).mean()

        user_emb0 = F.embedding(user_index, user_weight0)  # [B,T]
        pos_item_emb0 = F.embedding(pos_item_index, item_weight0)  # [B,T]
        neg_item_emb0 = F.embedding(neg_item_index, item_weight0)  # [B,T]
        regularizer = (torch.sum(user_emb0 ** 2)
                       + torch.sum(pos_item_emb0 ** 2)
                       + torch.sum(neg_item_emb0 ** 2))
        rl_loss = self.decay * regularizer / batch_size
        loss = bpr_loss + rl_loss

        return loss, bpr_loss

    def predict(self):
        all_feat, feat = self.forward()
        _, context = self.gru(self.dp(all_feat))
        context = context.squeeze()
        feat = self.SimpleHighway(feat, context)  # [B,T]
        user_weight, item_weight = torch.split(feat, [self.num_user, self.num_item], dim=0)

        # i = self.adj.indices()
        # row, col = i[0, :], i[1, :]
        # feat = torch.cat([user_weight, item_weight], dim=0).detach()
        # feat_i, feat_j = feat[row], feat[col]
        # src = F.cosine_similarity(feat_i, feat_j)
        # print(src.mean())
        return user_weight, item_weight

    def forward(self):
        adj = self.dropout_node()
        feat = self.E.weight
        all_feat = []
        for conv in self.LgcnConvs:
            feat = conv(adj, feat)  # [B,T]
            all_feat += [feat]
        all_feat = torch.stack(all_feat, 1)  # [node_num,n,T]\
        # idx = [1, 2, 0]
        idx = [2, 0, 1]
        # idx = [2, 1, 0]
        idx = torch.LongTensor(idx).to(all_feat.device)
        all_feat = all_feat.index_select(1, idx)

        return all_feat, feat
