'''z
This script handling the training process.
'''
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

cores = mp.cpu_count() // 32
Ks = [20]
K_max = max(Ks)


def test_one(x):
    top_k, pos = x[0], x[1]
    r = np.in1d(top_k, pos).astype(np.int32)
    max_r = np.zeros(K_max)
    length = min(K_max, len(pos))
    max_r[:length] = 1
    return r, max_r


def test_all(pool, top_k, tests, num_test):
    infor = zip(top_k, tests)
    # need optimize
    out = np.array(pool.map(test_one, infor))
    r, max_r = out[:, 0, :], out[:, 1, :]  # [B, K_max ]
    # not need optimize
    sum_r = np.sum(r, -1)  # [B]
    mean_r = sum_r / K_max  # [B]
    smooth = np.expand_dims(np.log2(np.arange(2, K_max + 2)), axis=0)  # [1,K_max]
    idcg = np.sum(max_r / smooth, -1)
    dcg = np.sum(r / smooth, -1)
    ndcg = dcg / (idcg + 1e-16)
    recall = sum_r / np.array(num_test)
    precision = mean_r
    hit_ratio = (sum_r > 0).astype(np.int32)
    result = {'recall': np.mean(recall), 'precision': np.mean(precision),
              'ndcg': np.mean(ndcg), 'hit_ratio': np.mean(hit_ratio)}
    return result


class Session(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.pool = mp.Pool(cores)

    def train(self, loader, optimizer):
        mean_loss = 0
        for i, batch_samples in enumerate(tqdm(loader)):
            optimizer.zero_grad()
            batch_samples = batch_samples.to(self.device)
            loss, bpr_loss = self.model.loss(batch_samples)
            loss.backward()
            optimizer.step()
            mean_loss += bpr_loss.item()
        mean_loss = mean_loss / len(loader)
        return mean_loss

    def test(self, loader):
        result = {'precision': 0, 'recall': 0, 'ndcg': 0, 'hit_ratio': 0}
        self.model.eval()
        user_weight, item_weight = self.model.predict()
        user_weight, item_weight = user_weight.detach(), item_weight.detach()
        for i, batch_samples in enumerate(tqdm(loader)):
            user, train_mask, tests, num_test = batch_samples
            user = user.to(self.device)
            train_mask = train_mask.to(self.device) == 1
            ratings = torch.mm(user_weight[user, :], torch.transpose(item_weight, 0, 1))
            ratings = ratings.masked_fill(train_mask, -np.inf)
            _, top_k = ratings.topk(K_max, 1, True, True)
            batch_result = test_all(self.pool, top_k.cpu().numpy(), tests, num_test)
            for key in result.keys():
                result[key] += batch_result[key]
        for key in result.keys():
            result[key] = result[key] / len(loader)
        self.model.train()
        return result
