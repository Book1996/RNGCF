import torch.utils.data
import random
import numpy as np


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rated, num_item):
        self.dataset = dataset
        self.rated = rated
        self.num_item = num_item

    def __getitem__(self, index):
        user = self.dataset[index]  # 1
        pos = random.choice(self.rated[user])
        neg = np.random.randint(0, self.num_item)
        while neg in self.rated[user]:
            neg = np.random.randint(0, self.num_item)
        return [user, pos, neg]

    def __len__(self):
        return len(self.dataset)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_U2I: dict, train_U2I: dict, num_item):
        self.U = list(test_U2I.keys())
        self.test_U2I = test_U2I
        self.train_U2I = train_U2I
        self.num_item = num_item

    def __getitem__(self, index):
        user = self.U[index]
        train_mask = np.zeros(self.num_item)
        rate = self.train_U2I[user]
        train_mask[rate] = 1
        test = self.test_U2I[user]
        tot = len(test)
        return [user, train_mask.tolist(), test, tot]

    def __len__(self):
        return len(self.U)
