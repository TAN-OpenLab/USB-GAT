import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import  DataLoader
import pickle


class BiGraphDataset(Dataset):

    def __init__(self, filepath, file_list):
        # 获得训练数据的总行
        self.filepath = filepath
        self.file_list = file_list
        self.number = len(self.file_list)

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        file = self.file_list[idx]
        x, A, y, user_x = pickle.load(open(os.path.join(self.filepath, file), 'rb'), encoding='utf-8')
        self.number = len(x)
        x = torch.tensor(x,dtype=torch.float32)
        A = torch.tensor(A,dtype=torch.long)
        y = torch.tensor(y,dtype=torch.long)
        user_x = torch.tensor(user_x,dtype=torch.float32)
        return Data(x=x, edge_index=A, y=y, root=x[0, :],
                    rootindex=torch.LongTensor([0]), user_x=user_x)

