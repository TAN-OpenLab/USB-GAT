import os
import sys
import torch
import copy
import math
import torch as th
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean
from torch_geometric.nn import GATConv

sys.path.append(os.getcwd())

a = torch.device('cuda:0')


class TDGAT(th.nn.Module):
    # 768 + 8 64 64
    def __init__(self, features, hidden, classes):
        super(TDGAT, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=8, add_self_loops=False)
        self.gat2 = GATConv(hidden * 8 + features, classes, add_self_loops=False)

    def forward(self, x, user_x, adj, data_batch):
        chuanbo_x = torch.cat((x, user_x), dim=1)
        x1 = copy.copy(chuanbo_x)
        chuanbo_x = self.gat1(chuanbo_x, adj)
        x2 = copy.copy(chuanbo_x)
        # 强化
        root_extend = th.zeros(len(data_batch), x1.size(1), device=a)
        root_extend = torch.as_tensor(root_extend, dtype=torch.float32, device=a)

        for num_batch in range(len(data_batch)):
            if num_batch > 0 and data_batch[num_batch] == data_batch[num_batch - 1]:
                if user_x[num_batch][7] == 0:
                    root_extend[num_batch] = temp.cuda(torch.device('cuda:0'))
                elif user_x[num_batch][7] == -1:
                    root_extend[num_batch] = x1[num_batch]
                else:
                    root_extend[num_batch] = temp.cuda(torch.device('cuda:0')) + x1[num_batch]
            else:
                root_extend[num_batch] = x1[num_batch]
                temp = root_extend[num_batch]

        chuanbo_x = th.cat((chuanbo_x, root_extend), 1)
        chuanbo_x = F.relu(chuanbo_x)
        chuanbo_x = F.dropout(chuanbo_x, training=self.training)
        chuanbo_x = self.gat2(chuanbo_x, adj)

        root_extend = th.zeros(len(data_batch), x2.size(1), device=a)
        root_extend = torch.as_tensor(root_extend, dtype=torch.float32, device=a)
        for num_batch in range(len(data_batch)):
            if num_batch > 0 and data_batch[num_batch] == data_batch[num_batch - 1]:
                if user_x[num_batch][7] == 0:
                    root_extend[num_batch] = temp.cuda(torch.device('cuda:0'))
                elif user_x[num_batch][7] == -1:
                    root_extend[num_batch] = x2[num_batch]
                else:
                    root_extend[num_batch] = temp.cuda(torch.device('cuda:0')) + x2[num_batch]
            else:
                root_extend[num_batch] = x2[num_batch]
                temp = root_extend[num_batch]

        chuanbo_x = th.cat((chuanbo_x, root_extend), 1)
        chuanbo_x = F.relu(chuanbo_x)
        x = scatter_mean(chuanbo_x, data_batch, dim=0)

        return x


class BUGAT(th.nn.Module):
    # 768 + 8 64 64
    def __init__(self, features, hidden, classes):
        super(BUGAT, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=8, add_self_loops=False)
        self.gat2 = GATConv(hidden * 8 + features, classes, add_self_loops=False)

    def forward(self, x, user_x, adj_T, data_batch):
        chuanbo_x = torch.cat((x, user_x), dim=1)
        x1 = copy.copy(chuanbo_x)
        chuanbo_x = self.gat1(chuanbo_x, adj_T)
        x2 = copy.copy(chuanbo_x)
        # 强化
        root_extend = th.zeros(len(data_batch), x1.size(1), device=a)
        root_extend = torch.as_tensor(root_extend, dtype=torch.float32, device=a)

        for num_batch in range(len(data_batch)):
            if num_batch > 0 and data_batch[num_batch] == data_batch[num_batch - 1]:
                if user_x[num_batch][7] == 0:
                    root_extend[num_batch] = temp.cuda(torch.device('cuda:0'))
                elif user_x[num_batch][7] == -1:
                    root_extend[num_batch] = x1[num_batch]
                else:
                    root_extend[num_batch] = temp.cuda(torch.device('cuda:0')) + x1[num_batch]
            else:
                root_extend[num_batch] = x1[num_batch]
                temp = root_extend[num_batch]

        chuanbo_x = th.cat((chuanbo_x, root_extend), 1)
        chuanbo_x = F.relu(chuanbo_x)
        chuanbo_x = F.dropout(chuanbo_x, training=self.training)
        chuanbo_x = self.gat2(chuanbo_x, adj_T)
        root_extend = th.zeros(len(data_batch), x2.size(1), device=a)
        root_extend = torch.as_tensor(root_extend, dtype=torch.float32, device=a)

        for num_batch in range(len(data_batch)):
            if num_batch > 0 and data_batch[num_batch] == data_batch[num_batch - 1]:
                if user_x[num_batch][7] == 0:
                    root_extend[num_batch] = temp.cuda(torch.device('cuda:0'))
                elif user_x[num_batch][7] == -1:
                    root_extend[num_batch] = x2[num_batch]
                else:
                    root_extend[num_batch] = temp.cuda(torch.device('cuda:0')) + x2[num_batch]
            else:
                root_extend[num_batch] = x2[num_batch]
                temp = root_extend[num_batch]
        chuanbo_x = th.cat((chuanbo_x, root_extend), 1)
        chuanbo_x = F.relu(chuanbo_x).cuda()
        x = scatter_mean(chuanbo_x, data_batch, dim=0)
        return x


class Position_Embeddings(th.nn.Module):
    def __init__(self):
        super(Position_Embeddings, self).__init__()

    def forward(self, data_batch):
        pe = torch.zeros(len(data_batch), 768, device=a).float()
        position = torch.zeros(len(data_batch), 1).float()
        for num_batch in range(len(data_batch)):
            if num_batch > 0 and data_batch[num_batch] == data_batch[num_batch - 1]:
                position[num_batch] = position[num_batch - 1] + 1
            else:
                position[num_batch] = 0
        div_term = (torch.arange(0, 768, 2).float() * -(math.log(10000.0) / 768)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = F.dropout(pe, training=self.training)
        return pe


class Lichang_Embeddings(th.nn.Module):
    def __init__(self):
        super(Lichang_Embeddings, self).__init__()

    def forward(self, data_batch, user_x):
        le = torch.zeros(len(data_batch), 768).float()
        for num_batch in range(len(data_batch)):
            if user_x[num_batch][7] == 0:
                continue
            elif user_x[num_batch][7] == 1:
                temp = torch.ones(1, 768).float()
                le[num_batch] = temp
            else:
                temp = np.repeat(-1, 768)
                temp = list(temp)
                temp = torch.LongTensor(temp)
                le[num_batch] = temp

        return le


class Net(th.nn.Module):
    def __init__(self, features_1, gat_hidden, gat_classes):
        super(Net, self).__init__()
        self.PBE = Position_Embeddings()
        self.LCE = Lichang_Embeddings()
        self.TDGAT = TDGAT(features_1 + 8, gat_hidden, gat_classes)
        self.BUGAT = BUGAT(features_1 + 8, gat_hidden, gat_classes)
        self.fc = th.nn.Linear((gat_hidden * 8 + gat_classes) * 2, 2)

    def forward(self, x, adj, data_batch, user_x, A_T):
        # 计算position_embedding
        position_eb = self.PBE(data_batch)
        position_eb = torch.as_tensor(position_eb, dtype=torch.float32, device=a)

        # 计算立场_embedding
        lichang_eb = self.LCE(data_batch, user_x)
        lichang_eb = torch.as_tensor(lichang_eb, dtype=torch.float32, device=a)

        # x + position_eb + lichang_eb
        TD_x = self.TDGAT(x + position_eb + lichang_eb, user_x, adj, data_batch)
        BU_x = self.BUGAT(x + position_eb + lichang_eb, user_x, A_T, data_batch)
        x = th.cat((TD_x, BU_x), 1)
        x = self.fc(x)
        return x + 1e-6
