import torch
import pickle
import os, sys
from BGAT import Net
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
from torch_geometric.data import Data
from dataloader_pyg import BiGraphDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class USB_GAT_model(object):
    def __init__(self, args, device):
        # parameters
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.batch_size = args.batch_size
        self.f_in = args.input_dim_G
        self.h_DUGAT = args.h_DUGAT
        self.channels = args.channels
        self.num_heads = args.num_heads
        self.num_nodes = args.num_nodes
        self.h_op = args.h_op
        self.h_UDGAT = args.h_UDGAT
        self.hidden_LSTM = args.hidden_LSTM
        self.dense_C = args.dense_C
        self.num_worker = args.num_worker
        self.save_dir = args.save_dir
        self.model_dir = args.model_dir
        self.lr = args.lr
        self.delay_rate = args.delay_rate
        self.weight_decay = args.weight_decay
        self.device = device
        self.b1, self.b2 = args.b1, args.b2
        self.features_1, self.features_2 = args.features_1, args.features_2
        self.hidden = args.hidden
        self.classes = args.classes

        self.gat_hidden = args.gat_hidden
        self.gat_classes = args.gat_classes
        self.gcn_hidden = args.gcn_hidden
        self.gcn_classes = args.gcn_classes

        print(self.device)
        self.Net = Net(self.features_1, self.gat_hidden, self.gat_classes).cuda(self.device)
        self.optimizer = torch.optim.Adam(self.Net.parameters(), lr=self.lr, weight_decay=self.delay_rate)

    # all_data_emb
    def train(self, datapath, start_epoch, ispath):
        trainlist = os.listdir(datapath + '/train')
        train_dataset = BiGraphDataset(datapath + '/train', trainlist)

        testlist = os.listdir(datapath + '/test')
        test_dataset = BiGraphDataset(datapath + '/test', testlist)

        evallist = os.listdir(datapath + '/eval')
        eval_dataset = BiGraphDataset(datapath + '/eval', evallist)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_worker,
                                  drop_last=True,
                                  pin_memory=True)

        eval_loader = DataLoader(dataset=eval_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_worker,
                                 drop_last=True,
                                 pin_memory=False)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_worker,
                                 drop_last=True,
                                 pin_memory=False)
        self.train_hist = {}
        self.train_hist['train_loss'] = []
        self.train_hist['test_loss'] = []
        self.train_hist['train_acc'] = []
        self.train_hist['test_acc'] = []
        self.train_hist['pre'] = []
        self.train_hist['recall'] = []
        self.train_hist['f1'] = []

        self.max_acc = 0
        decayRate = 0.96
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)
        for epoch in range(start_epoch, self.epochs):
            train_loss_value, acc_value = self.train_batch(epoch, train_loader)
            print('train_loss_value:%.8f acc_value:%.8f' % (train_loss_value, acc_value))
            with torch.no_grad():
                test_loss, test_acc, pre, recall, f1 = self.eval(epoch, eval_loader)
            self.train_hist['train_loss'].append(train_loss_value)
            self.train_hist['train_acc'].append(acc_value)
            self.train_hist['test_loss'].append(test_loss)
            self.train_hist['test_acc'].append(test_acc)
            self.train_hist['pre'].append(pre)
            self.train_hist['recall'].append(recall)
            self.train_hist['f1'].append(f1)
            if epoch % 10 == 0:
                torch.save(self.Net.state_dict(),
                           os.path.join(self.save_dir, self.model_dir,
                                        str(epoch) + '_classifier.pkl'))
                with open(os.path.join(self.save_dir, self.model_dir, 'predict.txt'), 'w') as f:
                    hist = [str(k) + ':' + str(self.train_hist[k]) for k in self.train_hist.keys()]
                    f.write('\n'.join(hist) + '\n')
                print('save classifer : %d epoch' % epoch)
            if test_acc > self.max_acc:
                torch.save(self.Net.state_dict(),
                           os.path.join(self.save_dir, self.model_dir,
                                        str(epoch) + '_classifier.pkl'))
                self.max_acc = test_acc
                print('maxacc: ' + str(test_acc) + ' , save classifer : ' + str(epoch) + 'epoch')
            with torch.no_grad():
                test_loss, test_acc, pre, recall, f1 = self.eval(epoch, test_loader)
                print('test_loss_value:%.8f acc_value:%.8f' % (test_loss, test_acc))
            scheduler.step()
            print(epoch, scheduler.get_last_lr()[0])

        with torch.no_grad():
            test_loss, test_acc, pre, recall, f1 = self.eval(epoch, test_loader)
            print('test_loss_value:%.8f acc_value:%.8f' % (test_loss, test_acc))
            with open(os.path.join(self.save_dir, self.model_dir, 'predict.txt'), 'a') as f:
                hist = [test_loss, test_acc, pre, recall, f1]
                hist = list(map(str, hist))
                f.write(
                    ' test_loss, test_acc, pre, recall, f1' + '\n' + ' '.join(hist) + '\n')

    def train_batch(self, epoch, dataloader):
        train_loss_value = 0
        acc_value = 0
        self.Net.train()
        for iter, sample in enumerate(dataloader):
            sample = sample.cuda(self.device)
            y = sample['y']
            A_T = sample.edge_index
            A_T[[0, 1], :] = A_T[[1, 0], :]
            # x, adj, data_batch, user_x, A_T
            out_labels = self.Net(sample.x, sample.edge_index, sample.batch, sample.user_x, A_T)
            criteria = nn.CrossEntropyLoss()
            finalloss = criteria(out_labels, y)
            loss = finalloss
            _, pred = out_labels.max(dim=-1)
            acc = (pred == y).sum() / len(y)
            train_loss_value += loss.item()
            acc_value += acc.item()

            # 梯度置0
            self.optimizer.zero_grad()
            # 反向传播
            loss.backward()

            nn.utils.clip_grad_norm_(self.Net.parameters(), max_norm=3, norm_type=2)
            self.optimizer.step()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [model loss: %f] [model acc: %f] "
                % (
                    epoch,
                    self.epochs,
                    iter,
                    len(dataloader),
                    loss.item(),
                    acc.item()
                )
            )
        train_loss_value = train_loss_value / (iter + 1)
        acc_value = acc_value / (iter + 1)

        return train_loss_value, acc_value

    def eval(self, epoch, dataloader):
        acc_value, pre_value, recall_value, f1_value, test_loss_value = 0, 0, 0, 0, 0
        self.Net.eval()
        for iter, sample in enumerate(dataloader):
            sample = sample.cuda(self.device)
            A_T = sample.edge_index
            A_T[[0, 1], :] = A_T[[1, 0], :]
            val_out = self.Net(sample.x, sample.edge_index, sample.batch, sample.user_x, A_T)
            criteria = nn.CrossEntropyLoss()
            val_loss = criteria(val_out, sample.y)
            val_pred = val_out.data.max(1)[1].cpu()
            y_ = sample.y.cpu()
            test_loss_value += val_loss.item()
            val_acc = accuracy_score(val_pred, y_)
            pre = precision_score(val_pred, y_, average='binary', zero_division=0)
            recall = recall_score(val_pred, y_, average='binary', zero_division=0)
            f1 = f1_score(val_pred, y_, average='binary', zero_division=0)
            pre_value += pre
            recall_value += recall
            f1_value += f1
            acc_value += val_acc

        test_loss_value = test_loss_value / (iter + 1)
        acc_value = acc_value / (iter + 1)
        pre_value = pre_value / (iter + 1)
        recall_value = recall_value / (iter + 1)
        f1_value = f1_value / (iter + 1)
        print(
            'test_loss:%0.8f, acc:%0.8f, pre:%0.8f, recall:%0.8f, f1:%0.8f' % (
                test_loss_value, acc_value, pre_value, recall_value, f1_value))
        return test_loss_value, acc_value, pre_value, recall_value, f1_value

    def load(self, start_model):
        self.classifier_1.load_state_dict(
            torch.load(os.path.join(start_model), map_location=self.device,
                       encoding='utf-8'))
        name = start_model.split("/")
        model_name = name[-1].split("_")
        start_epoch = int(model_name[1])
        start_epoch += 1

        return start_epoch
