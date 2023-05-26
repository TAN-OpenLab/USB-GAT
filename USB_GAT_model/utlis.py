import torch
from torch.autograd import Function,Variable
import torch.nn as nn
import torch.nn.functional as F



class BinarizedF(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        a = torch.ones_like(input)
        b = torch.zeros_like(input)
        output = torch.where(input > 0, a, b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        input_g =  0.5 - torch.abs((input-0.5))
        ones = torch.ones_like(input)
        zeros = -torch.ones_like(input)
        grad_output = torch.where(input >= 0.5, torch.mul(ones, input_g), torch.mul(zeros,input_g))
        return grad_output


# class BinarizedModule(nn.Module):
#     def __init__(self):
#         super(BinarizedModule, self).__init__()
#         self.BF = BinarizedF()
#     def forward(self,input):
#         #print(input.shape)
#         output = self.BF(input)
#         return output

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__()

        self.gamma= gamma
        self.reduction= reduction
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, (float, int)):  # 仅仅设置第一类别的权重
                self.alpha = torch.zeros(class_num)
                self.alpha[0] += alpha
                self.alpha[1:] += (1 - alpha)
                self.alpha = Variable(self.alpha)
            if isinstance(alpha, list):  # 全部权重自己设置
                self.alpha = Variable(torch.Tensor(alpha))

    def forward(self, inputs, targets):

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, targets, weight=self.alpha, reduction= self.reduction)

        return loss




class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self):
        self.weight_list = self.get_weight(self.model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
