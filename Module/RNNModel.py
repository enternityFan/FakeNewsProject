# @Time : 2022-02-24 15:36
# @Author : Phalange
# @File : RNNModel.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

def mlp(num_inputs,num_hiddens):
    net = []

    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs,num_hiddens))
    net.append(nn.PReLU())
    net.append(nn.Linear(num_hiddens,num_hiddens))
    net.append(nn.PReLU())
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens,num_hiddens))
    net.append(nn.PReLU())

    return nn.Sequential(*net)




class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self,vocab,embed_size,num_hiddens,mode='train',**kwargs):
        super(RNNModel,self).__init__(**kwargs)
        self.mode = mode
        self.rnn_hpy = nn.RNN(embed_size,num_hiddens,batch_first=True)
        self.rnn_prem = nn.RNN(embed_size,num_hiddens,batch_first=True)
        self.mlp = mlp(num_hiddens * 2,200) # 隐藏层是200
        self.vocab_size = len(vocab)
        self.num_hiddens = self.rnn_hpy.hidden_size
        self.embedding = nn.Embedding(len(vocab),embed_size)
        # 单向的RNN网络
        self.num_directions = 1
        self.linear = nn.Linear(200,3)

    def forward(self,inputs,state):
        premises,hypotheses = inputs

        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        # (batch_size,A/B的词元数,embed_size)变为(A/B的词元数，batch_size,embed_size)

        Y_hpy,state_hpy = self.rnn_hpy(A,state) # params'num:4015K
        Y_prem,state_prem = self.rnn_prem(B,state)
        #if self.mode == 'predict':
        #    state_hpy = torch.squeeze(state_hpy,dim=1)
        #    state_prem = torch.squeeze(state_prem,dim=1)
        #elif self.mode == 'train':
        state_hpy = torch.squeeze(state_hpy)
        state_prem = torch.squeeze(state_prem)
        output1 = self.mlp(torch.cat([state_hpy,state_prem],1))  # params'num:160.K
        output = self.linear(output1) # params'num: 600
       # output = F.softmax(output,dim=-1)
        return output

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
