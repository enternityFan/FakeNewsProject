# @Time : 2022-02-24 20:04
# @Author : Phalange
# @File : LSTMRNNModel.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import torch
from torch import nn
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

class LSTMRNNModel(nn.Module):
    """LSTMRNNModel"""
    def __init__(self,vocab,embed_size,num_hiddens,mode='train',**kwargs):
        super(LSTMRNNModel, self).__init__(**kwargs)
        self.mode = mode
        self.lstm_hpy = nn.LSTM(embed_size,num_hiddens,batch_first=True)
        self.lstm_prem = nn.LSTM(embed_size,num_hiddens,batch_first=True)
        self.mlp = mlp(num_hiddens * 2,200)
        self.vocab_size = len(vocab)
        self.num_hiddens = self.lstm_hpy.hidden_size
        self.embedding = nn.Embedding(len(vocab),embed_size)
        # 单项的LSTM网络
        self.num_directions = 1
        self.linear = nn.Linear(200,3)

    def forward(self,inputs,state):
        premises, hypotheses = inputs

        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        # (batch_size,A/B的词元数,embed_size)变为(A/B的词元数，batch_size,embed_size)

        Y_hpy, state_hpy = self.lstm_hpy(A, state)
        Y_prem, state_prem = self.lstm_prem(B, state)
        if self.mode == 'predict':
            state_hpy = torch.squeeze(state_hpy[0], dim=1)
            state_prem = torch.squeeze(state_prem[0], dim=1)
        elif self.mode == 'train':
            state_hpy = torch.squeeze(state_hpy[0])
            state_prem = torch.squeeze(state_prem[0])
        output1 = self.mlp(torch.cat([state_hpy, state_prem], 1))
        output = self.linear(output1)
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
