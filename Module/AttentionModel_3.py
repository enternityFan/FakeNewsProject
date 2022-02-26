# @Time : 2022-02-26 17:02
# @Author : Phalange
# @File : AttentionModel_3.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import math
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.3))
    net.append(nn.Linear(num_inputs, num_hiddens*2))

    net.append(nn.PReLU())

    if flatten:
        net.append(nn.Flatten(start_dim=1))

    net.append(nn.Dropout(0.4))
    net.append(nn.Linear(num_hiddens*2, num_hiddens))
    net.append(nn.PReLU())
    net.append(nn.Dropout(0.4))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.PReLU())

    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)

class Attend(nn.Module):
    def __init__(self, ffn_num_inputs, ffn_num_hiddens,ffn_num_outputs, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = PositionWiseFFN(ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs)

    def forward(self, A, B):
        # A/B的形状：（批量大小，序列A/B的词元数，embed_size）
        # f_A/f_B的形状：（批量大小，序列A/B的词元数，num_hiddens）
        f_A = self.f(A) # params'num:100.6K
        f_A = torch.bmm(f_A,f_A.permute(0,2,1))
        f_B = self.f(B)
        f_B = torch.bmm(f_B,f_B.permute(0,2,1))
        # e的形状：（批量大小，序列A的词元数，序列B的词元数）
        # beta的形状：（批量大小，序列A的词元数，embed_size），
        # 意味着序列B被软对齐到序列A的每个词元(beta的第1个维度)
        beta = torch.bmm(F.softmax(f_B.permute(0,2,1), dim=-1), B)
        # beta的形状：（批量大小，序列B的词元数，embed_size），
        # 意味着序列A被软对齐到序列B的每个词元(alpha的第1个维度)
        alpha = torch.bmm(F.softmax(f_A.permute(0, 2, 1), dim=-1), A)
        return beta, alpha

class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2)) # params'num:120.6k
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True) # 160.6K
        self.linear = nn.Linear(num_hiddens, num_outputs) # 600

    def forward(self, V_A, V_B):
        # 对两组比较向量分别求和
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # 将两个求和结果的连结送到多层感知机中
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat

class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)

        num_inputs_compare = embed_size * 2
        num_inputs_agg = num_hiddens * 2
        ffn_num_inputs = embed_size
        ffn_num_hiddens = embed_size
        ffn_num_outputs = embed_size
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout=0.3)
        self.attend = Attend(ffn_num_inputs, ffn_num_hiddens,ffn_num_outputs)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # 有3种可能的输出：蕴涵、矛盾和中性
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.pos_encoding(self.embedding(premises) * math.sqrt(self.num_hiddens))
        B = self.pos_encoding(self.embedding(hypotheses) * math.sqrt(self.num_hiddens))
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self,ffn_num_input,ffn_num_hiddens,ffn_num_outputs,**kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.activate = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens,ffn_num_outputs)

    def forward(self,X):
        return self.dense2(self.activate(self.dense1(X)))


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self,num_hiddens,dropout,max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1,max_len,num_hiddens))
        X = torch.arange(max_len,dtype=torch.float32).reshape(-1,1) / torch.pow(10000,torch.arange(
            0,num_hiddens,2,dtype=torch.float32)/num_hiddens)
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)

    def forward(self,X):
        X = X + self.P[:,:X.shape[1],:].to(X.device)
        return self.dropout(X)