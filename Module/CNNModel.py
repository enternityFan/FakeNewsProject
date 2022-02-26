# @Time : 2022-02-26 8:48
# @Author : Phalange
# @File : CNNModel.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import torch
from torch import nn
from d2l import torch as d2l


class TextCNN(nn.Module):
    def __init__(self,vocab_size,embed_size,kernel_sizes,num_channels,**kwargs):
        super(TextCNN,self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels),3)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.active = nn.PReLU()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c,k in zip(num_channels,kernel_sizes):
            self.convs.append(nn.Conv1d(embed_size,c,k))

    def forward(self,inputs):
        premises, hypotheses = inputs
        embeddings = torch.cat((self.embedding(premises),self.embedding(hypotheses)),dim=1)
        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0,2,1)
        # 每一个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        # params'num:当conv为[3,4,5]时,输入通道是[100,100,100]时，120K
        encoding = torch.cat([
            torch.squeeze(self.active(self.pool(conv(embeddings))),dim=-1)
            for conv in self.convs],dim=1)
        outputs = self.decoder(self.dropout(encoding)) # params' num: 900
        return outputs

