# @Time : 2022-02-22 18:55
# @Author : Phalange
# @File : train.py.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import random
import DataProcess
import timer
data = pd.read_csv("./Data/train.csv")
#test_data = pd.read_csv("./Data/test.csv")
# 取前90%的数据进行训练

data = data.iloc[:,[3,4,-1]]
#test_data = test_data.iloc[:,[0,1,2,3,4]]
print(data.iloc[:5,])
print(data.shape)
data.dropna(axis=0,how='all')
num_data = data.shape[0]
features = list(data.iloc[:,0] +"split"+ data.iloc[:,1])
labels = list(data.iloc[:,2])
print(features[:5])
print(labels[:5])

"""
参数配置
"""
batch_size = 256
lr = 0.01


def Replace(text,old,new): #替换列表的字符串
    res = []
    for line in text:
        for ch in old:
            line = str(line).replace(ch,new)
        res.append(line)
    return res


def load_data(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size,num_examples)])
        yield features[batch_indices,:-1],labels[batch_indices,:-1]

# 数据预处理   这里发现这些中文的标点符号全角和半角一定要注意。。
#sign = '!"#$%&()*+,-./:;<=>?@[\\]^_，“：‘！？……”《》{|}~\xa0'
sign =  "\xa0！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

features = Replace(features,sign,' ')
#print(features[500:600])

print("数据预处理成功！")

# 词元化
#tokens  = DataProcess.tokenize(features)

#print(tokens[:5])
#print("词元化成功！")

# 词表

#Timer = d2l.Timer()
#vocab = DataProcess.Vocab(tokens)
#print(list(vocab.token_to_idx.items())[:10])
#print("词表制作成功！")

corpus,vocab = DataProcess.load_corpus_fake_news(features)
train_features,train_labels = load_data(batch_size,features[:round(num_data*0.9)],labels[:round(num_data*0.9)])
test_features,test_labels = load_data(batch_size,features[round(num_data*0.9):],labels[round(num_data*0.9):])
print(train_features.shape)
print(test_features.shape)


