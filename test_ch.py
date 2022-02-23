# @Time : 2022-02-22 18:56
# @Author : Phalange
# @File : test_ch.py.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import os
import pandas as pd
import torch
from d2l import torch as d2l
import DataProcess
import Module.AttentionModel
import Module.evalScript
from torch import nn
import numpy as np
import jieba_fast as jieba
from tqdm import *
import pickle

weight_path = "./Cache/epoch_20_ch.pth"
glove_file ="./Data/wiki.zh.glove.Mode"
train_vocab_path = "./Data/train_vocab.pkl"
test_vocab_path = "./Data/test_vocab.pkl"

label_set = {'disagreed': 0, 'agreed': 1, 'unrelated': 2}
label_list = ['disagreed','agreed','unrelated']

if not os.path.exists(weight_path):
    print("请检查权重路径是否正确！")
    raise FileNotFoundError


test_data = pd.read_csv("./Data/test.csv")
test_data = test_data.iloc[:,[0,3,4]]# id 前提 假设
test_data = list(test_data.values)

def preprocess(features):
    """
        传入的应该是一个[data.iloc[:,0] , data.iloc[:,1],data.iloc[:,2]]列表
        返回一个三个列表组成的元组:(id,premises,hypotheses)
    """
    # 去掉字符串
    premises,hypotheses = Replace_ch(features, ' ')
    id = [int(line.tolist()[0]) for line in features]


    return id,premises, hypotheses

def Replace_ch(text,new): #替换列表的字符串
    premises,hypotheses = [],[]
    sign = "\xa0！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    for line in text:
        line1,line2 = line[1],line[2]
        for ch in sign:
            line1 = str(line1).replace(ch,new)
            line2 = str(line2).replace(ch,new)
        premises.append(line1)
        hypotheses.append(line2)
    return premises,hypotheses

test_data = list(preprocess(test_data))
predict_label = []

# 读取事先保存的vocab
vocab = DataProcess.Vocab()
with open(train_vocab_path,'rb') as f:
    vocab = pickle.loads(f.read())
print("读取vocab成功")

#vocab = DataProcess.Vocab(DataProcess.tokenize(test_data[1])+DataProcess.tokenize(test_data[2]),min_freq=5, reserved_tokens=['<pad>'])
#print("vocab makes success!")

embed_size, num_hiddens, devices = 300, 200, d2l.try_all_gpus()
net = Module.AttentionModel.DecomposableAttention(vocab, embed_size, num_hiddens)



net.load_state_dict(torch.load(weight_path))
#下面这个glove层应该就不用加载了，因为保存的时候就是有的。
glove_embedding =DataProcess.Embedding(glove_file)
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.to(device=devices[0])
print("模型加载成功！！准备预测。。。")
net.eval()
save_data = []
for i in tqdm(range(len(test_data[0]))):
    label = Module.evalScript.predict_fake_news(net, vocab, jieba.lcut(test_data[1][i]), jieba.lcut(test_data[2][i]))

    save_data.append([test_data[0][i],label])

print(predict_label[:5])

# 保存submission.csv
print("saving data....")
df = pd.DataFrame(save_data,columns=["Id","Category"])
df.to_csv("./Data/submission_ch.csv",index=False)
print("data saving success!!")

