# @Time : 2022-02-23 9:12
# @Author : Phalange
# @File : evalScript.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D
from d2l import torch as d2l
import torch

label_list = ['disagreed','agreed','unrelated']
#@save
def predict_fake_news(net, vocab, premise, hypothesis):
    """预测前提和假设之间的逻辑关系"""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return label_list[label]