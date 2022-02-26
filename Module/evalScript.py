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
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu()).long()
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu()).long()
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return label_list[label]


def predict_fake_news_rnn(net, vocab, premise, hypothesis,num_steps=50):
    """预测前提和假设之间的逻辑关系"""
    state = None
    net.eval()
    premise = torch.tensor([d2l.truncate_pad(vocab[premise], num_steps, vocab['<pad>'])],device=d2l.try_gpu()).long()

    hypothesis = torch.tensor([d2l.truncate_pad(vocab[hypothesis], num_steps, vocab['<pad>'])],device=d2l.try_gpu()).long()
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))],state), dim=1)
    return label_list[label]

def predict_fake_news_transformer(net, vocab, premise, hypothesis,num_steps=50):
    """预测前提和假设之间的逻辑关系"""
    net.eval()
    premise = torch.tensor([d2l.truncate_pad(vocab[premise], num_steps, vocab['<pad>'])],device=d2l.try_gpu()).long()
    prem_valid_len = (premise != vocab['<pad>']).type(torch.int32).sum(1)
    hypothesis = torch.tensor([d2l.truncate_pad(vocab[hypothesis], num_steps, vocab['<pad>'])],device=d2l.try_gpu()).long()
    hyp_valid_len = (premise !=vocab['<pad>']).type(torch.int32).sum(1)
    X = [premise.reshape((1,-1)),prem_valid_len,hypothesis.reshape((1,-1)),hyp_valid_len]

    label = torch.argmax(net(X), dim=1)
    return label_list[label]

def predict_fake_news_cnn(net, vocab, premise, hypothesis,num_steps=50,):
    """预测前提和假设之间的逻辑关系"""
    net.eval()
    premise = torch.tensor([d2l.truncate_pad(vocab[premise], num_steps, vocab['<pad>'])], device=d2l.try_gpu()).long()

    hypothesis = torch.tensor([d2l.truncate_pad(vocab[hypothesis], num_steps, vocab['<pad>'])],
                              device=d2l.try_gpu()).long()


    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return label_list[label]