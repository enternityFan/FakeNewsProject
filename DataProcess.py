# @Time : 2022-02-22 21:14
# @Author : Phalange
# @File : DataProcess.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D

import collections
import re
from d2l import torch as d2l
import jieba_fast as jieba
import torch
import random
import os
import re
def preprocess(features,label_set,mode='ch'):
    """传入的应该是一个[data.iloc[:,0] , data.iloc[:,1],data.iloc[:,2]]列表,还有一个label_set"""


    labels = [ label_set[line.tolist()[2]]for line in features if line.tolist()[2] in label_set]
    # 去掉字符串
    if mode=='ch':
        premises,hypotheses = Replace_ch(features, ' ')
    elif mode =='en':
        premises,hypotheses = Replace_en(features,' ')

    return premises, hypotheses, labels

def Replace_en(text,new):
    premises, hypotheses = [], []
    for line in text:
        line1,line2 = str(line[0]),str(line[1])
        premises.append(re.sub(('[^A-Za-z0-9]+', ' ', line1).strip().lower()))
        hypotheses.append(re.sub(('[^A-Za-z0-9]+', ' ', line2).strip().lower()))

    return premises,hypotheses

def Replace_ch(text,new): #替换列表的字符串
    premises,hypotheses = [],[]
    sign = "\xa0！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    for line in text:
        line1,line2 = line[0],line[1]
        for ch in sign:
            line1 = str(line1).replace(ch,new)
            line2 = str(line2).replace(ch,new)
        premises.append(line1)
        hypotheses.append(line2)
    return premises,hypotheses

def tokenize(lines,mode='ch'):
    """将文本行拆分为单词词元"""
    if mode=='ch':
        return [jieba.lcut(line) for line in lines]
    elif mode =='en':
        return [line.split() for line in lines]

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

class FakeNewsDataset_seq2seq:
    """用于加载SNLI数据集的自定义数据集"""
    def __init__(self, dataset, num_steps, vocab=None,mode='ch'):
        self.num_steps = num_steps
        all_premise_tokens = tokenize(dataset[0],mode)
        all_hypothesis_tokens = tokenize(dataset[1],mode)
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.prem_valid_len = (self.premises != vocab['<pad>']).type(torch.int32).sum(1)
        self.hyp_valid_len = (self.hypotheses != vocab['<pad>']).type(torch.int32).sum(1)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx],self.prem_valid_len[idx], self.hypotheses[idx],self.hyp_valid_len[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)



#@save
class FakeNewsDataset:
    """用于加载SNLI数据集的自定义数据集"""
    def __init__(self, dataset, num_steps, vocab=None,mode='ch'):
        self.num_steps = num_steps
        all_premise_tokens = tokenize(dataset[0],mode)
        all_hypothesis_tokens = tokenize(dataset[1],mode)
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)

        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)

def count_corpus(tokens):  # @save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)



class Embedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        """Defined in :numref:`sec_synonyms`"""
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = embedding_name
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(embedding_name, 'r',encoding='utf-8') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)