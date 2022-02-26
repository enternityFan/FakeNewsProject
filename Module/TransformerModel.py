# @Time : 2022-02-25 9:16
# @Author : Phalange
# @File : TransformerModel.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


import math
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

class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self,**kwargs):
        super(Encoder, self).__init__(**kwargs)
    def forward(self,X,*args):
        raise NotImplementedError

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)
        self.prem_encoder = TransformerEncoder(vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,
                                               ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,use_bias)
        self.hpy_encoder = TransformerEncoder(vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,
                                               ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,use_bias)
        #self.full_encoder = TransformerEncoder(vocab_size,key_size,query_size,value_size,num_hiddens,norm_shape,
        #                                       ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,use_bias)
        self.addAttention = AdditiveAttention(key_size,query_size,num_hiddens,dropout)
        self.dense = nn.Linear(num_hiddens,3) # 全连接层
        #self.mlp = mlp()

    def forward(self,X):
        premises,prem_valid_len, hypotheses,hpy_valid_len = X
        premises = self.prem_encoder(premises,prem_valid_len)
        hypotheses = self.hpy_encoder(hypotheses,hpy_valid_len)
        # 做一个合理的叠加的操作
        prem_hpy = torch.cat([premises,hypotheses],dim=1) # prem_hpy的shape:(batch_size,A+B的词元数，词向量）
        prem_hpy_valid_len = torch.tensor([max(prem_valid_len[i],hpy_valid_len[i]) for i in range(len(prem_valid_len))],device=premises.device)
        output = self.addAttention(prem_hpy,prem_hpy,prem_hpy,prem_hpy_valid_len) # params'num:20.1K

        #prem_hpy = self.full_encoder(prem_hpy,prem_hpy_valid_len)
        output = output.sum(dim=2) # output.shape = (batch_size,A+B的词元数）
        output = self.dense(output) # output.shape = (batch_size,3)
        return output




class TransformerEncoder(Encoder):
    """transformer编码器"""
    def __init__(self,vocab_size,key_size,query_size,value_size,
                 num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,
                 num_heads,num_layers,dropout,use_bias=False,**kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddnes = num_hiddens
        self.embedding = nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens,dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 EncoderBlock(key_size,query_size,value_size,num_hiddens,
                                              norm_shape,ffn_num_input,ffn_num_hiddens,
                                              num_heads,dropout,use_bias))

    def forward(self,X,valid_lens,*args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddnes))

        self.attention_weights = [None] * len(self.blks)
        for i,blk in enumerate(self.blks):
            X = blk(X,valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X



class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self,key_size,query_size,value_size,num_hiddens,
                 norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,
                 dropout,use_bias=False,**kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)
        self.addnorm1 = AddNorm(norm_shape,dropout)
        self.ffn = PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm2 = AddNorm(norm_shape,dropout)

    def forward(self,X,valid_lens):
        Y = self.addnorm1(X,self.attention(X,X,X,valid_lens))
        return self.addnorm2(Y,self.ffn(Y))




class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self,ffn_num_input,ffn_num_hiddens,ffn_num_outputs,**kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.activate = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens,ffn_num_outputs)

    def forward(self,X):
        return self.dense2(self.activate(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self,normalized_shape,dropout,**kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self,X,Y):
        return self.ln(self.dropout(Y) + X)



class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self,key_size,query_size,value_size,num_hiddens,
                 num_heads,dropout,bias=False,**kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=bias)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias=bias)
        self.W_o = nn.Linear(num_hiddens,num_hiddens,bias=bias)

    def forward(self,queries,keys,values,valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries),self.num_heads)
        keys = transpose_qkv(self.W_k(keys),self.num_heads)
        values = transpose_qkv(self.W_v(values),self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

            # output的形状:(batch_size*num_heads，查询的个数，
            # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)



class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=False)
        self.w_v = nn.Linear(num_hiddens,1,bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self,queries,keys,values,valid_lens):
        queries,keys = self.W_q(queries),self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # queries在第三维拓展一个维度，keys在第二维拓展一个维度
        #print("queries's size : " + str(queries.shape))
        #print("keys's size : " + str(keys.shape))
        #print("features's size : "+str(features.shape))
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：（batch_size,查询个数，“键-值”对的个数）
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores,valid_lens)
        # values的形状：(batch_size,"键-值“对的个数，值的维度）
        return torch.bmm(self.dropout(self.attention_weights),values) #  bmm函数的第一位是batch，要求相等




class DotProductAttention(nn.Module):
    def __init__(self,dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

        # queries的形状：(batch_size，查询的个数，d)
        # keys的形状：(batch_size，“键－值”对的个数，d)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)

    def forward(self,queries,keys,values,valid_lens=None):

        d = queries.shape[-1]
        # 设置transpose_b = True 是为了交换keys的最后两个维度
        scores = torch.bmm(queries,keys.transpose(1,2) / math.sqrt(d))
        self.attention_weights = masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)




def masked_softmax(X,valid_lens):
    """ 通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3d张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X,dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens,shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1,shape[-1]),valid_lens,value=-1e6)
        return nn.functional.softmax(X.reshape(shape),dim=-1)

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



def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X






def transpose_qkv(X,num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)

    # 输出X的形状：(batch_size,num_heads,查询或者"键-值"对的个数，
    # num_hiddens/num_heads)
    X = X.permute(0,2,1,3)

    # 最终输出的形状：（batch_size*num_heads,查询或者"键-值“对的个数，
    # num_hiddens/num_heads)
    return X.reshape(-1,X.shape[2],X.shape[3])

def transpose_output(X,num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X = X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)