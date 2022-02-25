# @Time : 2022-02-25 14:10
# @Author : Phalange
# @File : train_en_transformer.py
# @Software: PyCharm
# C'est la vie,enjoy it! :D


import pandas as pd
import torch
from d2l import torch as d2l
import DataProcess
import Module.TransformerModel
from torch import nn
import os
import pickle
import Module.trick


weight_path = "./Cache/epoch_10_en.pth"
train_vocab_path = "./Data/train_vocab_en.pkl"
test_vocab_path = "./Data/test_vocab_en.pkl"
# 下面两个路径是进行符号预处理后的
train_data_path = "./Data/train_data_en.pkl"
test_data_path = "./Data/test_data_en.pkl"

label_set = {'disagreed': 0, 'agreed': 1, 'unrelated': 2}
label_list = ['disagreed','agreed','unrelated']
data = pd.read_csv("./Data/train.csv")
# 取前90%的数据进行训练

data = data.iloc[:,[5,6,-1]]
#test_data = test_data.iloc[:,[0,1,2,3,4]]
#print(data.iloc[:5,])
#print(data.shape)
data.dropna(axis=0,how='all')
#data = data[:5000]
num_data = data.shape[0]

features = list(data.values)





def load_data(batch_size,features,num_steps=50,train_vocab=None,test_vocab=None):

    num_workers = d2l.get_dataloader_workers()
    if not os.path.exists(train_data_path):
        train_data = DataProcess.preprocess(features[:round(num_data * 0.9)],label_set)
        train_data_file = open(train_data_path,'wb')
        pickle.dump(train_data,train_data_file)
        train_data_file.close()
    else:

        with open(train_data_path,'rb') as f:
            train_data = pickle.load(f, encoding='bytes')
        print("train data 加载成功！")
    if not os.path.exists(test_data_path):
        test_data = DataProcess.preprocess(features[round(num_data * 0.9):],label_set)
        test_data_file = open(test_data_path,'wb')
        pickle.dump(test_data,test_data_file)
        test_data_file.close()
    else:

        with open(test_data_path,'rb') as f:
            test_data = pickle.load(f,encoding='bytes')
        print("test data 加载成功！")


    train_set = DataProcess.FakeNewsDataset_seq2seq(train_data,num_steps,train_vocab,mode='en')
    test_set = DataProcess.FakeNewsDataset_seq2seq(test_data,num_steps,test_vocab,mode='en')
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab



if __name__ == "__main__":



    if os.path.exists(weight_path):
        train = False
    else:
        train = True
    #train = True
    batch_size = 64
    num_steps = 50
    if os.path.exists(train_vocab_path) and os.path.exists(test_vocab_path):
        train_vocab = DataProcess.Vocab()
        with open(train_vocab_path, 'rb') as f:
            train_vocab = pickle.loads(f.read())
        print("读取train vocab成功")
        test_vocab = DataProcess.Vocab()
        with open(train_vocab_path, 'rb') as f:
            test_vocab = pickle.loads(f.read())
        print("读取test vocab成功")
        train_iter, test_iter, vocab = load_data(batch_size, features, num_steps,train_vocab,test_vocab)
    else:
        train_iter, test_iter, vocab = load_data(batch_size, features, num_steps)
        # vocab词表对象通过pickle保存
        output_hal = open(train_vocab_path, 'wb')
        str = pickle.dumps(vocab)
        output_hal.write(str)
        output_hal.close()
        output_hal = open(test_vocab_path,'wb')
        str = pickle.dumps(vocab)
        output_hal.write(str)
        output_hal.close()

    #print(vocab.type)
    embed_size,  devices = 100, d2l.try_all_gpus()
    num_hiddens, num_layers, dropout  = 100, 1, 0.3
    ffn_num_input, ffn_num_hiddens, num_heads = 100, 100, 4
    key_size, query_size, value_size = 100, 100, 100
    norm_shape = [100]

    net = Module.TransformerModel.TransformerModel(len(vocab),key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout,use_bias=False)
    glove_embedding =d2l.TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.prem_encoder.embedding.weight.data.copy_(embeds)
    net.hpy_encoder.embedding.weight.data.copy_(embeds)
    if train:

        print("start training...")
        lr, num_epochs = 0.001, 10
        trainer = torch.optim.Adam(net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0,2.0,10.0],device=devices[0]).float(),reduction="none")
        cosScheduler = Module.trick.CosineScheduler(max_update=10, warmup_steps=5,base_lr=lr, final_lr=0.00007)
        Module.trick.train_transformer_scheduler(net, train_iter, test_iter, loss, trainer, num_epochs,
                       devices,scheduler=cosScheduler)
        d2l.plt.show()
        print("train success!")
        torch.save(net.state_dict(), './Cache/AttentionWeights.pth')
    else:
        print("只能训练。。。sorry")