# FakeNewsProject
这个仓库用来做kaggle的FakeNews竞赛

## 基于注意力的自然语言推断方法
### baseline
基于《动手学习深度学习》的代码，模型部分没有改变，只优化了一下
DecomposableAttention类中的num_inputs_attend，num_inputs_compare，
num_inputs_agg这几个变量，本来代码是默认赋值，这里用他们与embed_size的关系进行修改，
方便对中文文本和英文文本的处理模型的调用。

#### 中文结果（迭代20次）
batch_size=256,lr=0.001,在公共数据集上实现了0.48510的结果

#### 英文结果（迭代20次）
batch_size=256,lr=0.001,在公共数据集上实现了0.62478 0.58261



考虑二者差别比较大（主要是训练中文数据集时等待分词时间太长了。。。懒得等了。。），下面的修改都是修改英文数据的模型，
并且以迭代10次为最终迭代次数

#### 英文结果（迭代10次）使用余弦退火学习率
batch_size=2048,CosineScheduler(max_update=10, warmup_steps=5,base_lr=lr, final_lr=0.00007)
在公共数据集上实现了0.62512的结果。


在处理英文数据的模式下，默认词元是以单词的形式进行。


### AttentionModel_2
修改激活函数为PReLu()，修改MLP层数为3，把隐藏单元数量改为300，同时增大dropout的概率。

我使用的是train_.csv的前90%做训练集，后面的做测试集，然后我查看前90%数据样本的分布情况：
`label distribute :
2    196797
1     84226
0      7474`

#### 第一次实验
发现他们的占比为：68%，29%，0.0259%，为交叉熵损失函数添加[1,2,10]的权重,同时因为MLP的参数变多了，显存不够了，就把
batch_size调整为了1024.

epoch10_en_v2_1.x为本次的结果
在公共数据集上实现了0.71128的结果,卧槽，高兴死了，不过本次结果很反常跟之前3次提交相比，那就是他的
公共数据集的分数比个人数据集的分数高，我估计这是因为交叉熵损失函数的权重调整起作用了。

#### 第二次实验
这次我把交叉熵损失函数的权重都缩小，变为[1,1.7,4],并且改大学习率，因为在第一次实验的时候发现loss下降的没有以前多
lr=0.01,Module.trick.CosineScheduler(max_update=10, warmup_steps=5,base_lr=lr, final_lr=0.0007)

这个实验的效果远没有上次的好，但这次的loss下降的很快，Score: 0.59519 Public score: 0.63574

#### 第三次实验
可以进行第三次实验，这一次因为该吃饭了，电脑也空闲，尝试训练30次迭代，
权重变为[1,3,12],lr=0.01,Module.trick.CosineScheduler(max_update=30, warmup_steps=5,base_lr=lr, final_lr=0.0007)

这次实验失败了，这个参数根本就无法学习，一直loss不下降，然后我就又进行了一个实验：

权重变为[1,3,12],Module.trick.CosineScheduler(max_update=20, warmup_steps=5,base_lr=lr, final_lr=0.0001)
还是30次训练。这次结果是Score: 0.58440 Public score: 0.66348，效果不好，不过也是出现了公共数据集分数比私有数据集分数高的情况

#### 第四次实验
这次只是重复实验1的参数，只不过是迭代了30次，出去跑步了，电脑闲的没事搁着干活吧。
Score: 0.69389
Public score: 0.72253
不错不错！上升咯

### RNN_Model
在v2的基础上修改，~~首先根据原论文《A Decomposable Attention Model for Natural Language Inference》实现with intra-sentence attention的版本论文解释参数没解释清楚，在网上搜了开源代码也没有做这个的，就放弃了~~



#### 第一次实验
在V2的基础上修改，不过交叉熵损失的权重没有给。根据SNL数据集的论文和实验方法，首先尝试RNN的实验，实验中发现了loss下降比较缓慢的现象。。甚至可以说迭代10次，6次的loss都不动。
太神奇了，我就把MLP部分的tanh()激活函数修改为了PReLU()激活函数，我发现loss下降了。。精度上升了，但是有疑问，就是原论文其实是使用tanh的。
本次实验数据为epoch10_en_rnn_1,OK啦，实验结果为：Score: 0.51205 Public score: 0.55162

#### 第二次实验
我只是把交叉熵损失的权重给改为了[1,2,10]，研究一下这个交叉熵损失的权重是不是有关系。测试结束，loss不下降，哈哈哈。。。奇奇怪怪。有点神奇啊。。。我不知道动这啥了。。就是不给我训练了。。
可能是因为我在forward()里面错误的判断语句，那就以后test的时候注释掉这两行，train的时候注释掉那两行，这个样子。
epoch10_en_rnn_2是这几次做实验的图，我不知道犯了什么错要这么折磨我，train_acc = 0.682...一直都是这个数。。。求求了。。loss下降吧,不过也有可能，我在网上搜，
说是可能batch_size比较大，所以前期因为梯度平均，所以下降的速度过慢。我把batch_size从4096调小到1024，训练还是loss不下降，不过，我把交叉熵损失的权重给去除之后。。loss终于下降了。好事。

### LSTM_Model
这个实验的交叉损失权重是[1,2,10]的。
前4轮迭代都没有下降loss。。第5轮开始下降了，batch_size为1024
Score: 0.57845 Public score: 0.66675 这个结果还是可以说非常好的


### transformer_Model（貌似是。。我也不知道是不是
#### 第一次训练

这个训练的速度是最慢最慢的，把他的训练过程给弄出来，以后闲了可以训练训练这个，因为下架速度也是很客观的。

> the 0 epochs success!,the loss: 1.925,train acc 0.697
>
> the 1 epochs success!,the loss: 1.521,train acc 0.764
>
> the 2 epochs success!,the loss: 1.470,train acc 0.762
>
> the 3 epochs success!,the loss: 1.383,train acc 0.778
>
> the 4 epochs success!,the loss: 1.302,train acc 0.794
>
> the 5 epochs success!,the loss: 1.237,train acc 0.804
>
> the 6 epochs success!,the loss: 1.182,train acc 0.815
>
> the 7 epochs success!,the loss: 1.111,train acc 0.825
>
> the 8 epochs success!,the loss: 1.036,train acc 0.838
>
> the 9 epochs success!,the loss: 0.965,train acc 0.849

Score: 0.66404 Private score: 0.57750

哎,没我想象的结果好,不过也不错了其实,白亏了我训练这么久。

#### 第二次训练

这次训练我把num_layers给设置为2，《动手学习深度学习》书上的参数，并且把batch_size也从64改为了128，然后训练20次，希望结果惊艳我！！！

Score: 0.57612 Public score: 0.64924

好吧，更垃圾了。。

### CNN_Model

这个训练速度挺快的嘿！

#### 第一次实验

参数设置就不说了，这个的train代码是和train_en里面的一模一样，test_en也是一样的，这两个的读取数据的要求都一样，真好

Score: 0.57215 Public score: 0.60753

数据比我想象的要垃圾，因为我在自己的电脑上训练感觉还是很不错的。。哎。不过把也是，反正我感觉我的这个网络的结构，根本没有获得两句话的关系。。

```python
# kernel_sizes, nums_channels = [3,4,5], [100, 100, 100] # v1和v2的数据
```

#### 第二次实验

只是修改了交叉熵损失函数让他有权重了，我还是认为这个交叉熵损失的权重是挺好用的。

Score: 0.57942 Public score: 0.66770

```python
# kernel_sizes, nums_channels = [3,4,5], [100, 100, 100] # v1和v2的数据
```

#### 第三次实验

稍微修改一下网路的卷积核的步数，并且提高了batch_size的大小，训练30次。

```python
kernel_sizes, nums_channels = [3,5,7,9,11], [100, 100, 100]
```

Score: 0.57766 Public score: 0.66529结果不尽人意哈哈。

### Attention+CNN Model

在最初的Attention_Model的基础上，把mlp给替换成cnn进行尝试。

f()函数不改变输入的大小。g()函数也不改变输入的大小。



# 实验结果整理

注：没有特殊说明，实验结果都是在迭代10次得到的。

| Model               | Private Score | Public Score | 描述                                                         |
| ------------------- | ------------- | ------------ | ------------------------------------------------------------ |
| Attention_Model_ch  | 0.48510       | 0.52344      | 用的中文的数据做的，不过后来因为分词太慢了，并且效果也很垃圾。。使用的模型就再也不处理中文的了。 |
| Attention_Model_en  | 0.62478       | 0.58261      | 使用的英文的数据进行的。                                     |
| Attention_schaduler | 0.65485       | 0.62512      | 这个就是在上面的基础上添加了学习率调度器，效果还是提升了不少的。 |
| AttentionModel_2    | **0.69389**   | **0.72253**  | 这个是训练了30次得到的结果。主要修改了部分网络的结构。       |
| RNN_Model           | 0.51205       | 0.55162      | 这个结果还是很差的，并且有一个很麻烦的问题，不过也可能是我batch_size设置的太大了，导致前几批次loss不下降，并且修改lr后会10次迭代loss都不下降 |
| LSTM_Model          | 0.57750       | 0.66404      | 中规中矩的一个结果。                                         |
| transformer_Model   | 0.57750       | 0.66404      | 基于transformer改的一个模型，训练速度很慢，并且结果也没预期的那么好，我以为这个结果会是最好的，哈哈，或者起码上70吧。 |
| CNN_Model           | 0.57942       | 0.66770      | CNN模型，速度很快，使用有权重的损失函数的结果，好了很多。    |
|                     |               |              |                                                              |







# 其他资料



NLLLoss
不同损失函数的讲解
https://zhuanlan.zhihu.com/p/61379965



孪生神经网络



## nn.Sequential与nn.ModuleList的区别

nn.ModuleList 这个类，你可以把任意 nn.[Module](https://so.csdn.net/so/search?q=Module&spm=1001.2101.3001.7020) 的子类添加到这个list里面，方法和Python 自带的 list 一样，但与一般的list的区别在于，加入到 nn.ModuleList 里面的 module 是会注册到整个网络上的，同时 module 的 parameters 也会自动添加到整个网络中。它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。但是，我们需要注意到，nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言，执行顺序还是按照forward里面的写法来执行的。

nn.Sequential，不同于 nn.ModuleList，它已经实现的 forward 函数，而且里面的模块是按照顺序进行排列的，所以我们必须确保前一个模块的输出大小和下一个模块的输入大小是一致的，并且需要保证顺序，因为不能够再定制forward函数的顺序。

参考《PyTorch 中的 ModuleList 和 Sequential: 区别和使用场景》csdn博客。



# 错误记录

## RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
这个错误在test.en.py中预测标签的哪个步骤遇到了，通过

`premise = torch.tensor(vocab[premise], device=d2l.try_gpu()).long()`
使用.long进行转换。