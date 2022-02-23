# FakeNewsProject
这个仓库用来做kaggle的FakeNews竞赛

## 基于注意力的自然语言推断方法
### baseline
基于《动手学习深度学习》的代码，模型部分没有改变，只优化了一下
DecomposableAttention类中的num_inputs_attend，num_inputs_compare，
num_inputs_agg这几个变量，本来代码是默认赋值，这里用他们与embed_size的关系进行修改，
方便对中文文本和英文文本的处理模型的调用。


在处理英文数据的模式下，默认词元是以单词的形式进行。

### 修改激活函数为PReLu()



# 错误记录

## RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
这个错误在test.en.py中预测标签的哪个步骤遇到了，通过

`premise = torch.tensor(vocab[premise], device=d2l.try_gpu()).long()`
使用.long进行转换。