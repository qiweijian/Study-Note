#! https://zhuanlan.zhihu.com/p/656952541
## 写在前面
> 日期： 2023-09-18 
> 最近深感自己基础薄弱，准备顺着UMass的Advanced NLP CS685 的[大纲](https://people.cs.umass.edu/~miyyer/cs685_f22/schedule.html)，补齐一些NLP基础知识。

# 1 原版Attention 论文阅读笔记 《NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE》

神经机器翻译 Neural Machine Translation，顾名思义，用神经网络做翻译的任务就是神经机器翻译。传统的机器翻译方法大多是基于短语的。

如上图所示，大部分此类网络都是编码器解码器架构。中间状态用固定长度的向量保存，造成的模型的瓶颈。如果测试阶段输入句子比训练集句子长，性能就会大幅度下降。

本工作的motivation就是避开这种有损压缩，用一组向量代替一个向量。

> In order to address this issue, we introduce an extension to the encoder–decoder model which learns to align and translate jointly.

每一次解码器输出一个词，它就从源输入中寻找最相关的词的位置。
具体来讲，不用一个定长向量做embedding，而是嵌入成一组向量，让解码器自己决定用哪些向量的信息。

## 背景知识

机器翻译任务的可以建模为一个概率问题
$$
\arg \max _{\mathbf{y}} p(\mathbf{y} \mid \mathbf{x}) \quad {\small \mathbf{x,y}是输入输出序列}
$$
用神经网络的方法已经取得了很好的性能，例如Ilya的Sequence to sequence learning with neural networks. (nips 2014)

### RNN enc-dec
在RNN的一个encoder单元中
$$
h_t=f\left(x_t, h_{t-1}\right)
$$
接收上一个时间的状态和当前的输入，得到当前状态 $h_t$。
多个时间的状态作为输入，得到上下文向量 $c$
$$
c=q\left(\left\{h_1, \cdots, h_{T_x}\right\}\right),
$$

例如Ilya那一篇将LSTM作为$f$，$q$直接取最后时间的$h_T$。

decoder在预测下一词任务上训练。
$$
p(\mathbf{y})=\prod_{t=1}^T p\left(y_t \mid\left\{y_1, \cdots, y_{t-1}\right\}, c\right)
$$

## 方法

### encoder —— 双向RNN
相当于两个RNN，一个前向一个后向。第$j$位的输出是
$$
h_j=\left[\overrightarrow{h}_j^{\top} ; \overleftarrow{h}_j^{\top}\right]^{\top}
$$

### decoder

$$
p\left(y_i \mid \underbrace{y_1, \ldots, y_{i-1}}_{\text{前i-1步的输出}}, \mathbf{x}\right) = g\left(y_{i-1}, \underbrace{s_i}_{\text{当前状态}}, \underbrace{c_i}_{\text{上下文}}\right)
$$

当前状态 $s_i$ 是RNN第$i$步的输出。
$$
s_i=f\left(s_{i-1}, y_{i-1}, c_i\right)
$$

$c_i$ 是本模型和传统encdec模型的区别，对于每一个$y_i$都有一个单独的$c_i$，而不是前文中的一个固定向量$c$。

$$
c_i=\sum_{j=1}^{T_x} \underbrace{\alpha_{i j}}_{\text{注意力分数}} h_j \\
$$
$$
\mathrm{Probability} \underbrace{\alpha_{i j}}_{第i位输出受到第j输入的影响}=\frac{\exp \left(e_{ij}\right)}{\sum_{k=1}^{T_x} \exp \left(e_{i k}\right)} 
$$
$$
\mathrm{Energy} \underbrace{e_{i j}}_{\text{未归一化的}}=\underbrace{a}_{\text{对齐模型}}\left(s_{i-1}, h_j\right)
$$

### 对齐模型

#### 输入
$s_{i-1}$ 这是解码器在时间步 i−1 的隐藏状态。
$h_j$ 是编码器在第j步的隐藏状态。
这两个状态来自于不同模型，是解码器在解码第i步时最重要的两个依据。

#### 对其模型$a$
这里用的就是一个前馈神经网络。作者在这里把神经网络实现的堆砌和统计机器翻译里显式定义的“对齐”做了区分：不再使用一个固定或难以推断的对齐，模型计算一个"软对齐"（soft alignment）。表示该单词对当前输出单词的重要性。这些权重是连续的，可以是任何值（通常在0和1之间）。这些权重反向传播时同步更新。

---
实验部分就没什么好说的了。