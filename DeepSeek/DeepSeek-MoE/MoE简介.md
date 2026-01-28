# MoE简介

当前的流行LLM均使用了MoE(Mixture of Experts)架构，用于**条件计算**，LLM推理的时候通过门口单元，只经过稀疏的几个专家网络块计算，然后输出结果，大大降低了计算时间和推理难度，几乎是当前LLM使用必须的一个流程。下面介绍几篇MoE的经典论文及其原理

## MoE开山之作

首先介绍LLM的MoE开山之作——[《OUTRAGEOUSLY LARGE NEURAL NETWORKS:  THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER》](https://arxiv.org/abs/1701.06538)。由Google Brain团队在2017年发表于ICLR，距离Transformer提出的时间很接近。文章提出的**稀疏门控混合专家层**是当前各种MoE变体的基础。下面介绍数学公式原理

MoE层包含了 $n$ 个专家网路 $E_1,\cdots,E_n$ 和一个门控网络 $G$ 输出一个 $n$ 维的稀疏向量。如下图所示

![MoE Layer架构](MoE%20Layer.png)

专家块 $\text{Expert}_i$ 实际上就是神经网络，均支持同样大小的输入和输出。假设门控网络和第 $i$ 个专家块的输出为 $G(x)$ 和 $E_i(x)$，则MoE模型的输出为
$$
\begin{align}
y = \sum_{i=1}^n G(x)_iE_i(x) \tag{1}
\end{align}
$$
其中条件计算的体现在于 $G(x)_i= 0$ 则不计算 $E_i(x)$。下面主要介绍几种门控网络的设计

### 门控网络设计
**Softmax Gating**：一种简单的非稀疏的门控函数，输出乘以可训练权重矩阵 $W_g$ 再计算一下Softmax函数如下所示
$$
G_{\sigma}(x) = \text{Softmax}(x\cdot W_g)\tag{2}
$$
**Noisy Top-K Gating**：对上述Softmax门控函数加入**稀疏性**和**噪声**。在取软Softmax函数前，我们先添加**可调高斯噪声**，然后只保留Top-k个值，其余设为 $-\infty$（这使得相应的门值在经过Softmax后为0）。噪声项有助于负载均衡，每个分量的噪声量由第二个可训练权重矩阵 $W_{\text{noise}}$ 控制。
$$
\begin{align}
&G(x) = \text{Softmax} (\text{KeepTopK}(H(x),k))\\
&H(x)_i = (x\cdot W_g)_i + \text{StandardNormal()}\cdot\text{Softplus}((x\cdot W_{\text{noise}})_i)\\
& \text{KeepTopK}(v,k)_i = \left\{
\begin{aligned}
&v_i,\quad \text{if}\ v_i\ \text{is in the top}\ k \ \text{element of}\ v \\
&-\infty, \quad \text{otherwise}.
\end{aligned}
\right.
\end{align}\tag{3}
$$
其中两个重要函数 $\text{Softmax}$ 和 $\text{Softplus}$ ，其中 $\text{Softmax}$ 用于多分类概率问题，将 $NN$ 的输出转化为概率分布
$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{i=1}^n e^{x_i}}
$$
而 $\text{Softplus}$ 是激活函数，为 $\text{ReLu}$ 的平滑近似
$$
\text{Softplus}(x ) = \ln (1 + e^x)
$$

最后训练门控网络直接采用向后传播，同时训练网络的其他部分。

### 解决性能挑战

##### 批量缩小问题
现在的CPUs和GPUs的使用过程中，大批量对于计算效率非常重要的，而对于MoE模型，则批量会变为 $\frac{kb}{n}\ll b$，使得计算效率大大下降。解决办法为让原来的批量尽可能大，有一下几个方法

**数混合据并行和模型并行**：
这种方法将标准层按数据并行分布，但让所有设备同步处理不同的数据批次，并将这些批次中发往同一专家的样本合并。通过让每个专家接收来自所有设备的汇总样本，专家处理的批次大小提升了 $d$ 倍（设备数量），从而在增加模型参数的同时保持设备负担恒定。

**利用卷积特性**：
在语言模型中，将同一 MoE 应用于前一层的所有时间步，通过等待前一层全部完成后再统一处理，将所有时间步的样本合并为一个大批次。这种做法能使进入 MoE 层的批次大小直接翻倍，倍数等同于序列展开的时间步数。

**增加循环MoE的批次大小**：
对于专家层存在递归依赖（如 LSTM）导致无法合并时间步的情况，通过引入重新计算前向激活值的技术来减少内存占用。这种权衡空间与时间的策略，允许系统在有限的资源下容纳更大的输入批次，从而提升循环 MoE 结构的训练效率。

##### 网络带宽
网络带宽是影响性能的主要因素之一，由于稀疏专家的计算，所以主要通信消耗在于将专家的输入和输出发送到网络的两端。于是**专家计算量与输入输出大小的比例必须超过计算设备计算容量与网络容量的比值**。比如对于GPUs，为几千比一。实际上就是增大隐藏层的宽度。

### 专家平衡利用
在训练过程中发现门控网络总会收敛到一个状态，使得某些专家的权重很大，导致其他没有很好利用。这里采用一个软约束犯法，定义专家相对于一批训练样本的重要性为该专家门值的批量总和。定义损失函数 $L_{\text{importance}}$，加入整体损失函数中，其值和上述定义相同，并乘以一个人工缩放因子 $w_{\text{importance}}$，让每个专家有同等的重要性
$$
\begin{align}
\text{Importance}(X)  &= \sum_{x\in X}G(x),\\
L_{\text{importance}}(X) &= w_{\text{importance}}\cdot CV(\text{Importance}(X))^2
\end{align}
$$
其中CV的含义是变异系数，衡量数据离散程度的统计量，这里代表**不同专家之间重要性的差异**，通过最小化这个函数，来达到每个专家都有近乎相同权重的效果。