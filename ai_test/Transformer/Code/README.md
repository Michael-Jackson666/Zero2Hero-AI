# Transformer 代码实现

本目录包含完整的 Transformer Encoder-Decoder 实现, 代码已模块化分解为多个文件, 便于理解、学习和复用.

---

## 📁 完整文件结构

```
Code/
├── 核心模块 (推荐版本 - 模块化设计)
│   ├── __init__.py                    # 模块初始化, 统一导出接口
│   ├── attention.py                   # 注意力机制 (ScaledDotProductAttention, MultiHeadAttention)
│   ├── embedding.py                   # 位置编码 (PositionalEncoding)
│   ├── feedforward.py                 # 前馈网络和归一化 (FeedForward, ResidualLayerNorm)
│   ├── mask.py                        # 掩码工具函数 (make_pad_mask, make_subsequent_mask)
│   ├── layers.py                      # Encoder/Decoder 层 (EncoderLayer, DecoderLayer)
│   └── transformer.py                 # 完整 Transformer 模型 (Transformer)
│
├── 单独模块 (教学版本 - 便于单独学习)
│   ├── ScaledDotProductAttention.py   # 缩放点积注意力 (独立文件)
│   └── MultiHeadAttention.py          # 多头注意力 (独立文件)
│
├── 测试与示例
│   ├── test_attention.py              # 注意力模块测试 (5个测试场景)
│   └── ATTENTION_USAGE.md             # 注意力模块使用指南 (详细示例)
│
├── 参考文档
│   ├── README.md                      # 本文件 (总体说明)
│   └── Combined.py                    # 原始合并版本 (已弃用, 仅供参考)
```

---

## 🎯 使用指南

### 选择合适的版本

#### 1. **推荐版本** (核心模块)
适用场景:
- ✅ 构建完整的 Transformer 模型
- ✅ 集成到项目中使用
- ✅ 理解模块间的关系
- ✅ 面试时展示架构设计能力

使用方式:
```python
from transformer import Transformer
# 或
from attention import MultiHeadAttention
from layers import EncoderLayer, DecoderLayer
```

#### 2. **教学版本** (单独模块)
适用场景:
- ✅ 学习单个组件的实现
- ✅ 面试时手撕代码 (逐个实现)
- ✅ 深入理解某个模块
- ✅ 快速测试和调试

使用方式:
```python
from ScaledDotProductAttention import ScaledDotProductAttention
from MultiHeadAttention import MultiHeadAttention
```

---

## 📚 模块详细说明

### 核心模块 (推荐版本)

#### 1. `__init__.py` - 模块初始化

**功能**: 统一导出所有公共接口

**导出内容**:
```python
__all__ = [
    'ScaledDotProductAttention',    # 缩放点积注意力
    'MultiHeadAttention',            # 多头注意力
    'PositionalEncoding',            # 位置编码
    'FeedForward',                   # 前馈网络
    'ResidualLayerNorm',             # 残差连接 + LayerNorm
    'make_pad_mask',                 # Padding 掩码工具
    'make_subsequent_mask',          # 因果掩码工具
    'EncoderLayer',                  # Encoder 层
    'DecoderLayer',                  # Decoder 层
    'Transformer',                   # 完整模型
]
```

**使用示例**:
```python
# 导入所有组件
from transformer_module import *

# 或选择性导入
from transformer_module import Transformer, MultiHeadAttention
```

---

#### 2. `attention.py` - 注意力机制 ⭐⭐⭐

**包含类**:
- `ScaledDotProductAttention`: 缩放点积注意力 (Attention 的核心)
- `MultiHeadAttention`: 多头注意力 (Transformer 的关键组件)

**核心公式**:
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**关键特性**:
- ✅ 支持任意形状的 Q, K, V
- ✅ 支持 Padding Mask 和 Causal Mask
- ✅ 多头并行计算
- ✅ Dropout 正则化

**代码行数**: ~150 行

**使用示例**:
```python
from attention import MultiHeadAttention

# 创建多头注意力
mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)

# 自注意力 (Self-Attention)
out, attn_weights = mha(x, x, mask=None)

# 交叉注意力 (Cross-Attention)
out, attn_weights = mha(query, key_value, mask=cross_mask)
```

**面试重点**:
- Q, K, V 的来源和作用
- 为什么除以 $\sqrt{d_k}$
- 多头的分割和合并过程
- Mask 的应用时机

---

#### 3. `embedding.py` - 位置编码 ⭐⭐

**包含类**:
- `PositionalEncoding`: 正弦/余弦位置编码

**核心公式**:
$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{aligned}
$$

**关键特性**:
- ✅ 固定编码, 无需训练参数
- ✅ 支持任意长度序列 (最大 `max_len`)
- ✅ 直接加到 Token Embedding 上

**代码行数**: ~40 行

**使用示例**:
```python
from embedding import PositionalEncoding

pos_enc = PositionalEncoding(d_model=512, max_len=5000, dropout=0.1)
x = pos_enc(x)  # x: (B, T, d_model)
```

**面试重点**:
- 为什么需要位置编码
- 正弦/余弦的优势
- 与可学习位置编码的区别

---

#### 4. `feedforward.py` - 前馈网络和归一化 ⭐⭐

**包含类**:
- `FeedForward`: Position-wise Feed-Forward Network
- `ResidualLayerNorm`: 残差连接 + Layer Normalization

**核心公式**:
$$
\begin{aligned}
\text{FFN}(x) &= W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2 \\
\text{ResidualLN}(x, f) &= \text{LayerNorm}(x + f(x))
\end{aligned}
$$

**关键特性**:
- ✅ 支持 ReLU 和 GELU 激活函数
- ✅ 通常 $d_{\text{ff}} = 4 \times d_{\text{model}}$
- ✅ Post-LN 实现 (也可改为 Pre-LN)

**代码行数**: ~60 行

**使用示例**:
```python
from feedforward import FeedForward, ResidualLayerNorm

ffn = FeedForward(d_model=512, d_ff=2048, activation='gelu')
norm = ResidualLayerNorm(d_model=512)

# 使用
out = ffn(x)
x = norm(x, out)  # 残差 + 归一化
```

**面试重点**:
- FFN 的作用 (非线性变换)
- Post-LN vs Pre-LN 的区别
- 为什么需要残差连接

---

#### 5. `mask.py` - 掩码工具函数 ⭐⭐

**包含函数**:
- `make_pad_mask()`: 构造 Padding Mask
- `make_subsequent_mask()`: 构造 Causal Mask (下三角)

**掩码类型**:

| 掩码类型 | 用途 | 形状 | 示例 |
|---------|------|------|------|
| **Padding Mask** | 屏蔽 PAD token | `(B, 1, T_q, T_k)` | Encoder/Decoder |
| **Causal Mask** | 防止看到未来 | `(1, 1, T, T)` | Decoder Self-Attn |
| **Cross Mask** | 屏蔽源端 PAD | `(B, 1, T_t, T_s)` | Decoder Cross-Attn |

**代码行数**: ~50 行

**使用示例**:
```python
from mask import make_pad_mask, make_subsequent_mask

# Padding mask
pad_mask = make_pad_mask(T, T, src_pad, src_pad)

# Causal mask (下三角)
causal_mask = make_subsequent_mask(T)

# 组合 (Decoder 自注意力)
tgt_mask = pad_mask & causal_mask if pad_mask is not None else causal_mask
```

**面试重点**:
- 为什么需要 Padding Mask
- Causal Mask 的作用 (自回归)
- 掩码的取值约定 (1=可见, 0=屏蔽)

---

#### 6. `layers.py` - Encoder/Decoder 层 ⭐⭐⭐

**包含类**:
- `EncoderLayer`: Transformer Encoder 层
- `DecoderLayer`: Transformer Decoder 层

**架构对比**:

| 组件 | EncoderLayer | DecoderLayer |
|------|--------------|--------------|
| **子层1** | Self-Attention | Masked Self-Attention |
| **子层2** | FFN | Cross-Attention |
| **子层3** | - | FFN |
| **掩码** | Padding Mask | Causal + Padding Mask |
| **输入** | 源序列 | 目标序列 + Encoder 输出 |

**代码行数**: ~120 行

**使用示例**:
```python
from layers import EncoderLayer, DecoderLayer

# Encoder Layer
enc_layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)
x, attn = enc_layer(x, src_mask)

# Decoder Layer
dec_layer = DecoderLayer(d_model=512, num_heads=8, d_ff=2048)
y, (self_attn, cross_attn) = dec_layer(y, memory, tgt_mask, mem_mask)
```

**面试重点**:
- Encoder 和 Decoder 的结构差异
- Cross-Attention 的 Q/K/V 来源
- 每层的输入输出形状

---

#### 7. `transformer.py` - 完整 Transformer 模型 ⭐⭐⭐

**包含类**:
- `Transformer`: 完整的 Encoder-Decoder 模型

**核心方法**:
- `encode()`: Encoder 前向传播
- `decode()`: Decoder 前向传播
- `forward()`: 完整前向传播 (训练)
- `greedy_decode()`: 贪心解码 (推理)

**模型参数**:
```python
Transformer(
    src_vocab=10000,      # 源端词表大小
    tgt_vocab=10000,      # 目标端词表大小
    d_model=512,          # 模型维度
    num_heads=8,          # 注意力头数
    d_ff=2048,            # FFN 隐藏层维度
    num_layers=6,         # Encoder/Decoder 层数
    dropout=0.1,          # Dropout 概率
    max_len=5000          # 最大序列长度
)
```

**代码行数**: ~200 行

**使用示例**:
```python
from transformer import Transformer

model = Transformer(src_vocab=10000, tgt_vocab=10000, d_model=512)

# 训练
logits = model(src, tgt_inp, src_pad, tgt_pad)

# 推理
output = model.greedy_decode(src, bos_id=1, eos_id=2, max_new_tokens=50)
```

**面试重点**:
- Encoder-Decoder 的交互方式
- 训练和推理的区别
- Teacher Forcing 机制

---

### 单独模块 (教学版本)

#### 8. `ScaledDotProductAttention.py` - 独立注意力模块

**特点**:
- ✅ 单文件实现, 无外部依赖
- ✅ 适合逐行讲解
- ✅ 面试手撕首选

**代码行数**: ~55 行

**使用场景**:
- 面试时从零实现注意力机制
- 学习注意力的核心计算

---

#### 9. `MultiHeadAttention.py` - 独立多头注意力

**特点**:
- ✅ 依赖 `ScaledDotProductAttention.py`
- ✅ 完整的多头实现
- ✅ 包含分头和合头逻辑

**代码行数**: ~110 行

**使用场景**:
- 在实现单头注意力后扩展到多头
- 理解多头的分割和合并

---

### 测试与示例

#### 10. `test_attention.py` - 注意力模块测试

**包含测试**:
1. ✅ 缩放点积注意力测试
2. ✅ 多头注意力测试
3. ✅ 自注意力测试 (Q=K=V)
4. ✅ 交叉注意力测试 (Q≠K,V)
5. ✅ 因果掩码测试 (Causal Mask)

**代码行数**: ~250 行

**运行方式**:
```bash
python test_attention.py
```

**输出示例**:
```
============================================================
测试 ScaledDotProductAttention
============================================================
输入形状:
  Q: torch.Size([2, 4, 5, 8])
  K: torch.Size([2, 4, 6, 8])
  V: torch.Size([2, 4, 6, 8])
...
✅ 所有测试完成!
```

---

#### 11. `ATTENTION_USAGE.md` - 使用指南

**包含内容**:
- ✅ 快速使用示例
- ✅ 4 个详细场景 (自注意力、交叉注意力、掩码等)
- ✅ 参数说明表格
- ✅ 维度变换流程图
- ✅ 常见问题解答

**适用场景**:
- 快速上手注意力模块
- 查阅参数和返回值
- 理解不同使用场景

---

### 参考文档

#### 12. `Combined.py` - 原始合并版本

**说明**: 
- ⚠️ 已弃用, 仅供参考
- ❌ 包含多个错误 (已在其他文件中修正)
- ℹ️ 可用于对比模块化前后的差异

**已修正的错误** (16 处):
1. `nn.Moduel` → `nn.Module`
2. `super.__init__()` → `super().__init__()`
3. `self.W_k(x_kv)` → `self.W_v(x_kv)`
4. ... (见下文完整列表)

---

## 🚀 快速开始

### 方式 1: 使用完整模型 (推荐)

#### 安装依赖

```bash
pip install torch
```

#### 导入并使用

```python
from transformer import Transformer
import torch

# 创建模型
model = Transformer(
    src_vocab=10000,
    tgt_vocab=10000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    dropout=0.1
)

# 准备数据
B, T_s, T_t = 32, 20, 25
src = torch.randint(0, 10000, (B, T_s))      # 源序列
tgt_inp = torch.randint(0, 10000, (B, T_t))  # 目标序列输入
tgt_out = torch.randint(0, 10000, (B, T_t))  # 目标序列输出 (用于计算损失)

# 训练模式
model.train()
logits = model(src, tgt_inp)  # (B, T_t, tgt_vocab)

# 计算损失
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
loss = criterion(logits.view(-1, 10000), tgt_out.view(-1))
print(f"Loss: {loss.item():.4f}")

# 推理模式
model.eval()
output = model.greedy_decode(src, bos_id=1, eos_id=2, max_new_tokens=30)
print(f"Generated output shape: {output.shape}")
```

---

### 方式 2: 使用单个组件

#### 只使用注意力模块

```python
from attention import MultiHeadAttention
import torch

# 创建多头注意力
mha = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)

# 自注意力
x = torch.randn(2, 10, 64)  # (B, T, d_model)
out, attn_weights = mha(x, x)

print(f"输出形状: {out.shape}")           # (2, 10, 64)
print(f"注意力权重: {attn_weights.shape}")  # (2, 8, 10, 10)
```

#### 组合多个组件

```python
from attention import MultiHeadAttention
from feedforward import FeedForward, ResidualLayerNorm
from embedding import PositionalEncoding
import torch

# 创建组件
pos_enc = PositionalEncoding(d_model=64, max_len=100)
mha = MultiHeadAttention(d_model=64, num_heads=8)
ffn = FeedForward(d_model=64, d_ff=256)
norm1 = ResidualLayerNorm(d_model=64)
norm2 = ResidualLayerNorm(d_model=64)

# 模拟一个 Encoder Layer
x = torch.randn(2, 10, 64)
x = pos_enc(x)

# Self-Attention + Residual + Norm
attn_out, _ = mha(x, x)
x = norm1(x, attn_out)

# FFN + Residual + Norm
ffn_out = ffn(x)
x = norm2(x, ffn_out)

print(f"最终输出: {x.shape}")  # (2, 10, 64)
```

---

### 方式 3: 使用教学版本 (单文件)

```python
from ScaledDotProductAttention import ScaledDotProductAttention
from MultiHeadAttention import MultiHeadAttention
import torch

# 1. 测试缩放点积注意力
attn = ScaledDotProductAttention(dropout=0.1)
Q = torch.randn(2, 4, 5, 8)  # (B, H, T_q, d_k)
K = torch.randn(2, 4, 6, 8)  # (B, H, T_k, d_k)
V = torch.randn(2, 4, 6, 8)  # (B, H, T_k, d_v)
out, attn_weights = attn(Q, K, V)
print(f"Attention 输出: {out.shape}")

# 2. 测试多头注意力
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)
out, attn_weights = mha(x, x)
print(f"MHA 输出: {out.shape}")
```

---

### 运行测试

```bash
# 测试注意力模块
python test_attention.py

# 预期输出:
# ============================================================
# 测试 ScaledDotProductAttention
# ============================================================
# 输入形状:
#   Q: torch.Size([2, 4, 5, 8])
#   ...
# ✅ 所有测试完成!
```

---

## 📊 模型架构与参数量

### Transformer 整体架构

```
输入序列 (src)
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────────────┐
│     Encoder (N 层堆叠)              │
│  ┌───────────────────────────────┐  │
│  │ Multi-Head Self-Attention     │  │
│  │         ↓                     │  │
│  │ Add & Norm                    │  │
│  │         ↓                     │  │
│  │ Feed Forward Network          │  │
│  │         ↓                     │  │
│  │ Add & Norm                    │  │
│  └───────────────────────────────┘  │
│         (重复 N 次)                 │
└─────────────────────────────────────┘
    ↓
Encoder Output (Memory)
    ↓ ─────────────────────────┐
    │                          │
目标序列 (tgt)                  │
    ↓                          │
Token Embedding + Pos Encoding  │
    ↓                          │
┌─────────────────────────────────────┐
│     Decoder (N 层堆叠)              │
│  ┌───────────────────────────────┐  │
│  │ Masked Multi-Head Self-Attn   │  │ (防止看到未来)
│  │         ↓                     │  │
│  │ Add & Norm                    │  │
│  │         ↓                     │  │
│  │ Multi-Head Cross-Attention ←──┼──┘ (查询 Encoder)
│  │         ↓                     │  │
│  │ Add & Norm                    │  │
│  │         ↓                     │  │
│  │ Feed Forward Network          │  │
│  │         ↓                     │  │
│  │ Add & Norm                    │  │
│  └───────────────────────────────┘  │
│         (重复 N 次)                 │
└─────────────────────────────────────┘
    ↓
Linear (投影到词表)
    ↓
Softmax
    ↓
输出概率分布
```

---

### 参数量计算

以 BERT-Base 配置为例: $d_{\text{model}}=768$, $H=12$, $N=12$, $V=30000$

| 组件 | 参数量公式 | 具体数值 | 占比 |
|------|-----------|---------|------|
| **Token Embedding** | $V \times d$ | $30000 \times 768 = 23.04\text{M}$ | ~21% |
| **Position Encoding** | 0 | 0 | 0% |
| **MHA** (单层) | $4d^2$ | $4 \times 768^2 = 2.36\text{M}$ | - |
| **FFN** (单层) | $8d^2$ | $8 \times 768^2 = 4.72\text{M}$ | - |
| **LayerNorm** (单层×2) | $4d$ | $4 \times 768 = 3072$ | ~0% |
| **单层 Encoder** | $12d^2 + 4d$ | $\approx 7.08\text{M}$ | - |
| **N 层 Encoder** | $N(12d^2 + 4d)$ | $12 \times 7.08\text{M} = 84.96\text{M}$ | ~78% |
| **输出投影** | $d \times V$ | $768 \times 30000 = 23.04\text{M}$ | - |
| **总计** | $\approx 2Vd + 12Nd^2$ | $\approx 110\text{M}$ | 100% |

**说明**:
- 实际 BERT-Base 约 110M 参数 (与计算相符)
- GPT-2 (117M): $d=768$, $N=12$, $V=50257$
- GPT-3 (175B): $d=12288$, $N=96$, $V=50257$

---

### 典型配置对比

| 模型 | $d_{\text{model}}$ | $H$ | $N$ | $d_{\text{ff}}$ | 参数量 |
|------|-------------------|-----|-----|----------------|--------|
| **Transformer-Base** | 512 | 8 | 6 | 2048 | ~65M |
| **Transformer-Big** | 1024 | 16 | 6 | 4096 | ~213M |
| **BERT-Base** | 768 | 12 | 12 | 3072 | ~110M |
| **BERT-Large** | 1024 | 16 | 24 | 4096 | ~340M |
| **GPT-2** | 768 | 12 | 12 | 3072 | ~117M |
| **GPT-2 Large** | 1280 | 20 | 36 | 5120 | ~774M |
| **GPT-3** | 12288 | 96 | 96 | 49152 | ~175B |

---

### 计算复杂度分析

对于单层 Transformer, 序列长度 $T$, 模型维度 $d$:

| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| **Self-Attention** | $\mathcal{O}(T^2 \cdot d)$ | $\mathcal{O}(T^2)$ | 瓶颈: 注意力矩阵 |
| **FFN** | $\mathcal{O}(T \cdot d^2)$ | $\mathcal{O}(T \cdot d)$ | 当 $T>d$ 时较小 |
| **单层总计** | $\mathcal{O}(T^2 \cdot d + T \cdot d^2)$ | $\mathcal{O}(T^2 + T \cdot d)$ | - |
| **N 层总计** | $\mathcal{O}(N(T^2 \cdot d + T \cdot d^2))$ | $\mathcal{O}(NT^2)$ | 训练时需存储梯度 |

**关键影响因素**:
- **序列长度 $T$**: 二次增长 $\mathcal{O}(T^2)$ 🔥🔥🔥
- **模型维度 $d$**: 平方增长 $\mathcal{O}(d^2)$ 🔥
- **层数 $N$**: 线性增长 $\mathcal{O}(N)$ ✅

详见教程中的"面试高频问题详解"。

---

## 🔧 代码修正说明

原始 `Combined.py` 文件中存在的错误已全部修正:

1. ✅ `nn.Moduel` → `nn.Module` (拼写错误)
2. ✅ `super.__init__()` → `super().__init__()` (语法错误)
3. ✅ `self.W_k(x_kv)` → `self.W_v(x_kv)` (逻辑错误, V 投影错误使用了 K)
4. ✅ `torch.arange(0, d_model, 2)).float()` → `torch.arange(0, d_model, 2).float()` (括号错误)
5. ✅ `self.act = nn.ReLU()` → `self.act = nn.GELU()` (GELU 分支错误)
6. ✅ `d_modle` → `d_model` (拼写错误)
7. ✅ `q_pad is None` 判断逻辑优化
8. ✅ `__ini__` → `__init__` (拼写错误)
9. ✅ `super.__init__()` → `super().__init__()` (多处)
10. ✅ `self.tgt_vocab` → `self.tgt_embed` (变量名错误)
11. ✅ `self.encode_layers` → `self.encoder_layers` (一致性)
12. ✅ `self.decode_layers` → `self.decoder_layers` (一致性)
13. ✅ `forwad` → `forward` (拼写错误)
14. ✅ `torch.log` → `torch.long` (类型错误)
15. ✅ `tgt_vocab=None` → `tgt_pad=None` (参数名错误)
16. ✅ 缩进错误修正 (多个方法定义在类外)

---

## 🎯 使用建议

### 面试手撕建议

1. **优先级排序**:
   - 必须会: `ScaledDotProductAttention`, `MultiHeadAttention`
   - 重要: `EncoderLayer`, `DecoderLayer`
   - 次要: `PositionalEncoding`, `FeedForward`
   - 可简化: Mask 函数, 完整 Transformer 类

2. **简化技巧**:
   - 只实现 Encoder 或 Decoder (不必两者都写)
   - 忽略 Dropout
   - 假设没有 Padding (简化 Mask)
   - 使用伪代码描述复杂部分

3. **关键点强调**:
   - Attention 公式: $\frac{QK^T}{\sqrt{d_k}}$
   - 多头分割: `(B,T,d) → (B,H,T,d_k)`
   - Mask 机制: Padding + Causal
   - 残差连接 + LayerNorm

---

## 📖 参考资料

- 原论文: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- The Annotated Transformer: http://nlp.seas.harvard.edu/annotated-transformer/
- PyTorch 官方文档: https://pytorch.org/docs/stable/nn.html#transformer-layers

---

## 📝 TODO

- [ ] 添加训练脚本示例
- [ ] 添加 Beam Search 解码
- [ ] 添加模型可视化工具
- [ ] 添加单元测试
- [ ] 支持 FlashAttention
- [ ] 支持 Pre-LN 版本

---

**最后更新**: 2025年11月9日
