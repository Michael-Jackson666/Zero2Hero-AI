# Chapter 3: 注意力机制

本章深入探讨注意力机制 (Attention Mechanisms)，这是 Transformer 和现代 LLM 的核心组件。

---

## 📂 文件说明

```
chap3-attention_mechanisms/
├── self-attention.py          # 自注意力机制实现
├── multihead-attention.py     # 多头注意力实现
├── masked-attention.py        # 因果掩码注意力 (Causal Attention)
├── dropout.py                 # Dropout 正则化
└── saved-code-3.py            # 本章完整代码
```

---

## 🎯 学习目标

- ✅ 理解自注意力 (Self-Attention) 原理
- ✅ 实现多头注意力 (Multi-Head Attention)
- ✅ 掌握因果掩码 (Causal Masking)
- ✅ 应用 Dropout 防止过拟合

---

## 🚀 快速开始

### 1. 自注意力

```bash
python self-attention.py
```

**核心公式**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 2. 多头注意力

```bash
python multihead-attention.py
```

**特性**:
- 并行多个注意力头
- 捕获不同位置关系
- 增强模型表达能力

### 3. 因果掩码注意力

```bash
python masked-attention.py
```

**用途**:
- 防止看到未来信息
- GPT 解码器的关键组件
- 自回归生成

---

## 📚 核心概念

### Self-Attention (自注意力)

将输入序列的每个位置与所有位置进行交互，计算加权和。

**步骤**:
1. 计算 Q (Query), K (Key), V (Value)
2. 计算注意力分数: $QK^T / \sqrt{d_k}$
3. Softmax 归一化
4. 加权求和: Attention × V

### Multi-Head Attention (多头注意力)

并行运行多个注意力头，捕获不同的特征。

$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, ..., head_h)W^O$$

其中:
$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Causal Masking (因果掩码)

在解码时防止位置 $i$ 看到位置 $j > i$ 的信息。

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))
```

---

## 💡 代码示例

```python
import torch
from multihead_attention import MultiHeadAttention

# 初始化多头注意力
mha = MultiHeadAttention(
    d_model=512,    # 模型维度
    num_heads=8,    # 注意力头数
    dropout=0.1
)

# 输入
x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)

# 前向传播
output, attention_weights = mha(x, x, x)

print(f"输出形状: {output.shape}")           # (2, 10, 512)
print(f"注意力权重: {attention_weights.shape}") # (2, 8, 10, 10)
```

---

## 🔍 注意力可视化

注意力权重矩阵显示每个位置关注其他位置的程度：

```
        Token1  Token2  Token3  Token4
Token1  [0.4    0.3     0.2     0.1  ]
Token2  [0.2    0.5     0.2     0.1  ]
Token3  [0.1    0.2     0.6     0.1  ]
Token4  [0.1    0.1     0.2     0.6  ]
```

---

## 🔗 相关章节

- **上一章**: [Chapter 2 - 文本数据处理](../chap2-work_with_text_data/)
- **下一章**: [Chapter 4 - 实现 GPT 模型](../chap4-implement_gpt_model/)

---

**最后更新**: 2025年11月17日
