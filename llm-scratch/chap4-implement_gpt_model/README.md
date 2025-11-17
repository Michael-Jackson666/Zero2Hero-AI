# Chapter 4: 实现 GPT 模型

本章从零开始实现完整的 GPT (Generative Pre-trained Transformer) 模型。

---

## 📂 文件说明

```
chap4-implement_gpt_model/
├── gpt-model.py                  # 完整 GPT 模型实现
├── gpt.py                        # GPT 架构主文件
├── transformer-block.py          # Transformer 块实现
├── feed-forward.py               # 前馈网络 (FFN)
├── layer-normlization.py         # Layer Normalization
├── add-shortcut-connection.py    # 残差连接
├── generating-text.py            # 文本生成函数
├── DummyGPTModel.py             # 简化版 GPT (教学用)
└── saved-code-4.py              # 本章完整代码
```

---

## 🎯 学习目标

- ✅ 理解 GPT 架构组成
- ✅ 实现 Transformer Block
- ✅ 掌握残差连接和 LayerNorm
- ✅ 实现文本生成逻辑

---

## 🚀 快速开始

### 1. 构建 GPT 模型

```bash
python gpt-model.py
```

**模型配置**:
```python
GPT_CONFIG = {
    "vocab_size": 50257,      # 词汇表大小
    "context_length": 1024,   # 上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头数
    "n_layers": 12,          # Transformer 层数
    "drop_rate": 0.1,        # Dropout 比率
    "qkv_bias": False        # QKV 是否使用偏置
}
```

### 2. 文本生成

```bash
python generating-text.py
```

**生成方法**:
- Greedy Decoding (贪婪解码)
- Temperature Sampling (温度采样)
- Top-k Sampling (Top-k 采样)
- Top-p (Nucleus) Sampling

---

## 🏗️ GPT 架构

```
Input Text
    ↓
Token Embedding + Positional Embedding
    ↓
┌─────────────────────────────────┐
│  Transformer Block 1            │
│  ┌───────────────────────────┐  │
│  │ Multi-Head Self-Attention │  │
│  │         ↓                 │  │
│  │ Add & LayerNorm          │  │
│  │         ↓                 │  │
│  │ Feed Forward Network     │  │
│  │         ↓                 │  │
│  │ Add & LayerNorm          │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓
    ... (重复 N 层)
    ↓
LayerNorm
    ↓
Linear (投影到词汇表)
    ↓
Softmax
    ↓
Output Probabilities
```

---

## 📚 核心组件

### 1. Transformer Block

每个 Transformer Block 包含:
- Multi-Head Causal Self-Attention
- Feed-Forward Network
- 2 个 LayerNorm
- 2 个残差连接

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(...)
        self.ff = FeedForward(...)
        self.norm1 = LayerNorm(...)
        self.norm2 = LayerNorm(...)
    
    def forward(self, x):
        # Self-Attention + 残差
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = x + shortcut
        
        # Feed-Forward + 残差
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        
        return x
```

### 2. Feed-Forward Network

```python
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

通常 FFN 中间层维度是 `4 × emb_dim`。

### 3. Layer Normalization

```python
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

归一化每个样本的特征维度。

---

## 💡 代码示例

```python
from gpt_model import GPTModel
import torch

# 创建 GPT 模型
model = GPTModel(GPT_CONFIG)

# 输入
input_ids = torch.randint(0, 50257, (2, 10))  # (batch, seq_len)

# 前向传播
logits = model(input_ids)
print(f"输出形状: {logits.shape}")  # (2, 10, 50257)

# 文本生成
from generating_text import generate_text

prompt = "Once upon a time"
generated = generate_text(
    model=model,
    prompt=prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_k=40
)
print(generated)
```

---

## 🎯 文本生成策略

### Greedy Decoding
每次选择概率最高的 token。

### Temperature Sampling
调整概率分布的"锐度"：
```python
probs = torch.softmax(logits / temperature, dim=-1)
```
- `temperature < 1`: 更确定性
- `temperature > 1`: 更随机

### Top-k Sampling
只从概率最高的 k 个 token 中采样。

### Top-p (Nucleus) Sampling
从累积概率达到 p 的最小 token 集合中采样。

---

## 🔗 相关章节

- **上一章**: [Chapter 3 - 注意力机制](../chap3-attention_mechanisms/)
- **下一章**: [Chapter 5 - 预训练](../chap5-pretraining/)

---

**最后更新**: 2025年11月17日
