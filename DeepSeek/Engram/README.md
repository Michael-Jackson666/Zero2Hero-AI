# Engram: 条件记忆 —— LLM 的新稀疏方向

本目录包含 DeepSeek 论文 [*Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*](https://arxiv.org/abs/2601.07372) 的学习笔记和代码实现。

## 📖 核心思想

Engram 提出了**条件记忆 (Conditional Memory)** 的概念，作为 MoE（条件计算）的互补：

| 稀疏轴 | 功能定位 | 寻址依赖 | 复杂度 |
|--------|----------|----------|--------|
| **MoE** | 组合推理、逻辑泛化 | 动态依赖（Runtime State） | O(N) |
| **Engram** | 知识检索、事实查表 | 静态依赖（Input Token） | O(1) |

**一句话总结**：Engram 把"死记硬背"的参数从神经网络中剥离，用 O(1) 哈希查表实现，并可通过预取完全掩盖通信延迟。

## 📁 目录结构

```
Engram/
├── README.md                              # 本文件
├── Engram.md                              # 📝 详细学习笔记（公式推导、架构解析）
├── Engram.png                             # 🖼️ 架构图
├── Sparsity allocation and Engram scaling.png  # 📊 稀疏分配实验图
└── Code/                                  # 💻 PyTorch 代码实现
    ├── README.md                          # 代码文档
    ├── tokenizer_compression.py           # Token 压缩与 N-gram 提取
    ├── multi_head_hashing.py              # 多头哈希与 Embedding 查找
    ├── context_aware_gating.py            # 上下文感知门控
    ├── fusion.py                          # 深度卷积融合层
    └── engram.py                          # 完整 Engram 模块
```

## 🔧 Engram 架构五步流程

```
输入 Token → [1. 词表压缩] → [2. N-gram 提取] → [3. 多头哈希]
                                                      ↓
                                              [4. Embedding 查表]
                                                      ↓
隐藏状态 h_t → [5. 上下文门控] ← e_t (检索记忆)
                    ↓
              [6. 融合 + 残差] → 更新后的隐藏状态
```

## 📐 关键公式

**多头哈希检索**：
$$\mathbf{e}_t = \mathop{\Big\Vert}_{n=2}^N \mathop{\Big\Vert}_{k=1}^K \mathbf{E}_{n,k}[\varphi_{n,k}(g_{t,n})]$$

**上下文感知门控**：
$$\alpha_t = \sigma \left( \frac{\text{RMSNorm}(\mathbf{h}_t)^\top \text{RMSNorm}(\mathbf{W}_K \mathbf{e}_t)}{\sqrt{d}} \right)$$

**融合输出**：
$$\mathbf{Y} = \text{SiLU}(\text{Conv1D}(\text{RMSNorm}(\alpha_t \cdot \mathbf{W}_V \mathbf{e}_t))) + \alpha_t \cdot \mathbf{W}_V \mathbf{e}_t$$

## 🚀 快速开始

```python
from Code.engram import EngramModule, EngramConfig

config = EngramConfig(
    vocab_size=50_000,
    min_n=2, max_n=4,
    num_hash_heads=4,
    embedding_table_size=1_000_000,
    embedding_dim=64,
    hidden_dim=2048,
)

engram = EngramModule(config)

# 前向传播
updated_states, gating = engram(
    hidden_states=hidden_states,
    input_ids=input_ids,
    return_gating=True
)
```

## 💡 核心创新：零开销预取

由于 Engram 的索引只依赖输入 Token（而非运行时隐藏状态），可以实现完美的计算-存储流水线：

- **GPU**：计算 Layer 0-1
- **CPU**：同时预取 Layer 2 的 Engram Embedding
- **结果**：通信延迟被计算完全掩盖

## 📚 学习路线

1. **理论理解**：阅读 [Engram.md](Engram.md) 详细笔记
2. **代码实践**：运行 [Code/](Code/) 中的各模块
3. **论文原文**：[arXiv:2601.07372](https://arxiv.org/abs/2601.07372)

## 📄 License

MIT License
