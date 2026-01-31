# DeepSeek 论文精读与复现

本目录包含 DeepSeek 系列论文的学习笔记和代码实现。

## 📁 目录结构

```
DeepSeek/
├── DeepSeek-MoE/              # DeepSeekMoE 混合专家模型
│   ├── README.md              # 模块概述
│   ├── MoE简介.md             # 📝 MoE 基础知识
│   ├── DeepSeek-MoE.md        # 📝 DeepSeekMoE 详细笔记
│   ├── DeepSeekMoE.png        # 🖼️ 架构图
│   ├── MoE Layer.png          # 🖼️ MoE 层示意图
│   └── Code/                  # 💻 PyTorch 实现
│       ├── experts.py         # 专家网络 (SwiGLU FFN)
│       ├── router.py          # Top-K 路由与负载均衡
│       ├── moe_layer.py       # MoE 层 (共享+路由专家)
│       └── deepseek_moe.py    # 完整模型实现
│
├── Engram/                    # Engram 条件记忆架构
│   ├── README.md              # 模块概述
│   ├── Engram.md              # 📝 Engram 详细笔记
│   ├── Engram.png             # 🖼️ 架构图
│   ├── Sparsity allocation and Engram scaling.png
│   └── Code/                  # 💻 PyTorch 实现
│       ├── tokenizer_compression.py  # Token 压缩与 N-gram 提取
│       ├── multi_head_hashing.py     # 多头哈希与 Embedding 查找
│       ├── context_aware_gating.py   # 上下文感知门控
│       ├── fusion.py                 # 深度卷积融合层
│       └── engram.py                 # 完整 Engram 模块
│
├── mHC/                       # mHC 多头因果架构
│   ├── mHC.md                 # 📝 mHC 学习笔记
│   ├── mHC.png                # 🖼️ 架构图
│   └── Communication-Computation Overlapping for mHC.png
│
└── DeepThink.md               # 📝 深度思考笔记
```

## 📚 已完成内容

### 1. DeepSeekMoE - 极致专家特化

**论文**: [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)

| 内容 | 状态 |
|------|------|
| MoE 基础知识笔记 | ✅ 完成 |
| DeepSeekMoE 原理笔记 | ✅ 完成 |
| 代码实现 (前向传播) | ✅ 完成 |

**核心创新**:
- 细粒度专家分割 (1/m 大小)
- 共享专家隔离 (始终激活)
- Top-K 路由与负载均衡

### 2. Engram - 条件记忆

**论文**: [Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372)

| 内容 | 状态 |
|------|------|
| Engram 原理笔记 | ✅ 完成 |
| 代码实现 (前向传播) | ✅ 完成 |

**核心创新**:
- 条件记忆 (与 MoE 条件计算互补)
- N-gram 多头哈希检索
- 上下文感知门控
- 零开销预取机制

### 3. mHC - 多头因果架构

**论文**: DeepSeek-V3 系列

| 内容 | 状态 |
|------|------|
| mHC 学习笔记 | ✅ 完成 |
| 代码实现 | 📋 计划中 |

## 🚀 快速开始

```bash
cd DeepSeek

# 运行 DeepSeekMoE 示例
python DeepSeek-MoE/Code/deepseek_moe.py

# 运行 Engram 示例
python Engram/Code/engram.py
```

## 📄 License

MIT License
