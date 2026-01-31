# DeepSeek-MoE: 极致专家特化

本目录包含 DeepSeek 论文 [*DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*](https://arxiv.org/abs/2401.06066) 的学习笔记和代码实现。

## 📖 核心思想

DeepSeekMoE 通过两大策略解决传统 MoE 的专家特化不足问题：

| 问题 | 传统 MoE | DeepSeekMoE 解决方案 |
|------|----------|---------------------|
| **知识混合** | 单个专家承担多种知识 | 细粒度专家分割 (1/m 大小) |
| **知识冗余** | 多个专家重复学习通用知识 | 共享专家隔离 (始终激活) |

**一句话总结**：将专家切小、切多，并隔离出"通用知识专家"，让路由专家专注于特定领域。

## 📁 目录结构

```
DeepSeek-MoE/
├── README.md                    # 本文件
├── DeepSeek-MoE.md              # 📝 DeepSeekMoE 详细学习笔记
├── MoE简介.md                   # 📝 MoE 基础知识介绍
├── DeepSeekMoE.png              # 🖼️ 架构图
├── MoE Layer.png                # 🖼️ MoE 层示意图
└── Code/                        # 💻 PyTorch 代码实现
    ├── README.md                # 代码文档
    ├── experts.py               # 专家网络 (SwiGLU FFN)
    ├── router.py                # Top-K 路由与负载均衡
    ├── moe_layer.py             # MoE 层 (共享+路由专家)
    └── deepseek_moe.py          # 完整模型实现
```

## 🔧 DeepSeekMoE 架构

```
输入 Token → Self-Attention → MoE Layer → 输出
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              Shared Experts              Routed Experts
              (始终激活 K_s 个)            (Top-K 选择 mK-K_s 个)
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                              求和 + 残差
```

## 📐 关键公式

**DeepSeekMoE 输出**：
$$\mathbf{h}_t = \underbrace{\sum_{i=1}^{K_s} \text{FFN}_i(\mathbf{u}_t)}_{\text{共享专家}} + \underbrace{\sum_{i=K_s+1}^{mN} g_{i,t} \text{FFN}_i(\mathbf{u}_t)}_{\text{路由专家}} + \mathbf{u}_t$$

**门控机制**：
$$s_{i,t} = \text{Softmax}_i(\mathbf{u}_t^T \mathbf{e}_i), \quad g_{i,t} = \begin{cases} s_{i,t}, & \text{if in Top-K} \\ 0, & \text{otherwise} \end{cases}$$

## 📊 模型配置

| 模型 | 总参数 | 激活参数 | 共享专家 | 路由专家 | Top-K |
|------|--------|----------|----------|----------|-------|
| DeepSeekMoE-2B | 2.0B | 0.3B | 1 | 63 | 7 |
| DeepSeekMoE-16B | 16.4B | 2.8B | 2 | 64 | 6 |
| DeepSeekMoE-145B | 144.6B | 22.2B | 4 | 128 | 12 |

## 🚀 快速开始

```python
from Code.deepseek_moe import DeepSeekMoEModel, DeepSeekMoEConfig

config = DeepSeekMoEConfig(
    hidden_size=2048,
    num_shared_experts=2,
    num_routed_experts=64,
    num_experts_per_token=6,
)

model = DeepSeekMoEModel(config)
outputs = model(input_ids)
```

## 💡 关键发现

1. **组合爆炸**：64 个小专家 Top-8 选择有 44 亿种组合（vs 16 专家 Top-2 仅 120 种）
2. **极致特化**：移除任一专家性能显著下降（说明无冗余）
3. **U 型最优**：纯 Dense 和纯 MoE 都不是最优，混合策略效果最佳

## 📚 学习路线

1. **MoE 基础**：阅读 [MoE简介.md](MoE简介.md)
2. **DeepSeekMoE 原理**：阅读 [DeepSeek-MoE.md](DeepSeek-MoE.md)
3. **代码实践**：运行 [Code/](Code/) 中的模块
4. **论文原文**：[arXiv:2401.06066](https://arxiv.org/abs/2401.06066)

## 📄 License

MIT License
