# Transformer 学习资源

本目录包含 Transformer 模型的完整学习资料，包括交互式教程和模块化代码实现。

---

## 📂 目录结构

```
Transformer/
├── tutorial.ipynb          # 交互式教程 (包含面试高频问题详解)
└── Code/                   # 模块化代码实现
    ├── README.md           # 详细代码文档 (763行)
    ├── attention.py        # 注意力机制
    ├── embedding.py        # 位置编码
    ├── feedforward.py      # 前馈网络
    ├── mask.py            # 掩码工具
    ├── layers.py          # Encoder/Decoder层
    ├── transformer.py     # 完整模型
    └── test_attention.py  # 测试套件
```

---

## 🎯 快速开始

### 1. 交互式学习 (推荐新手)

打开 `tutorial.ipynb` Jupyter Notebook：
- 📖 Transformer 基础概念
- 🔍 面试高频问题详解 (复杂度、Embedding、序列长度等)
- 💡 带公式和可视化的详细解释

### 2. 代码实现学习

进入 `Code/` 目录，查看模块化实现：

```python
# 使用完整模型
from Code.transformer import Transformer

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)
```

详见 `Code/README.md` 获取完整使用指南。

---

## 📚 学习路径

1. **入门** → 阅读 `tutorial.ipynb` 理解概念
2. **深入** → 学习 `Code/attention.py` 和 `Code/embedding.py`
3. **实践** → 运行 `Code/test_attention.py` 测试
4. **面试** → 复习 `tutorial.ipynb` 中的面试问题

---

**最后更新**: 2025年11月11日
