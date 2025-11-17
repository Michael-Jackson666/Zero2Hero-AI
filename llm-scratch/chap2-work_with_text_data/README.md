# Chapter 2: 处理文本数据

本章介绍如何处理和准备文本数据用于大语言模型训练。

---

## 📂 文件说明

```
chap2-work_with_text_data/
├── data-preprocessing.py       # 文本数据预处理
├── BPE.py                     # Byte Pair Encoding (BPE) 分词算法
├── embeddings.py              # 词嵌入实现
├── the-verdict.txt            # 示例文本数据
└── saved-code-2.py            # 本章完整代码
```

---

## 🎯 学习目标

- ✅ 理解文本分词 (Tokenization)
- ✅ 掌握 BPE 算法原理和实现
- ✅ 学习词嵌入 (Word Embeddings)
- ✅ 数据预处理流程

---

## 🚀 快速开始

### 1. 文本预处理

```bash
python data-preprocessing.py
```

**功能**:
- 文本清洗和规范化
- 构建词汇表
- 数据集划分

### 2. BPE 分词

```bash
python BPE.py
```

**功能**:
- 实现 Byte Pair Encoding 算法
- 子词切分 (Subword Tokenization)
- 处理未登录词 (OOV)

### 3. 词嵌入

```bash
python embeddings.py
```

**功能**:
- 创建词嵌入矩阵
- Token ID 到向量的映射
- 位置编码

---

## 📚 核心概念

### Tokenization (分词)
将文本转换为模型可处理的 token 序列。

**常见方法**:
- Word-level: 按单词切分
- Character-level: 按字符切分
- **Subword-level**: BPE, WordPiece (推荐)

### BPE 算法
Byte Pair Encoding 通过迭代合并高频字符对来构建词汇表。

**优势**:
- 平衡词汇表大小和表达能力
- 有效处理未登录词
- 适用于多语言

### 词嵌入
将离散的 token 映射到连续的向量空间。

$$\text{embedding}: \text{token} \rightarrow \mathbb{R}^d$$

其中 $d$ 是嵌入维度（通常 256-1024）。

---

## 💡 代码示例

```python
# 简单的 BPE 示例
from BPE import BytePairEncoding

# 初始化 BPE
bpe = BytePairEncoding(vocab_size=1000)

# 训练
corpus = ["Hello world", "Hello there"]
bpe.train(corpus)

# 编码
tokens = bpe.encode("Hello world")
print(tokens)  # [72, 101, 108, ...]

# 解码
text = bpe.decode(tokens)
print(text)  # "Hello world"
```

---

## 🔗 相关章节

- **下一章**: [Chapter 3 - 注意力机制](../chap3-attention_mechanisms/)
- **上一章**: Chapter 1 - 介绍

---

**最后更新**: 2025年11月17日
