# Universal Translator

基于 Transformer 架构的机器学习项目集合，包含翻译模型、推理系统和AI算法学习资源。

---

## � 项目结构

```
Universal-Translator/
├── Eng2Fren/              # 英法翻译模型
│   ├── transformer.py              # Transformer 模型实现
│   ├── transformer-d2l.py          # 训练脚本
│   ├── simple_translator.py        # 交互式翻译器
│   └── batch_translate.py          # 批量翻译工具
│
├── ai_test/               # AI 算法学习与面试
│   ├── Transformer/                # Transformer 教程和代码实现
│   │   ├── tutorial.ipynb          # 交互式教程 (含面试题)
│   │   └── Code/                   # 模块化代码实现
│   ├── ACM/                        # ACM 算法练习
│   └── 2025/                       # 大厂机试题目集 (2025年)
│
├── llm-scratch/           # LLM 从零实现
│   ├── chap2-work_with_text_data/  # 文本数据处理
│   ├── chap3-attention_mechanisms/ # 注意力机制
│   ├── chap4-implement_gpt_model/  # GPT 模型实现
│   ├── chap5-pretraining/          # 预训练
│   ├── chap6-fine-tuning-for-classification/  # 分类任务微调
│   └── chap7-fine-tuning-to-follow-instruction/  # 指令微调
│
└── reasoning-scratch/     # 推理系统从零实现
    └── README.md                   # 逻辑推理、搜索算法等
```

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/Michael-Jackson666/Universal-Translator.git
cd Universal-Translator

# 安装依赖
pip install torch d2l numpy matplotlib
```

### 使用示例

#### 1. 英法翻译模型

```bash
# 训练模型
python Eng2Fren/transformer-d2l.py

# 交互式翻译
python Eng2Fren/simple_translator.py

# 批量翻译
python Eng2Fren/batch_translate.py --input source.txt --output translated.txt
```

#### 2. Transformer 学习

```bash
# 查看交互式教程
jupyter notebook ai_test/Transformer/tutorial.ipynb

# 运行测试
cd ai_test/Transformer/Code
python test_attention.py
```

#### 3. LLM 从零实现

```bash
# 学习各章节代码
cd llm-scratch/chap4-implement_gpt_model
python gpt-model.py
```

---

## � 主要模块说明

### Eng2Fren - 英法翻译
基于 Transformer 的机器翻译模型，支持训练、推理和批量翻译。

### ai_test - AI 学习资源
- **Transformer/**: 完整的 Transformer 教程和模块化实现
- **ACM/**: 算法练习题
- **2025/**: 大厂机试真题集

### llm-scratch - LLM 从零实现
逐章实现大语言模型，从文本处理到指令微调的完整流程。

### reasoning-scratch - 推理系统
逻辑推理、搜索算法、知识图谱等推理系统的实现。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

**最后更新**: 2025年11月15日