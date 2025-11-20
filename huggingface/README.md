# Hugging Face 大模型部署学习笔记

本目录包含使用 Hugging Face Transformers 库进行大模型部署和应用的学习代码和笔记。

---

## 📚 学习目标

- 🎯 掌握 Hugging Face Transformers 库的使用
- 🚀 学习如何部署和调用预训练大模型
- 💡 理解各种 NLP 任务的实现方法
- 🔧 实践模型推理和优化技巧

---

## 📂 文件说明

```
huggingface/
├── README.md                    # 本文件
├── api_test01.py               # 基础示例：文本生成、问答、翻译（推荐在终端运行）
├── huggingface_examples.py     # 完整示例：8种常用NLP任务（推荐在终端运行）
└── demo.ipynb                  # Gradio交互式演示（Jupyter Notebook）
```

### 文件用途

- **api_test01.py**: 演示3个基础NLP任务（文本生成、问答、翻译）
- **huggingface_examples.py**: 包含8个常用NLP任务的完整示例
- **demo.ipynb**: Gradio可视化界面演示，包含简化的情感分类示例

---

## 🚀 快速开始

### 环境准备

```bash
# 创建conda环境
conda create -n huggingface python=3.10 -y

# 激活环境
conda activate huggingface

# 安装依赖
pip install transformers datasets accelerate tokenizers huggingface_hub torch
```

### 运行示例

```bash
# 方式1: 运行Python脚本（推荐）
python api_test01.py                 # 基础示例
python huggingface_examples.py       # 完整示例

# 方式2: 运行Jupyter Notebook
jupyter notebook demo.ipynb          # Gradio交互式演示
```

**⚠️ 重要提示**：
- **Python脚本**：在终端运行，所有transformers功能正常
- **Jupyter Notebook**：由于PyTorch依赖问题，使用简化版演示
- 建议优先使用Python脚本学习Hugging Face模型

---

## 💡 核心概念

### 1. Pipeline API

Hugging Face 提供的高级接口，简化了模型使用流程：

```python
from transformers import pipeline

# 创建管道
generator = pipeline('text-generation', model='gpt2')

# 使用管道
result = generator("Hello world")
```

**优势**:
- 自动下载模型和分词器
- 处理输入预处理和输出后处理
- 支持批处理和流式输出

### 2. 常用任务类型

| 任务 | Pipeline名称 | 说明 |
|------|-------------|------|
| 文本生成 | `text-generation` | GPT系列模型 |
| 文本分类 | `text-classification` | 情感分析、主题分类 |
| 问答 | `question-answering` | 基于上下文的问答 |
| 翻译 | `translation_XX_to_YY` | 多语言翻译 |
| 摘要 | `summarization` | 文本摘要生成 |
| NER | `ner` | 命名实体识别 |
| 零样本分类 | `zero-shot-classification` | 无需训练的分类 |
| 填充 | `fill-mask` | BERT风格的掩码填充 |

### 3. 模型加载方式

```python
# 方式1: 使用 pipeline（推荐）
generator = pipeline('text-generation', model='gpt2')

# 方式2: 手动加载模型和分词器
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

# 使用模型
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs)
text = tokenizer.decode(outputs[0])
```

---

## 🎯 实践案例

### 案例1: 文本生成（api_test01.py）

```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
result = generator(
    "Explain the theory of relativity in simple terms.",
    max_length=100,
    num_return_sequences=1
)
print(result[0]['generated_text'])
```

**关键参数**:
- `max_length`: 生成文本的最大长度
- `num_return_sequences`: 返回结果数量
- `temperature`: 控制随机性（0.7-1.0）
- `top_k`: Top-K采样
- `top_p`: Nucleus采样

### 案例2: 问答系统

```python
qa = pipeline('question-answering')
result = qa(
    question="What is AI?",
    context="AI is artificial intelligence..."
)
print(result['answer'])
```

### 案例3: 多语言翻译

```python
translator = pipeline('translation_en_to_fr', model='t5-small')
result = translator("Hello, how are you?")
print(result[0]['translation_text'])
```

---

## 🔧 性能优化技巧

### 1. 使用 GPU/MPS 加速

```python
# 自动使用可用的加速设备
generator = pipeline('text-generation', model='gpt2', device=0)

# macOS 使用 MPS
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
generator = pipeline('text-generation', model='gpt2', device=device)
```

### 2. 批处理

```python
# 批量处理多个输入
texts = ["Text 1", "Text 2", "Text 3"]
results = generator(texts, batch_size=8)
```

### 3. 模型量化

```python
# 使用量化模型减少内存占用
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'gpt2',
    load_in_8bit=True,  # 8位量化
    device_map='auto'
)
```

### 4. 缓存管理

```python
# 模型默认缓存位置: ~/.cache/huggingface/

# 自定义缓存目录
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'

# 清理缓存
# rm -rf ~/.cache/huggingface/hub/*
```

---

## 📊 常用模型推荐

### 文本生成
- **GPT-2**: `gpt2`, `gpt2-medium`, `gpt2-large`
- **GPT-Neo**: `EleutherAI/gpt-neo-1.3B`, `EleutherAI/gpt-neo-2.7B`
- **BLOOM**: `bigscience/bloom-560m`, `bigscience/bloom-1b7`

### 问答系统
- **BERT**: `bert-large-uncased-whole-word-masking-finetuned-squad`
- **RoBERTa**: `deepset/roberta-base-squad2`
- **ELECTRA**: `google/electra-base-discriminator`

### 翻译
- **T5**: `t5-small`, `t5-base`, `t5-large`
- **mBART**: `facebook/mbart-large-50-many-to-many-mmt`
- **MarianMT**: `Helsinki-NLP/opus-mt-en-zh`

### 中文模型
- **ChatGLM**: `THUDM/chatglm-6b`, `THUDM/chatglm2-6b`
- **Qwen**: `Qwen/Qwen-7B-Chat`
- **Baichuan**: `baichuan-inc/Baichuan2-7B-Chat`

---

## 🛠️ 常见问题

### Q1: 模型下载失败？
```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地
git lfs install
git clone https://huggingface.co/gpt2
```

### Q2: 内存不足？
- 使用更小的模型（如 `gpt2` 而非 `gpt2-large`）
- 启用模型量化（8bit/4bit）
- 减小 `batch_size`
- 使用梯度检查点

### Q3: 速度太慢？
- 使用 GPU/MPS 加速
- 启用批处理
- 使用更小的 `max_length`
- 考虑使用 ONNX Runtime

### Q4: 如何使用私有模型？
```bash
# 登录 Hugging Face
huggingface-cli login

# 输入 Access Token
```

### Q5: Jupyter Notebook中transformers无法使用？
**问题描述**: notebook kernel中PyTorch损坏，导致 `AttributeError: module 'torch' has no attribute 'Tensor'`

**解决方案**:
```bash
# 方案1: 在终端运行Python脚本（推荐）
python api_test01.py

# 方案2: 使用demo.ipynb中的简化版Gradio演示
jupyter notebook demo.ipynb

# 方案3: 重建conda环境
conda env remove -n huggingface
conda create -n huggingface python=3.10 -y
conda activate huggingface
pip install transformers datasets accelerate tokenizers torch gradio
```

**注意**: 终端运行的Python脚本可以正常使用所有功能

---

## 📖 学习资源

### 官方文档
- [Transformers 文档](https://huggingface.co/docs/transformers/)
- [Hugging Face Hub](https://huggingface.co/models)
- [Datasets 文档](https://huggingface.co/docs/datasets/)

### 教程
- [Hugging Face Course](https://huggingface.co/course/)
- [Fine-tuning Tutorial](https://huggingface.co/docs/transformers/training)
- [Pipeline API](https://huggingface.co/docs/transformers/main_classes/pipelines)

### 社区
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [GitHub Issues](https://github.com/huggingface/transformers/issues)

---

## 📝 学习进度

- [x] 环境搭建和基础配置
- [x] Pipeline API 基础使用
- [x] 文本生成任务实践
- [x] 问答系统实现
- [x] 翻译功能测试
- [x] Gradio可视化界面创建
- [x] 基础NLP任务演示（8种任务）
- [ ] 模型微调（Fine-tuning）
- [ ] 自定义数据集训练
- [ ] 模型量化和优化
- [ ] 生产环境部署
- [ ] API服务搭建（FastAPI）
- [ ] 解决Jupyter Notebook中的PyTorch依赖问题

---

## 🎯 下一步计划

1. **模型微调**: 在自定义数据集上微调预训练模型
2. **性能优化**: 研究量化、剪枝等优化技术
3. **部署实践**: 使用 FastAPI 搭建模型推理服务
4. **多模态**: 探索图文、语音等多模态模型
5. **RAG系统**: 构建检索增强生成系统

---

## 💻 开发环境

- **系统**: macOS (Apple Silicon)
- **Python**: 3.10
- **加速**: MPS (Metal Performance Shaders)
- **主要依赖**:
  - transformers: 4.57.1
  - torch: 2.9.1
  - datasets: 2.12.0
  - accelerate: 1.11.0
  - gradio: 5.49.1
  - peft: 0.18.0
  - optimum: 2.0.0
  - sentencepiece: 0.2.1

### 已知问题

⚠️ **Jupyter Notebook PyTorch问题**:
- **症状**: `AttributeError: module 'torch' has no attribute 'Tensor'`
- **影响**: Notebook中无法使用transformers库
- **临时方案**: 使用终端运行Python脚本
- **状态**: 待解决

✅ **终端运行正常**:
- 所有Python脚本在终端中运行完全正常
- transformers、torch等库功能完整

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 改进学习笔记！

---

**最后更新**: 2025年11月20日

**学习状态**: 🚀 进行中

**推荐使用方式**: 
- 📝 学习：运行Python脚本（`api_test01.py`, `huggingface_examples.py`）
- 🎨 演示：使用Gradio界面（`demo.ipynb`，简化版）
- 🔧 生产：等待PyTorch问题解决后使用完整功能
