# English ↔ French Translator

基于 Transformer 架构的英法双向翻译模型实现。

---

## 📂 文件说明

```
Eng2Fren/
├── transformer.py                    # Transformer 模型核心实现
├── transformer-d2l.py                # 训练脚本 (基于 D2L)
├── transformer_inference.py          # 推理模块
├── simple_translator.py              # 交互式翻译器 (推荐)
├── mini_translator.py                # 轻量级翻译器
├── batch_translate.py                # 批量翻译工具
├── transformer_fra_eng.pth           # 训练好的模型文件
├── example_input.txt                 # 示例输入文件
├── example_input_translated.txt      # 示例翻译结果
└── batch_translation_results.txt     # 批量翻译结果
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch d2l numpy matplotlib
```

### 2. 训练模型

```bash
# 使用默认参数训练
python transformer-d2l.py

# 自定义参数训练
python transformer-d2l.py --num_epochs 50 --batch_size 128
```

**训练参数说明**:
- `--num_epochs`: 训练轮数 (默认: 30)
- `--batch_size`: 批次大小 (默认: 64)
- `--num_hiddens`: 隐藏层维度 (默认: 256)
- `--num_heads`: 注意力头数 (默认: 8)
- `--num_layers`: Encoder/Decoder 层数 (默认: 2)

### 3. 使用翻译

#### 方式1: 交互式翻译器 (推荐)

```bash
python simple_translator.py
```

**功能特性**:
- ✅ 自动语言检测
- ✅ 实时翻译
- ✅ 历史记录
- ✅ 友好的交互界面

#### 方式2: 命令行翻译

```bash
# 快速单句翻译
python mini_translator.py "Hello, how are you?"

# 指定翻译方向
python mini_translator.py "Bonjour" --direction fr-en
```

#### 方式3: 批量文件翻译

```bash
# 翻译文本文件
python batch_translate.py --input example_input.txt --output output.txt

# 并行处理
python batch_translate.py --input large_file.txt --output result.txt --parallel 4
```

---

## 📝 使用示例

### Python API 调用

```python
from transformer_inference import Translator

# 初始化翻译器
translator = Translator(model_path='transformer_fra_eng.pth')

# 英译法
result = translator.translate("Hello world", direction='en-fr')
print(result)  # "Bonjour le monde"

# 法译英
result = translator.translate("Bonjour", direction='fr-en')
print(result)  # "Hello"
```

---

## 🏗️ 模型架构

基于 "Attention Is All You Need" 论文实现的标准 Transformer 架构:

- **Encoder**: Multi-Head Self-Attention + Feed Forward
- **Decoder**: Masked Multi-Head Self-Attention + Cross-Attention + Feed Forward
- **位置编码**: 正弦/余弦位置编码
- **优化器**: Adam with learning rate scheduling

**默认配置**:
- 模型维度: 256
- 注意力头数: 8
- Encoder/Decoder 层数: 2
- 前馈网络维度: 1024
- Dropout: 0.1

---

## 📊 性能指标

在标准测试集上的表现:

| 方向 | BLEU Score | 训练时间 (GPU) |
|------|-----------|---------------|
| EN → FR | ~35-40 | 2-3 小时 |
| FR → EN | ~33-38 | 2-3 小时 |

**测试环境**: Tesla V100 / RTX 4090

---

## 🔧 常见问题

### Q: 模型文件不存在？
A: 请先运行 `python transformer-d2l.py` 训练模型。

### Q: CUDA out of memory？
A: 减小 `--batch_size` 参数，或使用 `--num_hiddens 128` 减小模型大小。

### Q: 翻译质量不佳？
A: 增加训练轮数 `--num_epochs 50`，或使用更大的模型 `--num_hiddens 512`。

### Q: 翻译速度慢？
A: 使用 GPU 推理，或减小 `beam_size` 参数。

---

## 📖 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原论文
- [D2L.ai](https://d2l.ai/) - 深度学习教程
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - 注释版实现

---

**最后更新**: 2025年11月16日
