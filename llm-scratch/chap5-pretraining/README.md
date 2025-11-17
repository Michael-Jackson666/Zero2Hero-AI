# Chapter 5: 预训练

本章实现 GPT 模型的预训练流程，包括训练循环、损失计算、模型保存和文本生成。

---

## 📂 文件说明

```
chap5-pretraining/
├── train-llm.py                  # 主训练脚本
├── complete_training.py          # 完整训练流程
├── load-train.py                 # 加载数据和训练
├── evaluate-text.py              # 模型评估
├── generate_new.py               # 新版文本生成
├── generate_old.py               # 旧版文本生成
├── cross_entropy_explanation.py  # 交叉熵损失详解
├── temperature-scaling.py        # 温度缩放实验
├── top-k_sampling.py             # Top-k 采样实现
├── load-save-weight.py           # 模型权重保存/加载
├── check_model_size.py           # 检查模型参数量
├── seed_experiment.py            # 随机种子实验
├── previous_chapters.py          # 前几章代码导入
├── the-verdict.txt               # 训练数据示例
├── loss-plot.pdf                 # 训练损失曲线
├── temperature-plot.pdf          # 温度采样对比
└── load-gpt-model/               # 预训练模型加载
```

---

## 🎯 学习目标

- ✅ 实现完整的预训练循环
- ✅ 理解交叉熵损失
- ✅ 掌握模型评估方法
- ✅ 学习各种采样策略
- ✅ 模型检查点保存和恢复

---

## 🚀 快速开始

### 1. 完整训练流程

```bash
python complete_training.py
```

**训练参数**:
```python
TRAINING_CONFIG = {
    "batch_size": 4,
    "max_epochs": 10,
    "learning_rate": 0.0001,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "grad_clip": 1.0
}
```

### 2. 评估模型

```bash
python evaluate-text.py
```

**评估指标**:
- 训练损失 (Training Loss)
- 验证损失 (Validation Loss)
- 困惑度 (Perplexity)

### 3. 生成文本

```bash
python generate_new.py
```

**生成示例**:
```python
from generate_new import generate

prompt = "The future of AI is"
output = generate(
    model=model,
    prompt=prompt,
    max_tokens=100,
    temperature=0.7,
    top_k=50
)
print(output)
```

---

## 📚 核心概念

### 预训练任务

GPT 使用**自回归语言建模** (Autoregressive Language Modeling):

给定前 $t-1$ 个 token，预测第 $t$ 个 token：

$$P(x_t | x_1, x_2, ..., x_{t-1})$$

### 损失函数

使用**交叉熵损失** (Cross-Entropy Loss):

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_1, ..., x_{t-1})$$

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1)
)
```

### 困惑度 (Perplexity)

衡量模型预测质量的指标：

$$\text{PPL} = \exp(\mathcal{L})$$

- 越低越好
- 表示模型对下一个 token 的"困惑程度"

---

## 🎯 训练流程

```python
def train_epoch(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    
    for batch_idx, (input_ids, targets) in enumerate(train_loader):
        # 前向传播
        logits = model(input_ids)
        
        # 计算损失
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=1.0
        )
        
        # 参数更新
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

---

## 📊 训练监控

### 损失曲线

查看 `loss-plot.pdf`:
- 训练损失应稳定下降
- 验证损失不应持续上升（过拟合警告）

### 生成质量

定期生成样本文本检查：
```bash
# Epoch 1: "The cat sat sat sat..."  (重复)
# Epoch 5: "The cat sat on mat."    (语法正确)
# Epoch 10: "The cat sat on the comfortable mat." (流畅)
```

---

## 💡 采样策略对比

### Temperature Scaling

```python
# temperature = 0.1 (更确定)
"The cat sat on the mat."

# temperature = 1.0 (平衡)
"The cat wandered through the garden."

# temperature = 2.0 (更随机)
"The purple elephant danced joyfully."
```

查看 `temperature-plot.pdf` 了解概率分布变化。

### Top-k Sampling

```python
# top_k = 1 (贪婪)
"The most common response is yes."

# top_k = 10
"The best answer might be yes."

# top_k = 50
"Perhaps the answer could be yes."
```

---

## 🔧 模型保存和加载

### 保存模型

```python
# 保存完整模型
torch.save(model.state_dict(), 'gpt_model.pth')

# 保存训练状态
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')
```

### 加载模型

```python
# 加载权重
model.load_state_dict(torch.load('gpt_model.pth'))

# 恢复训练
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 🔍 实用工具

### 检查模型大小

```bash
python check_model_size.py
```

输出:
```
Total parameters: 124,439,808 (124.4M)
Trainable parameters: 124,439,808
Model size: 474.4 MB
```

### 随机种子实验

```bash
python seed_experiment.py
```

验证不同随机种子对训练的影响。

---

## 🔗 相关章节

- **上一章**: [Chapter 4 - 实现 GPT 模型](../chap4-implement_gpt_model/)
- **下一章**: [Chapter 6 - 分类任务微调](../chap6-fine-tuning-for-classification/)

---

**最后更新**: 2025年11月17日
