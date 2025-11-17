# Chapter 6: 分类任务微调

本章展示如何将预训练的 GPT 模型微调用于文本分类任务（以垃圾邮件分类为例）。

---

## 📂 文件说明

```
chap6-fine-tuning-for-classification/
├── fine-tuning.py                      # 主微调脚本
├── fine-tuning-all.py                  # 完整微调流程
├── gpt_class_finetune.py               # GPT 分类器微调
├── add_classification_head.py          # 添加分类头
├── create_data_loaders.py              # 数据加载器
├── prepare_dataset.py                  # 数据集准备
├── load_weights.py                     # 加载预训练权重
├── gpt_download.py                     # 下载 GPT-2 模型
├── spam_classifier.py                  # 垃圾邮件分类器
├── spam_classifier_simple.py           # 简化版分类器
├── spam_classifier_inference.py        # 推理脚本
├── simple_spam_classifier.py           # 基础分类器实现
├── previous_chapters.py                # 前几章代码
├── spam_classifier_full_finetune.pth   # 微调后模型
├── train.csv                           # 训练集
├── validation.csv                      # 验证集
├── test.csv                            # 测试集
├── training_history_full.pkl           # 训练历史
├── loss-plot.pdf                       # 损失曲线
├── accuracy-plot.pdf                   # 准确率曲线
├── sms_spam_collection/                # SMS 垃圾邮件数据集
└── gpt2/                               # GPT-2 预训练模型
```

---

## 🎯 学习目标

- ✅ 理解迁移学习和微调
- ✅ 为 GPT 添加分类头
- ✅ 掌握微调策略（全参数 vs 部分参数）
- ✅ 实现文本分类流程
- ✅ 评估分类模型性能

---

## 🚀 快速开始

### 1. 准备数据集

```bash
# 下载并准备 SMS 垃圾邮件数据集
python prepare_dataset.py
```

**数据格式**:
```
Label,Text
spam,"Congratulations! You've won a $1000 prize..."
ham,"Hey, are we still meeting for lunch?"
```

### 2. 下载预训练模型

```bash
python gpt_download.py
```

下载 GPT-2 (124M) 预训练权重。

### 3. 微调模型

```bash
# 完整微调（所有参数）
python fine-tuning-all.py

# 或只微调最后几层
python fine-tuning.py --freeze_layers 8
```

### 4. 推理预测

```bash
python spam_classifier_inference.py
```

---

## 🏗️ 分类器架构

```
Input Text
    ↓
GPT-2 Encoder (预训练)
    ↓
Last Token Representation
    ↓
Classification Head
    ├── Linear(768 → 768)
    ├── GELU
    ├── Dropout
    └── Linear(768 → 2)
    ↓
Softmax
    ↓
[Spam, Ham] Probabilities
```

---

## 📚 核心概念

### 迁移学习

1. **预训练**: 在大规模语料上学习通用语言表示
2. **微调**: 在特定任务数据上调整参数

优势:
- 需要更少的标注数据
- 训练速度更快
- 性能通常更好

### 添加分类头

```python
class GPTClassifier(nn.Module):
    def __init__(self, gpt_model, num_classes=2):
        super().__init__()
        self.gpt = gpt_model
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, input_ids):
        # GPT 编码
        output = self.gpt(input_ids)  # (batch, seq_len, 768)
        
        # 取最后一个 token 的表示
        last_token = output[:, -1, :]  # (batch, 768)
        
        # 分类
        logits = self.classifier(last_token)  # (batch, 2)
        return logits
```

### 微调策略

#### 1. 全参数微调
微调所有参数（包括 GPT 主干）。

```python
# 所有参数可训练
for param in model.parameters():
    param.requires_grad = True
```

#### 2. 部分微调
冻结前 N 层，只微调后面的层。

```python
# 冻结前 8 层
for i in range(8):
    for param in model.gpt.layers[i].parameters():
        param.requires_grad = False
```

#### 3. 仅训练分类头
冻结整个 GPT，只训练新加的分类头。

```python
# 冻结 GPT
for param in model.gpt.parameters():
    param.requires_grad = False
    
# 分类头参数可训练
for param in model.classifier.parameters():
    param.requires_grad = True
```

---

## 💡 完整训练示例

```python
from gpt_class_finetune import GPTClassifier
from create_data_loaders import create_dataloaders

# 加载预训练 GPT-2
gpt_model = load_pretrained_gpt()

# 创建分类器
classifier = GPTClassifier(gpt_model, num_classes=2)

# 准备数据
train_loader, val_loader, test_loader = create_dataloaders(
    train_csv='train.csv',
    val_csv='validation.csv', 
    test_csv='test.csv',
    batch_size=8
)

# 训练
optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    # 训练循环
    classifier.train()
    for input_ids, labels in train_loader:
        logits = classifier(input_ids)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证
    classifier.eval()
    accuracy = evaluate(classifier, val_loader)
    print(f"Epoch {epoch}: Val Accuracy = {accuracy:.2%}")

# 保存模型
torch.save(classifier.state_dict(), 'spam_classifier.pth')
```

---

## 📊 性能评估

### 评估指标

```bash
python spam_classifier_inference.py --evaluate
```

输出:
```
Accuracy:  98.5%
Precision: 97.2%
Recall:    96.8%
F1-Score:  97.0%

Confusion Matrix:
              Predicted
            Ham    Spam
Actual Ham   892     8
     Spam     12    88
```

### 可视化

查看训练曲线:
- `loss-plot.pdf`: 训练/验证损失
- `accuracy-plot.pdf`: 训练/验证准确率

---

## 🔍 推理示例

```python
from spam_classifier_inference import SpamClassifier

# 加载模型
classifier = SpamClassifier('spam_classifier_full_finetune.pth')

# 预测
text = "Congratulations! You've won a free iPhone!"
result = classifier.predict(text)

print(f"Text: {text}")
print(f"Prediction: {result['label']}")  # "spam"
print(f"Confidence: {result['confidence']:.2%}")  # 98.5%
```

---

## 🎯 最佳实践

### 1. 数据准备
- 平衡数据集（spam/ham 比例）
- 清洗文本（去除特殊字符）
- 合理划分训练/验证/测试集

### 2. 超参数调整
- 学习率: `1e-5` ~ `5e-5`
- Batch Size: `4` ~ `16`
- Epochs: `3` ~ `10`
- Warmup Steps: `10%` 总步数

### 3. 防止过拟合
- 使用 Dropout (0.1 ~ 0.3)
- Early Stopping
- 权重衰减 (Weight Decay)
- 数据增强

### 4. 评估
- 不只看准确率，关注 F1-Score
- 分析混淆矩阵
- 测试边界样本

---

## 🔗 相关章节

- **上一章**: [Chapter 5 - 预训练](../chap5-pretraining/)
- **下一章**: [Chapter 7 - 指令微调](../chap7-fine-tuning-to-follow-instruction/)

---

**最后更新**: 2025年11月17日
