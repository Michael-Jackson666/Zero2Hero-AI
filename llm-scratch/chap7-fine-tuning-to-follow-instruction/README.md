# Chapter 7: 指令微调

本章介绍如何对 GPT 模型进行指令微调 (Instruction Fine-tuning)，使其能够遵循用户指令完成各种任务。

---

## 📂 文件说明

```
chap7-fine-tuning-to-follow-instruction/
├── fine-tuning.py                                      # 主微调脚本
├── gpt_instruction_finetuning.py                       # 指令微调实现
├── prepare_dataset.py                                  # 数据集准备
├── create_data_loaders.py                              # 数据加载器
├── organize_data.py                                    # 数据组织
├── evaluate_model.py                                   # 模型评估
├── extract_response.py                                 # 提取模型响应
├── load_weights.py                                     # 加载权重
├── gpt_download.py                                     # 下载 GPT-2
├── previous_chapters.py                                # 前几章代码
├── instruction-data.json                               # 指令数据集
├── instruction-data-full.json                          # 完整指令数据
├── instruction-data-with-response_gpt2-medium_355M.json # 带响应的数据
├── gpt2-medium355M-sft.pth                            # 微调后模型
├── loss-plot.pdf                                       # 损失曲线
└── gpt2/                                               # GPT-2 预训练模型
```

---

## 🎯 学习目标

- ✅ 理解指令微调 (Instruction Tuning)
- ✅ 准备指令数据集格式
- ✅ 实现监督式微调 (SFT)
- ✅ 评估指令遵循能力
- ✅ 掌握提示工程 (Prompt Engineering)

---

## 🚀 快速开始

### 1. 准备指令数据集

```bash
python prepare_dataset.py
```

**指令数据格式**:
```json
[
    {
        "instruction": "将以下句子翻译成法语",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?"
    },
    {
        "instruction": "总结以下文本",
        "input": "长文本...",
        "output": "摘要..."
    }
]
```

### 2. 下载预训练模型

```bash
python gpt_download.py --model gpt2-medium
```

下载 GPT-2 Medium (355M) 模型。

### 3. 指令微调

```bash
python gpt_instruction_finetuning.py \
    --data instruction-data.json \
    --epochs 3 \
    --batch_size 4 \
    --lr 5e-5
```

### 4. 评估和推理

```bash
# 评估模型
python evaluate_model.py

# 交互式测试
python fine-tuning.py --interactive
```

---

## 🏗️ 指令微调架构

```
User Instruction + Input
    ↓
"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ↓
GPT-2 Model (微调)
    ↓
Generated Response
    ↓
"### Response:\n{output}"
```

---

## 📚 核心概念

### 指令微调 vs 预训练

| 方面 | 预训练 | 指令微调 |
|------|--------|---------|
| **目标** | 学习语言模式 | 学习遵循指令 |
| **数据** | 无标注文本 | 指令-响应对 |
| **任务** | 下一个词预测 | 条件生成 |
| **能力** | 通用语言理解 | 特定任务执行 |

### 数据格式设计

**Alpaca 格式** (推荐):
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**优势**:
- 明确的结构化标记
- 模型易于识别各部分
- 支持有/无输入的指令

### 训练目标

只计算 Response 部分的损失：

```python
# 创建损失掩码
loss_mask = torch.zeros_like(input_ids)
response_start_idx = find_response_start(input_ids)
loss_mask[:, response_start_idx:] = 1

# 计算损失
loss = criterion(logits, targets) * loss_mask
loss = loss.sum() / loss_mask.sum()
```

---

## 💡 完整微调流程

```python
from gpt_instruction_finetuning import InstructionGPT
from create_data_loaders import create_instruction_dataloader

# 加载预训练模型
model = load_gpt2_medium()

# 创建指令微调包装
instruction_model = InstructionGPT(model)

# 准备数据
train_loader = create_instruction_dataloader(
    'instruction-data.json',
    batch_size=4,
    max_length=512
)

# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01
)

# 训练循环
for epoch in range(3):
    for batch in train_loader:
        # 前向传播
        logits = model(batch['input_ids'])
        
        # 只计算 response 部分损失
        loss = compute_instruction_loss(
            logits, 
            batch['labels'],
            batch['response_mask']
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")

# 保存微调模型
torch.save(model.state_dict(), 'gpt2-medium355M-sft.pth')
```

---

## 🎯 指令类型示例

### 1. 文本生成
```
Instruction: 写一首关于春天的诗
Input: 
Output: 春风拂面暖阳天，...
```

### 2. 问答
```
Instruction: 回答以下问题
Input: 地球上最高的山是什么？
Output: 珠穆朗玛峰，海拔 8,848.86 米。
```

### 3. 文本转换
```
Instruction: 将以下句子改写为正式语气
Input: 这东西真的超级好用！
Output: 该产品具有优异的性能和实用价值。
```

### 4. 分类
```
Instruction: 判断以下评论的情感倾向（正面/负面）
Input: 这部电影太精彩了！
Output: 正面
```

### 5. 摘要
```
Instruction: 总结以下文章的主要内容
Input: [长文本]
Output: 本文主要讨论了...
```

---

## 📊 评估方法

### 1. 自动评估

```python
from evaluate_model import evaluate_instructions

metrics = evaluate_instructions(
    model,
    test_data='test_instructions.json'
)

print(f"ROUGE-L: {metrics['rouge_l']:.2f}")
print(f"BLEU: {metrics['bleu']:.2f}")
print(f"Exact Match: {metrics['exact_match']:.2%}")
```

### 2. 人工评估

评估维度:
- ✅ **相关性**: 响应是否回答了指令
- ✅ **准确性**: 信息是否正确
- ✅ **流畅性**: 语言是否自然
- ✅ **完整性**: 是否完整回答

### 3. 对比测试

```bash
python evaluate_model.py --compare
```

对比微调前后的响应质量。

---

## 🔍 推理示例

```python
from extract_response import generate_instruction_response

# 加载微调模型
model = load_instruction_model('gpt2-medium355M-sft.pth')

# 构造提示
instruction = "将以下文本翻译成英语"
input_text = "你好，世界！"

# 生成响应
response = generate_instruction_response(
    model,
    instruction=instruction,
    input_text=input_text,
    max_tokens=50,
    temperature=0.7
)

print(f"Instruction: {instruction}")
print(f"Input: {input_text}")
print(f"Response: {response}")
# Output: "Hello, World!"
```

---

## 🎯 提示工程技巧

### 1. 清晰具体的指令
❌ "处理这个文本"
✅ "将以下文本翻译成法语"

### 2. 提供示例 (Few-Shot)
```
Instruction: 将句子改为疑问句

Example 1:
Input: 他喜欢篮球。
Output: 他喜欢篮球吗？

Example 2:
Input: [实际输入]
Output:
```

### 3. 分步骤指令
```
Instruction: 
1. 阅读以下文本
2. 提取关键信息
3. 用三句话总结
```

### 4. 指定输出格式
```
Instruction: 以 JSON 格式提取以下信息：
- 姓名
- 年龄
- 职业
```

---

## 🔧 高级技术

### 1. RLHF (人类反馈强化学习)
下一步可以使用人类反馈进一步优化。

### 2. LoRA (低秩适应)
仅微调部分参数，减少计算和存储。

### 3. Prompt Tuning
只优化提示词嵌入，冻结模型参数。

---

## 📖 数据集资源

公开的指令数据集:
- **Alpaca**: 52K 指令-响应对
- **Dolly**: 15K 高质量样本
- **FLAN**: 多任务指令集合
- **Self-Instruct**: 自动生成指令

---

## 🔗 相关章节

- **上一章**: [Chapter 6 - 分类任务微调](../chap6-fine-tuning-for-classification/)
- **附录**: 高级训练技术

---

**最后更新**: 2025年11月17日
