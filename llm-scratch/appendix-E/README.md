# Appendix E: LoRA 参数高效微调

本附录介绍 LoRA (Low-Rank Adaptation)，一种参数高效的微调方法。

---

## 📂 文件说明

```
appendix-E/
├── LoRA.py                  # LoRA 实现
├── gpt_download.py          # 下载 GPT-2 模型
├── previous_chapters.py     # 前几章代码
├── loss-plot.pdf            # 训练损失曲线
├── train.csv                # 训练数据
├── validation.csv           # 验证数据
├── test.csv                 # 测试数据
├── sms_spam_collection/     # 垃圾邮件数据集
└── gpt2/                    # GPT-2 模型
```

---

## 🎯 学习目标

- ✅ 理解 LoRA 原理
- ✅ 实现低秩矩阵分解
- ✅ 掌握参数高效微调
- ✅ 对比全参数微调和 LoRA

---

## 🚀 快速开始

### 1. 下载预训练模型

```bash
python gpt_download.py
```

### 2. 使用 LoRA 微调

```bash
python LoRA.py \
    --model gpt2 \
    --rank 8 \
    --alpha 16 \
    --epochs 5
```

---

## 📚 核心概念

### LoRA 原理

LoRA 通过低秩矩阵分解来减少可训练参数：

$$W' = W + \Delta W = W + BA$$

其中:
- $W \in \mathbb{R}^{d \times k}$: 原始权重矩阵（冻结）
- $B \in \mathbb{R}^{d \times r}$: 低秩矩阵 B（可训练）
- $A \in \mathbb{R}^{r \times k}$: 低秩矩阵 A（可训练）
- $r \ll \min(d, k)$: 秩（通常 4-64）

**参数对比**:
- 原始参数: $d \times k$
- LoRA 参数: $d \times r + r \times k$
- 当 $r=8$, $d=k=768$ 时: $589,824$ → $12,288$ (98% 减少！)

### 优势

✅ **参数效率**: 只需微调 0.1%-1% 的参数  
✅ **存储效率**: 每个任务只需保存小的 LoRA 权重  
✅ **推理效率**: 可以合并权重，无额外开销  
✅ **多任务**: 一个基座模型 + 多个 LoRA 模块  

---

## 💡 LoRA 实现

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank=8,
        alpha=16,
        dropout=0.1
    ):
        super().__init__()
        
        # 原始权重（冻结）
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False
        
        # LoRA 参数
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # 缩放因子
        self.scaling = alpha / rank
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x):
        # 原始输出
        result = self.linear(x)
        
        # LoRA 增量
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        lora_out = self.dropout(lora_out)
        
        return result + lora_out
    
    def merge_weights(self):
        """合并 LoRA 权重到原始权重（推理优化）"""
        self.linear.weight.data += (
            self.lora_B @ self.lora_A.T * self.scaling
        )
```

### 应用 LoRA 到 GPT

```python
def apply_lora_to_gpt(model, rank=8, alpha=16):
    """将 LoRA 应用到 GPT 的注意力层"""
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 只替换注意力层的 QKV 投影
            if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
                in_features = module.in_features
                out_features = module.out_features
                
                # 创建 LoRA 层
                lora_layer = LoRALayer(
                    in_features,
                    out_features,
                    rank=rank,
                    alpha=alpha
                )
                
                # 复制原始权重
                lora_layer.linear.weight.data = module.weight.data.clone()
                
                # 替换
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, lora_layer)
    
    return model
```

---

## 📊 参数对比

### GPT-2 Base (124M 参数)

| 方法 | 可训练参数 | 百分比 | 存储大小 |
|------|-----------|-------|---------|
| **全参数微调** | 124M | 100% | ~500 MB |
| **LoRA (r=4)** | 0.3M | 0.24% | ~1.2 MB |
| **LoRA (r=8)** | 0.6M | 0.48% | ~2.4 MB |
| **LoRA (r=16)** | 1.2M | 0.97% | ~4.8 MB |

### GPT-2 Large (774M 参数)

| 方法 | 可训练参数 | 百分比 | 存储大小 |
|------|-----------|-------|---------|
| **全参数微调** | 774M | 100% | ~3 GB |
| **LoRA (r=8)** | 4M | 0.52% | ~16 MB |

---

## 🎯 训练示例

```python
from LoRA import apply_lora_to_model, train_with_lora

# 加载预训练 GPT-2
model = load_gpt2()

# 应用 LoRA
model = apply_lora_to_model(
    model,
    rank=8,
    alpha=16,
    target_modules=['q_proj', 'v_proj']  # 只在 QV 投影使用 LoRA
)

# 冻结非 LoRA 参数
for name, param in model.named_parameters():
    if 'lora_' not in name:
        param.requires_grad = False

# 查看可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params:.2%})")

# 训练
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

train_with_lora(model, train_loader, optimizer, epochs=5)

# 保存 LoRA 权重（仅几 MB）
save_lora_weights(model, 'lora_weights.pt')
```

---

## 🔧 高级技巧

### 1. 选择目标层

```python
# 只在注意力层使用 LoRA
target_modules = ['q_proj', 'v_proj']

# 也可以在 FFN 使用
target_modules = ['q_proj', 'v_proj', 'fc1', 'fc2']
```

### 2. 调整秩 (Rank)

- **r=4**: 最少参数，适合小任务
- **r=8**: 平衡选择（推荐）
- **r=16+**: 更强表达能力，接近全参数微调

### 3. Alpha 缩放

```python
scaling = alpha / rank

# alpha=16, rank=8 → scaling=2 (推荐)
# alpha=rank → scaling=1
```

### 4. 推理优化

```python
# 训练后合并权重
model.merge_lora_weights()

# 推理时无额外开销
output = model(input)
```

---

## 📖 参考资料

- [LoRA 论文](https://arxiv.org/abs/2106.09685): LoRA: Low-Rank Adaptation of Large Language Models
- [PEFT 库](https://github.com/huggingface/peft): Hugging Face 的参数高效微调库

---

## 🔗 相关章节

- [Chapter 6 - 分类任务微调](../chap6-fine-tuning-for-classification/)
- [Chapter 7 - 指令微调](../chap7-fine-tuning-to-follow-instruction/)

---

**最后更新**: 2025年11月17日
