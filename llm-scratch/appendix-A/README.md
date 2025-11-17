# Appendix A: 分布式数据并行训练

本附录介绍如何使用 PyTorch 的分布式数据并行 (DDP) 来加速大模型训练。

---

## 📂 文件说明

```
appendix-A/
├── DDP-script.py              # 基础 DDP 训练脚本
└── DDP-script-torchrun.py     # 使用 torchrun 的 DDP 脚本
```

---

## 🎯 学习目标

- ✅ 理解分布式训练原理
- ✅ 掌握 DDP (DistributedDataParallel) 使用
- ✅ 学习多 GPU 训练配置
- ✅ 优化训练效率

---

## 🚀 快速开始

### 单机多 GPU 训练

#### 方式 1: 使用 torch.distributed.launch (旧版)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    DDP-script.py
```

#### 方式 2: 使用 torchrun (推荐)

```bash
torchrun \
    --standalone \
    --nproc_per_node=4 \
    DDP-script-torchrun.py
```

### 多机多 GPU 训练

```bash
# 主节点 (rank 0)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    DDP-script-torchrun.py

# 工作节点 (rank 1)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    DDP-script-torchrun.py
```

---

## 📚 核心概念

### DistributedDataParallel (DDP)

DDP 通过数据并行实现模型训练加速：
- 每个 GPU 持有模型的完整副本
- 数据在 GPU 间分片
- 梯度在反向传播后同步

### 与 DataParallel 的区别

| 特性 | DataParallel (DP) | DistributedDataParallel (DDP) |
|------|------------------|------------------------------|
| **多机支持** | ❌ 否 | ✅ 是 |
| **效率** | 较低 (单进程) | 高 (多进程) |
| **梯度同步** | 主 GPU 收集 | All-Reduce |
| **推荐** | ❌ 不推荐 | ✅ 推荐 |

---

## 💡 DDP 代码示例

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """初始化进程组"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理进程组"""
    dist.destroy_process_group()

def train(rank, world_size):
    # 初始化
    setup(rank, world_size)
    
    # 创建模型并移到对应 GPU
    model = GPTModel(config).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 创建分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 训练循环
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 重要！确保每个 epoch 数据不同
        
        for batch in dataloader:
            batch = batch.to(rank)
            
            # 前向传播
            loss = model(batch)
            
            # 反向传播（自动同步梯度）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if rank == 0:  # 只在主进程打印
                print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # 清理
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

---

## 🔧 关键配置

### 环境变量

```bash
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0
```

### 进程组初始化

```python
# NCCL backend (推荐用于 GPU)
dist.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=world_size,
    rank=rank
)

# Gloo backend (CPU 或跨平台)
dist.init_process_group(
    backend='gloo',
    init_method='env://',
    world_size=world_size,
    rank=rank
)
```

---

## 📊 性能优化

### 1. Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 3. 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

---

## 🎯 最佳实践

1. **使用 torchrun**: 比旧版 launch 更稳定
2. **NCCL backend**: GPU 训练首选
3. **设置随机种子**: 确保可复现性
4. **合理批次大小**: `total_batch = per_gpu_batch × num_gpus`
5. **主进程操作**: 保存模型、日志等只在 rank 0 执行

---

## 🔗 相关资源

- [PyTorch DDP 官方教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/)

---

**最后更新**: 2025年11月17日
