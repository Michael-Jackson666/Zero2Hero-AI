# DeepSeekMoE: Ultimate Expert Specialization

PyTorch implementation of the DeepSeekMoE architecture from the paper: [*DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*](https://arxiv.org/abs/2401.06066).

## 🧠 Overview

DeepSeekMoE addresses the **expert specialization** problem in traditional MoE models through two key innovations:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DeepSeekMoE Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input Token                                                           │
│       │                                                                 │
│       ▼                                                                 │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │                    Self-Attention                              │    │
│   └───────────────────────────────────────────────────────────────┘    │
│       │                                                                 │
│       ▼                                                                 │
│   ┌───────────────────────────────────────────────────────────────┐    │
│   │                      MoE Layer                                 │    │
│   │  ┌─────────────────┐    ┌─────────────────────────────────┐   │    │
│   │  │ Shared Experts  │    │      Routed Experts (Top-K)     │   │    │
│   │  │   (Always On)   │    │  ┌───┐ ┌───┐ ┌───┐     ┌───┐   │   │    │
│   │  │  ┌───┐  ┌───┐   │ +  │  │ E │ │ E │ │ E │ ... │ E │   │   │    │
│   │  │  │ S │  │ S │   │    │  │ 1 │ │ 2 │ │ 3 │     │ N │   │   │    │
│   │  │  └───┘  └───┘   │    │  └───┘ └───┘ └───┘     └───┘   │   │    │
│   │  └─────────────────┘    └─────────────────────────────────┘   │    │
│   └───────────────────────────────────────────────────────────────┘    │
│       │                                                                 │
│       ▼                                                                 │
│   Output                                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔑 Key Innovations

### 1. Fine-Grained Expert Segmentation

Split each standard FFN expert into **m smaller experts** (typically m=4):

- **Before**: N experts, each with intermediate size H
- **After**: mN experts, each with intermediate size H/m
- **Benefit**: Exponentially more expert combinations → better specialization

| Configuration | Experts | Top-K | Combinations |
|---------------|---------|-------|--------------|
| Standard MoE  | 16      | 2     | 120          |
| DeepSeekMoE   | 64      | 8     | 4,426,165,368|

### 2. Shared Expert Isolation

Isolate **K_s experts** that are **always activated** (no routing):

- **Shared Experts**: Capture common knowledge (grammar, frequent patterns)
- **Routed Experts**: Specialize in specific knowledge domains
- **Benefit**: Reduces redundancy among routed experts

## 📁 Project Structure

```
Code/
├── experts.py        # Expert FFN networks (SwiGLU, fine-grained)
├── router.py         # Top-K routing with load balancing
├── moe_layer.py      # Complete MoE layer with shared & routed experts
├── deepseek_moe.py   # Full DeepSeekMoE model
└── README.md         # This file
```

## 📐 Mathematical Formulation

**Standard MoE Layer:**
$$\mathbf{h}_t^l = \sum_{i=1}^N g_{i,t} \text{FFN}_i(\mathbf{u}_t^l) + \mathbf{u}_t^l$$

**DeepSeekMoE Layer:**
$$\mathbf{h}_t^l = \underbrace{\sum_{i=1}^{K_s} \text{FFN}_i(\mathbf{u}_t^l)}_{\text{Shared Experts}} + \underbrace{\sum_{i=K_s+1}^{mN} g_{i,t} \text{FFN}_i(\mathbf{u}_t^l)}_{\text{Routed Experts}} + \mathbf{u}_t^l$$

**Routing Mechanism:**
$$s_{i,t} = \text{Softmax}_i(\mathbf{u}_t^{l^T} \mathbf{e}_i^l)$$
$$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{TopK}(\{s_{j,t}\}, mK - K_s) \\ 0, & \text{otherwise} \end{cases}$$

## 🚀 Quick Start

```python
from deepseek_moe import DeepSeekMoEModel, DeepSeekMoEConfig

# Create configuration
config = DeepSeekMoEConfig(
    vocab_size=102400,
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_shared_experts=2,
    num_routed_experts=64,
    num_experts_per_token=6,
)

# Create model
model = DeepSeekMoEModel(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
outputs = model(input_ids)

# Get logits
logits = outputs["logits"]

# Generation
generated = model.generate(input_ids, max_new_tokens=100)
```

## ⚖️ Load Balancing

Two-level auxiliary loss for balanced expert utilization:

### Expert-Level Balance Loss
$$\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i$$

Where:
- $f_i$ = fraction of tokens routed to expert $i$
- $P_i$ = average routing probability for expert $i$

### Device-Level Balance Loss
$$\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{d=1}^{D} f'_d P'_d$$

Ensures balanced computation across distributed devices.

## 📊 Model Configurations

| Model | Total Params | Active Params | Shared | Routed | Top-K |
|-------|--------------|---------------|--------|--------|-------|
| DeepSeekMoE-2B | 2.0B | 0.3B | 1 | 63 | 7 |
| DeepSeekMoE-16B | 16.4B | 2.8B | 2 | 64 | 6 |
| DeepSeekMoE-145B | 144.6B | 22.2B | 4 | 128 | 12 |

## 🔬 Key Findings

1. **U-Shaped Optimum**: Neither pure dense nor pure MoE is optimal; the best results come from mixing shared and routed experts.

2. **Extreme Specialization**: Removing any single expert causes significant performance degradation (unlike traditional MoE where experts are redundant).

3. **Iso-FLOPs Advantage**: At equal compute budget, DeepSeekMoE significantly outperforms GShard and approaches dense model performance.

## 🛠️ Module Details

### `experts.py`
- `Expert`: Single fine-grained FFN with SwiGLU activation
- `SharedExpert`: Always-active expert
- `ExpertGroup`: Batched expert computation

### `router.py`
- `TopKRouter`: Standard Top-K routing
- `NoisyTopKRouter`: With learnable noise for exploration
- `LoadBalanceLoss`: Expert-level balance loss
- `DeviceBalanceLoss`: Device-level balance loss

### `moe_layer.py`
- `DeepSeekMoELayer`: Complete layer with shared + routed experts
- `DeepSeekMoEBlock`: Transformer block with attention + MoE

### `deepseek_moe.py`
- `DeepSeekMoEModel`: Full transformer model
- `DeepSeekMoEConfig`: Configuration dataclass
- Helper functions for standard configurations

## 📚 References

- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668)

## 📄 License

MIT License
