# Engram: Conditional Memory via Scalable Lookup

PyTorch implementation of the Engram architecture from DeepSeek's paper: [*Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*](https://arxiv.org/abs/2601.07372).

## 🧠 Overview

Engram introduces **Conditional Memory** as a new axis of sparsity for LLMs, complementing the existing **Conditional Computation** (MoE). While MoE handles compositional reasoning, Engram offloads knowledge retrieval to O(1) hash-based lookups.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Two Axes of Sparsity                         │
├─────────────────────────────┬───────────────────────────────────────┤
│      MoE (Computation)      │         Engram (Memory)               │
├─────────────────────────────┼───────────────────────────────────────┤
│  Compositional Reasoning    │  Knowledge Retrieval                  │
│  Dynamic Expert Selection   │  Static Hash Lookup                   │
│  Runtime State Dependent    │  Input Token Dependent                │
│  Cannot Prefetch            │  Zero-Overhead Prefetching            │
│  O(N) Matrix Multiply       │  O(1) Table Lookup                    │
└─────────────────────────────┴───────────────────────────────────────┘
```

## 📁 Project Structure

```
Code/
├── tokenizer_compression.py   # Token normalization & N-gram extraction
├── multi_head_hashing.py      # Multi-head hash & embedding tables
├── context_aware_gating.py    # Cross-attention based gating mechanism
├── fusion.py                  # Depthwise conv & residual integration
└── engram.py                  # Complete Engram module
```

## 🔧 Module Components

### 1. Tokenizer Compression (`tokenizer_compression.py`)

Normalizes semantically equivalent tokens to reduce vocabulary size (~23% reduction):

$$x'_t = \mathcal{P}(x_t)$$

- Case normalization (`Apple` → `apple`)
- Leading space normalization (`_Apple` → `Apple`)
- Unicode normalization

### 2. Multi-Head Hashing (`multi_head_hashing.py`)

Maps N-grams to embedding indices using K independent hash functions:

$$z_{t,n,k} = \varphi_{n,k}(g_{t,n})$$
$$\mathbf{e}_{t,n,k} = \mathbf{E}_{n,k}[z_{t,n,k}]$$
$$\mathbf{e}_t = \mathop{\Big\Vert}_{n=2}^N \mathop{\Big\Vert}_{k=1}^K \mathbf{e}_{t,n,k}$$

Multiple hash heads reduce collision probability (similar to Bloom Filters).

### 3. Context-Aware Gating (`context_aware_gating.py`)

Cross-attention based gating to suppress noise from hash collisions:

$$\mathbf{k}_t = \mathbf{W}_K \mathbf{e}_t, \quad \mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t$$
$$\alpha_t = \sigma \left( \frac{\text{RMSNorm}(\mathbf{h}_t)^\top \text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}} \right)$$
$$\tilde{\mathbf{v}}_t = \alpha_t \cdot \mathbf{v}_t$$

### 4. Fusion (`fusion.py`)

Local temporal smoothing via depthwise convolution:

$$\mathbf{Y} = \text{SiLU}(\text{Conv1D}(\text{RMSNorm}(\tilde{\mathbf{v}}_t))) + \tilde{\mathbf{v}}_t$$
$$\mathbf{H}^{(\ell)} \leftarrow \mathbf{H}^{(\ell)} + \mathbf{Y}$$

### 5. Complete Module (`engram.py`)

Integrates all components into a plug-and-play module.

## 🚀 Quick Start

```python
from engram import EngramModule, EngramConfig

# Configure Engram
config = EngramConfig(
    vocab_size=50_000,
    min_n=2,
    max_n=4,
    num_hash_heads=4,
    embedding_table_size=1_000_000,
    embedding_dim=64,
    hidden_dim=2048,
)

# Create module
engram = EngramModule(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

# Option 1: Standard forward
updated_states, gating = engram(
    hidden_states=hidden_states,
    input_ids=input_ids,
    return_gating=True
)

# Option 2: With prefetching (for pipeline parallelism)
memory_embeddings = engram.prefetch_embeddings(input_ids)
# ... compute other layers on GPU ...
updated_states, _ = engram(
    hidden_states=hidden_states,
    memory_embeddings=memory_embeddings
)
```

## 🔑 Key Innovation: Zero-Overhead Prefetching

Unlike MoE where routing depends on runtime hidden states, Engram's indices only depend on input tokens:

```
┌────────────────────────────────────────────────────────────────────┐
│                      Pipeline Design                                │
├────────────────────────────────────────────────────────────────────┤
│  Time ────────────────────────────────────────────────────────>    │
│                                                                     │
│  GPU:  [Layer 0] ──> [Layer 1] ──> [Layer 2 + Engram] ──> ...     │
│                           │              ▲                          │
│                           │              │                          │
│  CPU:  [Prefetch Engram Embeddings] ─────┘                         │
│        (Happens while GPU computes Layer 0-1)                       │
└────────────────────────────────────────────────────────────────────┘
```

**Result:** Communication latency is completely hidden by computation.

## 📊 Sparsity Allocation (U-Shaped Scaling Law)

The paper discovers optimal allocation between MoE and Engram:

- **Pure MoE (100%)**: Suboptimal, experts waste capacity on memorization
- **Pure Engram (0%)**: Poor, lacks reasoning capability  
- **Optimal (~20-25% Engram)**: Best of both worlds

## 🛠️ Integration Example

```python
class TransformerWithEngram(nn.Module):
    def __init__(self, config, engram_config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        # Insert Engram at Layer 2 (as recommended in the paper)
        self.engram = EngramModule(engram_config)
        self.engram_layer_idx = 2
    
    def forward(self, input_ids, hidden_states):
        # Prefetch Engram embeddings immediately
        memory_embeddings = self.engram.prefetch_embeddings(input_ids)
        
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            
            if idx == self.engram_layer_idx:
                hidden_states, _ = self.engram(
                    hidden_states=hidden_states,
                    memory_embeddings=memory_embeddings
                )
        
        return hidden_states
```

## 📚 References

- [Conditional Memory via Scalable Lookup (arXiv:2601.07372)](https://arxiv.org/abs/2601.07372)
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

## 📄 License

MIT License
