"""
Complete DeepSeekMoE Model Implementation

This module implements the full DeepSeekMoE language model architecture,
including embeddings, multiple transformer blocks with MoE layers, and
the language modeling head.

Reference: DeepSeekMoE: Towards Ultimate Expert Specialization (arXiv:2401.06066)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from moe_layer import DeepSeekMoEBlock


@dataclass
class DeepSeekMoEConfig:
    """Configuration for DeepSeekMoE model."""
    
    # Model dimensions
    vocab_size: int = 102400
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    
    # MoE configuration
    num_shared_experts: int = 2
    num_routed_experts: int = 64
    num_experts_per_token: int = 6
    moe_layer_freq: int = 1  # Apply MoE every N layers (1 = every layer)
    first_moe_layer: int = 1  # First layer to use MoE (0-indexed)
    
    # Other settings
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Auxiliary loss
    aux_loss_alpha: float = 0.01
    
    def __post_init__(self):
        # Validate configuration
        assert self.num_experts_per_token <= self.num_routed_experts
        

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self._build_cache(max_position_embeddings)
        
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        seq_len = x.shape[1]
        
        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len)
            
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepSeekAttention(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings
        )
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        else:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class DeepSeekMLP(nn.Module):
    """Standard MLP for non-MoE layers."""
    
    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekDecoderLayer(nn.Module):
    """Decoder layer with optional MoE."""
    
    def __init__(self, config: DeepSeekMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Attention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepSeekAttention(config)
        
        # FFN or MoE
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Determine if this layer uses MoE
        self.use_moe = (
            layer_idx >= config.first_moe_layer and
            (layer_idx - config.first_moe_layer) % config.moe_layer_freq == 0
        )
        
        if self.use_moe:
            from moe_layer import DeepSeekMoELayer
            self.ffn = DeepSeekMoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_shared_experts=config.num_shared_experts,
                num_routed_experts=config.num_routed_experts,
                num_experts_per_token=config.num_experts_per_token,
                dropout=config.hidden_dropout
            )
        else:
            self.ffn = DeepSeekMLP(config)
            
        self.dropout = nn.Dropout(config.hidden_dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # FFN or MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        aux_loss = None
        if self.use_moe:
            hidden_states, _, aux_loss = self.ffn(hidden_states)
        else:
            hidden_states = self.ffn(hidden_states)
            hidden_states = residual + hidden_states
            
        return hidden_states, aux_loss


class DeepSeekMoEModel(nn.Module):
    """
    Complete DeepSeekMoE Model.
    
    This implements the full transformer architecture with:
    - Token embeddings
    - Multiple decoder layers (with MoE)
    - Final layer norm
    - LM head for next token prediction
    """
    
    def __init__(self, config: DeepSeekMoEConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DeepSeekDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM Head (weight tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Attention mask
            position_ids: Position IDs
            labels: Labels for language modeling loss
            
        Returns:
            Dictionary containing:
                - logits: LM logits
                - loss: Total loss (if labels provided)
                - aux_loss: Auxiliary MoE balance loss
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Accumulate auxiliary loss
        total_aux_loss = 0.0
        num_moe_layers = 0
        
        # Forward through layers
        for layer in self.layers:
            hidden_states, aux_loss = layer(hidden_states, attention_mask, position_ids)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
                num_moe_layers += 1
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # Add auxiliary loss
            if num_moe_layers > 0:
                avg_aux_loss = total_aux_loss / num_moe_layers
                loss = loss + self.config.aux_loss_alpha * avg_aux_loss
        
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss / max(num_moe_layers, 1) if num_moe_layers > 0 else None
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Prompt token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-K sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated token IDs
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Apply top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus sampling)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
        return input_ids
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Count MoE layers
        moe_layers = sum(1 for layer in self.layers if layer.use_moe)
        dense_layers = len(self.layers) - moe_layers
        
        # Count expert parameters
        expert_params = 0
        for layer in self.layers:
            if layer.use_moe:
                expert_params += sum(p.numel() for p in layer.ffn.parameters())
        
        return {
            "total_params": total_params,
            "total_params_b": total_params / 1e9,
            "num_layers": len(self.layers),
            "moe_layers": moe_layers,
            "dense_layers": dense_layers,
            "expert_params": expert_params,
            "expert_params_ratio": expert_params / total_params,
            "hidden_size": self.config.hidden_size,
            "num_heads": self.config.num_attention_heads,
            "vocab_size": self.config.vocab_size
        }


def create_deepseek_moe_2b() -> DeepSeekMoEModel:
    """Create DeepSeekMoE-2B configuration."""
    config = DeepSeekMoEConfig(
        vocab_size=102400,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_shared_experts=1,
        num_routed_experts=63,
        num_experts_per_token=7,
        max_position_embeddings=4096
    )
    return DeepSeekMoEModel(config)


def create_deepseek_moe_16b() -> DeepSeekMoEModel:
    """Create DeepSeekMoE-16B configuration."""
    config = DeepSeekMoEConfig(
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_shared_experts=2,
        num_routed_experts=64,
        num_experts_per_token=6,
        max_position_embeddings=4096
    )
    return DeepSeekMoEModel(config)


if __name__ == "__main__":
    print("=== DeepSeekMoE Model Demo ===\n")
    
    # Create a small model for testing
    config = DeepSeekMoEConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_shared_experts=1,
        num_routed_experts=8,
        num_experts_per_token=2,
        max_position_embeddings=512
    )
    
    print("Model Configuration:")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")
    
    # Create model
    model = DeepSeekMoEModel(config)
    
    # Get stats
    print("\nModel Statistics:")
    stats = model.get_model_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Forward pass
    batch_size = 2
    seq_len = 32
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Training forward
    model.train()
    outputs = model(input_ids, labels=labels)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    if outputs['aux_loss'] is not None:
        print(f"Aux loss: {outputs['aux_loss'].item():.4f}")
    
    # Generation
    print("\n--- Generation Demo ---")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"Prompt length: {prompt.shape[1]}")
    print(f"Generated length: {generated.shape[1]}")
    print(f"New tokens: {generated.shape[1] - prompt.shape[1]}")
