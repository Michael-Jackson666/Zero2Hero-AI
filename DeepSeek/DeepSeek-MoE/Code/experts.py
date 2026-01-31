"""
Expert Networks for DeepSeekMoE Architecture

This module implements the Feed-Forward Network (FFN) experts used in MoE layers.
DeepSeekMoE uses fine-grained expert segmentation where each expert is 1/m the size
of a standard FFN.

Reference: DeepSeekMoE: Towards Ultimate Expert Specialization (arXiv:2401.06066)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU activation function used in modern LLMs.
    
    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
    
    Where Swish(x) = x * sigmoid(x) = SiLU(x)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__()
        out_features = out_features or in_features
        
        # Gate projection
        self.w_gate = nn.Linear(in_features, hidden_features, bias=bias)
        # Up projection
        self.w_up = nn.Linear(in_features, hidden_features, bias=bias)
        # Down projection
        self.w_down = nn.Linear(hidden_features, out_features, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up, then down project
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Expert(nn.Module):
    """
    Single Expert Network (FFN).
    
    In DeepSeekMoE, experts are fine-grained (1/m size of standard FFN).
    Each expert has the same structure but smaller hidden dimension.
    
    Architecture:
        x -> Linear(d, h) -> Activation -> Linear(h, d) -> output
    
    Where h = intermediate_size (typically 4 * d / m for fine-grained experts)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swiglu",
        dropout: float = 0.0
    ):
        """
        Args:
            hidden_size: Model hidden dimension (d)
            intermediate_size: FFN intermediate dimension (h)
            activation: Activation function ("swiglu", "gelu", "relu")
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        if activation == "swiglu":
            self.ffn = SwiGLU(
                in_features=hidden_size,
                hidden_features=intermediate_size,
                out_features=hidden_size,
                bias=False
            )
        else:
            # Standard FFN with specified activation
            act_fn = {
                "gelu": nn.GELU(),
                "relu": nn.ReLU(),
                "silu": nn.SiLU()
            }.get(activation, nn.GELU())
            
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=False),
                act_fn,
                nn.Linear(intermediate_size, hidden_size, bias=False)
            )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.
        
        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor, shape (batch, seq_len, hidden_size)
        """
        return self.dropout(self.ffn(x))


class SharedExpert(Expert):
    """
    Shared Expert that is always activated regardless of routing.
    
    Shared experts capture common knowledge across all inputs,
    reducing redundancy in routed experts.
    
    Structurally identical to regular Expert, but treated differently
    in the MoE layer (always activated, no gating).
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swiglu",
        dropout: float = 0.0
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            dropout=dropout
        )


class ExpertGroup(nn.Module):
    """
    Group of experts for efficient batched computation.
    
    Instead of creating N separate Expert modules, this uses a single
    batched parameter tensor for all experts, enabling efficient
    parallel computation on GPU.
    """
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swiglu"
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Batched weights for all experts
        # Shape: (num_experts, hidden_size, intermediate_size)
        self.w_gate = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.w_up = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.w_down = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        for param in [self.w_gate, self.w_up, self.w_down]:
            nn.init.kaiming_uniform_(param, a=2.236)  # sqrt(5)
            
    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through selected experts.
        
        Args:
            x: Input tensor, shape (num_tokens, hidden_size)
            expert_indices: Which expert each token routes to, shape (num_tokens,)
            
        Returns:
            Output tensor, shape (num_tokens, hidden_size)
        """
        # Gather weights for selected experts
        # x: (num_tokens, hidden_size)
        # expert_indices: (num_tokens,)
        
        batch_size = x.shape[0]
        
        # Get weights for each token's expert
        w_gate = self.w_gate[expert_indices]  # (num_tokens, hidden_size, intermediate_size)
        w_up = self.w_up[expert_indices]      # (num_tokens, hidden_size, intermediate_size)
        w_down = self.w_down[expert_indices]  # (num_tokens, intermediate_size, hidden_size)
        
        # Compute SwiGLU
        # gate: (num_tokens, intermediate_size)
        gate = torch.bmm(x.unsqueeze(1), w_gate).squeeze(1)
        up = torch.bmm(x.unsqueeze(1), w_up).squeeze(1)
        
        hidden = F.silu(gate) * up
        
        # Down projection
        output = torch.bmm(hidden.unsqueeze(1), w_down).squeeze(1)
        
        return output


if __name__ == "__main__":
    print("=== Expert Networks Demo ===\n")
    
    # Configuration
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    intermediate_size = 64  # Fine-grained (1/4 of typical 4*hidden_size)
    num_experts = 8
    
    # Single expert
    expert = Expert(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="swiglu"
    )
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = expert(x)
    
    print(f"Single Expert:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in expert.parameters()):,}")
    
    # Expert group
    print(f"\nExpert Group ({num_experts} experts):")
    expert_group = ExpertGroup(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size
    )
    
    # Simulate routing: flatten batch and assign random experts
    x_flat = x.view(-1, hidden_size)  # (batch*seq, hidden_size)
    expert_indices = torch.randint(0, num_experts, (x_flat.shape[0],))
    
    output_group = expert_group(x_flat, expert_indices)
    
    print(f"  Input shape: {x_flat.shape}")
    print(f"  Expert indices shape: {expert_indices.shape}")
    print(f"  Output shape: {output_group.shape}")
    print(f"  Parameters: {sum(p.numel() for p in expert_group.parameters()):,}")
