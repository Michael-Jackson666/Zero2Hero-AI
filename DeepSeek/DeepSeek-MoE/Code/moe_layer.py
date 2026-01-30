"""
DeepSeekMoE Layer Implementation

This module implements the complete MoE layer with DeepSeek's innovations:
1. Fine-Grained Expert Segmentation (mN experts, each 1/m size)
2. Shared Expert Isolation (K_s always-active experts)

Reference: DeepSeekMoE: Towards Ultimate Expert Specialization (arXiv:2401.06066)

Key equation:
    h_t = Σ_{i=1}^{K_s} FFN_i(u_t)  +  Σ_{i=K_s+1}^{mN} g_{i,t} FFN_i(u_t)  +  u_t
          \_________________/          \________________________________/
            Shared Experts                    Routed Experts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from experts import Expert, SharedExpert
from router import TopKRouter, NoisyTopKRouter, LoadBalanceLoss


class DeepSeekMoELayer(nn.Module):
    """
    DeepSeekMoE Layer with Shared and Routed Experts.
    
    Architecture:
        1. Shared Experts: K_s experts always activated
        2. Routed Experts: mN - K_s experts, top (mK - K_s) selected per token
        3. Output = Shared_output + Routed_output + Residual
    
    Key innovations:
        - Fine-grained experts (1/m size) for more flexible combinations
        - Shared experts to capture common knowledge
        - Top-K routing for sparse activation
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_shared_experts: int = 2,
        num_routed_experts: int = 64,
        num_experts_per_token: int = 6,
        expert_intermediate_size: Optional[int] = None,
        activation: str = "swiglu",
        use_noisy_routing: bool = False,
        dropout: float = 0.0,
        normalize_expert_weights: bool = True
    ):
        """
        Args:
            hidden_size: Model hidden dimension (d)
            intermediate_size: Standard FFN intermediate size (for reference)
            num_shared_experts: Number of shared experts (K_s), always activated
            num_routed_experts: Number of routed experts (mN - K_s)
            num_experts_per_token: Number of routed experts activated per token (mK - K_s)
            expert_intermediate_size: Intermediate size per expert (default: intermediate_size / 4)
            activation: Activation function
            use_noisy_routing: Whether to use noisy Top-K routing
            dropout: Dropout probability
            normalize_expert_weights: Whether to renormalize routing weights
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_token = num_experts_per_token
        
        # Fine-grained expert size (typically 1/4 of standard FFN)
        self.expert_intermediate_size = expert_intermediate_size or intermediate_size // 4
        
        # Shared Experts (always activated)
        self.shared_experts = nn.ModuleList([
            SharedExpert(
                hidden_size=hidden_size,
                intermediate_size=self.expert_intermediate_size,
                activation=activation,
                dropout=dropout
            )
            for _ in range(num_shared_experts)
        ])
        
        # Routed Experts
        self.routed_experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                intermediate_size=self.expert_intermediate_size,
                activation=activation,
                dropout=dropout
            )
            for _ in range(num_routed_experts)
        ])
        
        # Router
        if use_noisy_routing:
            self.router = NoisyTopKRouter(
                hidden_size=hidden_size,
                num_experts=num_routed_experts,
                top_k=num_experts_per_token,
                normalize_expert_weights=normalize_expert_weights
            )
        else:
            self.router = TopKRouter(
                hidden_size=hidden_size,
                num_experts=num_routed_experts,
                top_k=num_experts_per_token,
                normalize_expert_weights=normalize_expert_weights
            )
        
        # Load balance loss calculator
        self.balance_loss = LoadBalanceLoss(
            num_experts=num_routed_experts,
            top_k=num_experts_per_token,
            alpha=0.01
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through DeepSeekMoE layer.
        
        Args:
            hidden_states: Input tensor, shape (batch, seq_len, hidden_size)
            output_router_logits: Whether to return router logits
            
        Returns:
            output: Updated hidden states
            router_logits: Router logits (if requested)
            aux_loss: Auxiliary balance loss (if training)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        
        # ========== Shared Experts (always activated) ==========
        # h_shared = Σ_{i=1}^{K_s} FFN_i(u_t)
        shared_output = torch.zeros_like(hidden_states)
        for shared_expert in self.shared_experts:
            shared_output = shared_output + shared_expert(hidden_states)
        
        # ========== Routed Experts (Top-K selected) ==========
        # Get routing weights
        gate_weights, expert_indices, router_logits = self.router(hidden_states)
        # gate_weights: (batch, seq, num_routed_experts)
        # expert_indices: (batch, seq, top_k)
        
        # Compute routed expert outputs
        # h_routed = Σ_{i=K_s+1}^{mN} g_{i,t} FFN_i(u_t)
        routed_output = self._compute_routed_output(
            hidden_states, gate_weights, expert_indices
        )
        
        # ========== Combine and add residual ==========
        output = shared_output + routed_output + residual
        
        # Compute auxiliary loss for training
        aux_loss = None
        if self.training:
            aux_loss = self.balance_loss(router_logits, expert_indices)
        
        if output_router_logits:
            return output, router_logits, aux_loss
        return output, None, aux_loss
    
    def _compute_routed_output(
        self,
        hidden_states: torch.Tensor,
        gate_weights: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute output from routed experts.
        
        This implementation iterates over experts for clarity.
        For production, use batched operations or torch.compile.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Initialize output
        routed_output = torch.zeros_like(hidden_states)
        
        # Flatten for easier indexing
        hidden_flat = hidden_states.view(-1, hidden_dim)  # (batch*seq, hidden)
        gate_flat = gate_weights.view(-1, self.num_routed_experts)  # (batch*seq, num_experts)
        
        # Process each expert
        for expert_idx, expert in enumerate(self.routed_experts):
            # Find tokens routed to this expert
            expert_mask = gate_flat[:, expert_idx] > 0  # (batch*seq,)
            
            if expert_mask.sum() == 0:
                continue
            
            # Get tokens for this expert
            expert_input = hidden_flat[expert_mask]  # (num_tokens, hidden)
            expert_weights = gate_flat[expert_mask, expert_idx]  # (num_tokens,)
            
            # Forward through expert
            expert_output = expert(expert_input.unsqueeze(1)).squeeze(1)  # (num_tokens, hidden)
            
            # Weight by gate value
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            
            # Scatter back to output
            routed_output.view(-1, hidden_dim)[expert_mask] += weighted_output
        
        return routed_output
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get statistics about expert configuration."""
        total_experts = self.num_shared_experts + self.num_routed_experts
        activated_per_token = self.num_shared_experts + self.num_experts_per_token
        
        return {
            "total_experts": total_experts,
            "shared_experts": self.num_shared_experts,
            "routed_experts": self.num_routed_experts,
            "activated_per_token": activated_per_token,
            "activation_ratio": activated_per_token / total_experts,
            "expert_intermediate_size": self.expert_intermediate_size
        }


class DeepSeekMoEBlock(nn.Module):
    """
    Complete Transformer block with DeepSeekMoE.
    
    Structure:
        x -> LN -> Self-Attention -> + -> LN -> MoE -> + -> output
                         |________________|      |_________|
                              residual            residual
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_shared_experts: int = 2,
        num_routed_experts: int = 64,
        num_experts_per_token: int = 6,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MoE Layer
        self.moe = DeepSeekMoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_shared_experts=num_shared_experts,
            num_routed_experts=num_routed_experts,
            num_experts_per_token=num_experts_per_token,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE block.
        
        Args:
            hidden_states: Input, shape (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            
        Returns:
            output: Updated hidden states
            aux_loss: Auxiliary balance loss
        """
        # Self-Attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            attn_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MoE
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states, _, aux_loss = self.moe(hidden_states)
        # Note: MoE already includes residual connection
        
        return hidden_states, aux_loss


if __name__ == "__main__":
    print("=== DeepSeekMoE Layer Demo ===\n")
    
    # Configuration (similar to DeepSeekMoE-16B)
    hidden_size = 256
    intermediate_size = 1024
    num_shared_experts = 2
    num_routed_experts = 64
    num_experts_per_token = 6
    
    batch_size = 2
    seq_len = 32
    
    # Create MoE layer
    moe_layer = DeepSeekMoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_shared_experts=num_shared_experts,
        num_routed_experts=num_routed_experts,
        num_experts_per_token=num_experts_per_token
    )
    
    # Print configuration
    print("MoE Configuration:")
    stats = moe_layer.get_expert_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Forward pass
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"\nInput shape: {hidden_states.shape}")
    
    # Training mode (with aux loss)
    moe_layer.train()
    output, router_logits, aux_loss = moe_layer(hidden_states, output_router_logits=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Router logits shape: {router_logits.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in moe_layer.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    
    # Parameters breakdown
    shared_params = sum(
        sum(p.numel() for p in expert.parameters())
        for expert in moe_layer.shared_experts
    )
    routed_params = sum(
        sum(p.numel() for p in expert.parameters())
        for expert in moe_layer.routed_experts
    )
    router_params = sum(p.numel() for p in moe_layer.router.parameters())
    
    print(f"  Shared experts: {shared_params / 1e6:.2f}M ({100*shared_params/total_params:.1f}%)")
    print(f"  Routed experts: {routed_params / 1e6:.2f}M ({100*routed_params/total_params:.1f}%)")
    print(f"  Router: {router_params / 1e6:.4f}M ({100*router_params/total_params:.2f}%)")
    
    # Test complete block
    print("\n=== DeepSeekMoE Block Demo ===\n")
    
    block = DeepSeekMoEBlock(
        hidden_size=hidden_size,
        num_heads=8,
        intermediate_size=intermediate_size,
        num_shared_experts=num_shared_experts,
        num_routed_experts=num_routed_experts,
        num_experts_per_token=num_experts_per_token
    )
    
    output, aux_loss = block(hidden_states)
    print(f"Block output shape: {output.shape}")
    print(f"Block aux loss: {aux_loss.item():.4f}")
    
    block_params = sum(p.numel() for p in block.parameters())
    print(f"Block total parameters: {block_params / 1e6:.2f}M")
