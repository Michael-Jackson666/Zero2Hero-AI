"""
Context-Aware Gating Module for Engram Architecture

This module implements the cross-attention based gating mechanism that determines
how much to trust the retrieved memory embeddings based on the current hidden state.

Reference: DeepSeek's "Conditional Memory via Scalable Lookup" Paper (arXiv:2601.07372)

Key equations:
    k_t = W_K @ e_t                                              # Key projection
    v_t = W_V @ e_t                                              # Value projection
    α_t = σ(RMSNorm(h_t)^T @ RMSNorm(k_t) / sqrt(d))            # Gating coefficient
    ṽ_t = α_t * v_t                                             # Gated value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    
    More efficient than LayerNorm as it doesn't compute mean for centering.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
    
    def forward_no_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize without learnable weight (for gating computation)."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms


class ContextAwareGating(nn.Module):
    """
    Context-Aware Gating: Determines how much to trust retrieved memory.
    
    This is essentially a Cross-Attention variant where:
    - Query: Current hidden state h_t
    - Key/Value: Retrieved memory embedding e_t
    
    The gating coefficient α_t ∈ (0, 1) suppresses noise from hash collisions
    when the memory is irrelevant to the current context.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        num_branches: int = 1,
        shared_value: bool = True
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states (d)
            memory_dim: Dimension of retrieved memory embeddings (d_e)
            num_branches: Number of parallel branches in multi-branch architecture (M)
            shared_value: Whether to share W_V across branches (recommended)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_branches = num_branches
        self.shared_value = shared_value
        self.scale = math.sqrt(hidden_dim)
        
        # Key projection: W_K (one per branch for flexibility)
        # W_K^{(m)} maps memory_dim -> hidden_dim
        self.W_K = nn.ModuleList([
            nn.Linear(memory_dim, hidden_dim, bias=False)
            for _ in range(num_branches)
        ])
        
        # Value projection: W_V (shared across branches to save memory)
        if shared_value:
            self.W_V = nn.Linear(memory_dim, hidden_dim, bias=False)
        else:
            self.W_V = nn.ModuleList([
                nn.Linear(memory_dim, hidden_dim, bias=False)
                for _ in range(num_branches)
            ])
        
        # RMSNorm for normalization (without learnable weights for gating)
        self.norm = RMSNorm(hidden_dim)
        
    def compute_gating(
        self,
        hidden_states: torch.Tensor,
        memory_embeddings: torch.Tensor,
        branch_idx: int = 0
    ) -> torch.Tensor:
        """
        Compute gating coefficient α_t for a specific branch.
        
        α_t^{(m)} = σ(RMSNorm(h_t^{(m)})^T @ RMSNorm(W_K^{(m)} @ e_t) / sqrt(d))
        
        Args:
            hidden_states: Current hidden states h_t, shape (batch, seq, hidden_dim)
            memory_embeddings: Retrieved memory e_t, shape (batch, seq, memory_dim)
            branch_idx: Index of the branch (m)
            
        Returns:
            Gating coefficients α_t, shape (batch, seq, 1)
        """
        # Key projection
        keys = self.W_K[branch_idx](memory_embeddings)  # (batch, seq, hidden_dim)
        
        # Normalize both hidden states and keys
        h_norm = self.norm.forward_no_weight(hidden_states)  # (batch, seq, hidden_dim)
        k_norm = self.norm.forward_no_weight(keys)           # (batch, seq, hidden_dim)
        
        # Compute attention score (dot product)
        # Sum over hidden_dim, result is (batch, seq)
        scores = (h_norm * k_norm).sum(dim=-1) / self.scale
        
        # Apply sigmoid to get gating coefficient in (0, 1)
        alpha = torch.sigmoid(scores).unsqueeze(-1)  # (batch, seq, 1)
        
        return alpha
    
    def compute_value(
        self,
        memory_embeddings: torch.Tensor,
        branch_idx: int = 0
    ) -> torch.Tensor:
        """
        Compute value projection v_t = W_V @ e_t.
        
        Args:
            memory_embeddings: Retrieved memory e_t, shape (batch, seq, memory_dim)
            branch_idx: Index of the branch (only used if not shared)
            
        Returns:
            Value vectors v_t, shape (batch, seq, hidden_dim)
        """
        if self.shared_value:
            return self.W_V(memory_embeddings)
        else:
            return self.W_V[branch_idx](memory_embeddings)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_embeddings: torch.Tensor,
        branch_idx: int = 0,
        return_gating: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full gating forward pass.
        
        Args:
            hidden_states: Current hidden states, shape (batch, seq, hidden_dim)
            memory_embeddings: Retrieved memory, shape (batch, seq, memory_dim)
            branch_idx: Index of the branch
            return_gating: Whether to return gating coefficients
            
        Returns:
            Gated values ṽ_t = α_t * v_t, shape (batch, seq, hidden_dim)
            Optionally, gating coefficients α_t
        """
        # Compute gating coefficient
        alpha = self.compute_gating(hidden_states, memory_embeddings, branch_idx)
        
        # Compute value projection
        values = self.compute_value(memory_embeddings, branch_idx)
        
        # Apply gating
        gated_values = alpha * values
        
        if return_gating:
            return gated_values, alpha
        return gated_values, None


class MultibranchGating(nn.Module):
    """
    Multi-branch Gating for modern LLM architectures like DeepSeek-V3.
    
    Implements parameter sharing strategy:
    - Shared: Embedding tables E and Value projection W_V
    - Independent: Key projections {W_K^{(m)}} for each branch
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        num_branches: int = 4
    ):
        super().__init__()
        self.num_branches = num_branches
        
        # Shared value projection
        self.W_V = nn.Linear(memory_dim, hidden_dim, bias=False)
        
        # Independent key projections for each branch
        self.W_K_list = nn.ModuleList([
            nn.Linear(memory_dim, hidden_dim, bias=False)
            for _ in range(num_branches)
        ])
        
        self.norm = RMSNorm(hidden_dim)
        self.scale = math.sqrt(hidden_dim)
        
    def forward(
        self,
        hidden_states_list: list,
        memory_embeddings: torch.Tensor
    ) -> list:
        """
        Compute gated outputs for all branches.
        
        Args:
            hidden_states_list: List of hidden states for each branch,
                               each shape (batch, seq, hidden_dim)
            memory_embeddings: Shared retrieved memory, shape (batch, seq, memory_dim)
            
        Returns:
            List of gated outputs for each branch
        """
        assert len(hidden_states_list) == self.num_branches
        
        # Compute shared value projection once
        values = self.W_V(memory_embeddings)  # (batch, seq, hidden_dim)
        
        outputs = []
        for m, hidden_states in enumerate(hidden_states_list):
            # Compute branch-specific key
            keys = self.W_K_list[m](memory_embeddings)
            
            # Normalize
            h_norm = self.norm.forward_no_weight(hidden_states)
            k_norm = self.norm.forward_no_weight(keys)
            
            # Gating
            scores = (h_norm * k_norm).sum(dim=-1) / self.scale
            alpha = torch.sigmoid(scores).unsqueeze(-1)
            
            # Apply gating to shared value
            gated_output = alpha * values
            outputs.append(gated_output)
            
        return outputs


if __name__ == "__main__":
    print("=== Context-Aware Gating Demo ===\n")
    
    # Configuration
    batch_size = 2
    seq_len = 16
    hidden_dim = 256
    memory_dim = 768  # (max_n - min_n + 1) * num_hash_heads * embedding_dim
    num_branches = 4
    
    # Create gating module
    gating = ContextAwareGating(
        hidden_dim=hidden_dim,
        memory_dim=memory_dim,
        num_branches=num_branches,
        shared_value=True
    )
    
    # Simulate inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    memory_embeddings = torch.randn(batch_size, seq_len, memory_dim)
    
    # Forward pass
    gated_values, alpha = gating(
        hidden_states, 
        memory_embeddings, 
        branch_idx=0,
        return_gating=True
    )
    
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Memory embeddings shape: {memory_embeddings.shape}")
    print(f"Gated values shape: {gated_values.shape}")
    print(f"Gating coefficients shape: {alpha.shape}")
    print(f"Gating coefficient stats: min={alpha.min():.3f}, max={alpha.max():.3f}, mean={alpha.mean():.3f}")
    
    # Multi-branch demo
    print("\n=== Multi-branch Gating Demo ===\n")
    
    multibranch = MultibranchGating(
        hidden_dim=hidden_dim,
        memory_dim=memory_dim,
        num_branches=num_branches
    )
    
    hidden_states_list = [
        torch.randn(batch_size, seq_len, hidden_dim)
        for _ in range(num_branches)
    ]
    
    outputs = multibranch(hidden_states_list, memory_embeddings)
    
    print(f"Number of branches: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Branch {i} output shape: {out.shape}")
