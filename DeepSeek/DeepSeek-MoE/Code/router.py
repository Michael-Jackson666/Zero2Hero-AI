"""
Router / Gating Network for DeepSeekMoE Architecture

This module implements the Top-K routing mechanism that selects which experts
process each token. Includes both standard Top-K gating and noisy Top-K gating.

Reference: DeepSeekMoE: Towards Ultimate Expert Specialization (arXiv:2401.06066)

Key equations:
    s_{i,t} = Softmax_i(u_t^T @ e_i)           # Affinity score
    g_{i,t} = s_{i,t} if in TopK else 0        # Sparse gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class TopKRouter(nn.Module):
    """
    Top-K Router for selecting experts.
    
    Each token computes affinity scores with all expert centroids,
    then selects the top-K experts with highest scores.
    
    s_{i,t} = Softmax(u_t^T @ e_i)
    g_{i,t} = s_{i,t} if s_{i,t} in TopK else 0
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        normalize_expert_weights: bool = True
    ):
        """
        Args:
            hidden_size: Model hidden dimension (d)
            num_experts: Total number of routed experts (mN - K_s)
            top_k: Number of experts to activate per token (mK - K_s)
            normalize_expert_weights: Whether to renormalize gate weights
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_expert_weights = normalize_expert_weights
        
        # Expert centroids (embeddings): e_i for each expert
        # Shape: (num_experts, hidden_size)
        self.expert_centroids = nn.Parameter(
            torch.empty(num_experts, hidden_size)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize expert centroids."""
        nn.init.kaiming_uniform_(self.expert_centroids, a=math.sqrt(5))
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: Input tensor, shape (batch, seq_len, hidden_size)
            
        Returns:
            gate_weights: Sparse gate values, shape (batch, seq_len, num_experts)
            expert_indices: Top-K expert indices, shape (batch, seq_len, top_k)
            router_logits: Raw logits for auxiliary loss, shape (batch, seq_len, num_experts)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute affinity scores: u_t^T @ e_i for all experts
        # hidden_states: (batch, seq, hidden_size)
        # expert_centroids: (num_experts, hidden_size)
        # router_logits: (batch, seq, num_experts)
        router_logits = torch.matmul(
            hidden_states,
            self.expert_centroids.t()
        )
        
        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-K experts
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        
        # Optionally renormalize weights
        if self.normalize_expert_weights:
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Create sparse gate tensor
        gate_weights = torch.zeros_like(router_probs)
        gate_weights.scatter_(-1, top_k_indices, top_k_weights)
        
        return gate_weights, top_k_indices, router_logits


class NoisyTopKRouter(nn.Module):
    """
    Noisy Top-K Router with learnable noise for better load balancing.
    
    Adds Gaussian noise before Top-K selection to encourage exploration
    and prevent routing collapse.
    
    H(x)_i = (x @ W_g)_i + StandardNormal() * Softplus((x @ W_noise)_i)
    G(x) = Softmax(KeepTopK(H(x), k))
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        noise_std: float = 1.0,
        normalize_expert_weights: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.normalize_expert_weights = normalize_expert_weights
        
        # Gate projection
        self.w_gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Noise projection (learnable noise magnitude)
        self.w_noise = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts with optional noise.
        
        Args:
            hidden_states: Input tensor, shape (batch, seq_len, hidden_size)
            add_noise: Whether to add noise (typically True during training)
            
        Returns:
            gate_weights: Sparse gate values
            expert_indices: Top-K expert indices
            router_logits: Raw logits for auxiliary loss
        """
        # Compute gate logits
        router_logits = self.w_gate(hidden_states)  # (batch, seq, num_experts)
        
        # Add noise during training
        if add_noise and self.training:
            noise_logits = self.w_noise(hidden_states)
            noise_std = F.softplus(noise_logits)
            noise = torch.randn_like(router_logits) * noise_std * self.noise_std
            router_logits = router_logits + noise
        
        # Apply softmax
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-K experts
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        
        # Renormalize
        if self.normalize_expert_weights:
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Create sparse gate tensor
        gate_weights = torch.zeros_like(router_probs)
        gate_weights.scatter_(-1, top_k_indices, top_k_weights)
        
        return gate_weights, top_k_indices, router_logits


class LoadBalanceLoss(nn.Module):
    """
    Auxiliary loss for load balancing across experts.
    
    Expert-Level Balance Loss:
        L_ExpBal = α * Σ_i (f_i * P_i)
        
    Where:
        f_i = fraction of tokens routed to expert i
        P_i = average probability assigned to expert i
        
    This encourages uniform distribution of tokens across experts.
    """
    
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        alpha: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        
    def forward(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balance loss.
        
        Args:
            router_logits: Raw router logits, shape (batch, seq, num_experts)
            expert_indices: Selected expert indices, shape (batch, seq, top_k)
            
        Returns:
            Scalar loss value
        """
        batch_size, seq_len, _ = router_logits.shape
        num_tokens = batch_size * seq_len
        
        # f_i: fraction of tokens selecting each expert
        # Count how many times each expert is selected
        expert_indices_flat = expert_indices.view(-1)  # (batch * seq * top_k,)
        
        # One-hot encode and sum
        expert_mask = F.one_hot(expert_indices_flat, self.num_experts).float()
        expert_counts = expert_mask.sum(dim=0)  # (num_experts,)
        
        # Normalize to get fraction
        f_i = expert_counts / (num_tokens * self.top_k / self.num_experts)
        
        # P_i: average probability for each expert
        router_probs = F.softmax(router_logits, dim=-1)  # (batch, seq, num_experts)
        P_i = router_probs.mean(dim=[0, 1])  # (num_experts,)
        
        # Balance loss
        loss = self.alpha * (f_i * P_i).sum() * self.num_experts
        
        return loss


class DeviceBalanceLoss(nn.Module):
    """
    Device-Level Balance Loss for distributed training.
    
    Instead of balancing individual experts, this balances the load
    across groups of experts (deployed on different devices).
    
    L_DevBal = α * Σ_d (f'_d * P'_d)
    """
    
    def __init__(
        self,
        num_experts: int,
        num_devices: int,
        top_k: int,
        alpha: float = 0.05
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_devices = num_devices
        self.experts_per_device = num_experts // num_devices
        self.top_k = top_k
        self.alpha = alpha
        
    def forward(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute device-level balance loss.
        """
        batch_size, seq_len, _ = router_logits.shape
        
        # Map expert indices to device indices
        device_indices = expert_indices // self.experts_per_device
        
        # Compute f'_d and P'_d for each device
        device_indices_flat = device_indices.view(-1)
        device_mask = F.one_hot(device_indices_flat, self.num_devices).float()
        device_counts = device_mask.sum(dim=0)
        
        f_d = device_counts / (device_counts.sum() + 1e-9)
        
        # P'_d: sum of probabilities for experts on each device
        router_probs = F.softmax(router_logits, dim=-1)
        P_d = torch.zeros(self.num_devices, device=router_logits.device)
        for d in range(self.num_devices):
            start_idx = d * self.experts_per_device
            end_idx = start_idx + self.experts_per_device
            P_d[d] = router_probs[:, :, start_idx:end_idx].sum(dim=-1).mean()
        
        loss = self.alpha * (f_d * P_d).sum() * self.num_devices
        
        return loss


if __name__ == "__main__":
    print("=== Router Demo ===\n")
    
    # Configuration
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_experts = 64
    top_k = 6
    
    # Standard Top-K Router
    router = TopKRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k
    )
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    gate_weights, expert_indices, router_logits = router(hidden_states)
    
    print("Top-K Router:")
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Gate weights shape: {gate_weights.shape}")
    print(f"  Expert indices shape: {expert_indices.shape}")
    print(f"  Non-zero gates per token: {(gate_weights > 0).sum(dim=-1).float().mean():.1f}")
    
    # Noisy Router
    print("\nNoisy Top-K Router:")
    noisy_router = NoisyTopKRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k
    )
    noisy_router.train()
    
    gate_weights_noisy, _, _ = noisy_router(hidden_states, add_noise=True)
    print(f"  Non-zero gates per token: {(gate_weights_noisy > 0).sum(dim=-1).float().mean():.1f}")
    
    # Load balance loss
    print("\nLoad Balance Loss:")
    balance_loss = LoadBalanceLoss(num_experts=num_experts, top_k=top_k, alpha=0.01)
    loss = balance_loss(router_logits, expert_indices)
    print(f"  Expert-level balance loss: {loss.item():.4f}")
    
    # Check expert utilization
    expert_counts = torch.zeros(num_experts)
    for idx in expert_indices.view(-1):
        expert_counts[idx] += 1
    print(f"  Expert utilization std: {expert_counts.std():.2f}")
    print(f"  Min/Max expert count: {expert_counts.min():.0f}/{expert_counts.max():.0f}")
