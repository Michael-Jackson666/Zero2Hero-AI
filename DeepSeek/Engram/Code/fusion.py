"""
Fusion Module for Engram Architecture

This module implements the fusion layer that processes gated memory values
through depthwise convolution for local temporal smoothing, then adds back
to the residual stream.

Reference: DeepSeek's "Conditional Memory via Scalable Lookup" Paper (arXiv:2601.07372)

Key equations:
    Y = SiLU(Conv1D(RMSNorm(ṽ_t))) + ṽ_t    # Fusion with residual
    H^{(ℓ)} ← H^{(ℓ)} + Y                    # Add to residual stream
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (SiLU) / Swish activation function.
    
    SiLU(x) = x * σ(x) = x * (1 / (1 + exp(-x)))
    
    Properties:
    - Smooth and non-monotonic (unlike ReLU)
    - Allows small negative values to pass through
    - Implicit gating property (outputs ~0 for very negative x)
    - Better gradient flow in deep networks
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class DepthwiseConv1D(nn.Module):
    """
    Depthwise 1D Convolution for local temporal smoothing.
    
    Each channel is convolved independently with its own kernel,
    enabling local context aggregation without cross-channel mixing.
    
    For position t and channel j:
        Conv1D(X)_{t,j} = Σ_{i=0}^{k-1} W_i * X_{t - floor(k/2) + i, j}
    
    This smooths the sequence locally, mitigating discontinuities
    from N-gram retrieval.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: str = 'same',
        causal: bool = True
    ):
        """
        Args:
            channels: Number of input/output channels (d)
            kernel_size: Size of convolutional kernel (k), typically 3 or 5
            padding: Padding mode ('same' for output size = input size)
            causal: If True, use causal (left) padding for autoregressive models
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.causal = causal
        
        # Depthwise convolution: groups=channels means each channel has its own kernel
        # Weight shape: (channels, 1, kernel_size)
        if causal:
            # For causal: pad only on the left
            self.padding = kernel_size - 1
            self.conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=0,  # We handle padding manually for causal
                groups=channels,
                bias=False
            )
        else:
            self.padding = kernel_size // 2
            self.conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=self.padding,
                groups=channels,
                bias=False
            )
        
        # Initialize with small values
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply depthwise convolution.
        
        Args:
            x: Input tensor, shape (batch, seq_len, channels)
            
        Returns:
            Output tensor, shape (batch, seq_len, channels)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        
        if self.causal:
            # Pad on the left for causal convolution
            x = F.pad(x, (self.padding, 0))
        
        out = self.conv(x)
        
        if self.causal:
            # Remove extra positions from the right
            out = out[:, :, :x.size(-1) - self.padding]
        
        return out.transpose(1, 2)  # (batch, seq_len, channels)


class EngramFusion(nn.Module):
    """
    Engram Fusion Layer: Processes gated memory and fuses with residual stream.
    
    Architecture:
        1. RMSNorm: Normalize gated values
        2. DepthwiseConv1D: Local temporal smoothing
        3. SiLU: Non-linear activation
        4. Residual: Add back the original gated values
    
    Y = SiLU(Conv1D(RMSNorm(ṽ_t))) + ṽ_t
    """
    
    def __init__(
        self,
        hidden_dim: int,
        conv_kernel_size: int = 3,
        causal: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            hidden_dim: Dimension of the hidden states (d)
            conv_kernel_size: Kernel size for depthwise convolution
            causal: Whether to use causal convolution
            dropout: Dropout probability after fusion
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Normalization
        self.norm = RMSNorm(hidden_dim)
        
        # Depthwise convolution for local smoothing
        self.conv = DepthwiseConv1D(
            channels=hidden_dim,
            kernel_size=conv_kernel_size,
            causal=causal
        )
        
        # Activation
        self.activation = SiLU()
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(
        self,
        gated_values: torch.Tensor,
        return_pre_residual: bool = False
    ) -> torch.Tensor:
        """
        Apply fusion to gated memory values.
        
        Args:
            gated_values: Gated values ṽ_t, shape (batch, seq_len, hidden_dim)
            return_pre_residual: If True, also return the output before residual addition
            
        Returns:
            Fused output Y, shape (batch, seq_len, hidden_dim)
        """
        # Step 1: Normalize
        normed = self.norm(gated_values)
        
        # Step 2: Depthwise convolution
        conv_out = self.conv(normed)
        
        # Step 3: SiLU activation
        activated = self.activation(conv_out)
        
        # Step 4: Apply dropout
        activated = self.dropout(activated)
        
        # Step 5: Residual connection
        output = activated + gated_values
        
        if return_pre_residual:
            return output, activated
        return output


class EngramResidualIntegration(nn.Module):
    """
    Full residual integration of Engram output into the transformer hidden states.
    
    H^{(ℓ)} ← H^{(ℓ)} + Y
    
    Includes optional scaling for stable training.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        fusion_kernel_size: int = 3,
        causal: bool = True,
        residual_scale: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Fusion layer
        self.fusion = EngramFusion(
            hidden_dim=hidden_dim,
            conv_kernel_size=fusion_kernel_size,
            causal=causal,
            dropout=dropout
        )
        
        # Learnable or fixed scaling factor
        self.residual_scale = residual_scale
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        gated_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate Engram output into hidden states.
        
        Args:
            hidden_states: Transformer hidden states H^{(ℓ)}, shape (batch, seq, hidden_dim)
            gated_values: Gated memory values ṽ_t, shape (batch, seq, hidden_dim)
            
        Returns:
            Updated hidden states, shape (batch, seq, hidden_dim)
        """
        # Apply fusion to get Y
        fused_output = self.fusion(gated_values)
        
        # Add to residual stream with optional scaling
        updated_states = hidden_states + self.residual_scale * fused_output
        
        return updated_states


if __name__ == "__main__":
    print("=== Engram Fusion Demo ===\n")
    
    # Configuration
    batch_size = 2
    seq_len = 16
    hidden_dim = 256
    
    # Create fusion module
    fusion = EngramFusion(
        hidden_dim=hidden_dim,
        conv_kernel_size=3,
        causal=True
    )
    
    # Simulate gated values
    gated_values = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Forward pass
    output, pre_residual = fusion(gated_values, return_pre_residual=True)
    
    print(f"Gated values shape: {gated_values.shape}")
    print(f"Pre-residual output shape: {pre_residual.shape}")
    print(f"Final output shape: {output.shape}")
    
    # Verify residual connection
    residual_component = output - gated_values
    print(f"\nResidual component (should match pre_residual):")
    print(f"  Max difference: {(residual_component - pre_residual).abs().max():.6f}")
    
    # Full integration demo
    print("\n=== Residual Integration Demo ===\n")
    
    integration = EngramResidualIntegration(
        hidden_dim=hidden_dim,
        fusion_kernel_size=3,
        causal=True,
        residual_scale=1.0
    )
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    updated_states = integration(hidden_states, gated_values)
    
    print(f"Original hidden states shape: {hidden_states.shape}")
    print(f"Updated hidden states shape: {updated_states.shape}")
    
    # Verify the update
    difference = updated_states - hidden_states
    print(f"Update magnitude: mean={difference.abs().mean():.4f}, std={difference.std():.4f}")
