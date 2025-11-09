"""
Encoder 和 Decoder 层
包含: EncoderLayer, DecoderLayer
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from attention import MultiHeadAttention
from feedforward import FeedForward, ResidualLayerNorm


class EncoderLayer(nn.Module):
    """
    Transformer Encoder 层
    
    结构:
        1. Multi-Head Self-Attention + Residual + LayerNorm
        2. Feed-Forward Network + Residual + LayerNorm
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: FFN 隐藏层维度
        dropout: Dropout 概率
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = ResidualLayerNorm(d_model)
        self.norm2 = ResidualLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        前向传播
        
        参数:
            x: 输入张量, 形状 (B, T, d_model)
            src_mask: 源序列掩码, 形状 (B, 1, T, T)
        
        返回:
            x: 输出张量, 形状 (B, T, d_model)
            sa_w: 自注意力权重, 形状 (B, H, T, T)
        """
        # 1. Self-Attention
        sa_out, sa_w = self.self_attn(x, x, src_mask)
        x = self.norm1(x, self.dropout(sa_out))
        
        # 2. Feed-Forward Network
        ff_out = self.ffn(x)
        x = self.norm2(x, self.dropout(ff_out))
        
        return x, sa_w


class DecoderLayer(nn.Module):
    """
    Transformer Decoder 层
    
    结构:
        1. Masked Multi-Head Self-Attention + Residual + LayerNorm
        2. Multi-Head Cross-Attention + Residual + LayerNorm
        3. Feed-Forward Network + Residual + LayerNorm
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: FFN 隐藏层维度
        dropout: Dropout 概率
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = ResidualLayerNorm(d_model)
        self.norm2 = ResidualLayerNorm(d_model)
        self.norm3 = ResidualLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        y: Tensor, 
        memory: Tensor, 
        tgt_mask: Optional[Tensor], 
        memory_mask: Optional[Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        前向传播
        
        参数:
            y: 目标序列输入, 形状 (B, T_t, d_model)
            memory: Encoder 的输出 (记忆), 形状 (B, T_s, d_model)
            tgt_mask: 目标序列掩码 (causal + padding), 形状 (B, 1, T_t, T_t)
            memory_mask: 记忆掩码 (padding), 形状 (B, 1, T_t, T_s)
        
        返回:
            y: 输出张量, 形状 (B, T_t, d_model)
            (sa_w, ca_w): 自注意力和交叉注意力权重
        """
        # 1. Masked Self-Attention (Decoder)
        sa_out, sa_w = self.self_attn(y, y, tgt_mask)
        y = self.norm1(y, self.dropout(sa_out))
        
        # 2. Cross-Attention: Q=decoder, K/V=encoder memory
        ca_out, ca_w = self.cross_attn(y, memory, memory_mask)
        y = self.norm2(y, self.dropout(ca_out))
        
        # 3. Feed-Forward Network
        ff_out = self.ffn(y)
        y = self.norm3(y, self.dropout(ff_out))
        
        return y, (sa_w, ca_w)
