"""
注意力机制模块
包含: ScaledDotProductAttention, MultiHeadAttention
"""
import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    
    计算公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    
    参数:
        dropout: Dropout 概率
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        前向传播
        
        参数:
            Q: Query 张量, 形状 (B, H, T_q, d_k)
            K: Key 张量, 形状 (B, H, T_k, d_k)
            V: Value 张量, 形状 (B, H, T_k, d_v)
            mask: 掩码张量, 形状 (B, 1, T_q, T_k) 或 (B, H, T_q, T_k)
                  1 表示可见, 0 表示遮挡
        
        返回:
            out: 注意力输出, 形状 (B, H, T_q, d_v)
            attn: 注意力权重, 形状 (B, H, T_q, T_k)
        """
        d_k = Q.size(-1)
        # 计算注意力分数: QK^T / sqrt(d_k)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, T_q, T_k)
        
        # 应用掩码 (将被遮挡位置设为 -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax 归一化
        attn = scores.softmax(dim=-1)  # (B, H, T_q, T_k)
        attn = self.dropout(attn)
        
        # 加权求和
        out = attn @ V  # (B, H, T_q, d_v)
        return out, attn


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    将输入投影到多个子空间, 并行计算注意力, 最后拼接
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        dropout: Dropout 概率
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        将输入张量分割为多个头
        
        参数:
            x: 输入张量, 形状 (B, T, d_model)
        
        返回:
            分头后的张量, 形状 (B, H, T, d_k)
        """
        B, T, _ = x.shape
        # (B, T, d_model) -> (B, T, H, d_k) -> (B, H, T, d_k)
        x = x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        return x
    
    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        将多个头的输出拼接
        
        参数:
            x: 多头张量, 形状 (B, H, T, d_k)
        
        返回:
            拼接后的张量, 形状 (B, T, d_model)
        """
        B, H, T, d_k = x.shape
        # (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, d_model)
        x = x.transpose(1, 2).contiguous().view(B, T, H * d_k)
        return x
    
    def forward(self, x_q: Tensor, x_kv: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        前向传播
        
        参数:
            x_q: Query 输入, 形状 (B, T_q, d_model)
            x_kv: Key/Value 输入, 形状 (B, T_k, d_model)
            mask: 掩码, 形状 (B, 1, T_q, T_k) 或 (B, H, T_q, T_k)
        
        返回:
            out: 输出张量, 形状 (B, T_q, d_model)
            attn: 注意力权重, 形状 (B, H, T_q, T_k)
        """
        # 线性投影并分头
        Q = self._split_heads(self.W_q(x_q))   # (B, H, T_q, d_k)
        K = self._split_heads(self.W_k(x_kv))  # (B, H, T_k, d_k)
        V = self._split_heads(self.W_v(x_kv))  # (B, H, T_k, d_k)

        # 计算注意力
        out, attn = self.attn(Q, K, V, mask)   # out: (B, H, T_q, d_k)
        
        # 合并多头
        out = self._combine_heads(out)         # (B, T_q, d_model)
        
        # 输出投影
        out = self.W_o(out)
        out = self.dropout(out)
        
        return out, attn
