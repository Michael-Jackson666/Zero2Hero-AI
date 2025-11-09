"""
缩放点积注意力模块
包含: ScaledDotProductAttention
"""
import math
from typing import Optional
import torch.nn as nn
from torch import Tensor


class ScaledDotProductAttention(nn.Module):  # 修正: nn.Moduel → nn.Module
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
    

