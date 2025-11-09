"""
位置编码模块
包含: PositionalEncoding
"""
import torch
import torch.nn as nn
from torch import Tensor
import math


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码
    
    使用固定的正弦和余弦函数生成位置编码, 无需学习参数
    
    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    参数:
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout 概率
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        
        # 位置索引 (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # 频率项: 10000^(-2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度使用 sin, 奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        
        # 注册为 buffer (不参与梯度更新)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量, 形状 (B, T, d_model)
        
        返回:
            加上位置编码后的张量, 形状 (B, T, d_model)
        """
        T = x.size(1)
        # 将位置编码加到输入上 (广播)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)
