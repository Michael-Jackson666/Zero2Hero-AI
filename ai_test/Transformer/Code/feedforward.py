"""
前馈网络和归一化模块
包含: FeedForward, ResidualLayerNorm
"""
import torch
import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    两层全连接网络, 对每个位置独立处理
    
    结构: Linear -> Activation -> Dropout -> Linear
    
    参数:
        d_model: 模型维度
        d_ff: 隐藏层维度 (通常是 d_model 的 4 倍)
        dropout: Dropout 概率
        activation: 激活函数类型 ('relu' 或 'gelu')
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = 'relu'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 选择激活函数
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('activation 必须是 relu 或 gelu')

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量, 形状 (B, T, d_model)
        
        返回:
            输出张量, 形状 (B, T, d_model)
        """
        # (B, T, d_model) -> (B, T, d_ff) -> (B, T, d_model)
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class ResidualLayerNorm(nn.Module):
    """
    残差连接 + Layer Normalization
    
    实现: LayerNorm(x + sublayer(x))  (Post-LN)
    
    参数:
        d_model: 模型维度
        eps: LayerNorm 的数值稳定项
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: Tensor, sublayer_out: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 残差连接的输入, 形状 (B, T, d_model)
            sublayer_out: 子层的输出, 形状 (B, T, d_model)
        
        返回:
            归一化后的输出, 形状 (B, T, d_model)
        """
        # 残差连接 + LayerNorm
        return self.ln(x + sublayer_out)
