"""
掩码工具函数
包含: make_pad_mask, make_subsequent_mask
"""
import torch
from torch import Tensor


def make_pad_mask(q_len: int, k_len: int, q_pad: Tensor | None, k_pad: Tensor | None) -> Tensor:
    """
    构造 Padding Mask
    
    用于屏蔽序列中的填充位置 (PAD token)
    
    参数:
        q_len: Query 序列长度
        k_len: Key 序列长度
        q_pad: Query 的 padding 标记, 形状 (B, T_q), 1 表示 PAD 位置
        k_pad: Key 的 padding 标记, 形状 (B, T_k), 1 表示 PAD 位置
    
    返回:
        mask: 形状 (B, 1, T_q, T_k), 1 表示可见, 0 表示屏蔽
    """
    # 如果都没有 padding, 返回 None
    if q_pad is None and k_pad is None:
        return None
    
    # 处理 q_pad
    if q_pad is None:
        q_mask = torch.zeros_like(k_pad)
    else:
        q_mask = q_pad
    
    # 处理 k_pad
    if k_pad is None:
        k_mask = torch.zeros_like(q_mask)
    else:
        k_mask = k_pad
    
    # 构造可见性掩码 (非 PAD 位置为可见)
    q_visible = (q_mask == 0).unsqueeze(2)  # (B, T_q, 1)
    k_visible = (k_mask == 0).unsqueeze(1)  # (B, 1, T_k)
    
    # 两者都可见时才可见
    mask = q_visible & k_visible  # (B, T_q, T_k)
    
    return mask.unsqueeze(1)  # (B, 1, T_q, T_k)


def make_subsequent_mask(T: int) -> Tensor:
    """
    构造下三角掩码 (Causal Mask)
    
    用于 Decoder 的自注意力, 防止看到未来信息
    位置 i 只能看到位置 <= i 的信息
    
    参数:
        T: 序列长度
    
    返回:
        mask: 形状 (1, 1, T, T), 下三角为 1 (可见), 上三角为 0 (屏蔽)
    
    示例 (T=5):
        [[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1]]
    """
    # 生成下三角矩阵
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    
    # 增加 batch 和 head 维度 (用于广播)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
