"""
Transformer 模块
包含完整的 Transformer Encoder-Decoder 实现

主要组件:
- attention.py: 注意力机制 (ScaledDotProductAttention, MultiHeadAttention)
- embedding.py: 位置编码 (PositionalEncoding)
- feedforward.py: 前馈网络和归一化 (FeedForward, ResidualLayerNorm)
- mask.py: 掩码工具函数 (make_pad_mask, make_subsequent_mask)
- layers.py: Encoder/Decoder 层 (EncoderLayer, DecoderLayer)
- transformer.py: 完整模型 (Transformer)

使用示例:
    from transformer import Transformer
    
    model = Transformer(
        src_vocab=10000,
        tgt_vocab=10000,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6
    )
    
    # 训练
    logits = model(src, tgt_inp, src_pad, tgt_pad)
    
    # 推理
    output = model.greedy_decode(src, bos_id=1, eos_id=2, max_new_tokens=50)
"""

from .attention import ScaledDotProductAttention, MultiHeadAttention
from .embedding import PositionalEncoding
from .feedforward import FeedForward, ResidualLayerNorm
from .mask import make_pad_mask, make_subsequent_mask
from .layers import EncoderLayer, DecoderLayer
from .transformer import Transformer

__all__ = [
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'PositionalEncoding',
    'FeedForward',
    'ResidualLayerNorm',
    'make_pad_mask',
    'make_subsequent_mask',
    'EncoderLayer',
    'DecoderLayer',
    'Transformer',
]

__version__ = '1.0.0'
