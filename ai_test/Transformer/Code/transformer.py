"""
完整 Transformer 模型
包含: Transformer (Encoder-Decoder 架构)
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from embedding import PositionalEncoding
from layers import EncoderLayer, DecoderLayer
from mask import make_pad_mask, make_subsequent_mask


class Transformer(nn.Module):
    """
    完整的 Transformer Encoder-Decoder 模型
    
    参数:
        src_vocab: 源端词表大小 (包含 PAD/BOS/EOS 等特殊 token)
        tgt_vocab: 目标端词表大小
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: FFN 隐藏层维度
        num_layers: Encoder/Decoder 层数
        dropout: Dropout 概率
        max_len: 最大序列长度
    """
    def __init__(
        self, 
        src_vocab: int, 
        tgt_vocab: int, 
        d_model: int = 256, 
        num_heads: int = 8,
        d_ff: int = 512, 
        num_layers: int = 4, 
        dropout: float = 0.1, 
        max_len: int = 512
    ):
        super().__init__()
        
        # Embedding 层
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # Encoder 层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # Decoder 层堆叠
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.out_proj = nn.Linear(d_model, tgt_vocab)

    def encode(self, src: Tensor, src_pad: Optional[Tensor] = None) -> tuple[Tensor, list[Tensor]]:
        """
        Encoder 前向传播
        
        参数:
            src: 源序列, 形状 (B, T_s)
            src_pad: 源序列 padding 标记, 形状 (B, T_s), 1 表示 PAD
        
        返回:
            x: Encoder 输出, 形状 (B, T_s, d_model)
            attn_weights: 每层的注意力权重列表
        """
        # Embedding + Positional Encoding
        x = self.pos_enc(self.src_embed(src))  # (B, T_s, d_model)
        
        # 构造 padding mask
        src_len = src.size(1)
        src_mask = make_pad_mask(src_len, src_len, src_pad, src_pad)  # (B, 1, T_s, T_s)
        
        # 通过所有 Encoder 层
        attn_weights = []
        for layer in self.encoder_layers:
            x, sa_w = layer(x, src_mask)
            attn_weights.append(sa_w)
        
        return x, attn_weights
    
    def decode(
        self, 
        tgt: Tensor, 
        memory: Tensor, 
        src_pad: Optional[Tensor] = None, 
        tgt_pad: Optional[Tensor] = None
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """
        Decoder 前向传播
        
        参数:
            tgt: 目标序列, 形状 (B, T_t)
            memory: Encoder 输出, 形状 (B, T_s, d_model)
            src_pad: 源序列 padding 标记, 形状 (B, T_s)
            tgt_pad: 目标序列 padding 标记, 形状 (B, T_t)
        
        返回:
            y: Decoder 输出, 形状 (B, T_t, d_model)
            attn_pairs: 每层的 (自注意力权重, 交叉注意力权重) 列表
        """
        # Embedding + Positional Encoding
        y = self.pos_enc(self.tgt_embed(tgt))  # (B, T_t, d_model)
        
        T_t = tgt.size(1)
        B, T_s = memory.size(0), memory.size(1)
        
        # 构造掩码
        # 1. Padding mask
        pad_mask = make_pad_mask(T_t, T_t, tgt_pad, tgt_pad)  # (B, 1, T_t, T_t)
        
        # 2. Causal mask (下三角)
        subs_mask = make_subsequent_mask(T_t).to(y.device)  # (1, 1, T_t, T_t)
        
        # 3. 目标序列掩码 = padding mask & causal mask
        tgt_mask = pad_mask & subs_mask if pad_mask is not None else subs_mask
        
        # 4. 交叉注意力掩码 (只需屏蔽源序列的 padding)
        mem_mask = make_pad_mask(T_t, T_s, tgt_pad, src_pad)  # (B, 1, T_t, T_s)
        
        # 通过所有 Decoder 层
        attn_pairs = []
        for layer in self.decoder_layers:
            y, (sa_w, ca_w) = layer(y, memory, tgt_mask, mem_mask)
            attn_pairs.append((sa_w, ca_w))
        
        return y, attn_pairs
    
    def forward(
        self, 
        src: Tensor, 
        tgt_inp: Tensor, 
        src_pad: Optional[Tensor] = None, 
        tgt_pad: Optional[Tensor] = None
    ) -> Tensor:
        """
        完整的前向传播 (训练时使用)
        
        参数:
            src: 源序列, 形状 (B, T_s)
            tgt_inp: 目标序列输入 (不包含最后一个 token), 形状 (B, T_t)
            src_pad: 源序列 padding 标记
            tgt_pad: 目标序列 padding 标记
        
        返回:
            logits: 输出 logits, 形状 (B, T_t, tgt_vocab)
        """
        # Encode
        memory, _ = self.encode(src, src_pad)
        
        # Decode
        y, _ = self.decode(tgt_inp, memory, src_pad, tgt_pad)
        
        # 投影到词表
        logits = self.out_proj(y)  # (B, T_t, tgt_vocab)
        
        return logits
    
    @torch.no_grad()
    def greedy_decode(
        self, 
        src: Tensor, 
        bos_id: int, 
        eos_id: int, 
        max_new_tokens: int,
        src_pad: Optional[Tensor] = None
    ) -> Tensor:
        """
        贪心解码 (推理时使用)
        
        每步选择概率最高的 token, 直到生成 EOS 或达到最大长度
        
        参数:
            src: 源序列, 形状 (B, T_s)
            bos_id: 开始符 BOS 的 token ID
            eos_id: 结束符 EOS 的 token ID
            max_new_tokens: 最多生成的新 token 数量
            src_pad: 源序列 padding 标记
        
        返回:
            ys: 生成的序列, 形状 (B, T_out), 包含 BOS 和生成的 tokens
        """
        self.eval()
        
        # Encode 一次 (不需要重复)
        memory, _ = self.encode(src, src_pad)
        
        B = src.size(0)
        # 初始化输出序列 (只有 BOS)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        
        # 逐步生成
        for _ in range(max_new_tokens):
            # Decode 当前序列
            y, _ = self.decode(ys, memory, src_pad, tgt_pad=None)
            
            # 投影到词表
            logits = self.out_proj(y)  # (B, T, tgt_vocab)
            
            # 取最后一个位置的 logits, 选择概率最大的 token
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1)
            
            # 拼接到输出序列
            ys = torch.cat([ys, next_token], dim=1)
            
            # 如果所有样本都生成了 EOS, 提前终止
            if (next_token == eos_id).all():
                break
        
        return ys
