import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional


class ScaledDotProductAttention(nn.Moduel):
    def __init__(self, dropout: float=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, mask:Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        d_k = Q.size(-1)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ V
        return out, attn
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0):
        super.__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x:Tensor) -> Tensor:
        # x: (B,T,d_model) -> (B,H,T,d_k)
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        return x
    
    def _combine_heads(self, x:Tensor) -> Tensor:
        # x: (B,H,T,d_k) -> (B,T,d_model)
        B, H, T, d_k = x.shape
        x = x.transpose(1,2).contiguous().view(B, T, H * d_k)
    
    def forwad(self, x_q:Tensor, x_kv:Tensor, mask:Optional[Tensor]=None) -> tuple[Tensor,Tensor]:
        """
        x_q: (B,T_q,d_model)
        x_kv: (B,T_k,d_model)
        mask: (B,1,T_q,T_k) or (B,H,T_q,T_k)
        return: (out,attn) 
        """
        Q = self._split_heads(self.W_q(x_q))
        K = self._split_heads(self.W_k(x_kv))
        V = self._split_heads(self.W_k(x_kv))

        out, attn = self.attn(Q, K, V, mask)
        out = self._combine_heads(out)
        out = self.W_o(out)
        out = self.dropout(out)
        return out, attn
    

class PositionEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000, dropout:float=0.0):
        super.__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)).float() * (-math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:Tensor) -> Tensor:
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)
    

class FeedForwad(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.0, activation:str='relu'):
        super.__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.ReLU()
        else:
            raise ValueError('activation must be relu or gelu')

    def forward(self, x:Tensor) -> Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class ResidualLayerNorm(nn.Module):
    def __init__(self, d_modle:int, eps:float=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d_modle, eps=eps)

    def forward(self, x:Tensor, sublayer_out:Tensor) -> Tensor:
        return self.ln(x + sublayer_out)
    

def make_pad_mask(q_len:int, k_len:int, q_pad:Tensor|None, k_pad:Tensor|None) -> Tensor:
    if q_pad is None and k_pad is None:
        return None
    elif q_pad is None:
        q_mask = torch.zeros_like(k_pad)
    else:
        q_mask = q_pad
    if k_pad is None:
        k_mask = torch.zeros_like(q_mask)
    else:
        k_mask = k_pad
    q_visible = (q_mask==0).unsqueeze(2)
    k_visible = (k_mask==0).unsqueeze(1)
    mask = q_visible & k_visible
    return mask.unsqueeze(1)

def make_subsequent_mask(T:int) -> Tensor:
    return torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)


class EncoderLayer(nn.Module):
    def __ini__(self, d_model:int, num_heads:int, d_ff:int, dropout:float=0.1):
        super.__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwad(d_model, d_ff, dropout)
        self.norm1 = ResidualLayerNorm(d_model)
        self.norm2 = ResidualLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:Tensor, src_mask:Optional[Tensor]=None) -> tuple[Tensor, Tensor]:
        # Self-Attention
        sa_out, sa_w = self.self_attn(x, x, src_mask)
        x = self.norm1(x, self.dropout(sa_out))
        # FFN
        ff_out = self.ffn(x)
        x = self.norm2(x, self.dropout(ff_out))
        return x, sa_w
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, dropout:float=0.1):
        super.__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwad(d_model, d_ff, dropout)
        self.norm1 = ResidualLayerNorm(d_model)
        self.norm2 = ResidualLayerNorm(d_model)
        self.norm3 = ResidualLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y:Tensor, memory:Tensor, tgt_mask:Optional[Tensor], memory_mask:Optional[Tensor]) -> tuple[Tensor, tuple[Tensor,Tensor]]:
        # Masked Self-Attention(Decoder)
        sa_out, sa_w = self.self_attn(y, y, tgt_mask)
        y = self.norm1(y, self.dropout(sa_out))
        # Cross-Attention: Q=decoder, K/V = encoder memory
        ca_out, ca_w = self.cross_attn(y, memory, memory_mask)
        y = self.norm2(y, self.dropout(ca_out))
        ff_out = self.ffn(y)
        y = self.norm3(y, self.dropout(ff_out))
        return y, (sa_w, ca_w)


class Transformer(nn.Module):
    def __init__(self, src_vocab:int, tgt_vocab:int, d_model:int, num_heads:int=8,
                 d_ff:int=512, num_layers:int=4, dropout:float=0.1, max_len:int=512):
        # src_vocab: 源端（输入）词表大小，例如包含 PAD/BOS/EOS 等特殊 token
        # tgt_vocab: 目标端（输出）词表大小，用于最后的线性投影到词表概率
        super.__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_vocab = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionEncoding(d_model, max_len, dropout)

        self.encode_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decode_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, tgt_vocab)

        def encode(self, src:Tensor, src_pad:Optional[Tensor]=None) -> tuple[Tensor, list[Tensor]]:
            x = self.pos_enc(self.src_embed(src))
            attn_weights = []
            src_len = src.size(1)
            src_mask = make_pad_mask(src_len, src_len, src_pad, src_pad)
            for layer in self.encoder_layers:
                x, sa_w = layer(x, src_mask)
                attn_weights.append(sa_w)
            return x, attn_weights
        
        def decode(self, tgt:Tensor, memory:Tensor, src_pad:Optional[Tensor]=None, tgt_pad:Optional[Tensor]=None) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
            y = self.pos_enc(self.tgt_embed(tgt))
            T_t = tgt.size(1)
            B, T_s = memory.size(0), memory.size(1)
            # masks
            pad_mask = make_pad_mask(T_t, T_t, tgt_pad, tgt_pad)
            subs_mask = make_subsequent_mask(T_t).to(y.device)
            tgt_mask = pad_mask & subs_mask if pad_mask is not None else subs_mask
            mem_mask = make_pad_mask(T_t, T_s, tgt_pad, src_pad)

            attn_pairs = []
            for layer in self.decode_layers:
                y, (sa_w, ca_w) = layer(y, memory, tgt_mask, mem_mask)
                attn_pairs.append((sa_w, ca_w))
            return y, attn_pairs
        
        @torch.no_grad()
        def greed_decode(self, src:Tensor, bos_id:int, eos_id:int, max_new_tokesn:int, 
                         src_pad:Optional[Tensor]=None) -> Tensor:
            self.eval()
            memory, _ = self.encode(src, src_pad)
            B = src.size(0)
            ys = torch.full((B, 1), bos_id, dtype=torch.log, device=src.device)
            for _ in range(max_new_tokesn):
                y, _ = self.decode(ys, memory, src_pad, tgt_vocab=None)
                logits = self.out_proj(y)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if (next_token == eos_id).all():
                    break
            return ys
        
