from ScaledDotProductAttention import ScaledDotProductAttention
import torch.nn as nn
from torch import Tensor
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

    def _split_haeds(self, x:Tensor) -> Tensor:
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        return x
