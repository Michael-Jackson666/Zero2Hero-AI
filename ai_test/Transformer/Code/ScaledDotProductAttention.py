import math
from typing import Optional
import torch.nn as nn
from torch import Tensor

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
    

