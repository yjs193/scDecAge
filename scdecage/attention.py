from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp

try:
    from flash_attn import flash_attn_qkvpacked_func
except (ImportError, OSError):
    flash_attn_qkvpacked_func = None


class FlashAttention(nn.Module):
    """Checkpoint-compatible attention with a native PyTorch fallback."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        dropout = self.attn_drop.p if self.training else 0.0
        if flash_attn_qkvpacked_func is not None and x.is_cuda:
            output = flash_attn_qkvpacked_func(qkv, dropout_p=dropout).reshape(batch, tokens, channels)
        else:
            query, key, value = qkv.unbind(dim=2)
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=dropout
            ).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class FlashAttentionBlock(nn.Module):
    """Transformer block matching the pretrained cell-encoder checkpoint."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, act_layer: type[nn.Module] = nn.GELU,
                 norm_layer: partial = partial(nn.LayerNorm, eps=1e-6)) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))
