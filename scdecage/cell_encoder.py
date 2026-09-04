from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from .attention import FlashAttentionBlock


class LearnableCellEncoder(nn.Module):
    """Encode gene-expression tokens into contextual cellular representations."""

    def __init__(self, vocab_size: int, padding_idx: int = 0, embed_dim: int = 128,
                 depth: int = 12, num_heads: int = 8) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.padding_idx = padding_idx
        self.gene_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.value_embed = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            FlashAttentionBlock(embed_dim, num_heads, 4, qkv_bias=True, attn_drop=0.1, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.cellular_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.output_dim = embed_dim

    def forward(
        self, gene_ids: torch.Tensor, expression_values: torch.Tensor
    ) -> torch.Tensor:
        valid = gene_ids.ne(self.padding_idx)
        x = self.gene_embed(gene_ids) + self.value_embed(expression_values.unsqueeze(-1))
        x = torch.cat(
            [self.cls_token.expand(gene_ids.shape[0], -1, -1), x], dim=1
        )
        valid_tokens = torch.cat(
            [torch.ones_like(valid[:, :1]), valid], dim=1
        )
        for block in self.blocks:
            x = block(x, valid_tokens)
        x = self.norm(x)
        valid_float = valid.unsqueeze(-1).to(x.dtype)
        gene_representation = (x[:, 1:] * valid_float).sum(dim=1)
        gene_representation = gene_representation / valid_float.sum(dim=1).clamp_min(1.0)
        pooled_gene_states = torch.cat([x[:, 0], gene_representation], dim=-1)
        return self.cellular_projection(pooled_gene_states)


def load_pretrained_cell_encoder(checkpoint_path: str | Path, vocab_size: int,
                                 padding_idx: int = 0) -> LearnableCellEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    encoder = LearnableCellEncoder(vocab_size=vocab_size, padding_idx=padding_idx)
    expected = set(encoder.state_dict())
    filtered = {key.removeprefix("encoder."): value for key, value in state.items()
                if key.removeprefix("encoder.") in expected}
    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    task_specific = {key for key in expected if key.startswith("cellular_projection.")}
    missing_backbone = [key for key in missing if key not in task_specific]
    if missing_backbone or unexpected:
        raise RuntimeError(f"Cell-encoder checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return encoder
