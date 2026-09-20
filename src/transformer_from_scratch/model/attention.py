"""Attention primitives implemented explicitly from the Transformer equations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention.

    Args:
        query: [batch, heads, query_len, head_dim]
        key: [batch, heads, key_len, head_dim]
        value: [batch, heads, key_len, head_dim]
        mask: Boolean tensor broadcastable to [batch, heads, query_len, key_len].
              True means the position is allowed to participate in attention.
        dropout: Optional dropout module applied to attention probabilities.

    Returns:
        output: Weighted values, shape [batch, heads, query_len, head_dim].
        attention_weights: Shape [batch, heads, query_len, key_len].
    """
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype is not torch.bool:
            raise TypeError("attention mask must be boolean")
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attention_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    output = attention_weights @ value
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention built from four linear projections and SDPA."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        attended, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout=self.attn_dropout if self.training else None,
        )
        output = self.out_proj(self._merge_heads(attended))
        return output, weights if need_weights else None
