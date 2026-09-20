import torch

from transformer_from_scratch.model.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)


def test_scaled_dot_product_attention_shapes_and_probabilities() -> None:
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 7, 8)
    v = torch.randn(2, 4, 7, 8)

    output, weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == (2, 4, 5, 8)
    assert weights.shape == (2, 4, 5, 7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6)


def test_causal_mask_blocks_future_positions() -> None:
    q = torch.zeros(1, 1, 4, 2)
    k = torch.zeros(1, 1, 4, 2)
    v = torch.randn(1, 1, 4, 2)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool)).view(1, 1, 4, 4)

    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    upper_triangle = torch.triu(weights[0, 0], diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle))


def test_multi_head_attention_output_shape() -> None:
    layer = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    x = torch.randn(3, 6, 32)
    output, weights = layer(x, x, x, need_weights=True)

    assert output.shape == (3, 6, 32)
    assert weights is not None
    assert weights.shape == (3, 4, 6, 6)
