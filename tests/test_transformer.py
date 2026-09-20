import torch

from transformer_from_scratch.data.synthetic import PAD_ID
from transformer_from_scratch.model.transformer import Transformer


def make_model() -> Transformer:
    return Transformer(
        src_vocab_size=23,
        tgt_vocab_size=23,
        pad_id=PAD_ID,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        max_len=32,
    )


def test_transformer_forward_shape() -> None:
    model = make_model()
    src = torch.tensor([[3, 4, 5, 2, 0], [6, 7, 8, 9, 2]])
    tgt = torch.tensor([[1, 5, 4, 3], [1, 9, 8, 7]])

    logits = model(src, tgt)

    assert logits.shape == (2, 4, 23)


def test_target_mask_is_causal_and_respects_padding() -> None:
    model = make_model()
    tgt = torch.tensor([[1, 5, 4, 0]])
    mask = model.make_tgt_mask(tgt)

    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 0]
    assert not mask[0, 0, 0, 1]
    assert not mask[0, 0, 3, 3]
