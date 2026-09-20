"""Full encoder-decoder Transformer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .layers import DecoderLayer, EncoderLayer
from .positional_encoding import SinusoidalPositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_id: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.src_position = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.tgt_position = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def make_src_mask(self, src: Tensor) -> Tensor:
        # [B, 1, 1, S], True = key is visible.
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: Tensor) -> Tensor:
        # Padding mask for target keys: [B, 1, 1, T].
        padding_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        # Causal mask: [1, 1, T, T].
        causal_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device)
        ).unsqueeze(0).unsqueeze(0)
        return padding_mask & causal_mask

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.src_position(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.tgt_position(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return x

    def forward(self, src: Tensor, tgt_input: Tensor) -> Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt_input)
        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt_input, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return self.generator(decoded)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
    ) -> Tensor:
        self.eval()
        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask)
        generated = torch.full(
            (src.size(0), 1), bos_id, dtype=torch.long, device=src.device
        )
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

        for _ in range(max_new_tokens):
            decoded = self.decode(generated, memory, memory_mask=src_mask)
            next_logits = self.generator(decoded[:, -1])
            next_token = next_logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(eos_id)
            if finished.all():
                break

        return generated
