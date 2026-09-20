"""Synthetic sequence-reversal dataset for validating a full seq2seq Transformer."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
TOKEN_OFFSET = 3


@dataclass(frozen=True)
class Example:
    src: Tensor
    tgt_input: Tensor
    tgt_output: Tensor


class ReverseSequenceDataset(Dataset[Example]):
    """Generate deterministic integer sequences and their reversed targets."""

    def __init__(
        self,
        num_samples: int,
        min_len: int = 4,
        max_len: int = 12,
        symbol_vocab_size: int = 20,
        seed: int = 42,
    ) -> None:
        if min_len < 1 or max_len < min_len:
            raise ValueError("invalid min_len/max_len")
        if symbol_vocab_size < 2:
            raise ValueError("symbol_vocab_size must be at least 2")

        rng = random.Random(seed)
        self.examples: list[Example] = []

        for _ in range(num_samples):
            length = rng.randint(min_len, max_len)
            symbols = [rng.randrange(symbol_vocab_size) + TOKEN_OFFSET for _ in range(length)]
            reversed_symbols = list(reversed(symbols))

            src = torch.tensor(symbols + [EOS_ID], dtype=torch.long)
            tgt_input = torch.tensor([BOS_ID] + reversed_symbols, dtype=torch.long)
            tgt_output = torch.tensor(reversed_symbols + [EOS_ID], dtype=torch.long)
            self.examples.append(Example(src, tgt_input, tgt_output))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def collate_batch(batch: list[Example]) -> tuple[Tensor, Tensor, Tensor]:
    src = pad_sequence([x.src for x in batch], batch_first=True, padding_value=PAD_ID)
    tgt_input = pad_sequence(
        [x.tgt_input for x in batch], batch_first=True, padding_value=PAD_ID
    )
    tgt_output = pad_sequence(
        [x.tgt_output for x in batch], batch_first=True, padding_value=PAD_ID
    )
    return src, tgt_input, tgt_output


def vocab_size(symbol_vocab_size: int) -> int:
    return TOKEN_OFFSET + symbol_vocab_size
