from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformer_from_scratch.data.synthetic import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    TOKEN_OFFSET,
    ReverseSequenceDataset,
    collate_batch,
)
from transformer_from_scratch.model.transformer import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy-decode examples from a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best.pt"))
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def strip_special(tokens: list[int]) -> list[int]:
    result = []
    for token in tokens:
        if token == EOS_ID:
            break
        if token not in {PAD_ID, BOS_ID}:
            result.append(token - TOKEN_OFFSET)
    return result


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = Transformer(
        src_vocab_size=checkpoint["vocab_size"],
        tgt_vocab_size=checkpoint["vocab_size"],
        pad_id=PAD_ID,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_len=config["max_len"] + 8,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = ReverseSequenceDataset(
        num_samples=args.num_examples,
        min_len=config["min_len"],
        max_len=config["max_len"],
        symbol_vocab_size=config["symbol_vocab_size"],
        seed=config["seed"] + 10_000,
    )
    loader = DataLoader(dataset, batch_size=args.num_examples, collate_fn=collate_batch)
    src, _, tgt_output = next(iter(loader))
    src = src.to(device)

    generated = model.greedy_decode(
        src,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        max_new_tokens=config["max_len"] + 2,
    ).cpu()

    exact = 0
    for i in range(args.num_examples):
        source = strip_special(src[i].cpu().tolist())
        target = strip_special(tgt_output[i].tolist())
        prediction = strip_special(generated[i].tolist())
        is_correct = prediction == target
        exact += int(is_correct)
        mark = "✓" if is_correct else "✗"
        print(f"{mark} src={source} | target={target} | pred={prediction}")

    print(f"exact-match accuracy: {exact / args.num_examples:.2%}")


if __name__ == "__main__":
    main()
