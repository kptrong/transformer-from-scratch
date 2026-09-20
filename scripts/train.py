from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from transformer_from_scratch.data.synthetic import (
    PAD_ID,
    ReverseSequenceDataset,
    collate_batch,
    vocab_size,
)
from transformer_from_scratch.model.transformer import Transformer
from transformer_from_scratch.training.engine import evaluate_loss, train_one_epoch
from transformer_from_scratch.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer on sequence reversal.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--symbol-vocab-size", type=int, default=20)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_noam_scheduler(
    optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ReverseSequenceDataset(
        args.train_size,
        min_len=args.min_len,
        max_len=args.max_len,
        symbol_vocab_size=args.symbol_vocab_size,
        seed=args.seed,
    )
    val_dataset = ReverseSequenceDataset(
        args.val_size,
        min_len=args.min_len,
        max_len=args.max_len,
        symbol_vocab_size=args.symbol_vocab_size,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    full_vocab_size = vocab_size(args.symbol_vocab_size)
    model = Transformer(
        src_vocab_size=full_vocab_size,
        tgt_vocab_size=full_vocab_size,
        pad_id=PAD_ID,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len + 8,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    # Base LR=1.0 because the Noam schedule itself supplies the full learning-rate scale.
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = make_noam_scheduler(optimizer, args.d_model, args.warmup_steps)

    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    print(f"device={device} | parameters={sum(p.numel() for p in model.parameters()):,}")

    for epoch in trange(1, args.epochs + 1, desc="epochs"):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} lr={lr:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "vocab_size": full_vocab_size,
                    "best_val_loss": best_val,
                },
                args.output_dir / "best.pt",
            )

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"best validation loss: {best_val:.4f}")
    print(f"checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
