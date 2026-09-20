# Transformer From Scratch

A clean, educational implementation of the **original encoder-decoder Transformer** in PyTorch. The goal is to understand every major component rather than call `torch.nn.Transformer`.

The first experiment trains the model on a synthetic **sequence reversal** task:

```text
input :  [4, 1, 9, 2, 7]
target:  [7, 2, 9, 1, 4]
```

This tiny task is deliberately chosen because it exercises the complete architecture: encoder self-attention, masked decoder self-attention, encoder-decoder cross-attention, padding masks, causal masks, teacher forcing, BOS/EOS tokens, and autoregressive decoding.

## What is implemented by hand

- Scaled dot-product attention: `softmax(QK^T / sqrt(d_k)) V`
- Multi-head attention and head split/merge
- Sinusoidal positional encoding
- Position-wise feed-forward network
- Residual connections and LayerNorm
- Encoder layer and encoder stack
- Masked decoder layer and decoder stack
- Padding and causal masks
- Full encoder-decoder Transformer
- Teacher-forced training
- Greedy autoregressive decoding
- Noam learning-rate schedule
- Unit tests and GitHub Actions CI

PyTorch is used for tensors, autograd, basic layers such as `Linear`, `Embedding`, and `LayerNorm`, and optimization. The project intentionally does **not** use `nn.Transformer` or PyTorch's fused scaled-dot-product attention in the learning implementation.

## Repository layout

```text
transformer-from-scratch/
├── .github/workflows/ci.yml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── src/transformer_from_scratch/
│   ├── data/synthetic.py
│   ├── model/
│   │   ├── attention.py
│   │   ├── layers.py
│   │   ├── positional_encoding.py
│   │   └── transformer.py
│   ├── training/engine.py
│   └── utils/seed.py
├── tests/
│   ├── test_attention.py
│   └── test_transformer.py
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 1. Local setup in VS Code

Python 3.10+ is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Then select the `.venv` interpreter in VS Code.

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 2. Run tests first

```bash
pytest -q
```

The tests verify attention shapes, normalization, causal masking, multi-head output shape, and the full Transformer forward pass.

## 3. Train the toy model

```bash
python scripts/train.py
```

Useful smaller smoke test:

```bash
python scripts/train.py --epochs 2 --train-size 1000 --val-size 200 --d-model 64 --d-ff 128
```

If CUDA is available it is selected automatically. You may force a device with `--device cpu`, `--device cuda`, or `--device mps`.

The best checkpoint is written to:

```text
outputs/best.pt
```

## 4. Evaluate autoregressive generation

```bash
python scripts/evaluate.py --checkpoint outputs/best.pt --num-examples 20
```

Example output after successful training:

```text
✓ src=[2, 8, 1, 5] | target=[5, 1, 8, 2] | pred=[5, 1, 8, 2]
...
exact-match accuracy: 100.00%
```

## 5. Run on Google Colab or Kaggle

Clone the repository in a notebook cell:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd transformer-from-scratch
pip install -e .
python scripts/train.py --device cuda
python scripts/evaluate.py --checkpoint outputs/best.pt
```

For this toy task, a GPU is convenient but not required. CPU is enough for debugging.

## Architecture walkthrough

### Scaled dot-product attention

For one attention head:

```text
scores = Q @ K^T / sqrt(d_k)
weights = softmax(scores + mask)
output = weights @ V
```

See `src/transformer_from_scratch/model/attention.py`.

### Multi-head attention

`Q`, `K`, and `V` are projected from `d_model` into `num_heads` independent heads. Each head performs scaled dot-product attention, then the heads are concatenated and projected back to `d_model`.

### Encoder

Each encoder layer contains:

1. Multi-head self-attention
2. Residual connection + LayerNorm
3. Position-wise feed-forward network
4. Residual connection + LayerNorm

### Decoder

Each decoder layer contains:

1. Causal masked self-attention
2. Encoder-decoder cross-attention
3. Position-wise feed-forward network

The causal mask prevents position `t` from observing target positions greater than `t`.

## Suggested learning order

Study and modify the project in this order:

1. `data/synthetic.py` — understand source and shifted target sequences.
2. `model/attention.py` — derive and test scaled dot-product attention.
3. `MultiHeadAttention` — inspect head splitting and tensor shapes.
4. `model/positional_encoding.py` — understand why position information is needed.
5. `model/layers.py` — compose attention, residuals, normalization, and FFN.
6. `model/transformer.py` — connect encoder and decoder stacks.
7. `training/engine.py` and `scripts/train.py` — teacher forcing, loss, optimization.
8. `greedy_decode` — understand autoregressive inference.

## Recommended next milestones

- Add attention-map visualization.
- Compare hand-written attention with `torch.nn.functional.scaled_dot_product_attention`.
- Add Pre-LN as an architecture option and compare stability.
- Replace the synthetic dataset with a small real translation dataset.
- Add beam search.
- Add mixed precision training.
- Add experiment tracking and benchmark tables.
- Implement a decoder-only GPT-style model in a separate branch after mastering encoder-decoder attention.

## Design notes

The code uses the post-LayerNorm ordering associated with the original Transformer architecture. The optimizer uses Adam with the inverse-square-root warmup schedule popularized by the original paper. The default model is intentionally small so it can be trained in VS Code, Colab, or Kaggle without specialized hardware.

## License

MIT
