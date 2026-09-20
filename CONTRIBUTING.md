# Contributing

1. Create a virtual environment.
2. Install development dependencies with `pip install -e ".[dev]"`.
3. Run `ruff check .` and `pytest -q` before opening a pull request.
4. Keep model code explicit and pedagogical; avoid replacing the hand-written attention path with `nn.Transformer`.
