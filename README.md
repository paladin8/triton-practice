# Triton Practice

Practice implementing GPU kernels in [Triton](https://github.com/openai/triton). The purpose is to compare the implementations of kernels in Torch vs Triton.

## Getting Started

1. Ensure you have Python 3.10+ available locally.
2. Install `uv` by following the instructions from the official repository.
3. Synchronize the environment and install dependencies:

   ```bash
   uv sync --dev
   ```

4. Run a quick sanity check:

   ```bash
   uv run python -c "from triton_practice import sanity_check; sanity_check()"
   ```

## Linting

This project uses `Ruff` for linting. After syncing dependencies you can lint or auto-fix via:

```bash
uv run ruff check --fix src/
```

Most editors can run that command or call Ruff's built-in code actions automatically on save.
