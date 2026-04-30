# Decoupled Speculation Example

This directory has one CLI entrypoint for decoupled speculative decoding:

- `run_decoupled_speculation.py`: run decoupled speculation from either `--prompt` or `--dataset-path`.
- `decoupled_spec_common.py`: shared helpers for Ray, draft actor launch, GPU allocation, prompt normalization, and dataset parsing.

Common modes:

```bash
# Single prompt, compare decoupled speculation against normal decode.
python examples/runtime/engine/decoupled_speculation/run_decoupled_speculation.py \
  --prompt "Write a short haiku about distributed systems." \
  --target-model-path Qwen/Qwen3-32B \
  --draft-model-path Qwen/Qwen3-0.6B \
  --target-tp-size 4 \
  --draft-tp-size 1 \
  --max-new-tokens 128

# Dataset batch, decoupled speculation only.
python examples/runtime/engine/decoupled_speculation/run_decoupled_speculation.py \
  --dataset-path /path/to/prompts.parquet \
  --batch-size 16 \
  --skip-decode \
  --target-model-path /path/to/target \
  --draft-model-path /path/to/draft \
  --target-tp-size 8 \
  --draft-tp-size 1 \
  --max-new-tokens 1024

# Print responses and write per-mode CSV/JSON outputs.
python examples/runtime/engine/decoupled_speculation/run_decoupled_speculation.py \
  --prompt "Explain speculative decoding." \
  --show-responses \
  --output-dir ./decoupled_spec_outputs \
  --target-model-path /path/to/target \
  --draft-model-path /path/to/draft \
  --target-tp-size 4 \
  --draft-tp-size 1 \
  --max-new-tokens 256
```
