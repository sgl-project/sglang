# Double Sparsity calibration

Generates the per-layer, per-KV-head channel-index JSON consumed by
`--double-sparsity-config` at SGLang startup.

## What it does

For each transformer layer, the script hooks the K-projection output and
accumulates per-channel `abs_mean` across a small calibration set. It
then writes the top-`S` indices per (layer, KV head) to a JSON file
matching `schema_version=1`.

The script is **offline, single-replica (TP=1)**: it loads the model
directly with `transformers.AutoModelForCausalLM`. SGLang slices the
calibration to per-rank KV heads at server startup, so a single
calibration file works for any TP size.

## Quick start (Llama-3.1-8B, real dataset)

```bash
python scripts/double_sparsity/calibrate.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output calib_llama_3_1_8b.json \
    --heavy-channels 32 \
    --n-samples 64 --seq-len 4096
```

Then start SGLang with DS enabled:

```bash
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-double-sparsity \
    --double-sparsity-config calib_llama_3_1_8b.json \
    --double-sparsity-heavy-channels 32 \
    --page-size 1 \
    --attention-backend fa3
```

## Synthetic mode (smoke tests, e2e fixtures)

For pipeline validation without dataset access — generates calibration
from random token ids. Lower quality, fine for correctness testing:

```bash
python scripts/double_sparsity/calibrate.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output calib_synth.json \
    --synthetic --n-samples 8 --seq-len 1024
```

## Local prompts file

Newline-separated prompts:

```bash
python scripts/double_sparsity/calibrate.py \
    --model <path> \
    --output calib.json \
    --prompts-file my_prompts.txt
```

## Schema

```json
{
  "schema_version": 1,
  "model_arch": "...",
  "model_name_or_path": "...",
  "head_dim": 128,
  "num_layers": 32,
  "num_heads": 32,
  "num_kv_heads": 8,
  "heavy_channels": 32,
  "channel_type": "k",
  "indexing": "global_kv_head_id",
  "calibration": {"dataset": "...", "n_samples": 64, ...},
  "channels": {"0": [[c0, c1, ..., cS-1], ... num_kv_heads rows], ..., "31": [...]}
}
```

`indexing: "global_kv_head_id"` is required: each layer carries indices
for **all** global KV heads, partitioned at server startup along the
head axis based on `--tp-size`.

## Validation

Calibration files are validated at server startup against:
- `schema_version == 1`
- `head_dim` / `num_layers` / `num_heads` / `num_kv_heads` match the
  loaded model.
- All channel indices are in `[0, head_dim)` and unique within a row.
- Each layer has exactly `num_kv_heads` rows of length `heavy_channels`.

Mismatches raise `ValueError` before CUDA graph capture.

## What NOT to commit

Calibration JSONs for real models are typically 60-200 KB and are
**not** committed to the SGLang tree. Tests use a small synthetic
fixture under `test/registered/unit/mem_cache/sparsity/_fixtures/`.
