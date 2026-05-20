# Double Sparsity Calibration

This guide documents how to produce a channel mask file for the standalone Double Sparsity path on DeepSeek-V3.2 (FP8). The calibration script ships in the SGLang repo at `python/sglang/srt/layers/attention/double_sparsity/calibrate.py`; the calibrated artifact is NOT tracked in the repo and is owned by the deploying team per DEC-4.

## When you need to recalibrate

- Once per model revision. The runtime validator content-hashes the file and refuses to load it if the payload bytes have been tampered.
- After any change in `label_dim` (currently fixed at 16). The schema admits other values but the kernels assume 16.
- When migrating between dtypes (`fp8_e4m3` ↔ `bfloat16`). The runtime validator refuses cross-dtype loads.

## Inputs you need

| Input | Purpose | Default |
|-------|---------|---------|
| `--model` | HuggingFace ID or local path to the model | required |
| `--dtype` | `fp8_e4m3` or `bfloat16` — must match `--kv-cache-dtype` at serving time | required |
| `--output` | Path to write the channel mask `safetensors` file | required |
| `--label-dim` | Width of the compressed projection (selector buffer) | 16 |
| `--page-size` | Must match serving `--page-size` | 64 |
| `--num-samples` | Calibration prompts | 64 |
| `--ctx-len` | Approx token length per prompt | 4096 |
| `--dataset` | External corpus path (newline-delimited prompts); default is NIAH-shaped synthetic | None |

## Recommended invocation

For DeepSeek-V3.2 FP8 on the 2-node H200 hardware (per DEC-1; see also `development/serve_native_nsa.sh`):

```bash
python -m sglang.srt.layers.attention.double_sparsity.calibrate \
    --model deepseek-ai/DeepSeek-V3.2 \
    --dtype fp8_e4m3 \
    --tp 8 \
    --output /models/dsv32-fp8-channel-mask.safetensors \
    --label-dim 16 \
    --page-size 64 \
    --num-samples 1024 \
    --ctx-len 4096 \
    -v
```

**Expected wall clock**: 30–90 minutes on 1× H200 with the model loaded in bf16-on-fp8-weights mode. The script is single-process (`--tp` is informational); the deploying team typically calibrates on a single rank then deploys to TP=8 / 16 at serving time. The validator does NOT carry TP world size in the artifact — DEC-9's per-rank reduction handles cross-rank consistency.

## What gets calibrated

The script collects per-channel L2-squared importance on the K-projection layer via a forward hook, accumulates over the calibration corpus, and selects the top-`label_dim` channels per (layer, head). It also normalizes the importance to produce `channel_weights`.

The schema written to the file (per DEC-4 / AC-4):

- Tensors: `channel_selection[L, H, label_dim]` int32, `channel_weights[L, H, label_dim]` float32.
- Metadata: `schema_version`, `dtype`, `head_dim`, `page_size`, `label_dim`, `created_at`, `content_sha256`. The `content_sha256` is recomputed at load and the loader refuses to mount a file whose payload bytes do not match.

## CI fixture

The repository's CI runs the same script with a tiny NSA-shaped model fixture (`--tp 1 --allow-synthetic`, `--num-samples 4`, `--ctx-len 256`, with `--num-layers`, `--num-heads`, `--head-dim` hints) so the loader's happy path is exercised in under a minute. The `--allow-synthetic` flag is an explicit opt-in: without it, a non-local `--model` value (such as a HuggingFace repo ID) goes through `AutoModelForCausalLM.from_pretrained` and will auto-fetch the repo. Synthetic statistics are deterministic and record `calibration_source=synthetic` in the file metadata — this CI artifact is suitable for unit tests but **must NOT** be used for production serving.

## NIAH-min sanity probe

After calibration, the runtime validator runs a one-needle / 512-token-haystack probe at server startup to verify the channel mask actually retrieves a planted needle. The probe is inconclusive while the selector is still the placeholder; production deployment requires the real selection kernels to be wired (see `development/loop1/REVIEWER_GUIDE.md`).

## Where the artifact lives

Per DEC-4: **not** in the SGLang repo. The deploying team owns artifact storage (typically a model registry or object store). The validator gates startup on the file existing at the path given by `--double-sparsity-config '{"channel_mask_path": "<...>"}'`.

## DEC-9 score all-reduce contract (no calibration impact)

The calibration step is per-rank-agnostic. At serving time, ranks compute scalar page scores from their local head shards, all-reduce SUM, and run independent top-K — so the channel mask file can be reused across any TP topology without recalibration.

## TP head sharding at bind time

The calibration artifact stores `channel_selection[L, H_full, label_dim]` where `H_full = num_attention_heads`. The runtime selector expects `[L, num_local_heads, label_dim]`. Before passing the loaded mask into `DoubleSparsitySelector.bind_runtime_data`, the deploying team's integration glue must slice the per-rank head block:

```python
from sglang.srt.layers.attention.double_sparsity import slice_per_rank, load_channel_mask

mask = load_channel_mask("/models/dsv32-fp8-channel-mask.safetensors")
rank_mask = slice_per_rank(
    mask,
    num_local_heads=num_local_heads,  # = num_attention_heads // tp_size
    rank=current_tp_rank,
    tp_size=tp_size,
)
selector.bind_runtime_data(page_signature_table, rank_mask)
```

`bind_runtime_data` rejects an un-sliced (H_full) mask with a clear error message that names this helper.

## Future-compat (per the schema memo)

The schema admits GLM-5.1, 128K ISL, and FP4 weights without rewrite (see `docs/advanced_features/double_sparsity_schema_memo.md`). Recalibration is needed when migrating between model revisions but not when changing TP or context length.
