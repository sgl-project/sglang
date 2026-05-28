# Round 11 Summary

## Work Completed

### AC-4 — Fix 1: MLA extraction reshape-before-slice

Added `_extract_mla_nope_prefix(tensor, num_heads, nope_dim, suffix_dim)` helper in
`calibrate.py`. MLA projections interleave per-head blocks — `kv_b_proj` layout is
`[K_nope_h0 | V_h0 | K_nope_h1 | V_h1 | ...]` — so flat-slicing the first
`H * nope_dim` columns before reshape selects V columns from later heads (same class
of bug fixed in `dsa_backend.py` Round 3).

Changed both hooks to use the helper:
- K hook: `_extract_mla_nope_prefix(t, num_heads, k_head_dim, v_head_dim)` — replaces
  `tensor[..., :prefix].reshape(-1, num_heads, k_head_dim)`.
- Q hook: `_extract_mla_nope_prefix(t, num_heads, k_head_dim, qk_rope_head_dim)` —
  replaces `tensor[..., :prefix].reshape(-1, num_heads, k_head_dim)`.
  `qk_rope_head_dim = head_dim - qk_nope_head_dim` (derived from config).

Also added `is_mla_q` flag to `_make_q_hook` so standard-attention Q (`q_proj` or
`wq`) still uses a direct reshape instead of the per-head splitting logic.

### AC-4 — Fix 2: Pile-val seed=42, 256×512 calibration dataset

Added `_pile_val_blocks(num_blocks, block_size, seed)` function that loads
`mit-han-lab/pile-val-backup` via `datasets`, shuffles with `seed=42`, and returns
`num_blocks` text examples.

Changed `calibrate()` default dataset path:
- If `args.dataset`: use custom corpus file (unchanged).
- Elif `args.allow_synthetic`: use NIAH synthetic prompts (unchanged; CI path).
- Else (production path): use Pile-val blocks.

Added `--block-size` (default 512) and `--seed` (default 42) to the parser.
Changed `--num-samples` default from 64 to 256.
Added `dataset_source`, `seed`, `block_size` to output metadata.
Threaded `block_size` into `_collect_channel_importance` for tokenizer truncation
(`max_length=block_size, truncation=True` when set).

### AC-4 — Fix 3: Sentinel regression tests

Added 2 tests to `TestCalibrateMethod1`:

1. `test_mla_k_extraction_ignores_v_columns`: 2-head MLA, `kv_b_proj` output with
   K_nope=1.0 and V=100.0 (poison). Asserts `importance.max() < 10.0`. Fails under
   old flat-slice (head 1 gets V0 → importance ≈ 100.0).

2. `test_mla_q_extraction_ignores_rope_columns`: 2-head MLA, `q_b_proj` output with
   Q_nope=1.0 and Q_rope=100.0 (poison). Asserts `importance.max() < 10.0`. Fails
   under old flat-slice (head 1 gets Q0_rope[0:4] → importance ≈ 100.0).

Also fixed `_make_fake_model`'s `expected_importance` to use reshape-then-slice (to
match the now-correct calibrate.py extraction), and added `block_size=512, seed=42`
to the `test_label_dim_exceeds_k_head_dim_raises` argparse.Namespace fixture.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` — added
  `_extract_mla_nope_prefix` + `_pile_val_blocks`; fixed K and Q hook extraction;
  added `block_size` parameter; updated `calibrate()` dataset path and metadata;
  added `--block-size` and `--seed` CLI args
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — fixed
  `_make_fake_model` expected_importance; added 2 sentinel tests; fixed argparse
  Namespace fixture

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
185 passed, 0 failed in 11.50s
```

Commit: `d2f82c93a`

## Remaining Items

- `task-ac4-hwrun`: H200 hardware run to generate `dsv32-fp8-channel-mask.safetensors`
  (analyze tag → Codex; hardware not available here)
- `task-ac5-tp`: TP=2 multiprocess all-reduce test
- `task-ac6-cuda-graph`: `req_to_token` through `capture_decode_step`
- Hardware / analyze gates: AC-1, AC-1b, AC-8, AC-9–12
- Queued: update `docs/advanced_features/double_sparsity_calibration.md` before
  `task-ac4-hwrun`

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-reshape-before-slice-mla

Notes: The flat-slice-before-reshape pattern on MLA projection outputs is a
recurring class of bug in this codebase. Round 3 fixed it in `dsa_backend.py`;
Round 11 fixed the same class in `calibrate.py`. The pattern manifests silently
— the resulting tensor has correct shape but wrong values for heads > 0. A
BitLesson should be written so future rounds catch this immediately rather than
waiting for Codex review.
