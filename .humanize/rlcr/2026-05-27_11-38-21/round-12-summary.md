# Round 12 Summary

## Objective

Fix three AC-4 calibration coding gaps identified by Codex Round 11 review so
`task-ac4-calibrate` is coding-complete and `task-ac4-hwrun` can proceed safely.

## Changes Made

### 1. `python/sglang/srt/layers/attention/double_sparsity/calibrate.py`

**Gap 1 — 3-D hook output handling in `_extract_mla_nope_prefix`:**
- Old code used `T = tensor.shape[0]` and then `tensor.reshape(T, num_heads, ...)`,
  which fails when real HF projection hooks emit `[batch, seq, width]` (3-D).
- Fixed by flattening all leading dimensions first:
  `flat = tensor.reshape(-1, tensor.shape[-1])`, then
  `flat.reshape(-1, num_heads, nope_dim + suffix_dim)[..., :nope_dim].contiguous()`
- Handles `[T, W]`, `[B, T, W]`, and any N-D shape where `shape[-1] == H*(nope+suffix)`.

**Gap 2 — Exact Pile-val token-block recipe:**
- Old code returned raw text strings truncated to `block_size` characters — not
  tokenized fixed-size blocks, and not concatenated across document boundaries.
- Added `_build_pile_val_token_blocks(tokenizer, num_blocks, block_size, seed)`:
  - Loads `mit-han-lab/pile-val-backup`, shuffles with `seed`
  - Tokenizes each doc with `add_special_tokens=False`
  - Concatenates token IDs across document boundaries
  - Splits into exactly `num_blocks` tensors of shape `[1, block_size]`
  - Raises `RuntimeError` if total tokens < `num_blocks * block_size`
- Added `use_pile_val` and `pile_val_seed` parameters to `_collect_channel_importance`
- When `use_pile_val=True`, calls `_build_pile_val_token_blocks` after tokenizer loads
  and feeds `model(input_ids=block.to(device))` for each block
- `calibrate()` production path (no `--dataset`, no `--allow-synthetic`) now sets
  `use_pile_val=True` — implements the exact Pile-val-256×512 recipe per AC-4

**Gap 3 — Module header updated:**
- Replaced K-only NIAH description with Method 1 Q+K noPE + Pile-val seed=42 recipe
- Production invocation example updated to match new parameters

### 2. `docs/advanced_features/double_sparsity_calibration.md`

- Inputs table: added `--block-size` (default 512), `--seed` (default 42),
  `--allow-synthetic` entries
- Recommended invocation: updated to include `--block-size 512 --seed 42`
- Dataset description: explains concatenated fixed-size block construction
- "What gets calibrated": describes Method 1 Q+K noPE with reshape-before-slice;
  removes stale K-only L2 + NIAH wording
- CI fixture: clarified `--allow-synthetic` opt-in and scope

### 3. `test/registered/unit/layers/attention/test_double_sparsity_unit.py`

Two new tests added to `TestCalibrateMethod1` (before `test_512d_channel_index_rejected`):

**`test_3d_hook_output_handled`**:
- Builds a fake model where `kv_b_proj` and `q_b_proj` return `[1, T, W]` (3-D
  with batch=1), using the same random values as `_make_fake_model` (seed=42)
- Runs `_collect_channel_importance` via `_run_calibration`
- Asserts `importance_3d[0]` is finite and `allclose` to `importance_2d[0]`
- Proves the flatten-before-reshape fix handles batch dimensions correctly

**`test_pile_val_blocks_concatenate_across_docs`**:
- Patches `datasets.load_dataset` with 3 short docs yielding 200 tokens each
  (IDs 0..199, 200..399, 400..599) — 600 total, need 512 for one block
- Calls `_build_pile_val_token_blocks(fake_tok, num_blocks=1, block_size=512, seed=42)`
- Asserts result is `[1, 512]` and that token at index 511 comes from doc 2
  (IDs 400..599 range), proving cross-document concatenation vs. truncation

## Test Results

```
187 passed, 0 failed (PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q)
```

Target was >= 187 (185 + 2 new). Target met exactly.

## Commit

`287a58231` — [AC-4] Fix 3-D hook outputs, implement Pile-val token-block recipe, update calibration doc

## BitLesson Delta

Action: none
Reason: `BL-20260527-reshape-before-slice-mla` covers the reshape-before-slice pattern
that this round extends to multi-dimensional inputs. No new distinct failure mode.

## AC Status After Round 12

- **AC-4** (`task-ac4-calibrate`): coding-complete — all three Round-11-review gaps
  closed; 187 tests pass. Ready for `task-ac4-hwrun` (H200 hardware run, analyze/Codex).
- All other ACs: unchanged from Round 11.
