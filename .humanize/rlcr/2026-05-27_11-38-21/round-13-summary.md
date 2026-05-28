# Round 13 Summary

## Objective

Fix the two AC-4 calibration coding blockers from Codex Round 12 review:
(1) Q RoPE width derivation broken on real V3.2 config shape;
(2) mask metadata dtype conflicts with FP8 serving validator.

## Changes Made

### 1. `python/sglang/srt/layers/attention/double_sparsity/calibrate.py`

**Blocker 1 — Q RoPE width derivation:**

Old code derived `qk_rope_head_dim = head_dim - qk_nope_head_dim` where `head_dim`
was itself derived from `hidden_size // num_heads` when the config lacks a `head_dim`
field. For V3.2 (`hidden_size=7168`, `num_heads=128`, `qk_nope=128`):
- `head_dim = 7168 // 128 = 56`
- `qk_rope_head_dim = 56 - 128 = -72`
- `full_mla_q_width = None` (negative guard), every `q_b_proj` hook skipped
- `_accumulate_method1` waited for Q indefinitely => "hooks did not fire" RuntimeError

New code reads `config.qk_rope_head_dim` directly when present. Only falls back to
`head_dim - qk_nope_head_dim` for configs that lack the explicit field. Raises a
clear RuntimeError with the derived values if the fallback would be non-positive.

**Blocker 2 — mask metadata dtype vs serving dtype:**

Added `mask_dtype = getattr(args, "kv_cache_dtype", None) or args.dtype`.
`save_channel_mask(dtype=mask_dtype, ...)` uses this value. `--dtype` remains
the model loading forward dtype; `--kv-cache-dtype` (optional, default None =>
falls back to `--dtype`) controls the mask metadata dtype.

**Parser update:** added `--kv-cache-dtype` optional arg; updated `--dtype` help.

**Module header:** production recipe now shows `--dtype bfloat16 --kv-cache-dtype fp8_e4m3`.

### 2. `docs/advanced_features/double_sparsity_calibration.md`

- Inputs table: updated `--dtype` description; added `--kv-cache-dtype` row.
- Recommended invocation: added `--kv-cache-dtype fp8_e4m3`.
- Explanation of the bf16 model-load vs fp8_e4m3 mask-metadata distinction.

### 3. `test/registered/unit/layers/attention/test_double_sparsity_unit.py`

**`test_dsv32_real_config_shape_q_hook_fires`** (new):
- Fake config: `qk_nope_head_dim=8`, `qk_rope_head_dim=4`, `v_head_dim=4`,
  `hidden_size=32`, `num_attention_heads=4` — no `head_dim` attribute.
- `hidden_size // num_heads = 8 != qk_nope + qk_rope = 12` — proves direct read.
- Asserts importance[0] shape is `(4, 8)`, is finite, and equals
  `mean(abs(Q_nope * K_nope))` ground truth.

## Test Results

```
188 passed, 0 failed (PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q)
```

Target was >= 188 (187 + 1 new). Target met exactly.

## Commit

`104fedcab` — [AC-4] Fix V3.2 qk_rope_head_dim derivation; add --kv-cache-dtype for mask dtype

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-mla-config-rope-dim-derivation
Notes: DeepSeek-V3.2 config has qk_rope_head_dim as an explicit field but no head_dim. Deriving qk_rope_head_dim from hidden_size//num_heads gives 56-128=-72, which silently skips all Q hooks. Fix reads config.qk_rope_head_dim directly first; only falls back to head_dim-qk_nope when the field is absent.

## AC Status After Round 13

- **AC-4** (`task-ac4-calibrate`): coding-complete — all blockers resolved across
  Rounds 10-13. `task-ac4-hwrun` unblocked for H200 generation with
  `--dtype bfloat16 --kv-cache-dtype fp8_e4m3`.
- All other ACs: unchanged from Round 12.
