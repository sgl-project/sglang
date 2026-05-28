# Round 13 Contract

## Mainline Objective

Fix the two remaining AC-4 calibration coding blockers identified by Codex Round 12
review so `task-ac4-calibrate` is truly coding-complete and `task-ac4-hwrun` can
safely generate `/models/dsv32-fp8-channel-mask.safetensors` on H200.

## Target ACs

- **AC-4** — (1) Fix Q RoPE width derivation for real V3.2 config; (2) resolve dtype
  contract between calibration model-load dtype and mask metadata dtype.
  CI suite ≥ 188 passed (187 + 1 new regression).

## Blocking Issues (both must be fixed this round)

### Blocker 1 — Q hook silently skips on real V3.2 config

The real DeepSeek-V3.2 config has `qk_rope_head_dim=64` as an explicit field but
**no `head_dim`** field. `calibrate.py` currently derives `head_dim` as
`hidden_size // num_attention_heads = 7168 // 128 = 56`, then computes
`qk_rope_head_dim = head_dim - qk_nope_head_dim = 56 - 128 = -72`.

Because `-72 <= 0`, `full_mla_q_width` is set to `None`, every `q_b_proj` hook
fires but falls through the width check and writes nothing to `_q_buf`, and
`_accumulate_method1` waits for Q indefinitely — causing
"Calibration hooks did not fire on N/N layers".

**Fix**: derive `qk_rope_head_dim` from `config.qk_rope_head_dim` when present;
only fall back to `head_dim - qk_nope_head_dim` for configs without the field;
raise a clear error if the fallback is non-positive on an MLA config.

### Blocker 2 — mask dtype conflicts with FP8 serving

`--dtype bfloat16` controls both the model-load forward dtype AND what goes into
the mask metadata. The DS launcher defaults `KV_CACHE_DTYPE=fp8_e4m3`; startup
validation rejects a mask whose metadata `dtype != --kv-cache-dtype`.

Following the production recipe as written would produce a content-valid mask that
cannot boot the Option-B server.

**Fix**: add `--kv-cache-dtype` argument (optional; defaults to `--dtype`). When
provided, it controls the mask metadata dtype; `--dtype` stays as the model loading
dtype. Production recipe: `--dtype bfloat16 --kv-cache-dtype fp8_e4m3`.

## Queued (Out of Scope This Round)

- `task-ac4-hwrun`: H200 hardware run
- `task-ac5-tp`, `task-ac6-cuda-graph`: next coding tasks after AC-4 is closed
- Hardware/analyze gates: AC-1b, AC-8, AC-9–12

## Success Criteria

1. `calibrate.py` reads `qk_rope_head_dim` from `config.qk_rope_head_dim` when the
   field is present; falls back to `head_dim - qk_nope_head_dim` only when absent;
   raises a clear error if the fallback produces a non-positive value on an MLA config.

2. `calibrate.py` accepts `--kv-cache-dtype` (optional, default=`None`).
   When provided, `save_channel_mask(dtype=kv_cache_dtype, ...)` is called;
   otherwise `save_channel_mask(dtype=dtype, ...)` as before.

3. Module header and `docs/advanced_features/double_sparsity_calibration.md`
   updated to show `--kv-cache-dtype fp8_e4m3` in the production recipe.

4. One new test `test_dsv32_real_config_shape_q_hook_fires` in `TestCalibrateMethod1`:
   - Fake config has `qk_nope_head_dim=8`, `qk_rope_head_dim=4`, `v_head_dim=4`,
     `hidden_size=32`, `num_attention_heads=4` — no `head_dim` field;
     `hidden_size // num_heads = 8 ≠ qk_nope + qk_rope = 12`.
   - Fake `q_b_proj` returns `[T, H*(qk_nope+qk_rope)]` (width=48).
   - Test proves importance[0] is finite, non-zero, and matches
     `mean(abs(Q_nope * K_nope))` ground truth (Method 1).

5. Full suite: `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`
   ≥ 188 passed, 0 failed.
