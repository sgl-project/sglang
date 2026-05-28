# Round 10 Summary

## Work Completed

### AC-4 — Method 1 Q+K joint calibration in `calibrate.py`

Replaced the K-only L2 accumulator in `_collect_channel_importance` with Method 1
`mean(abs(Q_nope * K_nope))` per channel, and added a Q_nope hook alongside the
existing `kv_b_proj` K-side hook.

**Core changes to `python/sglang/srt/layers/attention/double_sparsity/calibrate.py`:**

- Added `_q_buf` and `_k_buf` per-layer lists for coordinating Q and K from the
  same forward pass.
- `_accumulate_method1(idx)`: fires when both buffers are populated; computes
  `(q * k).abs().mean(dim=0)` and adds to `importance[idx]`.
- `_accumulate_k_only(idx, k)`: K-only L2 fallback; used when no Q projection is
  found; logged at WARNING level.
- Q projection discovery: tries `q_b_proj` (MLA) → `q_proj` → `wq`; falls back
  with a "no Q projection found" warning when none is found.
- K hook: registered on `kv_b_proj` / `k_proj` / `wk`; slices K-noPE prefix
  (`[..., :num_heads * k_head_dim]`) from the full `kv_b_proj` output when MLA;
  stores in `_k_buf[idx]`; calls `_accumulate_method1` (or `_accumulate_k_only`
  if no Q).
- Q hook: registered on `q_b_proj` / `q_proj` / `wq`; slices noPE prefix
  `[..., :num_heads * k_head_dim]` (same width as K); stores in `_q_buf[idx]`;
  calls `_accumulate_method1`.

**Added `TestCalibrateMethod1` to
`test/registered/unit/layers/attention/test_double_sparsity_unit.py` (4 tests):**

1. `test_qk_pairing_uses_method1_formula`: Builds a 1-layer MLA fake model with
   real `nn.Module` submodules (`_FixedOutLinear`) so PyTorch forward-hooks fire
   during `model(**inputs)`. Verifies `importance[0]` matches
   `mean(abs(Q_nope * K_nope))` (not `sum(K^2)`) with `atol=1e-5`.

2. `test_k_only_fallback_when_q_missing`: Builds the same model without `q_b_proj`.
   Asserts a "no Q projection" WARNING is logged, and `importance[0]` matches
   `mean(K_nope^2)`.

3. `test_512d_channel_index_rejected`: Saves a channel mask with `head_dim=128` but
   `channel_selection[0, 0, 0] = 512` (out of range). Asserts `load_channel_mask`
   raises `DoubleSparsityChannelMaskCorrupt` or `ValueError` containing "out of range".

4. `test_label_dim_exceeds_k_head_dim_raises`: Calls `calibrate()` with
   `label_dim=256` and `head_dim=128`; asserts `ValueError` with "label-dim".

**Key fixture design decision:** The original `_CapturingLinear` used a plain Python
class whose `register_forward_hook` stored callbacks but never fired them when
`model(**inputs)` (a `MagicMock.__call__`) was executed. Replaced with real
`nn.Module` hierarchy (`_FixedOutLinear`, `_FakeAttn`, `_FakeLayer`, `_FakeInner`,
`_FakeTopModel`) so PyTorch's hook dispatch fires naturally via `model(**inputs)`.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` — rewrote
  `_collect_channel_importance` to implement Method 1 Q+K joint hooks with K-only
  L2 fallback; 168 lines changed
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — added
  `TestCalibrateMethod1` class (4 tests, 245 new lines)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
183 passed, 0 failed in 11.44s
```

Commit: `8379cfdba`

## Remaining Items

- `task-ac4-hwrun`: H200 hardware run to generate `dsv32-fp8-channel-mask.safetensors`
  (analyze tag → Codex; hardware not available here)
- `task-ac5-tp`: TP=2 multiprocess all-reduce test
- `task-ac6-cuda-graph`: `req_to_token` through `capture_decode_step`
- Hardware / analyze gates: AC-1, AC-1b, AC-8, AC-9–12

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The real-`nn.Module`-for-hook-testing pattern is a natural extension of the
`_FakeProj` stub pattern already in `test_first_decode_after_short_prefill_selects_prefill_slots`
(Round 9, line ~4320). The new `_FixedOutLinear` is the module-level form of that
pattern. No new generalizable lesson beyond what is already captured.
