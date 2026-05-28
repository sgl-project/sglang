# Round 10 Contract

## Mainline Objective

Implement AC-4 calibration: replace K-only L2 statistics in `calibrate.py` with
Method 1 `mean(abs(Q_nope * K_nope))` per channel. Add Q_nope hook alongside
the existing `kv_b_proj` K-side hook. Add fixture tests for Q/K pairing, 512-d
rejection, and label-dim mismatch.

## Target ACs

- **AC-4** — Method 1 Q+K joint hooks in `calibrate.py`; fixture tests for
  Q/K pairing, label-dim mismatch, and 512-d channel index rejection;
  CI tiny-fixture (synthetic) path stays green.

## Blocking Issues

None. AC-7 is closed; all blocking side issues are resolved.

## Queued (Out of Scope This Round)

- `task-ac4-hwrun`: H200 hardware run to generate `dsv32-fp8-channel-mask.safetensors`
  (analyze tag → Codex; hardware not available here)
- `task-ac5-tp`: TP multiprocess all-reduce test
- `task-ac6-cuda-graph`: `req_to_token` through `capture_decode_step`
- Hardware / analyze gates (AC-1, AC-1b, AC-8, AC-9–12)

## Success Criteria

1. `_collect_channel_importance` registers Q_nope hooks (from `q_b_proj` for MLA,
   `q_proj` for standard attention) alongside the existing K hooks; importance is
   accumulated as `mean(abs(Q_nope * K_nope), dim=0)` when both hooks fire for the
   same pass; falls back to K-only L2 per-layer with a logged warning when Q is
   not accessible.
2. New `TestCalibrateMethod1` test class with tests:
   - `test_qk_pairing_uses_method1_formula`: construct a synthetic model with both
     `q_b_proj` (MLA-style) and `kv_b_proj`, set known Q/K tensors, verify the
     resulting importance matches `mean(abs(Q*K))` rather than `sum(K^2)`.
   - `test_k_only_fallback_when_q_missing`: model with `kv_b_proj` only (no `q_b_proj`/
     `q_proj`) falls back to K-only L2 accumulation with a logged warning; fixture passes.
   - `test_512d_channel_index_rejected`: `load_channel_mask` raises
     `DoubleSparsityChannelMaskCorrupt` for a mask with `head_dim=128` but channel
     indices that include a value >= 128.
   - `test_label_dim_exceeds_k_head_dim_raises`: `calibrate()` raises when
     `label_dim > head_dim`.
3. Full suite: `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` ≥ 183 passed (179 + 4 new tests), 0 failed.
