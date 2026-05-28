# Round 9 Contract

## Mainline Objective

Close AC-7 by adding the first-decode-after-short-prefill integration proof: a single test
that writes real labels via `_set_mla_kv_buffer` then runs `_select_topk_indices` in decode
mode and asserts the output contains non-`-1` physical prefill slots.

## Target ACs

- **AC-7** (MHA bypass + label write chain) — complete the missing decode-after-prefill proof.
  AC-7 remains open per the Round 8 Codex review until this test lands.

## Blocking Issues

- **Round 8 Codex finding**: `test_mha_label_write_fires_in_set_mla_kv_buffer` only proved
  the spy was called; it did not chain the written labels into the subsequent decode selector
  call.  Codex requires the end-to-end flow: MHA prefill writes → decode scoring reads →
  non-(-1) physical slot returned.

## Queued (Out of Scope This Round)

- `task-ac4-calibrate`: Method 1 Q·K calibration in `calibrate.py`
- `task-ac5-tp`: TP multiprocess all-reduce test
- `task-ac6-cuda-graph`: `req_to_token` through `capture_decode_step`
- Hardware / analyze gates (AC-1, AC-1b, AC-8, AC-9–12)

## Success Criteria

1. New test `test_first_decode_after_short_prefill_selects_prefill_slots` in `TestAC7MHABypass`:
   - Uses a real `TokenLabelTable` and real `DoubleSparsitySelector` bound with a synthetic
     `ChannelMask` (not mocks).
   - Calls `_select_topk_indices` under `use_mha=True` and asserts `None`.
   - Calls `_set_mla_kv_buffer` via the real `NativeSparseAttnBackend._write_token_labels`
     and asserts `table.written` and non-zero `signatures` for all prefill slots.
   - Flips `use_mha=False`, calls `_select_topk_indices` with real `req_to_token`, asserts
     result is non-None and contains at least one physical slot from the prefill set.
2. Full suite still passes: `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` ≥ 179 passed, 0 failed.
