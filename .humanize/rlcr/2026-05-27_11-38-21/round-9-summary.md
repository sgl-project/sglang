# Round 9 Summary

## Work Completed

### AC-7 — First-decode-after-short-prefill integration proof

Added `test_first_decode_after_short_prefill_selects_prefill_slots` to `TestAC7MHABypass`
in `test/registered/unit/layers/attention/test_double_sparsity_unit.py`.

The test uses a real `TokenLabelTable`, a real `DoubleSparsitySelector` bound with a
synthetic `ChannelMask` (not mocks), and a `NativeSparseAttnBackend` constructed via
`object.__new__` to provide the real `_write_token_labels` implementation.

**Steps proved end-to-end in order:**

1. **MHA bypass fires**: Under `forward_context(ForwardContext(attn_backend=backend))` with
   `backend.use_mha=True`, `_select_topk_indices` returns `None` without calling
   `retrieve_topk`. Asserts `bypass_result is None`.

2. **Labels written via MHA_ONE_SHOT path**: `_set_mla_kv_buffer` is called for 3 prefill
   tokens (physical slots 1, 2, 3). The DS hook at the end of `_set_mla_kv_buffer` calls
   `backend._write_token_labels(attn_layer, cache_loc, kv_a.unsqueeze(1))` using the real
   projection (`_FakeProj` stub returning `(x.float() @ W,)` in ColumnParallelLinear style).
   Asserts `table.written[0, slot] = True` for all 3 prefill slots and
   `table.signatures[0, slot0].abs().sum() > 0` for slot 1 (the strongest-signal token).

3. **Decode selection uses prefill labels**: Backend flipped to `use_mha=False`. Decode query
   `q_nope[0, :, 0] = 1.0` (strong in K_noPE channel 0) scores positively against prefill
   slot 1's label `[[5, 0], [5, 0]]`. `_select_topk_indices` is called with real
   `req_to_token` mapping logical positions 0–2 to physical slots 1–3. The real
   `retrieve_topk_via_labels` pipeline runs (scoring → `select_topk_sequence_order` →
   `logical_to_physical`). Asserts `decode_result` is not None and contains at least one
   entry from the prefill slot set `{1, 2, 3}`.

**What this test catches:**
- Removing the `_write_token_labels` call from `_set_mla_kv_buffer` → no labels written →
  scoring returns -inf for all slots → `decode_result` is all-(-1) → test fails.
- Removing the `has_forward_context()` bypass in `_select_topk_indices` → prefill step calls
  `retrieve_topk` instead of returning None → test fails on `assertIsNone`.
- Replacing real `retrieve_topk_via_labels` with a mock → physical-slot assertion may fail
  depending on what the mock returns.

## Files Changed

- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — added
  `test_first_decode_after_short_prefill_selects_prefill_slots` to `TestAC7MHABypass`
  (221 new lines, bringing the test class to 7 tests)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
179 passed, 0 failed in 11.58s
```

Commit: `e2552fbe9`

## Remaining Items

- `task-ac1-hwtest`: H200 hardware population test.
- `task-ac4-calibrate` / `task-ac4-hwrun`: Method 1 calibration + mask generation.
- `task-ac5-tp`: TP=2 multiprocess all-reduce test.
- `task-ac6-cuda-graph` / `task-ac6-hwrun`: Graph capture + H200 replay.
- `task-ac1b-probe`, `task-ac8-*`, `task-ac12-quality`: Hardware/analyze gates.
- `task-ac9-baseline`, `task-ac10-radix`, `task-ac11-compare`: Stretch.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The pattern (real table + real selector bound with synthetic ChannelMask + fake backend via object.__new__) already appears in `test_real_logical_domain_scoring_and_adapter_roundtrip` (line 4320). No new generalizable lesson beyond what is already in the codebase.
