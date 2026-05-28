# Round 4 Contract

## Mainline Objective

Complete AC-1 verification by adding call-site tests for `forward_extend`, `forward_decode`, and the TRT-LLM FP8 path that confirm the production `_write_token_labels` invocations fire with the correct tensors and populate the label table.

## Target AC

- **AC-1** (sole target): All 3 production hook call sites (`forward_extend` L1510, `forward_decode` L1709, `_forward_trtllm` L2233) exercised by unit tests with `save_kv_cache=True`; table rows at `out_cache_loc` populated with non-zero K-noPE values; TRT-LLM FP8 path asserted to pass pre-quantized latent k (not FP8 output) to the hook.

## Blocking Issues

1. **No call-site test exercises the production hook lines.** Round 3 tests call `_write_token_labels` directly. A future removal of one production call would not be caught. Fix: add `TestAC1CallSites` with forward_extend + forward_decode + TRT-LLM FP8 tests and one negative (`save_kv_cache=False`).

## Queued Issues (out of scope)

- `task-ac1-hwtest` (H200 hardware population test) — pending hardware access; cannot run here
- AC-2 lifetime/stale-slot, AC-3 ownership mask, AC-7 bypass, AC-4 calibration, all others

## Success Criteria

1. `TestAC1CallSites.test_forward_extend_writes_token_labels` — calls `forward_extend` with `save_kv_cache=True`; asserts table populated at `out_cache_loc`.
2. `TestAC1CallSites.test_forward_decode_writes_token_labels` — calls `forward_decode` with `save_kv_cache=True`; asserts table populated at `out_cache_loc`.
3. `TestAC1CallSites.test_trtllm_hook_receives_pre_quantized_k` — calls `_forward_trtllm` with FP8 dtype and `save_kv_cache=True`; spy asserts the tensor reaching `_write_token_labels` is the original float latent k, not the FP8-overwritten one.
4. `TestAC1CallSites.test_no_labels_when_save_kv_cache_false` — negative: `save_kv_cache=False` leaves table at zeros.
5. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` → 160 passed, 0 failed.
