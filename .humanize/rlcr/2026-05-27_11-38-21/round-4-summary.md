# Round 4 Summary

## Work Completed

### AC-1 call-site verification — `TestAC1CallSites`

Added four tests that exercise the three production `_write_token_labels` call sites:

**`test_forward_extend_writes_token_labels`**
Calls `forward_extend` with `use_mha=True` and `save_kv_cache=True`. The MHA path
was chosen because it routes through the KV-write block (lines 1496-1510) and then
hits `_forward_standard_mha` (patched), avoiding all complex transform/paging logic
while still firing the production hook. Asserts `table.written[0, 5]` and
`table.written[0, 10]` are True, and signatures are non-zero.

**`test_forward_decode_writes_token_labels`**
Calls `forward_decode` with `dsa_decode_impl="flashmla_kv"`, `SGLANG_DSA_FUSE_TOPK=1`
(which routes `page_table_1 = topk_indices`, skipping `transform_index_page_table_decode`
and its metadata requirements), and `save_kv_cache=True`. `_forward_flashmla_kv` is
patched to return a dummy output. Asserts table populated at slots 7 and 15.

**`test_trtllm_hook_receives_pre_quantized_k`**
Calls `_forward_trtllm` directly (same production call site) with
`kv_cache_dtype=torch.float8_e4m3fn`. Patches `mla_quantize_and_rope_for_fp8` to
return a `float8_e4m3fn` k tensor (simulating the FP8 overwrite), and patches
`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` to avoid hardware.
Replaces `backend._write_token_labels` with a spy that records the `k` argument.
Asserts: exactly one call; dtype is `float32` (not `float8_e4m3fn`); values equal
the original `torch.ones(T, 1, kv_lora_rank)` latent k — proving `k_for_labels` is
saved before `mla_quantize_and_rope_for_fp8` runs.

**`test_no_labels_when_save_kv_cache_false`**
Same `forward_extend` setup but `save_kv_cache=False`. Asserts table.written remains
all-False — the KV-write block is guarded by `if save_kv_cache:` and the hook must
not fire outside that guard.

## Files Changed

- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — add `TestAC1CallSites` (4 tests)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
160 passed, 0 failed
```

Commit: `ef16fa441`

## Remaining Items

- `task-ac1-hwtest`: hardware test on H200 with real `forward_extend` against V3.2 weights — pending hardware access; cannot be automated in unit suite.
- Next coding work (by dependency order): AC-2 lifetime/stale-slot, AC-3 M2 range mask, AC-7 short-seq bypass, AC-4 calibration.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Call-site test structure pattern (object.__new__ + instance-level patching + spy) is standard Python mocking; no new project-specific lesson needed.
