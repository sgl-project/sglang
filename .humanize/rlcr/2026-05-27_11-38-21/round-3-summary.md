# Round 3 Summary

## Work Completed

### AC-1 hook correctness fixes

Two bugs in `_write_token_labels` caused incorrect label population. Both are now fixed.

**Bug 1 — reshape before slice (wrong K extraction)**

`kv_b_proj` output layout is per-head `[K_nope | V]`: head 0 K, head 0 V, head 1 K, head 1 V, etc. The Round-2 code sliced `kv_proj_out[:, :H_local * nope_dim]` from the flat tensor before reshaping. For H_local=16, nope_dim=128, v_head_dim=128, the flat first 2048 values cover exactly 8 full heads worth of output (8 × 256), so the reshape to `(16, 128)` produces: head 0 → K0 (correct), head 1 → V0 (wrong), head 2 → K1 (correct), head 3 → V1 (wrong), etc.

Fix in `_write_token_labels`:
```python
# Before (wrong — flat slice then reshape)
k_nope = kv_proj_out[:, : H_local * nope_dim].view(T, H_local, nope_dim)

# After (correct — reshape first, then K-noPE prefix per head)
head_width = nope_dim + layer.v_head_dim
k_nope = kv_proj_out.view(T, H_local, head_width)[..., :nope_dim].contiguous()
```

`layer.v_head_dim` is set on `RadixAttention.attn_mha` in `deepseek_v2.py:1610` and equals `self.v_head_dim` (128 for V3.2).

**Bug 2 — TRT-LLM FP8 path passed post-quantized k to hook**

`_forward_trtllm` unconditionally called `_write_token_labels(layer, cache_loc, k)` after `mla_quantize_and_rope_for_fp8` overwrote `k` with FP8 cache data. The hook projects `k` through `kv_b_proj`, which expects the 512-d latent float K — feeding FP8 bytes produces garbage labels.

Fix: save `k_for_labels = k` before the FP8 block and pass it to the hook:
```python
k_for_labels = k  # preserve latent K before FP8 quantization
if self.kv_cache_dtype == torch.float8_e4m3fn:
    q, k, k_rope = mla_quantize_and_rope_for_fp8(q, q_rope, k.squeeze(1), ...)
...
if save_kv_cache:
    self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)
    self._write_token_labels(layer, cache_loc, k_for_labels)  # pre-quantized
```

For non-FP8 paths, `k_for_labels = k` and `k` is never reassigned, so the fix is a no-op on those paths.

### Sentinel regression test

Added `test_write_token_labels_extracts_k_nope_not_v_columns` to `TestAC1HookUnit`:
- `kv_b_proj` stub returns per-head layout `[K_nope | V_sentinel]` where V columns = 999.0
- Head 0 K = `[1,2,3,4]`, Head 1 K = `[5,6,7,8]`, all V = `[999,999,999,999]`
- Asserts: no 999.0 in signatures; head-0 = `[1,2,3,4]`; head-1 = `[5,6,7,8]`
- This test fails deterministically with the old flat-slice code

Also added `v_head_dim=nope_dim` to the existing `test_write_token_labels_populates_table` layer stub so it remains correct under the new `layer.v_head_dim` access.

## Files Changed

- `python/sglang/srt/layers/attention/dsa_backend.py` — reshape fix in `_write_token_labels`; `k_for_labels` save in `_forward_trtllm`
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — add `test_write_token_labels_extracts_k_nope_not_v_columns`; add `v_head_dim` to existing stub

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
156 passed, 0 failed
```

Commit: `9ac489af3`

## Remaining Items

`task-ac1-hwtest` (hardware forward test on H200) remains pending per the plan — AC-1 unit tests pass but Codex hardware verification is still pending. Next coding tasks by dependency order:

- AC-2: boot-time GB/rank log; stale-slot lifetime test
- AC-3: per-request range mask (M2); multi-request boundary test
- AC-7: short-seq MHA bypass
- AC-4: calibrate.py Method 1 Q+K hooks

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Both bugs were reshape ordering and tensor preservation errors; no new generalizable lesson beyond what standard MLA projection reshape patterns already imply.
