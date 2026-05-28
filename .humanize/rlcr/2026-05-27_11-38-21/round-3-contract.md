# Round 3 Contract

## Mainline Objective

Close AC-1 (M1 live token-label cache population) by fixing the two `_write_token_labels` bugs reopened by Round-2 Codex review.

## Target ACs

- **AC-1** (primary): After `forward_extend`, `forward_decode`, and TRT-LLM paths, `token_label_table.signatures[layer_id, out_cache_loc, :, :]` contains correct per-head K-noPE values (not V columns or quantized FP8 data).

## Blocking Issues (must fix before AC-1 can close)

1. **Wrong kv_b_proj extraction**: `_write_token_labels` slices flat output before reshaping. Since the layout is per-head `[K_nope | V]`, the flat slice writes V columns as K labels for the second half of heads. Fix: reshape to `[T, H_local, qk_nope_head_dim + v_head_dim]` first, then slice `[..., :qk_nope_head_dim]`.

2. **TRT-LLM FP8 path passes post-quantized k**: `_forward_trtllm` calls `_write_token_labels` with `k` after `mla_quantize_and_rope_for_fp8` has overwritten it with FP8 cache data. Fix: save the pre-quantized latent `k` before the FP8 block and pass it to the hook.

## Queued Issues (explicitly out of scope this round)

- AC-6 graph helper stale (`capture_decode_step` missing req_to_token)
- AC-8 observability page-named fields
- All other ACs (AC-2, AC-3, AC-7, AC-4, AC-5)

## Success Criteria

1. `_write_token_labels` uses `layer.v_head_dim` for correct per-head reshape.
2. TRT-LLM FP8 path passes pre-quantized latent k to the hook, not FP8 k.
3. New sentinel regression: `kv_b_proj` output with V sentinel values (999.0) → signatures contain zero/K values, not sentinels.
4. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` → 156+ passed, 0 failed.
5. All 3 hook sites (extend, decode, TRT-LLM) verified by tests.
