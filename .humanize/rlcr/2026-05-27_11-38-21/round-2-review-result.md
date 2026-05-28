# Round 2 Review Result

Mainline Progress Verdict: ADVANCED

Round 2 fixed the AC-0 slot-count authority bug: the label table is now sized from the physical token-to-KV slot address space, and the committed non-contiguous slot regression covers the failure that escaped Round 1. The 155-test unit suite passes.

AC-1 is not closed. The hooks are present at the three claimed `set_mla_kv_buffer` sites, but the new projection helper extracts K labels from `kv_b_proj` incorrectly, so live labels are wrong for multiple heads. Downstream AC-2/AC-3/AC-7/AC-5/AC-6/AC-8 cannot safely build on this hook yet.

## Mainline Gaps

1. **AC-1 hook writes V columns as K labels for some heads.**

   Evidence:
   - `python/sglang/srt/layers/attention/dsa_backend.py:1429-1433` does:

     ```python
     kv_proj_out = kv_b_proj(k_latent)[0]
     k_nope = kv_proj_out[:, : H_local * nope_dim].view(T, H_local, nope_dim)
     ```

   - The existing MLA code shows the actual `kv_b_proj` layout is per-head `[K_nope | V]`: `python/sglang/srt/layers/attention/tokenspeed_mla_backend.py:228-233` reshapes to `[-1, num_local_heads, qk_nope_head_dim + v_head_dim]` first, then slices `[..., : qk_nope_head_dim]`.

   Reproducer against the current hook:

   ```text
   kv_b_proj output per-head:
   h0 K=[11,12], h0 V=[91,92], h1 K=[21,22], h1 V=[93,94]

   current table.signatures[slot]:
   [[11.0, 12.0], [91.0, 92.0]]

   expected:
   [[11.0, 12.0], [21.0, 22.0]]
   ```

   Required fix: reshape before slicing:

   ```python
   head_width = nope_dim + layer.v_head_dim
   kv = kv_proj_out.view(T, H_local, head_width)
   k_nope = kv[..., :nope_dim].contiguous()
   ```

   Add a regression with sentinel V values so flat slicing fails deterministically.

2. **The TRT-LLM FP8 path calls the hook after overwriting `k` with FP8 cache data.**

   Evidence:
   - `python/sglang/srt/layers/attention/dsa_backend.py:2203-2213` reassigns `q, k, k_rope = mla_quantize_and_rope_for_fp8(...)`.
   - `python/sglang/srt/layers/attention/dsa_backend.py:2226-2227` then writes the KV cache and calls `_write_token_labels(layer, cache_loc, k)`.

   The label projection must consume the normalized latent K, not the post-quantized FP8 cache tensor. Save `k_for_labels = k` before the FP8 quantize/rope call, continue passing the FP8 `k` to `set_mla_kv_buffer`, and pass `k_for_labels` to `_write_token_labels`.

3. **AC-1 real-path verification is still pending.**

   The added tests call `_write_token_labels` directly via `object.__new__`, but AC-1 requires the actual `forward_extend` and `forward_decode` paths to populate `token_label_table.signatures[layer_id, out_cache_loc]`. Keep `task-ac1-hwtest` active after the hook fix; do not close AC-1 until at least the real forward-path population test passes.

4. **The original lower-bound ACs remain unfinished.**

   AC-1b, AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8, and AC-12 are still active required work. AC-9 through AC-11 remain stretch but tracked. Claude’s “Remaining Items” are not accepted deferrals and must not be treated as optional completion criteria.

## Blocking Side Issues

1. **Wrong `kv_b_proj` K extraction blocks all live-label work.**

   Fix `_write_token_labels` to match the established MLA reshape/slice pattern, then rerun the unit suite. This is blocking because every selector read after AC-1 depends on the label table containing projected K-noPE, not V channels or missing heads.

2. **TRT-LLM FP8 hook input is the wrong tensor.**

   Preserve the pre-quantized latent K for labels in `_forward_trtllm`. This blocks the “all three hook sites” claim for AC-1.

## Queued Side Issues

1. **AC-6 graph helper is still stale.**

   `capture_decode_step` still calls `selector.retrieve_topk` without the logical-to-physical `req_to_token` path. This remains queued until `task-ac6-cuda-graph`.

2. **AC-8 observability still uses page-named fields for token counts.**

   `_publish_ds_request_summary` reports `selected_pages` and computes sparsity against page counts while `valid_lengths` is now token top-K. This remains queued until AC-8 metrics cleanup.

## Required Implementation Plan

1. Fix `_write_token_labels` to reshape `kv_b_proj(k_latent)[0]` as `[T, H_local, qk_nope_head_dim + layer.v_head_dim]`, then slice `[..., :qk_nope_head_dim]`.
2. Add a per-head interleave regression where V columns have sentinel values and assert only K-noPE values are written.
3. In `_forward_trtllm`, preserve the original latent K before `mla_quantize_and_rope_for_fp8` and pass that preserved tensor to `_write_token_labels`.
4. Add tests that exercise the actual `forward_extend`, `forward_decode`, and TRT-LLM hook call sites, not only the private helper.
5. Rerun `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`.
6. Continue in dependency order: AC-1 hardware forward test, AC-2 lifetime/stale-slot checks, AC-3 ownership mask, AC-7 short-seq bypass, AC-4 calibration and H200 mask generation, AC-5 TP harness, AC-6 graph capture, AC-1b probe, AC-8 bench/quality smoke, and AC-12 full quality.

## Goal Alignment Summary

```text
ACs: 3/15 addressed | Forgotten items: 0 | Unjustified deferrals: 0
```

Closed after this review: AC-0 and AC-13. AC-1 was attempted but reopened. The remaining lower-bound ACs are tracked as active work, not accepted deferrals.

## Tracker Update

I updated the mutable section of `goal-tracker.md`:
- Marked AC-0 `task-ac0-slot-authority` verified in Round 2.
- Reopened `task-m1-hook`.
- Removed AC-1 from Completed and Verified.
- Added the flat `kv_b_proj` slicing bug and TRT-LLM FP8 hook-input bug to Blocking Side Issues.
- Added a Round 2 Review entry to the Plan Evolution Log.

Validation run:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
155 passed, 24 warnings in 11.41s
```

NOT COMPLETE
