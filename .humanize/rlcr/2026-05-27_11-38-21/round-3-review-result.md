# Round 3 Review Result

Mainline Progress Verdict: ADVANCED

Round 3 fixed the two concrete AC-1 hook correctness bugs from the Round 2 review. `_write_token_labels` now reshapes `kv_b_proj` output as per-head `[K_nope | V]` before slicing K-noPE (`python/sglang/srt/layers/attention/dsa_backend.py:1431-1436`), and the TRT-LLM FP8 path preserves the latent K before `mla_quantize_and_rope_for_fp8` (`python/sglang/srt/layers/attention/dsa_backend.py:2196-2233`). The new sentinel regression is targeted and would catch the old flat-slice bug.

AC-1 is still not closed. The code-level bug fixes are real, but Round 3 did not satisfy its own contract or the plan's live-path verification requirement: no committed test invokes the `forward_extend`, `forward_decode`, or TRT-LLM hook call sites with `save_kv_cache=True`, and the H200 real forward population test remains pending.

## Mainline Gaps

1. **AC-1 live hook call-site verification is missing.**

   Evidence:
   - The three production hook calls are at `python/sglang/srt/layers/attention/dsa_backend.py:1504-1510`, `python/sglang/srt/layers/attention/dsa_backend.py:1703-1709`, and `python/sglang/srt/layers/attention/dsa_backend.py:2232-2233`.
   - The new unit coverage calls `_write_token_labels` directly at `test/registered/unit/layers/attention/test_double_sparsity_unit.py:4482`, `:4516`, and `:4587`.
   - The only nearby `forward_decode` unit path calls `k=None` and `save_kv_cache=False`, so it explicitly skips the KV-write block and the label hook.
   - Round 3 contract success criterion 5 says "All 3 hook sites (extend, decode, TRT-LLM) verified by tests"; that criterion is not met.

   Impact: a future change could remove one production `_write_token_labels(...)` call, or regress the TRT-LLM `k_for_labels` argument, while the current 156-test suite would still pass. This blocks AC-1 closure because the plan requires live population after `forward_extend` and `forward_decode`, not just private-helper correctness.

2. **`task-ac1-hwtest` remains unfinished.**

   Claude's summary correctly says the H200 hardware forward test is pending. The original plan makes this part of AC-1 verification: run a real `forward_extend` path and assert `token_label_table.signatures[layer_id, out_cache_loc]` is non-zero for each written slot. Do not move AC-1 to completed until that evidence exists.

3. **The original lower-bound ACs are still incomplete.**

   AC-1b, AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8, and AC-12 remain required lower-bound work. AC-9 through AC-11 remain stretch but tracked. These are pending tasks, not accepted deferrals.

## Blocking Side Issues

1. **AC-1 cannot safely feed downstream work until the live paths are tested.**

   Required fix: add call-site tests before starting AC-2/AC-3/AC-7. Use a shared fake backend fixture with a real `TokenLabelTable`, fake `token_to_kv_pool`, non-contiguous `out_cache_loc`, and a fake layer whose `kv_b_proj` emits per-head K/V sentinel data.

   The required tests are:
   - `forward_extend` with `save_kv_cache=True`: patch the selected attention kernel method to return a dummy output, assert the fake KV pool was called, and assert the table rows at `out_cache_loc` were written with K-noPE values.
   - `forward_decode` with `save_kv_cache=True`: same assertion pattern for the decode call site.
   - TRT-LLM FP8 path: patch `mla_quantize_and_rope_for_fp8` and `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` to avoid hardware kernels, instrument `_write_token_labels`, and assert the tensor passed to the hook equals the pre-quantized latent K, not the quantized `k`.
   - Negative check: with the hook disabled or `save_kv_cache=False`, the table remains at initialization default.

   After those unit tests pass, run `task-ac1-hwtest` on H200 against real V3.2 and record the exact command/log evidence in the tracker.

## Queued Side Issues

1. **AC-6 graph helper is still stale.**

   `capture_decode_step` still needs the logical-to-physical `req_to_token` path before `task-ac6-cuda-graph`. This remains queued until AC-6.

2. **AC-8 observability still uses page-named token metrics.**

   `_publish_ds_request_summary` still reports token selections through page-named fields and computes sparsity against page counts. This remains queued until AC-8 metrics cleanup.

## Required Implementation Plan

1. Finish AC-1 verification now: add the three call-site tests above, rerun `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`, then run the H200 `forward_extend` population test.
2. Only after AC-1 is verified, move in dependency order: AC-2 lifetime/stale-slot checks, AC-3 ownership mask and boundary test, AC-7 short-seq bypass, AC-4 Method 1 Q+K calibration plus H200 mask generation, AC-5 TP multiprocess harness, AC-6 graph capture plus H200 replay, AC-1b chunked-prefill probe, AC-8 bench/quality smoke, and AC-12 full NIAH/MMLU quality gate.
3. Leave AC-9 through AC-11 tracked as stretch work, but do not claim loop completion until the lower bound is closed.

## Goal Alignment Summary

```text
ACs: 3/15 addressed | Forgotten items: 0 | Unjustified deferrals: 0
```

Closed: AC-0 and AC-13. Advanced but open: AC-1. The tracker contains the remaining plan tasks in Active; no original task is missing from Active/Completed/Deferred, and no explicit deferral is currently justified or accepted.

## Tracker Update

I updated the mutable section of `goal-tracker.md`:
- Kept `task-m1-hook` active as `partial-r3-review` instead of completed.
- Removed the AC-1 row from Completed and Verified.
- Added a Round 3 Review plan-evolution entry explaining why AC-1 remains open.
- Added the missing live hook call-site verification as a Blocking Side Issue.

Validation run:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
156 passed, 24 warnings in 11.47s
```

NOT COMPLETE
