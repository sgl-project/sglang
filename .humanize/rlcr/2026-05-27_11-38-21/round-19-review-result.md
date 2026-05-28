# Round 19 Goal Alignment Review

Mainline Progress Verdict: ADVANCED

Round 19 closed the two Round 18 blockers for AC-6: `_select_topk_indices` now reaches `retrieve_topk_graph_safe` through the real `ForwardContext` metadata path, and production `int64` `req_pool_indices` / `seq_lens` are handled through preallocated int32 scratch. The new `logical_to_physical` Triton path also works on CUDA and the reported unit suite passes.

AC-6 is still not closed. The physical output buffer (`ds_topk_indices_out`) still uses the old synthetic-object lookup path and can miss the metadata-owned buffer in real CUDA-graph capture.

## Part 1: Goal Tracker Audit

### 1.1 Acceptance Criteria Status

| AC | Status | Evidence if MET | Blocker if NOT MET | Justification if DEFERRED |
|----|--------|-----------------|--------------------|----------------------------|
| AC-0 | MET | Token-level architecture + slot authority verified in earlier reviews; current `test_double_sparsity_unit.py` passes 200/200. | - | - |
| AC-1 | PARTIAL | `task-m1-hook` call-site tests verified all three KV-write hooks. | `task-ac1-hwtest` and AC-8 selector-read smoke remain pending on real H200 server path. | - |
| AC-1b | NOT MET | - | Chunked-prefill probe has not run. | - |
| AC-2 | MET | Round 7 verified live invalidation before retrieve; stale-slot tests and production hook spy passed. | - | - |
| AC-3 | MET | Logical-domain `req_to_token` range-mask test and contamination negative were verified. | - | - |
| AC-4 | PARTIAL | Calibration coding path verified in Round 13. | H200 calibration run and `/models/dsv32-fp8-channel-mask.safetensors` generation/validation are pending. | - |
| AC-5 | MET | TP=2 multiprocess integration test verified positive, negative, and physical permutation cases. | - | - |
| AC-6 | PARTIAL | Round 19 target tests pass; graph-safe selector is reached via `ForwardContext`; production int64 inputs are copied into scratch. | `ds_topk_indices_out` still misses real `ForwardContext` metadata in CUDA-graph capture; full conc=64 V3.2 hardware run also pending. | - |
| AC-7 | MET | First-decode-after-short-prefill proof verified in Round 9. | - | - |
| AC-8 | NOT MET | - | `bench_serving` conc 16/32/64 and lightweight quality smoke have not run. | - |
| AC-9 | NOT MET | - | DSA baseline JSON not produced. | - |
| AC-10 | NOT MET | - | Radix-cache fixture and config flip not done. | - |
| AC-11 | NOT MET | - | Comparator row requires AC-9 and AC-10. | - |
| AC-12 | NOT MET | - | Full NIAH/MMLU quality gate not run. | - |
| AC-13 | MET | Regression suite now passes 200/200, exceeding the original 150-test gate after migration. | - | - |

### 1.2 Forgotten Items Detection

No original-plan task is missing from Active, Completed, or the plan-evolution history after this review update.

Tracker drift found and corrected:
- Claude marked `task-ac6-cuda-graph` complete in the Round 19 summary/tracker, but the production CUDA-graph physical-output buffer path is not verified and still falls back to a lazy `torch.empty_like` allocation when only real `ForwardContext` metadata is provided.

### 1.3 Deferred Items Audit

`Explicitly Deferred` is empty. This is correct. Hardware-gated work is still active/pending, not accepted as deferred.

### 1.4 Goal Completion Summary

```text
Acceptance Criteria: 6/15 met (0 deferred)
Partial ACs: 3 (AC-1, AC-4, AC-6)
Active Tasks: 11 remaining
Estimated remaining rounds: 6-9, assuming H200 cluster availability and no AC-8/AC-12 quality regression
Critical blockers: AC-6 output-buffer metadata path; H200 hardware gates for AC-1/AC-4/AC-6/AC-8/AC-12
```

## Part 2: Mainline Drift Audit

The current round objective is clear and singular: make the production `_select_topk_indices` path CUDA-graph safe for AC-6. Claude has been advancing the mainline, not mostly clearing unrelated side issues. The fresh `logical_to_physical` Triton path is adjacent to AC-6 because the adapter is inside the captured DS selection path.

The remaining issue is a mainline AC-6 gap, not a side issue.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 0
Queued Side Issues: 3
```

Queued side issues:
- AC-8 observability still reports token selections through page-named fields and computes sparsity against page counts in `_publish_ds_request_summary`.
- Stale DS bind/runtime comments still mention `req_to_token_pool.size`.
- `token_label_table.py` lifetime docs still describe overwrite-before-read rather than invalidate-before-selection.

## Part 3: Implementation Review

### Mainline Gap

1. **CUDA-graph capture still does not use the metadata-owned `ds_topk_indices_out` buffer.**

Evidence:
- Real full-graph capture creates a local `ForwardBatch` without `attn_backend` or DS output fields (`python/sglang/srt/model_executor/cuda_graph_runner.py:1020-1048`) and publishes the attention backend through `ForwardContext` (`cuda_graph_runner.py:1062-1076`).
- The DSA capture initializer allocates `DSAMetadata.ds_topk_indices_out` and `DSAMetadata.ds_graph_state` (`python/sglang/srt/layers/attention/dsa_backend.py:1044-1077`) but cannot attach either to the local `ForwardBatch` because `init_forward_metadata_capture_cuda_graph` does not receive that object.
- Round 19 fixed `ds_graph_state` by reading `ForwardContext(...).attn_backend.forward_metadata.ds_graph_state` (`python/sglang/srt/models/deepseek_v2.py:2136-2148`).
- The physical output buffer did not get the same fix. `_select_topk_indices` still resolves `ds_topk_indices_out` only from `forward_batch.ds_topk_indices_out`, then from synthetic `forward_batch.attn_backend.forward_metadata`, then allocates `torch.empty_like(selected_indices)` (`deepseek_v2.py:2234-2247`). Real `ForwardBatch` has no `attn_backend` field (`python/sglang/srt/model_executor/forward_batch_info.py:273-288`).
- The new CUDA graph test hides this by manually setting `forward_batch.ds_topk_indices_out = ds_topk_out` before capture (`test/registered/unit/layers/attention/test_double_sparsity_unit.py:3500-3508`).

Verification probe:

```text
ForwardContext metadata contained ds_topk_indices_out.
forward_batch.ds_topk_indices_out was intentionally absent.
torch.empty_like calls: 1
returned_is_metadata_buffer: False
returned_is_forward_batch_buffer: True
```

Consequence:
- The Round 19 claim that the production capture path uses the metadata-owned physical output buffer is false.
- In real `cuda_graph_runner` capture, the first warmup can attach a lazy buffer to the capture-local `ForwardBatch`; that object is not retained in `self.output_buffers`, while the metadata-owned buffer is retained on the attention backend but unused. A CUDA graph should write into persistent pre-owned buffers, not a warmup-only fallback that exists because metadata lookup missed the real source.
- This keeps `task-ac6-cuda-graph` open before `task-ac6-hwrun`.

Required fix:
- Resolve `ds_topk_indices_out` from the same real metadata source as `ds_graph_state`: first `forward_batch.ds_topk_indices_out`, then active `ForwardContext` backend `forward_metadata.ds_topk_indices_out`. Remove or stop relying on the synthetic `forward_batch.attn_backend` branch.
- Keep `ds_out` slicing by `bs`, but the base tensor should be the metadata-owned buffer in capture/replay.
- Add a regression that publishes both `ds_graph_state` and `ds_topk_indices_out` only through `ForwardContext`, does not pre-set `forward_batch.ds_topk_indices_out`, asserts `torch.empty_like` is not called, and asserts the returned tensor aliases metadata `ds_topk_indices_out`.
- Keep the existing CUDA graph replay allocation test, but remove its manual `forward_batch.ds_topk_indices_out` setup so it matches the real capture object path.

### Verified Claims

- `test_select_topk_indices_uses_graph_safe_via_forward_context` and `test_select_topk_indices_zero_allocs_production_path` both pass.
- Full unit suite passes:

```text
PYTHONPATH=python pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py
200 passed, 24 warnings in 11.91s
```

- A CUDA probe for `logical_to_physical` with an invalid pool index returned `error_count=1` and `-1` padded the bad row, so the new Triton adapter path is functional for the checked case.

## Part 4: Goal Tracker Update Requests

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 19 Review`.
- Replaced the Round 19 evolution row with the verified state: Round 18 blockers were fixed, but `ds_topk_indices_out` still misses the real metadata path.
- Changed `task-ac6-cuda-graph` from `complete` back to `pending`.

Rejected tracker change:
- Rejected Claude's implied completion of `task-ac6-cuda-graph` and unblocking of `task-ac6-hwrun` because the physical output buffer is still not resolved from production `ForwardContext` metadata.

## Part 5: Progress Stagnation Check

Not stagnating. Rounds 16-19 have stayed on AC-6 and each round closed concrete, verified blockers: host sync, scratch-backed selector, production dtype casts, production graph-state lookup, and int64 inputs. The recurring pattern is test fixtures being too synthetic around production object ownership, but development is still making measurable mainline progress.

## Action Items

Mainline Gaps:
- Fix `ds_topk_indices_out` lookup to use real `ForwardContext` metadata during CUDA-graph capture/replay, then add a regression without pre-setting `forward_batch.ds_topk_indices_out`.

Blocking Side Issues:
- None separate from the mainline AC-6 gap.

Queued Side Issues:
- Fix AC-8 observability before starting AC-8 server/quality smoke.
- Clean stale DS bind/runtime comments when touching that area.
- Update token-label lifetime docs when touching token-label cache code.

NOT COMPLETE
