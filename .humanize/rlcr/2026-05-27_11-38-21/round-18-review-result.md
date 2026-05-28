# Round 18 Code Review

Mainline Progress Verdict: ADVANCED

Round 18 made real AC-6 progress on the helper itself: the fp16/bf16 production-dtype allocation gap in `retrieve_topk_graph_safe` is fixed, and the new zero-allocation production-dtype test passes on CUDA. However, `task-ac6-cuda-graph` is still not complete. The claimed production wiring is attached to a synthetic `forward_batch.attn_backend` attribute that production does not provide, and the forced graph-safe path rejects the real production `req_pool_indices` dtype.

## Mainline Gaps

1. **The graph-safe selector is still not reached through the real production object path.**

   Evidence:
   - `_select_topk_indices` looks for graph metadata only at `forward_batch.attn_backend.forward_metadata.ds_graph_state` (`python/sglang/srt/models/deepseek_v2.py:2123-2133`).
   - Real `ForwardBatch` has no `attn_backend` field (`python/sglang/srt/model_executor/forward_batch_info.py:274-330`). The only repository hit for `forward_batch.attn_backend` is the new test plus this Round 18 code.
   - Production forwards expose the attention backend through `ForwardContext`, not through `ForwardBatch`: see `cuda_graph_runner.py:1059-1068`, `model_runner.py:2715-2717`, and `forward_context.py:35-67`.
   - `dsa_backend.init_forward_metadata` attaches only `ds_topk_indices_out` onto `forward_batch` (`python/sglang/srt/layers/attention/dsa_backend.py:779-783`); it stores `ds_graph_state` only inside `DSAMetadata` (`dsa_backend.py:737-776`).
   - The new regression test explicitly mocks `forward_batch.attn_backend` (`test/registered/unit/layers/attention/test_double_sparsity_unit.py:3384-3429`), so it does not prove the production path.

   Verification probe:
   - With `ds_graph_state` available through `ForwardContext(attn_backend=...)` and no synthetic `forward_batch.attn_backend`, a CUDA spy around `retrieve_topk_graph_safe` recorded `spy_call_count 0`. The code fell back to `DoubleSparsitySelector.retrieve_topk` / `retrieve_topk_via_labels`.

   Consequence: real CUDA graph capture still records the allocating selector path, so `task-ac6-hwrun` and AC-8 remain blocked.

   Required implementation plan:
   - In `_select_topk_indices`, resolve DS metadata from real runtime state. Use `forward_batch.ds_graph_state` if present, otherwise use `has_forward_context()` / `get_attn_backend().forward_metadata`, mirroring the existing MHA bypass source of truth. Do not depend on `forward_batch.attn_backend`.
   - In `dsa_backend.init_forward_metadata`, also expose `forward_batch.ds_graph_state = ds_graph_state` next to `forward_batch.ds_topk_indices_out` for dynamic non-graph forwards. For capture/replay, rely on `ForwardContext(attn_backend=...)` because the capture initializer does not receive `forward_batch`.
   - Replace the current spy test with one that publishes only a real `ForwardContext` and a `ForwardBatch` without `attn_backend`; the spy must be called exactly once.
   - Add a CUDA allocation regression around the actual `_select_topk_indices` call with production metadata, not only around `retrieve_topk_graph_safe`.

2. **The new graph-safe fast path asserts on production request-pool dtype.**

   Evidence:
   - `retrieve_topk_graph_safe` now asserts `req_pool_indices.dtype == torch.int32` and `seq_lens.dtype == torch.int32` (`selection_kernel.py:745-753`).
   - Production request-pool tensors are `int64`: `ForwardBatch` documents `req_pool_indices` as int64 (`schedule_batch.py:1507`), scheduler constructs it as int64 (`scheduler.py:2277-2279`), and CUDA graph input buffers allocate it as int64 (`cuda_graph_runner.py:175-180`).
   - A CUDA probe forcing the new synthetic `forward_batch.attn_backend` path with real graph-buffer dtype fails before selection with `AssertionError: req_pool_indices must be int32, got torch.int64`.

   Consequence: after the metadata lookup bug above is fixed, the first production CUDA graph decode using the graph-safe path will either crash with an uncaught `AssertionError` or force developers back to the allocating fallback. This still blocks AC-6.

   Required implementation plan:
   - Extend `DSGraphState` with `scratch_req_pool_indices: int32[max_bs]`. In `_select_topk_indices`, copy `forward_batch.req_pool_indices` into that scratch view with `copy_` and pass the scratch to `retrieve_topk_graph_safe`.
   - Pass an int32 sequence-length tensor from metadata (`DSAMetadata.cache_seqlens_int32[:bs]`) instead of `forward_batch.seq_lens` when metadata is available. If supporting the dynamic `forward_batch.ds_graph_state` path without metadata, add `scratch_seq_lens: int32[max_bs]` and copy into it.
   - Keep `.to(...)` out of the captured selector region. The fix must be preallocated scratch plus `copy_`, or a Triton kernel contract that directly accepts int64 without allocation.
   - Add a CUDA production-path test where `forward_batch.req_pool_indices` and `forward_batch.seq_lens` are int64, matching scheduler/cudagraph inputs. It must prove `_select_topk_indices` calls `retrieve_topk_graph_safe`, returns the expected logical-to-physical output, and performs zero new CUDA allocations after warmup.

## Blocking Side Issues

None separate from the mainline AC-6 gaps. The two findings above are the current mainline blockers.

## Queued Side Issues

1. AC-8 observability still reports token selections through page-named fields and computes sparsity against page counts in `_publish_ds_request_summary`.
2. Cleanup: stale DS bind/runtime comments still mention `req_to_token_pool.size` as max-token authority.
3. Cleanup: token-label lifetime docs still describe overwrite-before-read rather than invalidate-before-selection.

These remain non-blocking for the next round and must not displace the AC-6 production wiring fix.

## Goal Alignment Summary

```text
ACs: 9/15 addressed (6 met, 3 partial: AC-1, AC-4, AC-6) | Forgotten items: 0 | Unjustified deferrals: 0
```

Status by AC:
- Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.
- Partial: AC-1, AC-4, AC-6.
- Not met: AC-1b, AC-8, AC-9, AC-10, AC-11, AC-12.

Hardware-gated items remain active pending work, not accepted deferrals. `task-ac6-cuda-graph` must stay active because the production coding path is still not graph-safe.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 18 Review`.
- Added a Round 18 Review plan-evolution row reopening `task-ac6-cuda-graph`.
- Changed `task-ac6-cuda-graph` in Active Tasks from `complete` back to `pending`.

## Validation Run

```text
PYTHONPATH=python pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py -k 'zero_allocs_production_dtypes or select_topk_indices_uses_graph_safe'
2 passed, 197 deselected
```

Additional CUDA probes:
- `ForwardContext` metadata path, no synthetic `forward_batch.attn_backend`: `retrieve_topk_graph_safe` spy call count was 0.
- Forced synthetic metadata path with production `int64` `req_pool_indices`: `_select_topk_indices` raised `AssertionError: req_pool_indices must be int32, got torch.int64`.

NOT COMPLETE
