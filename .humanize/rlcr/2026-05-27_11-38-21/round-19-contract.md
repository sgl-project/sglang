# Round 19 Contract

## Mainline Objective

Make Round 18's production wiring actually reach the production code path
and survive production tensor dtypes. Two Codex Round-18-review gaps:

1. The graph-safe gate lives at `forward_batch.attn_backend.forward_metadata`
   — but production `ForwardBatch` has no `attn_backend` field. The attention
   backend (and therefore `DSAMetadata.ds_graph_state`) is published through
   `ForwardContext`, exactly like the AC-7 MHA bypass.

2. The new fast path asserts `req_pool_indices.dtype == torch.int32`, but
   production tensors are int64 (per `schedule_batch.py:1507` and the CUDA
   graph buffer at `cuda_graph_runner.py:178`). The current code therefore
   raises `AssertionError` the first time the graph-safe path is reached
   with real input.

## Target ACs

- **AC-6** — the production decode path must call `retrieve_topk_graph_safe`
  when the DSA backend has allocated `ds_graph_state`, with 0 new CUDA
  allocations after warmup at production tensor dtypes (`int64`
  `req_pool_indices`, `int64`/`int32` `seq_lens`, fp16 sig, bf16 queries,
  `int32` `sparse_mask`).

## Required Implementation

### Fix 1: Resolve `ds_graph_state` via ForwardContext (deepseek_v2.py)

- Replace the `forward_batch.attn_backend.forward_metadata.ds_graph_state`
  lookup in `_select_topk_indices` with a two-source resolution:
  1. `getattr(forward_batch, "ds_graph_state", None)` — primary, set by
     `dsa_backend.init_forward_metadata` for dynamic non-graph forwards.
  2. `has_forward_context() and get_attn_backend().forward_metadata
     .ds_graph_state` — fallback for the CUDA-graph capture/replay path
     where `init_forward_metadata_capture_cuda_graph` does not receive a
     `forward_batch`.
- Same fallback should resolve `ds_topk_indices_out` (so dynamic and graph
  paths agree).

### Fix 2: Expose `ds_graph_state` on forward_batch (dsa_backend.py)

- In `init_forward_metadata` (extend/decode, ~line 770), set
  `forward_batch.ds_graph_state = ds_graph_state` next to the existing
  `forward_batch.ds_topk_indices_out` assignment, so the dynamic path can
  reach scratch via `forward_batch` without depending on
  `ForwardContext`.
- The CUDA-graph capture path (`init_forward_metadata_capture_cuda_graph`)
  cannot do this because it gets no `forward_batch` — that path relies on
  `ForwardContext(attn_backend=...)` set by the cuda_graph_runner before
  capture/replay, which is the standard published path.

### Fix 3: int32 scratch for production int64 inputs (cuda_graph.py + selection_kernel.py)

- Add two new fields to `DSGraphState`:
  - `scratch_req_pool_indices: int32[max_bs]`
  - `scratch_seq_lens: int32[max_bs]`
- Allocate them in `allocate_graph_state` when `max_seq_len > 0`.
- In `_select_topk_indices`, **before** calling `retrieve_topk_graph_safe`:
  - `state.scratch_req_pool_indices[:bs].copy_(forward_batch.req_pool_indices)`
    — `copy_` casts int64→int32 in-place without allocation.
  - Prefer `DSAMetadata.cache_seqlens_int32[:bs]` when the metadata source
    is `ForwardContext`. When falling back to dynamic
    `forward_batch.ds_graph_state`, copy `forward_batch.seq_lens` into
    `scratch_seq_lens[:bs]`.
- Pass the int32 scratch tensors to `retrieve_topk_graph_safe`; keep the
  asserts in the fast path (they now document the caller contract).

### Fix 4: Tests at production dtypes

- Replace `test_select_topk_indices_uses_graph_safe_when_metadata_state_present`
  with a version that publishes only a real `ForwardContext(attn_backend=...)`
  (no synthetic `forward_batch.attn_backend`) and passes int64
  `req_pool_indices` matching production. The spy must be called exactly
  once.
- Add `test_select_topk_indices_zero_allocs_production_path` (CUDA-only):
  same `ForwardContext` setup, real int64 `req_pool_indices`, fp16 sig +
  bf16 queries, `int32 sparse_mask`. Warm up; wrap second
  `_select_topk_indices` call in `assert_no_alloc_in_region`.

## Tests

- Existing 199 tests must still pass.
- 2 new / replaced tests above. Expect ≥ 200 passed.

## Success Criteria

1. `_select_topk_indices` invokes `retrieve_topk_graph_safe` via the
   `ForwardContext` source of truth (verified by spy on the dynamic
   import inside the kernel module).
2. Production-path CUDA test with int64 `req_pool_indices` + int64
   `seq_lens` + fp16 sig + bf16 queries + int32 sparse_mask shows 0 new
   CUDA allocations on the second `_select_topk_indices` call after
   warmup.
3. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`
   ≥ 200 passed, 0 failed.

## Blocking Issues

None.

## Queued (out of scope)

- AC-8 observability page-vs-token unit mix in `_publish_ds_request_summary`.
- Stale DS bind/runtime comments mentioning `req_to_token_pool.size`.
- Token-label lifetime docs (overwrite-before-read vs invalidate-before-selection).
- `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1-hwtest`, `task-ac1b-probe`,
  `task-ac8-*`, `task-ac10-radix`, `task-ac11-compare`,
  `task-ac12-quality` — all hardware-gated or downstream of AC-6 closure.
