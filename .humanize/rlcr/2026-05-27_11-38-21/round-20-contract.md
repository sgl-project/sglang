# Round 20 Contract

## Mainline Objective

Resolve `ds_topk_indices_out` in `_select_topk_indices` through the same
real `ForwardContext` metadata path that Round 19 used for `ds_graph_state`,
so the production CUDA-graph capture/replay writes into the metadata-owned
buffer (not a per-call `torch.empty_like(...)` fallback).

Codex Round-19-review gap: the current `_select_topk_indices` resolves
`ds_out` via `forward_batch.ds_topk_indices_out` (None during capture
because the capture-time `ForwardBatch` does not carry it), then via the
non-existent `forward_batch.attn_backend.forward_metadata`, then falls
through to `torch.empty_like(selected_indices)`. Codex's CUDA probe
confirmed `torch.empty_like` is called and the returned tensor does NOT
alias the metadata-owned buffer published through `ForwardContext`.

The new Round 19 test masked this by manually setting
`forward_batch.ds_topk_indices_out = ds_topk_out` before capture —
which is not what `cuda_graph_runner.py` does in real production.

## Target ACs

- **AC-6** — production `_select_topk_indices` must write into the
  metadata-owned `ds_topk_indices_out` buffer when the only metadata
  source is `ForwardContext.get_attn_backend().forward_metadata`, with 0
  new CUDA allocations and 0 `torch.empty_like` calls.

## Required Implementation

### Fix 1: `ds_topk_indices_out` resolution via ForwardContext

- In `_select_topk_indices` (deepseek_v2.py), reuse the
  `_dsa_metadata` already resolved during the `ds_graph_state` lookup
  (or re-resolve through `ForwardContext` if it was not). Resolve
  `ds_out` in this priority order:
  1. `forward_batch.ds_topk_indices_out` (dynamic non-graph forward).
  2. `_dsa_metadata.ds_topk_indices_out` (CUDA-graph capture/replay
     path via `ForwardContext`).
  3. **Last resort only**: `torch.empty_like(selected_indices)` —
     gated to dynamic CPU tests where neither metadata source exists.
- Remove the unreachable `forward_batch.attn_backend.forward_metadata`
  branch, which production never satisfies and which Round 19 left in
  place inadvertently.
- Hoist the `_dsa_metadata` resolution slightly so both
  `ds_graph_state` and `ds_topk_indices_out` use the same source.

### Fix 2: Update the existing zero-alloc test

- `test_select_topk_indices_zero_allocs_production_path`: remove the
  manual `forward_batch.ds_topk_indices_out = ds_topk_out` setup; the
  buffer must be reached only via `ForwardContext`. The test must
  still pass — 5 replays with 0 new CUDA allocations.

### Fix 3: Add the regression test Codex requested

- `test_select_topk_indices_uses_metadata_ds_topk_indices_out_via_forward_context`:
  publish both `ds_graph_state` AND `ds_topk_indices_out` only through
  `ForwardContext.attn_backend.forward_metadata` (no
  `forward_batch.ds_topk_indices_out`). Spy `torch.empty_like` and
  assert the spy is not called by `_select_topk_indices`. Assert the
  returned `ds_out` is identical (same `data_ptr`) to the metadata-owned
  buffer.

## Tests

- Existing 200 tests must still pass.
- 1 new regression + 1 updated test (no manual buffer pre-set).
- Expect ≥ 201 passed.

## Success Criteria

1. `_select_topk_indices` returns the metadata-owned
   `ds_topk_indices_out` buffer (verified by `data_ptr` identity)
   when the only metadata source is `ForwardContext`.
2. `torch.empty_like` is NOT called inside `_select_topk_indices`
   during real-capture-style forwards (verified by spy).
3. The CUDA-graph replay zero-alloc test still passes with the
   manual buffer pre-set removed.
4. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`
   ≥ 201 passed, 0 failed.

## Blocking Issues

None.

## Queued (out of scope)

- AC-8 observability page-vs-token unit mix-up in `_publish_ds_request_summary`.
- Stale DS bind/runtime comments mentioning `req_to_token_pool.size`.
- Token-label lifetime docs (overwrite-before-read vs invalidate-before-selection).
- `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1-hwtest`, `task-ac1b-probe`,
  `task-ac8-*`, `task-ac10-radix`, `task-ac11-compare`,
  `task-ac12-quality` — hardware-gated.
