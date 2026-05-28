# Round 16 Contract

## Mainline Objective

Fix the two AC-6 CUDA graph capture-safety bugs identified by Codex review of Round 15:

**Gap 1 — Illegal host sync**: `seq_lens.max().item()` in `_compute_logical_token_scores`
(selection_kernel.py:361) is a host sync illegal during `torch.cuda.graph` capture.
Fix: add `max_seq_len: int = 0` parameter; use it when nonzero, skip `.item()`.

**Gap 2 — Dynamic allocations inside captured region**: `retrieve_topk` allocates 48 new
CUDA tensors during the captured region; graph replay may break on allocation reuse.
Fix: create `retrieve_topk_graph_safe` that writes into caller-owned `DSGraphState`
scratch buffers; extend `DSGraphState` with all fixed-shape scratch needed by logical scoring.

## Target ACs

- **AC-6** (CUDA graph decode path): `capture_decode_step` must produce a CUDA graph
  with ZERO host syncs and ZERO new allocations in the captured region.

## Scope

### Fix 1: Remove host sync from `_compute_logical_token_scores`
- File: `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
- Add `max_seq_len: int = 0` parameter to `_compute_logical_token_scores`
- When `max_seq_len > 0`, skip `seq_lens.max().item()`; use the static value directly
- Update call sites: `retrieve_topk_via_labels` and `retrieve_topk_graph_safe`

### Fix 2: Add `retrieve_topk_via_labels` max_seq_len threading
- File: `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
- Add `max_seq_len: int = 0` parameter to `retrieve_topk_via_labels`
- Pass through to `_compute_logical_token_scores`

### Fix 3: Create `retrieve_topk_graph_safe`
- File: `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
- New function that accepts pre-allocated scratch and writes results in-place
- Zero host syncs, zero dynamic allocations in the captured path
- Scratch fields: `scratch_scores`, `scratch_q_proj`, `scratch_physical`, `scratch_written`,
  `scratch_topk_values`, `scratch_topk_indices`, `scratch_sorted`

### Fix 4: Extend `DSGraphState` + `allocate_graph_state`
- File: `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py`
- Add scratch fields to `DSGraphState`: all tensors needed by `retrieve_topk_graph_safe`
- Add `max_seq_len: int = 0` field to `DSGraphState`
- Extend `allocate_graph_state` to accept `max_seq_len`, `num_local_heads`, `label_dim`

### Fix 5: Update `capture_decode_step`
- File: `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py`
- Accept `max_seq_len: int = 0` parameter
- On CUDA path: use `retrieve_topk_graph_safe` (no host syncs, no allocs)
- On CPU/eager path: keep existing `retrieve_topk` call (unchanged)

### Fix 6: Add two CUDA-only tests to `TestCUDAGraphCapture`
- File: `test/registered/unit/layers/attention/test_double_sparsity_unit.py`
- Test 1 (`test_cuda_graph_100_step_replay_matches_eager`, `@skipUnless CUDA`):
  capture on CUDA, replay 100 steps, every result bit-equal to eager path
- Test 2 (`test_cuda_graph_replay_zero_allocations`, `@skipUnless CUDA`):
  `assert_no_alloc_in_region` wrapping actual graph replay → 0 new allocations

## Blocking Issues

None.

## Success Criteria

1. `_compute_logical_token_scores` accepts `max_seq_len: int = 0`; when nonzero, no `.item()` call.
2. `retrieve_topk_graph_safe` exists, writes results into `DSGraphState` scratch, zero host syncs.
3. `DSGraphState` has all scratch fields; `allocate_graph_state` sizes them from `max_seq_len`, `num_local_heads`, `label_dim`.
4. `capture_decode_step` accepts `max_seq_len: int = 0`; CUDA path uses `retrieve_topk_graph_safe`.
5. Two new `@skipUnless CUDA` tests pass when CUDA is available.
6. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` ≥ 193 passed, 0 failed.
7. No regressions in existing 193 tests.
