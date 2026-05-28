# Round 17 Contract

## Mainline Objective

Implement a truly allocation-free `retrieve_topk_graph_safe` for AC-6: zero new CUDA
allocations when the function is called after a warm-up pass.

Two Codex gaps from Round 16 review:
1. `retrieve_topk_graph_safe` still calls `retrieve_topk_via_labels` internally → 42 new
   allocations even with pre-allocated output buffers.
2. `retrieve_topk_graph_safe` drops the `per_request_valid` / `sparse_mask` parameter →
   silently omits the M2 ownership mask from the captured graph path.

## Target ACs

- **AC-6** (CUDA graph decode path): zero-allocation warm-then-call path verified by
  `assert_no_alloc_in_region` wrapping the direct `retrieve_topk_graph_safe` call.

## Required Implementation

### Fix 1: New Triton kernel `_logical_score_kernel`
- File: `selection_kernel.py`
- Grid: `(bs, ceil(max_seq_len / TOKEN_BLOCK))`
- For each (batch, logical_pos): lookup `physical_slot = req_to_token[pool_idx, pos]`,
  gather signatures at that slot, compute max-over-heads dot product with projected query
  (using `ch_sel` as a gather index into queries), apply `written` and `seq_lens` masks.
- Write into pre-allocated `out [bs, max_seq_len]` fp32. Zero Python-level allocations.

### Fix 2: Completely rewrite `retrieve_topk_graph_safe`
- File: `selection_kernel.py`
- On CUDA + Triton available: Triton kernel fills `scratch_scores`, then allocation-free
  topk pipeline:
  1. `topk(scratch_scores, k, sorted=False, largest=True, out=(scratch_topk_values, scratch_topk_indices))`
  2. `isneginf(scratch_topk_values, out=scratch_invalid_mask)`; `masked_fill_` sentinels
  3. `topk(scratch_topk_indices, k, sorted=True, largest=False, out=(scratch_sorted_vals, scratch_topk_indices))`  ← ascending sort via "smallest-first topk"
  4. `ge(scratch_sorted_vals, max_seq_len, out=scratch_invalid_mask)`; `out_indices.masked_fill_(-1)` 
  5. `searchsorted(scratch_sorted_vals, scratch_boundary, out=scratch_valid_i64)`; copy to `out_lengths`
- CPU fallback: call existing `retrieve_topk_via_labels` (allocating, fine for unit tests).
- Add `per_request_valid: Optional[torch.Tensor]` parameter. Apply after Triton kernel +
  all-reduce using pre-allocated `scratch_pv_mask`.

### Fix 3: Extend `DSGraphState` + `allocate_graph_state`
- File: `cuda_graph.py`
- New fields: `scratch_scores`, `scratch_topk_values`, `scratch_topk_indices`,
  `scratch_invalid_mask`, `scratch_sorted_vals`, `scratch_boundary`, `scratch_valid_i64`,
  `scratch_pv_mask`.
- `allocate_graph_state` gains `num_local_heads: int = 0`, `label_dim: int = 0` params
  (present per Codex requirement; scratch is allocated when `max_seq_len > 0`).

### Fix 4: Update `capture_decode_step` CUDA path
- File: `cuda_graph.py`
- Inside `torch.cuda.graph(graph)`: if selector is bound AND `state.scratch_scores is not None`,
  call `retrieve_topk_graph_safe` with all scratch from `state` (writing directly into
  `state.selected_indices` / `state.valid_lengths`). Otherwise fall back to
  `selector.retrieve_topk`.
- Pass `per_request_valid=sparse_mask` to `retrieve_topk_graph_safe`.

## Tests

### Fix 5: Update two existing CUDA tests
- Pass `num_local_heads=1, label_dim=1` to `allocate_graph_state` so scratch is allocated.
- `test_cuda_graph_100_step_replay_matches_eager`: still 100-step bit-equal.
- `test_cuda_graph_replay_zero_allocations`: still 0 allocs on replay.

### Fix 6: Two new CUDA-only tests
- `test_retrieve_topk_graph_safe_zero_allocs_after_warmup` (@skipUnless CUDA):
  Warm up with one direct call to `retrieve_topk_graph_safe`. Wrap second call in
  `assert_no_alloc_in_region`. Must not raise (0 new allocs). Verify idx=[2,3], valid=2.
- `test_retrieve_topk_graph_safe_per_request_valid_masks_position` (@skipUnless CUDA):
  Set `per_request_valid` to mask out logical position 2 (score 9.0 via req_to_token).
  Verify position 2 NOT in output. Verify remaining valid positions from [0,1,3].

## Success Criteria

1. `assert_no_alloc_in_region` wrapping direct `retrieve_topk_graph_safe` call passes (0 allocs).
2. `per_request_valid` parameter works: masked position not returned.
3. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`
   ≥ 195 passed, 0 failed.
4. All new CUDA tests pass when CUDA is available.

## Blocking Issues

None.

## Queued

- `task-ac4-hwrun`, `task-ac1-hwtest`, `task-ac1b-probe`, `task-ac8-*`, `task-ac12-*` —
  hardware-gated, unchanged.
