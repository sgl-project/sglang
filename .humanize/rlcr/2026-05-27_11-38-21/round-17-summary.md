# Round 17 Summary

## Work Completed

### task-ac6-cuda-graph — Truly Allocation-Free Graph-Safe Path

Codex Round 16 review left two gaps open:

1. `retrieve_topk_graph_safe` internally wrapped `retrieve_topk_via_labels` →
   ~42 new CUDA allocations even with pre-allocated output buffers.
2. `retrieve_topk_graph_safe` had no `per_request_valid` parameter, silently
   dropping the M2 ownership mask from the graph-safe path.

Both gaps are now closed.

**New Triton kernel `_logical_score_kernel`** (`selection_kernel.py`):
Grid `(bs, ceil(max_seq_len / TOKEN_BLOCK))`. Per (batch, logical_pos):
look up `physical_slot = req_to_token[pool_idx, pos]`, gather signatures at
that slot, max-over-heads dot product with projected query (using `ch_sel`
as gather index into queries), apply `written` and `seq_lens` masks. Writes
directly into pre-allocated `out [bs, max_seq_len]` fp32 — zero Python-level
allocations.

**Completely rewrote `retrieve_topk_graph_safe`** (`selection_kernel.py`):
On CUDA + Triton + scratch available, runs an allocation-free pipeline:

1. `_logical_score_kernel` fills `scratch_scores` (Triton).
2. (optional) all-reduce in place on `scratch_scores`.
3. (optional) apply `per_request_valid` via `scratch_pv_mask.copy_(...)` +
   in-place `logical_not` + `masked_fill_(-inf)`.
4. `topk(scratch_scores, k, sorted=False, largest=True,
   out=(scratch_topk_values, scratch_topk_indices))`.
5. `isneginf(scratch_topk_values, out=scratch_invalid_mask)`;
   `masked_fill_(invalid, max_seq_len)` sentinels invalid entries.
6. `topk(scratch_topk_indices, k, sorted=True, largest=False,
   out=(scratch_sorted_vals, scratch_throwaway_idx))` — ascending sort via
   smallest-first topk.
7. `out_indices.copy_(scratch_sorted_vals)`; `ge(...)` + `masked_fill_(-1)`
   converts sentinels to `-1`.
8. `searchsorted(scratch_sorted_vals, scratch_boundary, right=False,
   out=scratch_valid_i64)`; `out_lengths.copy_()`.

On CPU or with scratch missing: falls back to legacy `retrieve_topk_via_labels`
(allocating, fine for unit tests).

**Extended `DSGraphState`** (`cuda_graph.py`): added fields
`scratch_scores`, `scratch_topk_values`, `scratch_topk_indices`,
`scratch_invalid_mask`, `scratch_sorted_vals`, `scratch_boundary`,
`scratch_valid_i64`, `scratch_pv_mask`, **`scratch_throwaway_idx`**.

`scratch_throwaway_idx` was added after debugging the per_request_valid test
failure (see below). PyTorch `torch.topk(input=A, ...,
out=(values=B, indices=A))` corrupts the read when output indices alias
input. Symptom: input `[3, 1]` produced output values `[0, 1]` instead of
`[1, 3]`. Fix: route throwaway gather indices into a dedicated scratch.

**Extended `allocate_graph_state`**: now takes `num_local_heads: int = 0`,
`label_dim: int = 0` (accepted for API parity; scratch sizing is driven by
`max_seq_len`). When `max_seq_len > 0`, the eight scratch tensors above are
allocated; otherwise None (graph-safe fast path skipped).

**Updated `capture_decode_step` CUDA path**: when the selector is bound AND
`state.scratch_scores is not None`, the captured region calls
`retrieve_topk_graph_safe` with all scratch + `per_request_valid=sparse_mask`.
Otherwise falls back to `selector.retrieve_topk`. Eager+capture both use the
same `_call_into_state()` closure to guarantee warmup and capture paths are
identical.

### Tests

- **`test_retrieve_topk_graph_safe_zero_allocs_after_warmup`** (CUDA-only):
  Warm up with one direct call; wrap second call in `assert_no_alloc_in_region`.
  Asserts 0 new allocations AND correctness (`idx=[2,3]`, `valid=2`).
- **`test_retrieve_topk_graph_safe_per_request_valid_masks_position`**
  (CUDA-only): Masks logical position 2 (the would-be top score 9.0).
  Asserts position 2 is NOT in the output and remaining picks come from
  `[0, 1, 3]` → expected `sorted([1, 3])`.
- Existing CUDA tests updated to pass `num_local_heads=1, label_dim=1` so
  scratch is allocated. The 100-step bit-equal replay and the zero-alloc
  replay tests still pass.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`:
  - Added Triton kernel `_logical_score_kernel`.
  - Added wrapper `_logical_score_triton`.
  - Completely rewrote `retrieve_topk_graph_safe` (allocation-free fast
    path on CUDA + scratch; legacy fallback on CPU).
  - Added optional `per_request_valid`, `scratch_*`, `scratch_pv_mask`,
    `scratch_throwaway_idx` parameters.
- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py`:
  - `DSGraphState`: added 9 scratch fields.
  - `allocate_graph_state`: added `num_local_heads`, `label_dim` params;
    allocates the scratch when `max_seq_len > 0`.
  - `capture_decode_step`: CUDA path routes through `retrieve_topk_graph_safe`
    when selector is bound + scratch available; passes
    `per_request_valid=sparse_mask`.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  - Added two CUDA-only tests above.
  - Updated existing CUDA tests with `num_local_heads=1, label_dim=1`.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
197 passed, 0 failed (was 195 before this round)
```

The two new CUDA tests pass on the available H200 (Triton path exercised).
Existing CUDA-only 100-step replay + zero-alloc replay tests still pass with
the new graph-safe path engaged.

Holistic loop audit (rounds 10–16 Codex reviews): no unresolved
HIGH-SEVERITY findings outside Round 17 scope. Hardware-gated items
(`task-ac4-hwrun`, `task-ac1-hwtest`, `task-ac8-*`, `task-ac12-*`) remain
queued unchanged.

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real
  V3.2 H200 cluster. The coding path is now complete (both Round 16 gaps
  closed; allocation-free Triton + topk pipeline verified locally).
- `task-ac4-hwrun`: hardware gate — H200 channel-mask calibration.
- Next coding task: `task-ac1-hwtest` or `task-ac10-radix` (per goal tracker).

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-torch-topk-aliasing-corrupts-input
Notes: PyTorch `torch.topk(input=A, ..., out=(values=B, indices=A))` is NOT
safe — output indices aliasing input corrupts the read. Symptom in this
loop: input `[3, 1]` produced sorted values `[0, 1]` (instead of `[1, 3]`).
Fix is to route throwaway output indices into a separate scratch tensor.
Worth recording because the contract explicitly suggested aliasing as an
allocation-saving trick, and the failure mode (looks like scoring returned
zeros) is misleading on the surface.
