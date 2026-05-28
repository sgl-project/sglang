# Round 17 Code Review

Mainline Progress Verdict: ADVANCED

Round 17 made real AC-6 progress: there is now a Triton logical-score kernel, a scratch-backed `retrieve_topk_graph_safe`, `per_request_valid` is threaded into that helper, and `capture_decode_step` can call it. However, `task-ac6-cuda-graph` is still not complete. The new path is not wired into the production DeepSeek decode path, and the helper still allocates with the production token-label dtype.

## Mainline Gaps

1. **The allocation-free selector is not used by the actual DS decode path.**

   Evidence:
   - Production DS decode still calls `self.double_sparsity_selector.retrieve_topk(...)` in `python/sglang/srt/models/deepseek_v2.py:2117-2125`.
   - `DoubleSparsitySelector.retrieve_topk` still dispatches to `retrieve_topk_via_labels(...)` in `python/sglang/srt/layers/attention/double_sparsity/selector.py:234-253`.
   - `retrieve_topk_via_labels` is still the allocating path: `_compute_logical_token_scores` allocates logical-position/gather/scoring tensors and `select_topk_sequence_order` allocates top-k/sort/output tensors (`selection_kernel.py:471-498`, `selection_kernel.py:391-425`).
   - `capture_decode_step` is not referenced by the production model path. `rg "capture_decode_step"` returns only `cuda_graph.py`, unit tests, and docs; no production caller routes full-model decode through the new helper.

   Consequence: a real V3.2 server with CUDA graphs enabled still captures `_select_topk_indices`, not `capture_decode_step`, so the full decode graph will still see `retrieve_topk_via_labels` allocations. This blocks `task-ac6-hwrun` and AC-8 server work even though the helper-level tests pass.

   Required implementation plan:
   - Extend `DSAMetadata` with a `ds_graph_state: Optional[DSGraphState]` field, or equivalent explicit scratch fields, allocated in the same metadata initialization paths that already allocate `ds_topk_indices_out` (`dsa_backend.py:715-754` and `dsa_backend.py:1015-1040`).
   - Allocate this state with `allocate_graph_state(max_bs=bs, max_top_k=self.ds_max_top_k, max_seq_len=<metadata page_table_1 width / capture max seq len>, device=cache_seqlens_int32.device)`. The graph state's `selected_indices` remain logical positions; `ds_topk_indices_out` remains the physical output buffer for FlashMLA.
   - In `deepseek_v2.py::_select_topk_indices`, before falling back to `DoubleSparsitySelector.retrieve_topk`, detect the metadata-owned `ds_graph_state` when the selector is bound and the tensors are CUDA. Call `retrieve_topk_graph_safe` directly with the bound table/mask, `per_request_valid=forward_batch.sparse_mask`, `req_to_token`, `seq_lens`, and the metadata scratch. Use `state.selected_indices[:bs]` / `state.valid_lengths[:bs]` as the logical selector result.
   - Keep `logical_to_physical(..., out=ds_topk_indices_out)` as the single physical conversion step. Do not allocate a replacement physical buffer in `_select_topk_indices`.
   - Add a production-path regression that spies or monkeypatches `retrieve_topk_graph_safe` and proves `_select_topk_indices` uses it when metadata scratch is present. Add a CUDA allocation test around this production path, not just around `capture_decode_step`.

2. **`retrieve_topk_graph_safe` still allocates with production dtypes.**

   Evidence:
   - Production binds the token label table with `dtype=torch.float16` in `python/sglang/srt/models/deepseek_v2.py:1921-1928`; `allocate_token_label_table` also defaults to fp16 (`token_label_table.py:68-76`).
   - The fast path casts `sig_layer` and non-fp32 `queries` with `.to(torch.float32)` inside `retrieve_topk_graph_safe` (`selection_kernel.py:736-743`). Those are fresh CUDA tensors.
   - The new zero-allocation test hides this by allocating its CUDA token-label table as `dtype=torch.float32` and using float32 queries (`test_double_sparsity_unit.py:3053-3056`, `test_double_sparsity_unit.py:3171`).
   - A CUDA probe using the default production fp16 `TokenLabelTable` and fp16 queries still fails after warmup:

   ```text
   prod-dtype-graph-safe-second-call: new CUDA allocation detected inside the captured region (2 new allocations)
   idx [2, 3] len 2
   ```

   Consequence: even if the helper is wired into production, the captured selector region is not allocation-free at the actual operating point.

   Required implementation plan:
   - Remove all `.to(...)` conversions from the CUDA fast path of `retrieve_topk_graph_safe`.
   - Let `_logical_score_kernel` load fp16/bf16 query and signature pointers directly and cast loaded values to `tl.float32` inside the kernel. This is already the right shape for `tl.load(...).to(tl.float32)`.
   - Treat `channel_selection` and `channel_weights` as bind-time invariants: they should already be int32 and fp32 on the target device after `bind_runtime_data`. Assert/fail fast if not, rather than converting inside the graph-safe function.
   - For `req_pool_indices`, `req_to_token`, and `seq_lens`, require int32 in the graph-safe API or add preallocated int32 scratch copies to `DSGraphState`; do not call `.to(torch.int32)` inside the fast path.
   - Change the CUDA zero-allocation test to use the same dtypes as production: default fp16 `TokenLabelTable`, fp16 or bf16 queries, int32 `sparse_mask` passed as `per_request_valid`, and the same scratch path used by `capture_decode_step`.

## Blocking Side Issues

None separate from the mainline AC-6 gaps. The two findings above directly block the current mainline objective.

## Queued Side Issues

1. AC-8 observability still reports token selections through page-named fields and computes sparsity against page counts in `_publish_ds_request_summary`.
2. Cleanup: stale DS bind/runtime comments still mention `req_to_token_pool.size` as max-token authority.
3. Cleanup: token-label lifetime docs still describe overwrite-before-read rather than invalidate-before-selection.

These remain non-blocking for Round 17 and should not replace the AC-6 fix.

## Goal Alignment Summary

```text
ACs: 9/15 addressed (6 met, 3 partial: AC-1, AC-4, AC-6) | Forgotten items: 0 | Unjustified deferrals: 0
```

Status by AC:
- Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.
- Partial: AC-1, AC-4, AC-6.
- Not met: AC-1b, AC-8, AC-9, AC-10, AC-11, AC-12.

Hardware-gated items are still active pending work, not accepted deferrals. `task-ac6-cuda-graph` must also stay active because the coding path is not production-ready.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 17 Review`.
- Added a Round 17 Review plan-evolution row reopening `task-ac6-cuda-graph`.
- Changed `task-ac6-cuda-graph` in Active Tasks from `complete` back to `pending`.

## Validation Run

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py::TestCUDAGraphCapture -q
11 passed

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py::TestTritonEquivalence -q
2 passed
```

Additional CUDA allocation probe:
- `retrieve_topk_graph_safe(...)` with the production fp16 token-label table and fp16 queries still triggers `assert_no_alloc_in_region` after warmup with 2 new CUDA allocations.

NOT COMPLETE
