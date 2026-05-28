# Round 18 Contract

## Mainline Objective

Close `task-ac6-cuda-graph` for real: the allocation-free `retrieve_topk_graph_safe`
must (1) be wired into the production `_select_topk_indices` decode path —
not just `capture_decode_step` — and (2) be **actually 0-alloc with production
dtypes** (fp16 token-label table, bf16/fp16 queries, int32 sparse_mask).

Two Codex gaps from the Round 17 review:

1. The graph-safe helper is invoked only by `cuda_graph.py::capture_decode_step`,
   which is not referenced by the production decode path. The DeepSeek
   `_select_topk_indices` still calls `DoubleSparsitySelector.retrieve_topk`
   → `retrieve_topk_via_labels` → 47 allocations per step.
2. `retrieve_topk_graph_safe` casts `sig_layer` and `queries` to `torch.float32`
   inside the fast path. Under production dtypes (fp16 table + fp16 queries),
   those `.to()` calls allocate (2 new allocs observed even after warmup).

## Target ACs

- **AC-6** (CUDA graph decode path): production `_select_topk_indices` must
  route through `retrieve_topk_graph_safe` when DS scratch is allocated by the
  DSA backend, and the fast path must show 0 new allocations after warmup at
  production dtypes (fp16 sig + bf16/fp16 queries + int32 mask).

## Required Implementation

### Fix 1: In-kernel dtype casts (selection_kernel.py)

- Remove every `.to(torch.float32)` / `.to(torch.int32)` call from the CUDA
  fast path of `retrieve_topk_graph_safe`. The fast path now requires:
  - `queries`: any of fp32 / fp16 / bf16; loaded as-is into the kernel and
    cast to fp32 via `tl.load(...).to(tl.float32)`.
  - `token_signatures[layer_id]`: any of fp32 / fp16 / bf16; same handling.
  - `channel_selection[layer_id]`: int32 (bind-time invariant).
  - `channel_weights[layer_id]`: fp32 (bind-time invariant).
  - `req_pool_indices`, `req_to_token`, `seq_lens`: int32 (caller contract).
- Update `_logical_score_kernel` so the q-gather and sig loads emit
  `.to(tl.float32)` on the loaded value rather than relying on host-side casts.

### Fix 2: Production wiring (deepseek_v2.py + dsa_backend.py)

- Add `ds_graph_state: Optional[DSGraphState] = None` to `DSAMetadata` in
  `dsa_backend.py`.
- In both metadata init sites (extend at `dsa_backend.py:715-754` and decode
  at `dsa_backend.py:1015-1040`), when `self.enable_double_sparsity`:
  allocate `ds_graph_state` via `allocate_graph_state(max_bs=bs,
  max_top_k=self.ds_max_top_k, max_seq_len=<page_table_1_width or
  req_to_token width>, device=cache_seqlens_int32.device)`.
- In `deepseek_v2.py::_select_topk_indices` (`_run`), after `invalidate_token_label_slots`
  and before `retrieve_topk`, detect `metadata.ds_graph_state` when:
  - selector is bound (`token_label_table is not None and channel_mask is not None`),
  - tensors live on CUDA,
  - `ds_graph_state.scratch_scores is not None`.
- When that condition holds, call `retrieve_topk_graph_safe` directly,
  passing the bound table/mask, `per_request_valid=sparse_mask`,
  `req_to_token`, `seq_lens`, and the metadata-owned scratch.
- Use `state.selected_indices[:bs]` / `state.valid_lengths[:bs]` as the
  logical selector result; flow into `logical_to_physical(...,
  out=ds_topk_indices_out)` unchanged.
- Fall back to `self.double_sparsity_selector.retrieve_topk(...)` when the
  graph-state condition does not hold (CPU tests, placeholder selectors).

### Fix 3: Update tests to production dtypes

- Update `test_retrieve_topk_graph_safe_zero_allocs_after_warmup` and the
  existing 100-step replay tests to construct the `TokenLabelTable` with
  `dtype=torch.float16` and use `bfloat16` (or fp16) queries to exercise the
  actual production-dtype path. `int32 sparse_mask` as `per_request_valid`.
- Add `test_select_topk_indices_uses_graph_safe_when_metadata_state_present`:
  monkeypatch / spy `retrieve_topk_graph_safe`; build a minimal
  `forward_batch` + `DSAMetadata(ds_graph_state=...)`, drive
  `_select_topk_indices`, assert the spy was called once and
  `retrieve_topk_via_labels` was NOT called.
- Add CUDA-only `test_select_topk_indices_zero_allocs_after_warmup`: warm up
  once, wrap second call in `assert_no_alloc_in_region`, assert no growth.

### Fix 4: Bind-time invariants (selector.py — light touch)

- In `DoubleSparsitySelector.bind_runtime_data`, assert
  `channel_selection.dtype == torch.int32` and `channel_weights.dtype ==
  torch.float32` and raise a clear error if not. Avoid silently doing the
  conversion inside the fast path on every call.

## Tests

- Existing 197 tests must continue to pass.
- New: production-path spy regression + CUDA zero-alloc regression at
  production dtypes. ≥ 200 total expected.

## Success Criteria

1. `_select_topk_indices` calls `retrieve_topk_graph_safe` when
   `metadata.ds_graph_state` is present (spy regression).
2. Manual CUDA probe (matches Codex's report) using default fp16
   `TokenLabelTable` + fp16/bf16 queries + int32 sparse_mask shows 0 new
   CUDA allocations after warmup when calling `retrieve_topk_graph_safe`
   directly.
3. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`
   ≥ 200 passed, 0 failed.
4. `git status` clean after commit; no .humanize/ files staged.

## Blocking Issues

None — all required components are already drafted in Round 17. This is a
re-wiring + dtype-discipline pass.

## Queued

- `task-ac8-*`, `task-ac4-hwrun`, `task-ac1-hwtest`, `task-ac1b-probe`,
  `task-ac6-hwrun`, `task-ac10-radix`, `task-ac11-compare`,
  `task-ac12-quality` — unchanged.
- AC-8 observability page-vs-token unit mix — queued behind `task-ac8-*`.
