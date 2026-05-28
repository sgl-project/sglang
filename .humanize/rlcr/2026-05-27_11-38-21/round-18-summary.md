# Round 18 Summary

## Work Completed

### task-ac6-cuda-graph — Production wiring + production dtypes

Codex Round 17 review left two open gaps:

1. The graph-safe `retrieve_topk_graph_safe` was only invoked from
   `cuda_graph.py::capture_decode_step` — a helper that is **not** referenced
   by the production DeepSeek decode path. Production
   `_select_topk_indices` was still calling
   `DoubleSparsitySelector.retrieve_topk` → `retrieve_topk_via_labels` (the
   allocating path, ~47 allocs per step).

2. The CUDA fast path of `retrieve_topk_graph_safe` cast `sig_layer` and
   non-fp32 `queries` with `.to(torch.float32)`. Under production dtypes
   (fp16 TokenLabelTable + bf16 / fp16 queries), those casts allocate fresh
   tensors. A Codex CUDA probe at production dtypes reported 2 new
   allocations after warmup.

Both are now closed.

#### Production wiring

- **`DSAMetadata` gained `ds_graph_state: Optional[DSGraphState]`** (dsa_backend.py).
- **Both metadata-init sites allocate it** when DS is enabled:
  - `init_forward_metadata` (extend / dynamic decode, ~line 715)
  - `init_forward_metadata_capture_cuda_graph` (CUDA graph capture, ~line 1015)
  - Sizing: `max_bs=bs, max_top_k=self.ds_max_top_k, max_seq_len=req_to_token.shape[1]`.
- **`deepseek_v2.py::_select_topk_indices`** detects
  `forward_batch.attn_backend.forward_metadata.ds_graph_state` and, when
  the selector is bound and tensors are CUDA, calls
  `retrieve_topk_graph_safe` directly with all scratch +
  `per_request_valid=sparse_mask`. Falls back to
  `DoubleSparsitySelector.retrieve_topk` only when scratch is absent (CPU
  tests / unbound selector / synthetic forward_batch without
  attn_backend).
- The downstream `logical_to_physical(..., out=ds_topk_indices_out)`
  conversion is unchanged.

#### Production dtypes

- **Removed all `.to(...)` casts** from the CUDA fast path of
  `retrieve_topk_graph_safe`. The `_logical_score_kernel` already loads
  fp16/bf16/fp32 q + sig pointers and casts via
  `tl.load(...).to(tl.float32)` inside the kernel.
- **Added contract asserts** in the fast path: `channel_selection int32`,
  `channel_weights fp32`, `req_pool_indices / req_to_token / seq_lens
  int32`. Any drift fails fast with a clear message instead of silently
  allocating per call.
- **Added bind-time asserts** in `DoubleSparsitySelector.bind_runtime_data`
  for `channel_selection.dtype == int32` and
  `channel_weights.dtype == float32`. The channel mask is the only tensor
  whose dtype the selector can enforce at bind time; the token
  signatures stay at the binder's choice (fp16 in production).

### Tests

- **`test_retrieve_topk_graph_safe_zero_allocs_production_dtypes`** (CUDA-only):
  Constructs a fp16 `TokenLabelTable` (production default) + bf16 queries
  + int32 `sparse_mask` (as `per_request_valid`). Warms up once; wraps
  the second call in `assert_no_alloc_in_region`. Closes Codex's
  prod-dtype CUDA probe (was 2 new allocs; is now 0).
- **`test_select_topk_indices_uses_graph_safe_when_metadata_state_present`**:
  Drives the actual `_select_topk_indices` method on a real-mode
  selector with a synthesized `forward_batch.attn_backend.forward_metadata.ds_graph_state`.
  Spies the dynamic import of `retrieve_topk_graph_safe` at the kernel
  module — asserts the spy is called exactly once. Closes Codex's
  production-path regression requirement.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`:
  removed `.to(torch.float32)` / `.to(torch.int32)` from CUDA fast path;
  added int32/fp32 contract asserts; documented the new caller contract.
- `python/sglang/srt/layers/attention/double_sparsity/selector.py`:
  added bind-time `channel_selection.dtype == int32` /
  `channel_weights.dtype == float32` asserts.
- `python/sglang/srt/layers/attention/dsa_backend.py`:
  added `DSGraphState` import; added
  `DSAMetadata.ds_graph_state: Optional[DSGraphState]`; allocated it in
  both `init_forward_metadata` sites (extend/decode dynamic + CUDA graph
  capture).
- `python/sglang/srt/models/deepseek_v2.py::_select_topk_indices`:
  added `_use_graph_safe` gate; when scratch is present + selector is
  bound + CUDA tensors, calls `retrieve_topk_graph_safe` directly with
  metadata-owned scratch; falls back to the legacy path otherwise.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  added `_make_bound_selector_cuda_fp16` helper; added the two new
  tests above.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
199 passed, 0 failed (was 197 before this round)
```

Targeted runs:
```
pytest -v -k zero_allocs_production           # 1 passed (CUDA fp16+bf16 0-alloc)
pytest -v -k test_select_topk_indices_uses_graph_safe  # 1 passed (production-path spy)
```

Commit: `38ef74765` — [AC-6] Wire allocation-free DS path into production + production dtypes.

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real
  V3.2 H200 cluster. The coding path is now production-ready (the
  graph-safe selector is invoked in the actual decode path, and the
  fast path is 0-alloc at production dtypes).
- `task-ac4-hwrun`: hardware gate — H200 channel-mask calibration.
- Hardware-gated: `task-ac1-hwtest`, `task-ac1b-probe`, `task-ac8-*`,
  `task-ac12-*`.
- Queued (non-blocking): AC-8 page-vs-token unit mix-up in
  `_publish_ds_request_summary`; stale DS bind/runtime comments;
  token-label lifetime docs.

## Push-to-remote Note

User requested per-commit push for crash safety. The RLCR loop's
`loop-bash-validator.sh` hook blocks `git push` unless the loop is
launched with `--push-every-round`. Commits are saved locally on
`dev/double-sparsity-standalone`; commit `38ef74765` is Round 18 and is
ahead of `jimmy/dev/double-sparsity-standalone` by 19 commits. Push
manually if you take over the shell, or relaunch the loop with the
`--push-every-round` flag.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 18 is a re-wiring round on top of Round 17's allocation-free
pipeline. The principle worth remembering — `tensor.to(dtype)` is NOT
allocation-free even when the caching allocator returns reused memory
(the `allocation.all.allocated` counter still increments) — is implicit
in the prior `BL-20260527-torch-topk-aliasing-corrupts-input` lesson's
"every host op must be allocation-free" framing. No new entry warranted.
