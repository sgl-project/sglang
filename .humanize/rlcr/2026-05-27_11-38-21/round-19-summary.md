# Round 19 Summary

## Work Completed

### task-ac6-cuda-graph — ForwardContext lookup + int64 input handling

Codex Round 18 review left two open gaps:

1. **Wrong production object path.** `_select_topk_indices` looked for
   `ds_graph_state` at `forward_batch.attn_backend.forward_metadata` — but
   real `ForwardBatch` has no `attn_backend` field. Production publishes the
   attention backend through `ForwardContext`, the same source the AC-7 MHA
   bypass already uses. Codex's spy probe recorded
   `retrieve_topk_graph_safe` call count 0 in production.

2. **Wrong dtype assumption.** Production `req_pool_indices` is int64
   (`schedule_batch.py:1507` + `cuda_graph_runner.py:178`); the new graph-
   safe fast path asserted int32. Codex's probe raised
   `AssertionError: req_pool_indices must be int32, got torch.int64`.

Both closed.

#### Metadata resolution

- `_select_topk_indices` resolves `ds_graph_state` in two steps (mirrors
  the AC-7 MHA bypass):
  1. `forward_batch.ds_graph_state` — primary, set by
     `dsa_backend.init_forward_metadata` for dynamic non-graph forwards.
  2. `has_forward_context() and get_attn_backend().forward_metadata.ds_graph_state`
     — fallback for the CUDA-graph capture/replay path.
- `dsa_backend.init_forward_metadata` now also assigns
  `forward_batch.ds_graph_state = ds_graph_state` next to
  `forward_batch.ds_topk_indices_out`.

#### int64 → int32 scratch via copy_

- Added two new fields to `DSGraphState`:
  `scratch_req_pool_indices: int32[max_bs]` and
  `scratch_seq_lens: int32[max_bs]`.
- `_select_topk_indices` does an in-place `copy_()` from
  production int64 `req_pool_indices` into the int32 scratch, then passes
  the scratch view to `retrieve_topk_graph_safe`.
- For `seq_lens`: prefers `DSAMetadata.cache_seqlens_int32[:bs]` when the
  metadata is reachable (the int32 view is already maintained per batch
  in `dsa_backend`); otherwise `copy_()` into `scratch_seq_lens`.

#### Bonus — allocation-free `logical_to_physical`

While tracing the production-path 0-alloc requirement we discovered
`logical_to_physical` allocated ~8 intermediates per call (clamp / sum /
where / full_like). Replaced the torch path with a single Triton kernel:

- `_logical_to_physical_kernel` (page_table_adapter.py): grid
  `(bs, ceil(max_top_k / BLOCK_K))`; gathers `req_to_token[safe_pool,
  safe_pos]`, masks padding + bad-pool rows to `-1`, atomically increments
  an int32 `error_scratch` for bad pool indices.
- `lp_error_scratch: int32[1]` is now part of `DSGraphState`. The
  `int(error_scratch.item())` host sync is skipped during stream capture
  (returns 0 conservatively); callers that need the count outside capture
  still get it.
- CPU + missing-scratch paths fall back to the original torch
  implementation — unit tests on CPU stay green.

#### Capture-safe `_publish_ds_request_summary`

The per-request CPU-side summary publication does
`valid_lengths.detach().to("cpu").tolist()` (a D2H sync that is illegal
during stream capture). Gated on
`not torch.cuda.is_current_stream_capturing()`, matching the
established pattern in `retrieve_topk_via_labels`.

### Tests

- **Replaced** `test_select_topk_indices_uses_graph_safe_when_metadata_state_present`
  with `..._via_forward_context`: publishes only a real
  `ForwardContext(attn_backend=...)`, no synthetic
  `forward_batch.attn_backend`. Production-dtype int64
  `req_pool_indices` + int64 `seq_lens` + int32 `sparse_mask`. Spies the
  dynamic import of `retrieve_topk_graph_safe` at the kernel module;
  asserts it is called exactly once.
- **Added** `test_select_topk_indices_zero_allocs_production_path`:
  captures `_select_topk_indices` into a real `torch.cuda.CUDAGraph`
  with the same production-dtype fixture; replays 5 times wrapped in
  `assert_no_alloc_in_region`; verifies 0 new CUDA allocations on every
  replay and that `ds_topk_indices_out` carries the expected physical
  slots from `logical_to_physical`. This mirrors how
  `cuda_graph_runner.py` captures the full decode forward in production.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py`:
  added `scratch_req_pool_indices`, `scratch_seq_lens`, `lp_error_scratch`
  fields to `DSGraphState`; allocated in `allocate_graph_state` under
  `max_seq_len > 0`.
- `python/sglang/srt/layers/attention/double_sparsity/page_table_adapter.py`:
  added Triton kernel `_logical_to_physical_kernel`; rewrote
  `logical_to_physical` with the alloc-free CUDA fast path + torch
  fallback; new `error_scratch` keyword argument; capture-safe error
  read.
- `python/sglang/srt/layers/attention/dsa_backend.py`:
  `init_forward_metadata` exposes `forward_batch.ds_graph_state` for
  the dynamic path.
- `python/sglang/srt/models/deepseek_v2.py::_select_topk_indices`:
  resolves `ds_graph_state` via `forward_batch` primary +
  `ForwardContext` fallback; `copy_()` int64 inputs into int32 scratch;
  passes `error_scratch` to `logical_to_physical`; gates
  `_publish_ds_request_summary` on capture state.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  added `_make_production_forward_batch` helper; replaced the
  attn_backend-mocked spy test with the `ForwardContext`-based one;
  added the CUDA-graph replay zero-alloc test.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
200 passed, 0 failed (was 199 before this round)
```

Targeted:
```
pytest -v -k "test_select_topk_indices_uses_graph_safe_via_forward_context"     # 1 passed
pytest -v -k "test_select_topk_indices_zero_allocs_production_path"             # 1 passed (5 replays x 0 allocs)
```

Commit: `8ab3c332a` — [AC-6] Resolve DS metadata via ForwardContext + int64 input dtypes.

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real
  V3.2 H200 cluster. The coding path is now production-ready: the
  graph-safe selector is reached via `ForwardContext`, production int64
  inputs are handled, `logical_to_physical` is alloc-free, and
  `_select_topk_indices` survives a real `torch.cuda.CUDAGraph` capture
  + replay with 0 new allocations.
- `task-ac4-hwrun`: hardware gate — H200 channel-mask calibration.
- Hardware-gated: `task-ac1-hwtest`, `task-ac1b-probe`, `task-ac8-*`,
  `task-ac12-*`.
- Queued (non-blocking): AC-8 page-vs-token unit mix-up in
  `_publish_ds_request_summary`; stale DS bind/runtime comments;
  token-label lifetime docs.

## Push-to-remote Status

User requested per-commit push to GitHub. The RLCR loop's
`loop-bash-validator.sh` hook still blocks `git push`; commits are saved
locally on `dev/double-sparsity-standalone` (Round 19 commit
`8ab3c332a`, 20 commits ahead of `jimmy/dev/double-sparsity-standalone`).
To enable per-round pushes, re-launch the loop with
`--push-every-round`. The commits will not be lost — local refs are
intact.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 19's lesson — "production metadata is published via
`ForwardContext`, not `forward_batch.attn_backend`" — is already
documented in the live AC-7 MHA bypass code at the top of
`_select_topk_indices`. The Round 19 fix copied that exact pattern.
No new BitLesson warranted; the existing inline comments in the bypass
make the pattern discoverable for future writers.
