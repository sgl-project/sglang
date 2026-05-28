# Round 20 Summary

## Work Completed

### task-ac6-cuda-graph — `ds_topk_indices_out` via ForwardContext

Codex Round 19 review found one remaining CUDA-graph production-object gap:

- Round 19 fixed `ds_graph_state` lookup to go through real `ForwardContext`
  metadata, but left `ds_topk_indices_out` looking only at the
  (non-existent) `forward_batch.attn_backend.forward_metadata`. In real
  CUDA-graph capture (`cuda_graph_runner.py`), the capture-time
  `ForwardBatch` has no DS fields and no `attn_backend` field; production
  publishes the attention backend through `ForwardContext`.
- The Round 19 zero-alloc test masked the bug by pre-setting
  `forward_batch.ds_topk_indices_out = ds_topk_out` before capture, which
  is not what production does.
- Codex's probe confirmed: with metadata-owned `ds_topk_indices_out` and
  no pre-set `forward_batch.ds_topk_indices_out`, `torch.empty_like` was
  called and the returned tensor did NOT alias the metadata buffer.

#### Fix

- **Hoisted `_dsa_metadata` resolution** in
  `_select_topk_indices` so it is always populated from
  `ForwardContext.get_attn_backend().forward_metadata` whenever a
  `ForwardContext` is published — both `ds_graph_state` and
  `ds_topk_indices_out` now share that single source.
- **Resolution order for `ds_topk_indices_out`:**
  1. `forward_batch.ds_topk_indices_out` (dynamic non-graph forwards;
     set by `dsa_backend.init_forward_metadata`).
  2. `_dsa_metadata.ds_topk_indices_out` (ForwardContext-published;
     CUDA-graph capture/replay).
  3. Last-resort lazy `torch.empty_like` (CPU unit tests only).
- **Removed the unreachable `forward_batch.attn_backend.forward_metadata`
  branch.** Production never satisfies it; it was a dead path Round 18
  inadvertently left in place.

### Tests

- **Added** `test_select_topk_indices_uses_metadata_ds_topk_indices_out_via_forward_context`:
  Publishes both `ds_graph_state` AND `ds_topk_indices_out` only via
  `ForwardContext.attn_backend.forward_metadata`. Spies `torch.empty_like`
  and asserts the spy is NOT called by `_select_topk_indices`. Asserts
  the returned `ds_out`'s `data_ptr` is identical to the metadata buffer's.
  This is exactly the regression Codex's review asked for.
- **Updated** `test_select_topk_indices_zero_allocs_production_path`:
  Removed the manual `forward_batch.ds_topk_indices_out = ds_topk_out`
  pre-set. The buffer is now reached only through `ForwardContext`,
  matching real capture. Still passes 5 replays with 0 new CUDA allocations.
- **Renamed** `test_select_topk_indices_reads_metadata_buffer_via_attn_backend`
  to `..._via_forward_context` and switched it to the real `ForwardContext`
  source. The old synthetic `forward_batch.attn_backend` path is gone.
- **Updated** `test_no_bypass_when_forward_context_use_mha_false`:
  Replaced `MagicMock()` backend stub with
  `SimpleNamespace(use_mha=False, forward_metadata=None)`. `MagicMock`'s
  auto-attributes would have polluted the new always-resolved
  `_dsa_metadata` lookup (returning a `MagicMock` instead of `None`).

## Files Changed

- `python/sglang/srt/models/deepseek_v2.py::_select_topk_indices`:
  hoisted `_dsa_metadata` resolution outside the `_ds_graph_state` block;
  reused for `ds_topk_indices_out` lookup; removed dead
  `forward_batch.attn_backend.forward_metadata` branch.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  added the new regression test; updated 3 existing tests as described.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
201 passed, 0 failed (was 200 before this round)
```

Targeted:
```
pytest -v -k "uses_metadata_ds_topk_indices_out_via_forward_context"             # 1 passed
pytest -v -k "test_select_topk_indices_zero_allocs_production_path"              # 1 passed
pytest -v -k "test_select_topk_indices_reads_metadata_buffer_via_forward_context"# 1 passed
pytest -v -k "test_no_bypass_when_forward_context_use_mha_false"                 # 1 passed
```

Commit: `5c636760f` — [AC-6] Resolve ds_topk_indices_out via ForwardContext (mirror Round 19).

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real
  V3.2 H200 cluster. The coding path is now production-ready: both DS
  scratch AND the physical output buffer are resolved through real
  `ForwardContext` metadata; no synthetic `forward_batch.attn_backend`
  attribute is read anywhere; `torch.empty_like` is verified not called
  in the production capture path; the CUDA-graph replay zero-alloc test
  passes with metadata-only lookup.
- `task-ac4-hwrun`: hardware gate — H200 channel-mask calibration.
- Hardware-gated: `task-ac1-hwtest`, `task-ac1b-probe`, `task-ac8-*`,
  `task-ac12-*`.
- Queued (non-blocking): AC-8 page-vs-token unit mix-up in
  `_publish_ds_request_summary`; stale DS bind/runtime comments;
  token-label lifetime docs.

## Push-to-remote Status

Branch `dev/double-sparsity-standalone` is 21 commits ahead of
`jimmy/dev/double-sparsity-standalone`. The RLCR loop's
`loop-bash-validator.sh` hook still blocks `git push`; commits are
saved locally only. To enable per-round pushes, re-launch the loop with
`--push-every-round`.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-ds-metadata-via-forward-context
Notes: After three consecutive rounds (R17→R20) of "wired into production
but actually still using a synthetic path" bugs, the recurring theme is:
production publishes the attention backend via `ForwardContext`, not via
`forward_batch.attn_backend`. New BitLesson captures the rule plus the
symptom (silent fallback to per-call `torch.empty_like`) so future writers
catch it in code review rather than after a Codex review cycle.
