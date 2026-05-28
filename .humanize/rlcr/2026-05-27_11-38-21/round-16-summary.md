# Round 16 Summary

## Work Completed

### task-ac6-cuda-graph — CUDA Capture-Safety Fix (two gaps from Round 15 Codex review)

**Gap 1 — Host sync in captured region (CRITICAL)**

`_compute_logical_token_scores` computed `max_seq_len = int(seq_lens.max().item())` inside the
path that `capture_decode_step` places inside `torch.cuda.graph(graph)`. The `.item()` is a
host sync that raises `CUDA error: operation not permitted when stream is capturing`.

Fix: Added `max_seq_len: int = 0` parameter to `_compute_logical_token_scores`. When nonzero,
the `.item()` call is skipped. Threaded the parameter up through:
- `retrieve_topk_via_labels` (`max_seq_len: int = 0`)
- `DoubleSparsitySelector.retrieve_topk` (`max_seq_len: int = 0`)
- `capture_decode_step` (`max_seq_len: int = 0`)

In `capture_decode_step`, a static `_max_seq_len` is resolved BEFORE the capture region:
```
priority: state.max_seq_len > max_seq_len parameter > one-time seq_lens.max().item()
```
That one-time `.item()` is safe because it happens before `torch.cuda.graph()` is entered.
All three call sites inside the function (eager warmup, CUDA warmup, CUDA capture) receive
`max_seq_len=_max_seq_len`.

**Gap 2 — Graph-safe selector API + DSGraphState extension**

Added `retrieve_topk_graph_safe` to `selection_kernel.py`: same contract as
`retrieve_topk_via_labels` but accepts `max_seq_len: int` (required, no default) and writes
results directly into caller-owned `out_indices` / `out_lengths` buffers. This is the
preferred API for graph capture call sites.

Extended `DSGraphState` with `max_seq_len: int = 0` (the static sequence width). Extended
`allocate_graph_state` to accept `max_seq_len: int = 0` and store it in the state. Callers
now pass `max_seq_len` at allocation time; `capture_decode_step` reads `state.max_seq_len`
automatically without requiring callers to pass the parameter twice.

**Two new CUDA-only tests** (decorated `@unittest.skipUnless(torch.cuda.is_available(), ...)`):

1. **`test_cuda_graph_100_step_replay_matches_eager`**: Creates a bound selector on CUDA with
   known sigs `[9.0, 8.0, 1.0, 2.0]` and `req_to_token = [[2, 3, 0, 1]]`. Captures a CUDA
   graph with `max_seq_len=4`. Calls `sel.retrieve_topk` eagerly to get reference result `[[2,3]]`.
   Replays the graph 100 times and verifies every replay is bit-equal to the eager reference.

2. **`test_cuda_graph_replay_zero_allocations`**: Same CUDA fixture. Wraps `replay()` in
   `assert_no_alloc_in_region("cuda-graph-replay")`. Verifies no `RuntimeError` is raised
   (i.e., 0 new CUDA allocations during graph replay). Also verifies correctness: `lens[0]=2`,
   `idx[0,:2] = [2, 3]`.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`:
  - `_compute_logical_token_scores`: added `max_seq_len: int = 0`; skip `.item()` when provided
  - `retrieve_topk_via_labels`: added `max_seq_len: int = 0`; threads to `_compute_logical_token_scores`
  - Added `retrieve_topk_graph_safe` (writes into `out_indices` / `out_lengths` in-place)
- `python/sglang/srt/layers/attention/double_sparsity/selector.py`:
  - `retrieve_topk`: added `max_seq_len: int = 0`; passes to `retrieve_topk_via_labels`
- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py`:
  - `DSGraphState`: added `max_seq_len: int = 0`
  - `allocate_graph_state`: added `max_seq_len: int = 0`; stored in state
  - `capture_decode_step`: added `max_seq_len: int = 0`; resolves static `_max_seq_len` before
    capture region; passes to selector in all three call sites
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  - Added `_make_bound_selector_cuda` helper (parametric device version of existing CPU helper)
  - Added `test_cuda_graph_100_step_replay_matches_eager` (CUDA-only)
  - Added `test_cuda_graph_replay_zero_allocations` (CUDA-only)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
195 passed, 0 failed (was 193 before this round)
```

Commit: `0ce54a98d` — [AC-6] Fix CUDA graph capture-safety: remove host sync + add graph-safe API

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real V3.2 H200 cluster.
  The coding path is now complete (both capture-safety gaps closed).
- `task-ac4-hwrun`: hardware gate — H200 CUDA OOM on available machine.
- Next coding task: `task-ac1-hwtest` or `task-ac10-radix`.
- Hardware-gated: `task-ac1b-probe`, `task-ac8-*`, `task-ac12-*`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The two AC-6 gaps (.item() host sync, graph-safe output buffers) are a known class
of CUDA graph capture requirements. No surprising failure mode warranting a new entry;
the fix is textbook and fully documented in the capture_decode_step docstring.
