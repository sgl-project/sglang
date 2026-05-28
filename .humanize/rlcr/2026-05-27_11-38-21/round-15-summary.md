# Round 15 Summary

## Work Completed

### task-ac6-cuda-graph — CUDA Graph Decode Path Fix + Tests

**Root cause fixed**: `capture_decode_step` was missing `req_to_token` in all three
`selector.retrieve_topk` call sites. Without it, the captured graph would use the
physical-domain fallback path instead of logical-domain selection, producing wrong top-K
results when physical slots differ across TP ranks.

**Fix**: Added `req_to_token: Optional[torch.Tensor] = None` to `capture_decode_step`
and threaded it through to all three call sites:
1. Eager CPU path (`_eager_replay` closure)
2. CUDA warmup pass
3. CUDA graph capture region

**New unit tests (5)** in `TestCUDAGraphCapture`:

1. **`test_req_to_token_threads_to_logical_domain`**: Uses a bound selector with physical
   sigs `[9.0, 8.0, 1.0, 2.0]` and `req_to_token = [[2,3,0,1]]`. With this mapping,
   logical scores are `[1.0, 2.0, 9.0, 8.0]`, so top-2 logical positions are `[2, 3]`.
   Verifies the replay returns `[[2, 3]]` (not `[[0, 1]]` which would be the physical-domain answer).

2. **`test_eager_replay_output_matches_direct_call`**: Calls replay once and separately
   calls `selector.retrieve_topk` directly with the same arguments; asserts bit-equal output.

3. **`test_eager_replay_100_steps_stable`**: Calls replay 100 times and verifies all
   outputs are identical (placeholder-mode determinism).

4. **`test_alloc_detector_raises_on_cuda_alloc_in_region`**: On CUDA, allocating a tensor
   inside `assert_no_alloc_in_region` raises `RuntimeError`. On CPU, the context manager
   is a no-op (verified it doesn't raise).

5. **`test_alloc_detector_silent_when_prealloc_before_region`**: Preallocating a buffer
   before the region and doing writes-only inside does not raise.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py`: added
  `req_to_token` parameter to `capture_decode_step`; threaded through all 3 call sites;
  updated docstring to explain the logical-domain requirement.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`: added
  `_make_bound_selector_with_known_sigs` helper and 5 new `TestCUDAGraphCapture` tests.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
193 passed, 0 failed (was 188 before this round)
```

Commit: `cd7c071f3` — [AC-6] Thread req_to_token through capture_decode_step; add AC-6 unit tests

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real V3.2 H200
  cluster. The coding path is complete.
- `task-ac4-hwrun`: hardware gate — H200 CUDA OOM on available machine.
- Next coding task: `task-ac1-hwtest` or `task-ac10-radix`. Hardware-gated tasks
  (`task-ac1b-probe`, `task-ac8-*`, `task-ac12-*`) need H200 cluster.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: No new lessons needed. The req_to_token threading omission was a straightforward
parameter gap, not a surprising failure mode warranting a BitLesson entry.
