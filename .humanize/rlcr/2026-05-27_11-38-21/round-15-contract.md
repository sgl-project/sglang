# Round 15 Contract

## Mainline Objective

Execute `task-ac6-cuda-graph` (coding/Claude): fix `capture_decode_step` to thread
`req_to_token` through to logical-domain selection, then add AC-6 unit tests:
- Preallocated-buffer replay correctness (100-step determinism, eager==direct call)
- `req_to_token` logical-domain verification
- `assert_no_alloc_in_region` alloc-detector negative test (CUDA-only)

The queued side issue "`capture_decode_step` calls selector without `req_to_token`" becomes
part of this implementation per the Codex review action item.

## Target ACs

- **AC-6** (CUDA graph decode path): `capture_decode_step` with preallocated buffers;
  100-step replay stable; eager output matches direct `retrieve_topk` call; alloc-detector
  negative test proves preallocated path avoids new allocations.

## Blocking Issues

None currently blocking mainline. The `req_to_token` omission in `capture_decode_step`
is the implementation gap being resolved this round.

## Queued (Out of Scope This Round)

- `task-ac4-hwrun`: hardware gate; no H200 cluster available
- `task-ac1-hwtest`: hardware gate
- `task-ac1b-probe`: analyze/Codex, depends on AC-6 hwrun
- `task-ac8-*`, `task-ac12-*`: hardware/analyze gates
- DS observability page-named fields fix: queued, needed before AC-8
- Stale comments cleanup: queued

## Success Criteria

1. `cuda_graph.py::capture_decode_step` accepts `req_to_token: Optional[torch.Tensor] = None`
   and passes it to `selector.retrieve_topk` in all three call sites.
2. New test: with bound real selector + req_to_token, replay output matches
   `retrieve_topk_via_labels` in logical-domain mode (bit-equal).
3. New test: 100 replay calls produce identical output (stability).
4. New test: state buffers after replay match a direct `selector.retrieve_topk` call.
5. New test (`@skipUnless CUDA`): `assert_no_alloc_in_region` raises `RuntimeError` when
   a tensor is allocated inside the region; silent outside.
6. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` ≥ 188 passed, 0 failed.
7. New tests in `TestCUDAGraphCapture` pass.
