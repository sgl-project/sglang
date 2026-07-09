---
paths:
  - "**/*.py"
---

# `ForwardBatch.init_new` must not mutate the ScheduleBatch

`init_new` (and any ForwardBatch factory) treats the input `ScheduleBatch` as
read-only: no field rebinds, no sub-object writes. Per-forward overrides go
through the kw-only parameters of `init_new` /
`TpModelWorker.forward_batch_generation`, never through a ScheduleBatch field
that `init_new` consumes. Regular batch-preparation writes (`out_cache_loc`,
`seq_lens_*`, ...) before calling `init_new` are out of scope.

Known object-sharing notes (do not add new exceptions):

- `init_new` writes `ret.sampling_info` sub-object attributes (grammars,
  canary ids); same object as `batch.sampling_info` until the sampling
  forward-copy op lands.
- `_expand_mrope_from_input` lazily fills
  `mm_input.mrope_position_delta_repeated_cache` on the ScheduleBatch-owned
  `MultimodalInputs` (pre-existing memoization; relocation is a candidate
  follow-up op).
