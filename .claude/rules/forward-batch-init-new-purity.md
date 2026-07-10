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

- `init_new` backfills `batch.seq_lens_sum` when it is `None` (an in-place
  write on the ScheduleBatch). Tolerated because the whole ScheduleBatch
  `seq_lens` family is slated for removal (moving to kv-committed lengths);
  revisit when that lands.
- `init_new` writes `batch.sampling_info` sub-object attributes (grammars,
  canary ids); same object shared with the ScheduleBatch until the sampling
  forward-copy op lands.
- `_expand_mrope_from_input` lazily fills
  `mm_input.mrope_position_delta_repeated_cache` on the ScheduleBatch-owned
  `MultimodalInputs` (pre-existing memoization; relocation is a candidate
  follow-up op).
