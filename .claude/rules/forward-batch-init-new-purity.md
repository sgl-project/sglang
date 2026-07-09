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

Temporary exceptions (do not add new ones):

- `seq_lens_sum` lazy backfill inside `init_new` — removed in a follow-up op.
- `sampling_info` sub-object writes inside `init_new` — until the sampling
  forward-copy op lands.
