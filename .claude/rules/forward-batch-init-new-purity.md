---
paths:
  - "**/*.py"
---

# `ForwardBatch.init_new` must not mutate the ScheduleBatch

`init_new` (and any ForwardBatch factory) treats the input `ScheduleBatch` as
read-only. Per-forward overrides go through the kw-only params of `init_new` /
`TpModelWorker.forward_batch_generation`, never a ScheduleBatch field. Batch-prep
writes (`out_cache_loc`, `seq_lens_*`, ...) before `init_new` are out of scope.

Tolerated exceptions (don't add new ones):

- `seq_lens_sum` backfill — the `seq_lens` family is slated for removal (→ kv-committed lengths).
- `sampling_info` sub-object writes (grammars, canary ids) — shared object, until the sampling forward-copy op.
- `_expand_mrope_from_input` memoizing `mrope_position_delta_repeated_cache` — pre-existing.
