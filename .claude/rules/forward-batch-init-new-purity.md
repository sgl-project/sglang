# `ForwardBatch.init_new` must not mutate the ScheduleBatch

`ForwardBatch.init_new` (and any ForwardBatch factory) must treat the input
`ScheduleBatch` as read-only: do not rebind its fields and do not write into
its sub-objects.

- Per-forward overrides go through explicit kw-only parameters of `init_new`
  (e.g. `capture_hidden_mode`, `return_hidden_states_before_norm`). Do not
  reintroduce the old pattern of writing a ScheduleBatch field and having
  `init_new` consume-and-reset it.
- Callers that need to influence a forward launched by
  `TpModelWorker.forward_batch_generation` pass the override through its
  kw-only parameters instead of writing onto the batch.

Temporary exceptions (do not add new ones):

- `seq_lens_sum` lazy backfill inside `init_new` — scheduled for removal in a
  follow-up op.
- `sampling_info` sub-object writes inside `init_new` — documented exception
  until the sampling forward-copy op lands.
