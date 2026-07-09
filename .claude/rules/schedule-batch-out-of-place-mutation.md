---
paths:
  - "python/sglang/srt/**/*.py"
---

# Mutate ScheduleBatch fields out of place

Never mutate a `ScheduleBatch` direct field in place — no `.extend()` /
`.append()`, no `+=` / `-=` / `|=`, no tensor `.add_()` / `.fill_()`, no
slice or index assignment. Build the new value and rebind the field:

```python
# Bad
self.reqs.extend(other.reqs)
self.seq_lens.add_(1)
self.extend_lens[i] -= encoder_len

# Good
self.reqs = self.reqs + other.reqs
self.seq_lens = self.seq_lens + 1
extend_lens = self.extend_lens[:]   # mutate a local copy in loops,
extend_lens[i] -= encoder_len       # then rebind once afterwards
self.extend_lens = extend_lens
```

Why:

- `ScheduleBatch.copy()` snapshots and the overlap scheduler's queued
  references rely on old objects staying frozen; in-place writes punch
  through those snapshots.
- Rebinding is the prerequisite for the per-step immutable ScheduleBatch
  refactor.
- Derived values (e.g. `seq_lens_sum`) can only be lazily recomputed safely
  when the underlying fields change by rebinding.

Out of scope (mutation is the intended semantics there): penalizer
cumulative buffers, CUDA graph static buffers, `ForwardBatch` fields, and
attention-backend metadata.
