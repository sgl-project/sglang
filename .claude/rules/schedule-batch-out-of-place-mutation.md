---
paths:
  - "python/sglang/srt/**/*.py"
---

# Mutate ScheduleBatch fields out of place

Never mutate a `ScheduleBatch` direct field in place — no `.extend()`/`.append()`,
`+=`/`-=`/`|=`, tensor `.add_()`/`.fill_()`, or slice/index assignment. Build the
new value and rebind the field:

```python
# Bad
self.reqs.extend(other.reqs)
self.seq_lens.add_(1)
self.extend_lens[i] -= encoder_len

# Good
self.reqs = self.reqs + other.reqs
self.seq_lens = self.seq_lens + 1
lens = self.extend_lens[:]; lens[i] -= encoder_len; self.extend_lens = lens  # loop a copy, rebind once
```

Why: `copy()` snapshots and the overlap scheduler's queued references rely on old
objects staying frozen (the prerequisite for the per-step immutable ScheduleBatch
refactor); derived values like `seq_lens_sum` only recompute safely when fields
are rebound.

Out of scope (mutation intended): penalizer buffers, CUDA graph static buffers,
`ForwardBatch` fields, attention-backend metadata.
