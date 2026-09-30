---
paths:
  - "**/*.py"
---

# Use `msgspec.Struct`, not `@dataclass`

Define data containers as `msgspec.Struct`. Do not add new
`dataclasses.dataclass` (or `attrs`) — they weaken strict type checking and
don't translate cleanly for multi-language support (e.g. the planned Rust migration).

```python
import msgspec

class LoadSnapshot(msgspec.Struct):   # prefer frozen= and omit_defaults=; kw_only= as needed
    dp_rank: int = 0
    tokens: list[int] = []            # mutable defaults are safe
```

- Methods / `@classmethod` constructors go on the `Struct`; see
  `python/sglang/srt/managers/load_snapshot.py`.
- New code only. Existing `@dataclass` is grandfathered — migrate opportunistically
  while editing the file, not in drive-by sweeps.
- Exception: values built or mutated inside a `torch.compile`d forward stay plain
  `__slots__` classes; Dynamo cannot construct or mutate a `Struct`.
