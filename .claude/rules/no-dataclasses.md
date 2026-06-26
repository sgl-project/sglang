---
paths:
  - "**/*.py"
---

# Use `msgspec.Struct`, not `@dataclass`

Define data containers as `msgspec.Struct`. Do not add new
`dataclasses.dataclass` (or `attrs`) — they weaken strict type checking and
don't map onto Rust structs for the planned Rust migration.

```python
import msgspec

class LoadSnapshot(msgspec.Struct):   # frozen=, kw_only=, omit_defaults= as needed
    dp_rank: int = 0
    tokens: list[int] = []            # mutable defaults are safe
```

- Methods / `@classmethod` constructors go on the `Struct`; see
  `python/sglang/srt/managers/load_snapshot.py`.
- New code only. Existing `@dataclass` is grandfathered — migrate opportunistically
  while editing the file, not in drive-by sweeps.
- If a third-party API forces `@dataclass`, keep it at that boundary only.
