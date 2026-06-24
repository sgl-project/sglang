from __future__ import annotations

from typing import Protocol


class FreeSpaceProvider(Protocol):
    """Capability the owned-kv allocator needs from whoever owns free space:
    make room for ``num_tokens`` and describe the shortage on OOM. The owned-kv
    allocator stays blind to where the space comes from (radix cache eviction,
    a no-op, etc.)."""

    def ensure_free(self, num_tokens: int) -> None: ...

    def describe_for_oom(self) -> str: ...
