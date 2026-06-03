"""Single source of truth for whether the unified_kv_triton attention backend
is active.

Activated by ``SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`` (the same env var
the existing FlashMLA backend dispatch reads). When inactive, NONE of the
unified_kv code paths must run — every hook in the memory pool / model / backend
is guarded by ``is_unified_kv_triton()`` so the default behaviour is byte-for-byte
unchanged.
"""

from __future__ import annotations

import functools
import os

_ENV = "SGLANG_HACK_FLASHMLA_BACKEND"
_VALUE = "unified_kv_triton"


@functools.lru_cache(maxsize=1)
def is_unified_kv_triton() -> bool:
    return os.environ.get(_ENV, "") == _VALUE
