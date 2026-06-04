from __future__ import annotations

import functools
import os


@functools.lru_cache(maxsize=1)
def is_unified_kv_triton() -> bool:
    return os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "") == "unified_kv_triton"
