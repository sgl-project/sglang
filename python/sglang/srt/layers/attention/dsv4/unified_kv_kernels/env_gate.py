from __future__ import annotations

import functools
import os

from sglang.srt.utils import is_hip


@functools.lru_cache(maxsize=1)
def is_unified_kv_triton() -> bool:
    # unified_kv_triton is only implemented on HIP (ROCm)
    return (
        is_hip()
        and os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "") == "unified_kv_triton"
    )
