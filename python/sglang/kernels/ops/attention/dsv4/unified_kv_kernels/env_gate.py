from __future__ import annotations

import functools

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip


@functools.lru_cache(maxsize=1)
def is_unified_kv_triton() -> bool:
    # unified_kv_triton is only implemented on HIP (ROCm)
    return is_hip() and envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "unified_kv_triton"


@functools.lru_cache(maxsize=1)
def use_flydsl_decode() -> bool:
    # Route unified_kv decode to the FlyDSL v4 bf16 kernel (aiter). ROCm only.
    return is_hip() and envs.SGLANG_USE_FLYDSL_DECODE.get()
