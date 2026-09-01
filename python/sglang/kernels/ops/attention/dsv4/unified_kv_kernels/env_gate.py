from __future__ import annotations

import functools

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip


@functools.lru_cache(maxsize=1)
def is_unified_kv_aiter() -> bool:
    """FP8 SoA unified_kv path driving aiter's mla_decode_fwd_v4_nm (#3112) and
    pa_sparse_prefill_fp8_opus (#3751). Shares ALL of the unified_kv metadata /
    radix plumbing (so ``is_unified_kv_triton`` also returns True), but swaps the
    single bf16 unified buffer for a two-buffer fp8-nope + bf16-rope SoA store and
    dispatches the aiter fp8 kernels instead of the vendored Triton ones."""
    return (
        is_hip() and envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "unified_kv_aiter"
    )


@functools.lru_cache(maxsize=1)
def is_unified_kv_triton() -> bool:
    # unified_kv is only implemented on HIP (ROCm). Both the Triton (bf16) and the
    # aiter (fp8 SoA) variants use the same unified_kv metadata/radix plumbing, so
    # this gate is True for either; is_unified_kv_aiter() selects the fp8 kernels.
    return is_hip() and envs.SGLANG_HACK_FLASHMLA_BACKEND.get() in (
        "unified_kv_triton",
        "unified_kv_aiter",
    )
