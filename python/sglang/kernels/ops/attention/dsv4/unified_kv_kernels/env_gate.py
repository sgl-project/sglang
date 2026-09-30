from __future__ import annotations

import functools
import logging

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.srt.utils.common import is_gfx1250_supported

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def is_unified_kv_triton() -> bool:
    # unified_kv_triton is only implemented on HIP (ROCm)
    return is_hip() and envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "unified_kv_triton"


@functools.lru_cache(maxsize=1)
def is_unified_kv_fp8() -> bool:
    # fp8 is a layout variant of the unified pool, so it can never outlive the
    # unified gate -- the sizing, the allocation and the writers all key off this
    # one call, so an unsupported device has to be turned away here or the three
    # will disagree.
    if not (is_unified_kv_triton() and envs.SGLANG_DSV4_UNIFIED_KV_FP8.get()):
        return False
    # two-pool fp8 is OCP e4m3 plus E8M0 tile scales, so it only means anything
    # where MX is native: on gfx94x sglang's fp8_dtype is e4m3fnuz (max 224, not
    # 448) and the writers would feed values the pool's own dtype misreads.
    if not is_gfx95_supported():
        logger.warning(
            "SGLANG_DSV4_UNIFIED_KV_FP8=1 needs an AMD gfx95 GPU; falling back to "
            "the bf16 unified_kv pool (see unified_fp8= in the DSV4 memory log)."
        )
        return False
    return True


@functools.lru_cache(maxsize=1)
def _decode_inv_rope_fusion_enabled() -> bool:
    # bf16 Triton paged decode only: the fp8 asm reader and gfx1250's aiter
    # pa_decode_sparse have no epilogue hook.
    return (
        envs.SGLANG_OPT_DSV4_DECODE_FUSED_INVROPE.get()
        and is_unified_kv_triton()
        and not is_unified_kv_fp8()
        and not is_gfx1250_supported()
    )


def unified_decode_fuses_inv_rope(forward_mode) -> bool:
    """Whether the attention output comes back inverse-RoPE'd. Static config +
    forward_mode only, so the decision is fixed per captured graph."""
    return _decode_inv_rope_fusion_enabled() and (
        forward_mode.is_decode_or_idle() or forward_mode.is_target_verify()
    )
