from __future__ import annotations

import functools
import logging

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, is_gfx1250_supported, is_hip

logger = logging.getLogger(__name__)

# Set before KV-pool sizing; V4.1 keeps the radix path.
_gfx1250_v4_unified = False


def maybe_enable_gfx1250_v4_unified_kv(model_type: str | None) -> None:
    """gfx1250 V4 Pro defaults to the unified-KV pool (aiter decode)."""
    global _gfx1250_v4_unified
    if _gfx1250_v4_unified:
        return
    if model_type != "deepseek_v4":
        return
    if envs.SGLANG_HACK_FLASHMLA_BACKEND.get() != "auto":
        return
    if not is_gfx1250_supported():
        return
    _gfx1250_v4_unified = True
    logger.info(
        "gfx1250 DeepSeek-V4 uses the unified-KV pool; set "
        "SGLANG_HACK_FLASHMLA_BACKEND=tilelang to keep the radix path."
    )


@functools.lru_cache(maxsize=1)
def _unified_kv_backend_selected() -> bool:
    # unified_kv_triton is only implemented on HIP (ROCm)
    return is_hip() and envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "unified_kv_triton"


def is_unified_kv_triton() -> bool:
    if _unified_kv_backend_selected():
        return True
    return _gfx1250_v4_unified


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
    # gfx1250 ASM decode has no 16-local-head variant; keep attn TP small.
    if not (is_gfx95_supported() or is_gfx1250_supported()):
        logger.warning(
            "SGLANG_DSV4_UNIFIED_KV_FP8=1 needs an AMD gfx95 or gfx1250 GPU; "
            "falling back to the bf16 unified_kv pool (see unified_fp8= in the "
            "DSV4 memory log)."
        )
        return False
    return True
