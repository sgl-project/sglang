from __future__ import annotations

import functools
import logging

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, is_gfx1250_supported, is_hip

logger = logging.getLogger(__name__)

# Armed before KV-pool sizing. V4 Pro (model_type deepseek_v4) on gfx1250 has no
# packed aiter_sparse reader; the unified bf16 pool's decode already calls aiter
# pa_decode_sparse once per compress stream. V4.1 must not set this.
_gfx1250_v4_unified = False


def maybe_enable_gfx1250_v4_unified_kv(model_type: str | None) -> None:
    """Default gfx1250 V4 Pro onto the unified-KV pool so decode uses aiter.

    V4.1 keeps the radix path. An explicit SGLANG_HACK_FLASHMLA_BACKEND other
    than auto is left alone, so tilelang stays available.
    """
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
        "gfx1250 DeepSeek-V4 decode uses the unified-KV pool "
        "(aiter pa_decode_sparse). V4.1 is unchanged. Set "
        "SGLANG_HACK_FLASHMLA_BACKEND=tilelang to keep the packed radix path."
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
    # gfx1250 also carries the OCP e4m3fn fp8_dtype (see
    # sglang.kernels.ops.quantization.fp8_kernel.is_fp8_fnuz, which only
    # special-cases "gfx94"), and aiter's gfx1250 decode/prefill kernels for
    # this layout (mla_decode_fwd_v4_nm, mla_sparse_prefill_fp8_asm /
    # pa_sparse_prefill_fp8_opus) already ship for this arch, so it is allowed
    # here alongside gfx95. The ASM decode kernel only has rows for GQA in
    # {32, 64, 128} at qSeqLen=1 on gfx1250 (no 16-head row); callers must keep
    # attn_tp_size small enough (e.g. --attn-dp-size == --tp) that local heads
    # land on one of those, or the aiter call raises "no shipped variant".
    if not (is_gfx95_supported() or is_gfx1250_supported()):
        logger.warning(
            "SGLANG_DSV4_UNIFIED_KV_FP8=1 needs an AMD gfx95 or gfx1250 GPU; "
            "falling back to the bf16 unified_kv pool (see unified_fp8= in the "
            "DSV4 memory log)."
        )
        return False
    return True
