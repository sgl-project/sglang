"""ROCm part of the inkling override declarations: the gfx950 guard, the MegaMoE
refusal, the decode KV-split defaults and exact all-reduces. inkling calls it on HIP;
it is pure Python and never imports aiter."""

import logging
import os
from typing import Any

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported

logger = logging.getLogger(__name__)


def rocm_overrides(cfg: Any) -> None:
    """Inkling2's ROCm launch checks and env defaults."""
    if (
        not is_gfx95_supported()
        and not envs.SGLANG_INKLING2_ALLOW_UNVALIDATED_ROCM.get()
    ):
        raise ValueError(
            "Inkling2 on ROCm is validated on gfx950 (MI350X / MI355X) only; its "
            "kernels are sized and tuned there. Set "
            "SGLANG_INKLING2_ALLOW_UNVALIDATED_ROCM=1 to run it anyway."
        )
    if "megamoe" in (cfg.moe_a2a_backend, cfg.moe_runner_backend):
        raise ValueError(
            "Inkling2's MegaMoE runs DeepGEMM kernels over CUDA symmetric "
            "memory; on ROCm its experts run their own kernels with an "
            "expert-parallel all-reduce. Drop --moe-a2a-backend / "
            "--moe-runner-backend megamoe."
        )
    # decode: KV splits of 32 keys, up to 128 per request (swept on MI350X)
    if not envs.SGLANG_TRITON_DECODE_SPLIT_MIN_TOKENS.is_set():
        envs.SGLANG_TRITON_DECODE_SPLIT_MIN_TOKENS.set("32")
    if not envs.SGLANG_TRITON_DECODE_SPLIT_MAX.is_set():
        envs.SGLANG_TRITON_DECODE_SPLIT_MAX.set("128")
    # the image's INT8 quick reduce quantizes prefill all-reduces, which moves
    # outputs off the exact-reduction reference
    quick_reduce = os.environ.get("ROCM_QUICK_REDUCE_QUANTIZATION", "NONE")
    if quick_reduce != "NONE" and not envs.SGLANG_INKLING2_ALLOW_QUICK_REDUCE.get():
        logger.info(
            "Inkling2 keeps exact all-reduces on ROCm: ignoring "
            f"ROCM_QUICK_REDUCE_QUANTIZATION={quick_reduce} (set "
            "SGLANG_INKLING2_ALLOW_QUICK_REDUCE=1 to keep it)."
        )
        os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = "NONE"
