"""ROCm part of the inkling override declarations: the gfx950 guard, the MegaMoE
refusal and exact all-reduces. inkling calls it on HIP; it is pure Python and never
imports aiter."""

import logging
import os
from typing import Any

from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


def rocm_overrides(cfg: Any) -> None:
    """Inkling2's ROCm launch checks."""
    # its kernels are sized and tuned for gfx950
    if not get_platform().is_gfx95:
        raise ValueError("Inkling2 on ROCm requires gfx950 (MI350X / MI355X)")
    if "megamoe" in (cfg.moe_a2a_backend, cfg.moe_runner_backend):
        raise ValueError(
            "Inkling2's MegaMoE runs DeepGEMM kernels over CUDA symmetric "
            "memory; on ROCm its experts run their own kernels with an "
            "expert-parallel all-reduce. Drop --moe-a2a-backend / "
            "--moe-runner-backend megamoe."
        )
    # the INT8 quick reduce the ROCm images export quantizes prefill all-reduces,
    # which moves outputs off the exact-reduction reference
    level = "ROCM_QUICK_REDUCE_QUANTIZATION"
    if os.environ.get(level, "NONE") != "NONE":
        logger.info(f"Inkling2 keeps exact all-reduces on ROCm: ignoring {level}")
        os.environ[level] = "NONE"
