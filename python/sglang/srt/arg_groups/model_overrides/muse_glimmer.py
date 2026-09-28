"""Config-time override declarations for muse_glimmer.

Architectures: MuseGlimmerForCausalLM, MuseGlimmerForConditionalGeneration.
"""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for("MuseGlimmerForConditionalGeneration", "MuseGlimmerForCausalLM")
def _muse_glimmer_fp4_gemm_runner_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if get_platform().is_sm120 and cfg.fp4_gemm_runner_backend == "auto":
        logger.info("Use marlin as FP4 GEMM runner backend on SM120 for Muse Glimmer")
        return {"fp4_gemm_runner_backend": "marlin"}
    return {}
