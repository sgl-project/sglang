"""Config-time override declarations for minimax_m2.

Architectures: MiniMaxM2ForCausalLM.
"""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    model_config_of,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for("MiniMaxM2ForCausalLM")
def _minimax_m2_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides = {"enable_tf32_matmul": True}
    logger.info(
        "Enable TF32 matmul for MiniMaxM2ForCausalLM model to improve gate gemm performance."
    )
    if (
        get_platform().is_sm100
        and cfg.moe_runner_backend == "auto"
        and model_config_of(server_args).quantization == "modelopt_fp4"
    ):
        overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
        logger.info(
            "Use flashinfer_trtllm_routed as MoE runner backend on SM10X "
            "for MiniMaxM2ForCausalLM with modelopt_fp4."
        )
    return overrides
