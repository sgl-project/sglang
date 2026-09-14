"""Config-time override declarations for cohere2_moe.

Architectures: Cohere2MoeForCausalLM, Cohere2VisionForConditionalGeneration.
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


def _is_nvfp4_pack_quantized(hf_config: Any) -> bool:
    # Note(mmangkad): nvfp4-pack-quantized is llm-compressor's output format.
    qc = getattr(hf_config, "quantization_config", None)
    if not isinstance(qc, dict):
        return False
    groups = qc.get("config_groups") or {}
    formats = [qc.get("format", "")] + [
        g.get("format", "") for g in groups.values() if isinstance(g, dict)
    ]
    return any("nvfp4" in str(fmt) for fmt in formats)


@_register_for(
    "Cohere2VisionForConditionalGeneration",
    "Cohere2MoeForCausalLM",
)
def _cohere2_moe_runner_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend != "auto":
        return {}
    if not get_platform().is_sm100:
        return {}
    if model_config_of(server_args).quantization is not None:
        if not _is_nvfp4_pack_quantized(hf_config):
            return {}
    logger.info(
        "Command-A-Plus on SM10X: moe_runner_backend=flashinfer_trtllm "
        "(trtllm-gen fused MoE)."
    )
    return {"moe_runner_backend": "flashinfer_trtllm"}
