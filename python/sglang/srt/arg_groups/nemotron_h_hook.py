from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.utils.common import get_device_capability, is_cuda, is_sm100_supported

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_nemotron_h_defaults(server_args: ServerArgs, model_arch: str) -> None:
    """Apply NemotronH model-specific server arg defaults and constraints."""
    model_config = server_args.get_model_config()
    is_modelopt = model_config.quantization in [
        "modelopt",
        "modelopt_fp8",
        "modelopt_fp4",
        "modelopt_mixed",
    ]
    if is_modelopt:
        assert model_config.hf_config.mlp_hidden_act == "relu2"
        if model_config.quantization == "modelopt":
            quant_algo = model_config.hf_config.quantization_config["quant_algo"]
            if quant_algo == "MIXED_PRECISION":
                server_args.quantization = "modelopt_mixed"
            else:
                server_args.quantization = (
                    "modelopt_fp4" if quant_algo == "NVFP4" else "modelopt_fp8"
                )
        else:
            server_args.quantization = model_config.quantization

    if (is_modelopt or model_config.quantization is None) and (
        server_args.moe_runner_backend == "auto"
    ):
        if is_sm100_supported() and server_args.moe_a2a_backend == "none":
            server_args.moe_runner_backend = "flashinfer_trtllm"
            logger.info(
                f"Use flashinfer_trtllm as MoE runner backend on sm100 for {model_arch}"
            )
        elif (
            (
                model_config.quantization in ("modelopt_fp4", "modelopt_mixed")
                or server_args.quantization == "modelopt_fp4"
            )
            and is_cuda()
            and (8, 0) <= get_device_capability() < (10, 0)
        ):
            server_args.moe_runner_backend = "marlin"
            logger.info(
                "Use marlin as MoE runner backend on SM80-SM90 for "
                f"{model_arch} {model_config.quantization}"
            )
        else:
            server_args.moe_runner_backend = "flashinfer_cutlass"

    server_args._handle_mamba_radix_cache(
        model_arch=model_arch,
        sm100_default_attention_backend="flashinfer",
        fallback_attention_backend="flashinfer",
    )
    assert server_args.attention_backend != "triton", (
        "NemotronHForCausalLM does not support triton attention backend,"
        "as the first layer might not be an attention layer"
    )
