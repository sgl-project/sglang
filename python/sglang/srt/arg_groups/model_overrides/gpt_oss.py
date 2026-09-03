"""Config-time override declarations for gpt_oss.

Architectures: GptOssForCausalLM.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import (
    get_nvidia_driver_version,
    is_cpu,
    is_mps,
    is_triton_kernels_available,
)

logger = logging.getLogger(__name__)


@_register_for("GptOssForCausalLM")
def _gpt_oss_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    # Set attention backend for GPT-OSS
    if is_attention_backend_not_set(cfg):
        if get_platform().is_sm100:
            overrides["attention_backend"] = "trtllm_mha"
        elif get_platform().is_sm90:
            overrides["attention_backend"] = "fa3"
        elif is_cpu() and get_platform().has_amx:
            overrides["attention_backend"] = "intel_amx"
        elif get_platform().is_xpu:
            overrides["attention_backend"] = "intel_xpu"
        elif get_platform().is_hip:
            overrides["attention_backend"] = "aiter"
        elif not (is_mps() and use_mlx()):
            # Exempt MLX only -- it owns attention in its own runner.  macOS
            # without MLX still falls through to triton and fails fast below,
            # rather than landing on torch_native (no sliding window, no sinks).
            overrides["attention_backend"] = "triton"
    if get_platform().is_xpu:
        # Check for bf16 dtype on Intel XPU. Reads the pristine dtype request,
        # which equals the legacy mid-branch read: dtype had no earlier writer
        # for this arch.
        if cfg.dtype == "auto":
            logger.warning(
                "GptOssForCausalLM on Intel XPU currently supports bfloat16 dtype only"
            )
        elif cfg.dtype not in ["bfloat16"]:
            raise NotImplementedError(
                f"GptOssForCausalLM on Intel XPU only supports bfloat16 dtype, "
                f"but got '{cfg.dtype}'. Please use --dtype bfloat16 or remove --dtype to use auto."
            )
    quantization_config = getattr(hf_config, "quantization_config", None)
    is_mxfp4_quant_format = (
        quantization_config is not None
        and quantization_config.get("quant_method") == "mxfp4"
    )
    if is_mxfp4_quant_format:
        # use bf16 for mxfp4 triton kernels
        overrides["dtype"] = "bfloat16"
    if cfg.moe_runner_backend == "auto":
        if get_platform().is_sm100 and is_mxfp4_quant_format:
            overrides["moe_runner_backend"] = "flashinfer_mxfp4"
            logger.warning(
                "Detected SM100 and MXFP4 quantization format for GPT-OSS model, enabling FlashInfer MXFP4 MOE kernel."
            )
        elif get_platform().is_sm120 and is_mxfp4_quant_format:
            overrides["moe_runner_backend"] = "flashinfer_mxfp4"
            logger.warning(
                "Detected SM120 and MXFP4 quantization format for GPT-OSS model, "
                "enabling FlashInfer CUTLASS MXFP4 MOE kernel."
            )
        elif (
            get_platform().is_hip and envs.SGLANG_USE_AITER.get()
        ) and is_mxfp4_quant_format:
            overrides["moe_runner_backend"] = "auto"
            logger.warning(
                "Detected ROCm and MXFP4 quantization format for GPT-OSS model, enabling aiter MXFP4 MOE kernel."
            )
            ## The AITER MXFP4 fused-MoE path for GPT-OSS expects the
            ## SEPARATED gate/up tile layout (matches the
            ## `gptoss_fp4_tuned_fmoe.csv` flydsl entries and the
            ## Mxfp4MoEMethod weight shuffle). Other AITER MXFP4
            ## callers default to INTERLEAVE; opt this path out
            ## unless the user explicitly overrode it.
            # envs.SGLANG_USE_AITER_MOE_GU_ITLV.set(False)
        elif get_platform().is_hip and envs.SGLANG_USE_AITER.get():
            # For GPT-OSS bf16 on ROCm with aiter, use triton backend
            # because aiter CK kernel doesn't support all GEMM dimensions
            overrides["moe_runner_backend"] = "triton"
            logger.warning(
                "Detected ROCm with SGLANG_USE_AITER for GPT-OSS bf16 model, using triton MOE kernel."
            )
        elif get_platform().is_musa and envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
            overrides["moe_runner_backend"] = "deep_gemm"
            logger.warning(
                "Detected MUSA with SGLANG_DEEPEP_BF16_DISPATCH for bf16 model, using deep_gemm kernel."
            )
        elif (
            cfg.ep_size == 1
            and is_triton_kernels_available()
            and cfg.quantization is None
            and not (is_cpu() and get_platform().has_amx)
        ):
            # The triton_kernels package segfaults on Blackwell (B200)
            # with NVIDIA driver >= 595. Fall back to triton backend.
            if get_platform().is_blackwell and get_nvidia_driver_version() >= (595,):
                overrides["moe_runner_backend"] = "triton"
                logger.warning(
                    "Detected GPT-OSS model on Blackwell with driver >= 595, "
                    "using triton MOE kernel to avoid triton_kernels SIGSEGV."
                )
            else:
                overrides["moe_runner_backend"] = "triton_kernel"
                logger.warning(
                    "Detected GPT-OSS model, enabling triton_kernels MOE kernel."
                )
    return overrides
