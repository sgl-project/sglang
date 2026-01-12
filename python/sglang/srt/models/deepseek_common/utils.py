import logging
import math
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
    is_nvidia_cublas_cu12_version_ge_12_9,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()
_is_gfx95_supported = is_gfx95_supported()

_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported

_is_cublas_ge_129 = is_nvidia_cublas_cu12_version_ge_12_9()

logger = logging.getLogger(__name__)

NVFP4_CKPT_FP8_ATTN_QUANT_MODULES = ["q_b_proj"]

FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "nsa",
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
]


def enable_nextn_moe_bf16_cast_to_fp8(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    return (
        envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get()
        and quant_config is not None
        and quant_config.get_name() == "modelopt_fp4"
        and get_moe_runner_backend().is_deep_gemm()
    )


def add_forward_absorb_core_attention_backend(backend_name: str) -> None:
    if backend_name not in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
        FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.")


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    return scaling[..., None, None]
