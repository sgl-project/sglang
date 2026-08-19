"""
Sonic MoE integration for SGLang.
Sonic MoE is a high-performance Mixture-of-Experts implementation optimized
for NVIDIA Hopper GPUs (H100/H200). This module provides integration with
SGLang's MoE infrastructure.
Reference: https://github.com/Dao-AILab/sonic-moe
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import StandardTopKOutput

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()

if _is_cuda:
    pass
elif _is_cpu and _is_cpu_amx_available:
    pass


padding_size = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0

logger = logging.getLogger(__name__)

# Lazy import for optional Sonic MoE dependency
_SONICMOE_AVAILABLE: bool | None = None


def _check_sonicmoe_available() -> bool:
    """Check if Sonic MoE is available."""
    global _SONICMOE_AVAILABLE
    if _SONICMOE_AVAILABLE is None:
        try:
            import sonicmoe  # noqa: F401

            _SONICMOE_AVAILABLE = True
        except ImportError:
            _SONICMOE_AVAILABLE = False
            logger.warning_once(
                "Sonic MoE is not installed. Install it from "
                "https://github.com/Dao-AILab/sonic-moe for Hopper GPU "
                "optimized MoE kernels."
            )
    return _SONICMOE_AVAILABLE


def _is_hopper_gpu() -> bool:
    """Check if the current GPU is a Hopper architecture GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 9  # Hopper is compute capability 9.0
    except Exception:
        return False


def is_sonic_moe_supported(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    moe_runner_config: MoeRunnerConfig,
    *,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[List[int]] = None,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
) -> bool:
    """
    Check if Sonic MoE can be used for the given configuration.
    Requirements:
    - Sonic MoE package must be installed
    - Must be running on Hopper GPU (H100/H200)
    - Hidden states must be bfloat16 or float16
    - Weights must be in compatible format
    """
    if not _check_sonicmoe_available():
        logger.debug("SonicMoE disabled: sonicmoe package not available.")
        return False
    if not is_cuda() or not _is_hopper_gpu():
        logger.debug("SonicMoE disabled: requires Hopper (H100/H200) GPU.")
        return False

    # dtype / layout
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        logger.debug(
            f"SonicMoE disabled: hidden_states must be bfloat16 or float16, "
            f"got {hidden_states.dtype}."
        )
        return False
    if w1.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
        return False
    if w1.dim() != 3 or w2.dim() != 3:
        logger.debug(
            f"SonicMoE disabled: weights must be 3D, got w1.dim={w1.dim()}, "
            f"w2.dim={w2.dim()}."
        )
        return False

    if b1 is not None or b2 is not None:
        return False
    if use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16:
        return False
    if per_channel_quant or block_shape is not None:
        return False

    # EP / filter_expert: SonicMoE doesn't support expert_map
    if (
        moe_runner_config.num_experts is not None
        and moe_runner_config.num_local_experts is not None
        and moe_runner_config.num_experts != moe_runner_config.num_local_experts
    ):
        return False

    if not moe_runner_config.is_gated:
        return False
    if moe_runner_config.activation not in ("silu", "gelu", "relu"):
        return False

    if moe_runner_config.apply_router_weight_on_input:
        return False

    return True


# TODO(yuan-luo): This function needs to be revised.
def _convert_weights_to_sonic_format(
    w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert SGLang weight format to Sonic MoE format.
    SGLang format: [E, N, K] (num_experts, intermediate_size, hidden_size)
    Sonic format: [I, H, E] where the weight is accessed via permute(1, 2, 0)
    For Sonic MoE's functional API, weights are passed as [I, H, E] format
    which is then internally permuted to [E, H, I] for computation.
    """
    # Make sure original weights are contiguous in their original layout.
    w1 = w1.contiguous()  # [E, 2N, K]
    w2 = w2.contiguous()  # [E, K, N]

    # Keep as a strided view (no .contiguous() here)
    w1_sonic = w1.permute(1, 2, 0)  # [2N, K, E], stride typically (K, 1, 2N*K)
    w2_sonic = w2.permute(1, 2, 0)  # [K, N, E],  stride typically (N, 1, K*N)
    return w1_sonic, w2_sonic


def sonic_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    routed_scaling_factor: Optional[float],
) -> torch.Tensor:
    """
    Refer to sonicmoe.functional TC routing + Up/DownProjection
    """
    from sonicmoe.count_cumsum import count_cumsum
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import (
        TC_topk_router_metadata,
        _DownProjection,
        _UpProjection,
    )

    # shape
    M, K = hidden_states.shape
    num_experts = w1.size(0)
    topk = topk_ids.size(1)

    # SonicMoE supports int32 expert ids
    if topk_ids.dtype != torch.int32:
        topk_ids_i32 = topk_ids.to(torch.int32)
    else:
        topk_ids_i32 = topk_ids

    # convert weight format (with cache)
    w1_sonic, w2_sonic = _convert_weights_to_sonic_format(w1, w2)

    # activation
    act_map = {"silu": "SWIGLU", "gelu": "GEGLU", "relu": "RELU"}
    act_enum = getattr(ActivationType, act_map.get(activation, "SWIGLU"))

    # routing metadata
    topk_indices_flat = topk_ids_i32.reshape(-1)
    _expert_freq, expert_freq_offset = count_cumsum(
        topk_indices_flat, num_experts, do_cumsum=True
    )
    (
        expert_freq_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    ) = TC_topk_router_metadata(topk_ids_i32, expert_freq_offset, topk)

    total_expert_freq = M * topk
    stream_id = torch.cuda.current_stream().cuda_stream

    # Up projection (GLU fused)
    y1, z = _UpProjection.apply(
        hidden_states,
        w1_sonic,
        None,  # b1
        expert_freq_offset,
        total_expert_freq,
        topk,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        False,  # is_varlen_K
        act_enum,
        True,  # is_inference_mode_enabled
    )

    # Down projection: Sonic internally weight+reduce
    o = _DownProjection.apply(
        y1,
        z,
        w2_sonic,
        None,  # b2
        topk_weights,
        expert_freq_offset,
        M,
        topk,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        False,  # is_varlen_K
        act_enum,
    )

    if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
        o = o.mul(routed_scaling_factor)

    return o


def sonic_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig = MoeRunnerConfig(),
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Compliant with sglang.fused_moe.fused_moe signature
    """
    topk_weights, topk_ids, _ = topk_output

    assert is_sonic_moe_supported(
        hidden_states,
        w1,
        w2,
        moe_runner_config,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        b1=b1,
        b2=b2,
    ), "SonicMoE unsupported for this configuration (fallback recommended)."

    assert w1_scale is None and w2_scale is None
    assert w1_zp is None and w2_zp is None
    assert a1_scale is None and a2_scale is None
    assert moe_runner_config.gemm1_alpha is None
    assert moe_runner_config.gemm1_clamp_limit is None

    out = sonic_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=moe_runner_config.activation,
        routed_scaling_factor=moe_runner_config.routed_scaling_factor,
    )

    if moe_runner_config.inplace:
        hidden_states.copy_(out)
        return hidden_states
    return out
