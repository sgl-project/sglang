from __future__ import annotations

"""
MoE runner backend powered by HPC-Ops (https://github.com/Tencent/hpc-ops),
a production-grade operator library for LLM inference developed by the
Tencent Hunyuan AI Infra team.

The backend wraps the monolithic FP8 fused-MoE kernels ``fuse_moe_blockwise``
(128x128 block-quantized weights + per-token-group-128 activations, e.g.
Qwen3-FP8 / Hy3-FP8 style checkpoints) and ``fuse_moe`` (per-tensor weight and
static per-tensor activation quantization). Both kernels fuse
gather -> grouped gate_up GEMM -> SiLU-and-mul -> grouped down GEMM -> weighted
reduce into one call and consume *global* top-k expert ids together with
``rank_ep`` / ``num_expert_total``, so expert parallelism with contiguous
expert partitioning works without a local-expert remap.

Only supported on NVIDIA Hopper / Blackwell (sm90+). Enable it explicitly with
``--moe-runner-backend hpc_ops``.
"""

import functools
import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo, register_fused_func

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

# The HPC-Ops group GEMM tiles N by 128 and the blockwise path quantizes
# activations in groups of 128, so every participating dim must be 128-aligned.
HPC_OPS_BLOCK_SIZE = 128
# The K dim of the blockwise weight scales must be padded to a multiple of 4
# (see hpc-ops tests: (k // 128 + 3) // 4 * 4).
_SCALE_K_ALIGN = 4


@functools.cache
def has_hpc_ops() -> bool:
    """Return True if the ``hpc`` package (HPC-Ops) is installed."""
    return importlib.util.find_spec("hpc") is not None


def pad_hpc_ops_block_scale(scale: torch.Tensor) -> torch.Tensor:
    """Pad the K dim (last dim) of a [E, N/128, K/128] block scale to %4."""
    k = scale.shape[-1]
    k_pad = (k + _SCALE_K_ALIGN - 1) // _SCALE_K_ALIGN * _SCALE_K_ALIGN
    if k == k_pad and scale.is_contiguous():
        return scale
    padded = scale.new_zeros((*scale.shape[:-1], k_pad))
    padded[..., :k].copy_(scale)
    return padded


@dataclass
class HpcOpsMoeQuantInfo(MoeQuantInfo):
    """Quant payload for the HPC-Ops fused MoE kernels.

    ``block_quant`` selects between the two kernels:
    - True: ``fuse_moe_blockwise`` with ``w13_weight_scale_inv`` /
      ``w2_weight_scale_inv`` ([E, N/128, K/128], K dim padded to %4) and
      dynamic per-token-group-128 activation quantization.
    - False: ``fuse_moe`` with per-expert dequant alphas
      ``gate_up_alphas = w13_weight_scale * w13_input_scale`` ([E]),
      ``down_alphas = w2_weight_scale * w2_input_scale`` ([E]) and the static
      activation scales ``w13_input_scale`` / ``w2_input_scale`` (scalars).
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    block_quant: bool
    global_num_experts: int
    moe_ep_rank: int
    # Blockwise path
    w13_weight_scale_inv: Optional[torch.Tensor] = None
    w2_weight_scale_inv: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None
    # Per-tensor path
    gate_up_alphas: Optional[torch.Tensor] = None
    down_alphas: Optional[torch.Tensor] = None
    w13_input_scale: Optional[torch.Tensor] = None
    w2_input_scale: Optional[torch.Tensor] = None


def _check_runner_config_supported(runner_config: MoeRunnerConfig) -> None:
    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise ValueError(
            "The hpc_ops MoE runner backend only supports the gated silu "
            f"activation, got activation={runner_config.activation}, "
            f"is_gated={runner_config.is_gated}."
        )
    if runner_config.num_fused_shared_experts != 0:
        raise ValueError(
            "The hpc_ops MoE runner backend does not support fused shared experts."
        )
    if runner_config.apply_router_weight_on_input:
        raise ValueError(
            "The hpc_ops MoE runner backend does not support "
            "apply_router_weight_on_input."
        )
    if runner_config.no_combine:
        raise ValueError(
            "The hpc_ops MoE runner backend does not support no_combine "
            "(the fused kernel always reduces over top-k experts)."
        )
    if (
        runner_config.gemm1_alpha is not None
        or runner_config.gemm1_clamp_limit is not None
    ):
        raise ValueError(
            "The hpc_ops MoE runner backend does not support gemm1_alpha / "
            "gemm1_clamp_limit."
        )


@register_fused_func("none", "hpc_ops")
def fused_experts_none_to_hpc_ops(
    dispatch_output: StandardDispatchOutput,
    quant_info: HpcOpsMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    import hpc

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.quantization.fp8_kernel import (
        scaled_fp8_quant,
        sglang_per_token_group_quant_fp8,
    )

    _check_runner_config_supported(runner_config)

    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output

    assert x.dtype == torch.bfloat16, (
        "The hpc_ops MoE runner backend only supports bf16 hidden states, "
        f"got {x.dtype}."
    )

    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)

    if quant_info.block_quant:
        assert quant_info.block_shape == [
            HPC_OPS_BLOCK_SIZE,
            HPC_OPS_BLOCK_SIZE,
        ], (
            "The hpc_ops MoE runner backend only supports 128x128 block "
            f"quantization, got {quant_info.block_shape}."
        )
        x_q, x_scale = sglang_per_token_group_quant_fp8(x, HPC_OPS_BLOCK_SIZE)
        output = hpc.fuse_moe_blockwise(
            x_q,
            x_scale,
            quant_info.w13_weight,
            quant_info.w13_weight_scale_inv,
            quant_info.w2_weight,
            quant_info.w2_weight_scale_inv,
            topk_ids,
            topk_weights,
            quant_info.moe_ep_rank,
            quant_info.global_num_experts,
        )
    else:
        x_q, _ = scaled_fp8_quant(x, quant_info.w13_input_scale)
        act_and_mul_scale = 1.0 / quant_info.w2_input_scale.reshape(1)
        output = hpc.fuse_moe(
            x_q,
            quant_info.w13_weight,
            quant_info.w2_weight,
            quant_info.gate_up_alphas,
            quant_info.down_alphas,
            act_and_mul_scale,
            topk_ids,
            topk_weights,
            quant_info.moe_ep_rank,
            quant_info.global_num_experts,
        )

    if runner_config.routed_scaling_factor is not None:
        output *= runner_config.routed_scaling_factor

    return StandardCombineInput(hidden_states=output)
