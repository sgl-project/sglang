"""HPC-ops MoE runner backend.

This module integrates Tencent's hpc-ops fused MoE kernels into SGLang's
MoE runner framework. It supports two FP8 quantization schemes:

  * **Per-tensor** — ``hpc.fuse_moe`` via :func:`hpc_fuse_moe`
  * **Block-wise** — ``hpc.fuse_moe_blockwise`` via :func:`hpc_fuse_moe_blockwise`

The backend is registered as a **fused-only** runner (no ``RunnerCore``),
following the same pattern as ``marlin.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

_FP8_E4M3_MAX = 448.0

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class HpcRunnerInput(RunnerInput):
    """Input bundle passed to the HPC runner core (unused in fused path)."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.HPC


@dataclass
class HpcRunnerOutput(RunnerOutput):
    """Output bundle returned from the HPC runner core (unused in fused path)."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.HPC


@dataclass
class HpcMoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the HPC MoE backend.

    Supports both per-tensor and block-wise FP8 quantization schemes.

    **Per-tensor fields** (used when ``is_block_quantized is False``):
        w13_weight: Gate+up expert weights, fp8.
        w2_weight: Down expert weights, fp8.
        w13_scale: Per-tensor scale for gate+up GEMM.
        w2_scale: Per-tensor scale for down GEMM.
        act_and_mul_scale: Scale at the activation boundary.

    **Block-wise fields** (used when ``is_block_quantized is True``):
        w13_weight: Gate+up expert weights, fp8.
        w2_weight: Down expert weights, fp8.
        w13_weight_scale: Per-block weight scale for gate+up.
        w2_weight_scale: Per-block weight scale for down.
        x_scale: Per-block input activation scale.
    """

    # Common
    w13_weight: torch.Tensor = None
    w2_weight: torch.Tensor = None
    is_block_quantized: bool = False

    # Per-tensor FP8 fields
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    act_and_mul_scale: Optional[torch.Tensor] = None

    # Block-wise FP8 fields
    w13_weight_scale: Optional[torch.Tensor] = None
    w2_weight_scale: Optional[torch.Tensor] = None
    x_scale: Optional[torch.Tensor] = None

    # EP metadata
    rank_ep: int = 0
    num_expert_total: int = -1

    # Optional shared-expert output to add in-place
    shared_output: Optional[torch.Tensor] = None


@register_fused_func("none", "hpc")
def fused_experts_none_to_hpc(
    dispatch_output: StandardDispatchOutput,
    quant_info: HpcMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Fused MoE entry point for HPC-ops backend.

    Dispatches to ``hpc.fuse_moe`` (per-tensor) or ``hpc.fuse_moe_blockwise``
    (block-wise) depending on ``quant_info.is_block_quantized``.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.utils.hpc import hpc_fuse_moe, hpc_fuse_moe_blockwise

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_ids = topk_output.topk_ids
    topk_scale = topk_output.topk_weights

    if quant_info.is_block_quantized:
        # Quantize x to FP8 with per-block scaling (dynamic)
        num_seq, hidden_size = hidden_states.shape
        block_k = hidden_size // quant_info.w13_weight_scale.shape[-1]
        x_reshaped = hidden_states.view(num_seq, -1, block_k)
        x_max = x_reshaped.abs().amax(dim=-1).clamp(min=1e-12)
        x_scale = (x_max / _FP8_E4M3_MAX).to(torch.float32)
        x_fp8 = (
            (x_reshaped / x_scale.unsqueeze(-1))
            .to(torch.float8_e4m3fn)
            .view_as(hidden_states)
        )

        output = hpc_fuse_moe_blockwise(
            x=x_fp8,
            x_scale=x_scale,
            gate_up_weight=quant_info.w13_weight,
            gate_up_weight_scale=quant_info.w13_weight_scale,
            down_weight=quant_info.w2_weight,
            down_weight_scale=quant_info.w2_weight_scale,
            topk_ids=topk_ids,
            topk_scale=topk_scale,
            rank_ep=quant_info.rank_ep,
            num_expert_total=quant_info.num_expert_total,
            shared_output=quant_info.shared_output,
        )
    else:
        # Per-tensor: quantize x to FP8 dynamically
        # quant_info.act_and_mul_scale is the model's w13_input_scale (a13_scale)
        # used for GEMM1 input quantization. If None, compute dynamically.
        if quant_info.act_and_mul_scale is not None:
            x_scale = quant_info.act_and_mul_scale
        else:
            x_scale = (
                (hidden_states.abs().max() / _FP8_E4M3_MAX)
                .clamp(min=1e-12)
                .to(torch.float32)
            )
        x_fp8 = (hidden_states / x_scale).to(torch.float8_e4m3fn)

        # GEMM1 dequantization: output = x_fp8 @ w13_fp8 * (x_scale * w13_scale)
        combined_gate_up_scale = (quant_info.w13_scale * x_scale).to(torch.float32)

        # HPC kernel's act_and_mul_scale is MULTIPLIED with the activation
        # (SiLU(gate) * up) BEFORE converting to FP8 (see hpc-ops naive reference).
        # So it must be the INVERSE of the quantization scale: 1/x_scale.
        hpc_act_and_mul_scale = (1.0 / x_scale).to(torch.float32)

        # GEMM2 dequantization must compensate for the pre-quantization:
        #   down_output = (act * hpc_act_and_mul_scale)_fp8 @ w2_fp8 * down_scale
        # For correct dequant: down_scale = w2_scale / hpc_act_and_mul_scale
        #                                 = w2_scale * x_scale
        hpc_down_scale = (quant_info.w2_scale * x_scale).to(torch.float32)

        output = hpc_fuse_moe(
            x=x_fp8,
            gate_up_weight=quant_info.w13_weight,
            down_weight=quant_info.w2_weight,
            gate_up_scale=combined_gate_up_scale,
            down_scale=hpc_down_scale,
            act_and_mul_scale=hpc_act_and_mul_scale,
            topk_ids=topk_ids,
            topk_scale=topk_scale,
            rank_ep=quant_info.rank_ep,
            num_expert_total=quant_info.num_expert_total,
            use_bf16_mul=True,
            shared_output=quant_info.shared_output,
        )

    return StandardCombineInput(
        hidden_states=output,
    )
