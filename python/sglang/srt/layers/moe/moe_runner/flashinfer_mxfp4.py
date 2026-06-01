"""FlashInfer SM90 cutlass mixed-input W4A16 MXFP4 MoE fused func.

Registered for ``("none", "flashinfer_mxfp4")``. Drives FlashInfer's
``cutlass_fused_moe(use_w4_group_scaling=True)`` (PR #3084 in flashinfer,
SM90 only). Quant methods build the quant_info each forward and call
``MoeRunner.run(dispatch_output, quant_info)``.

Two production call sites share this fused func:
  - GPT-OSS via :class:`Mxfp4MoEMethod` (input pad/output trim + per-expert
    SwiGLU scalars + per-expert bias)
  - DSv4 via :class:`Mxfp4FlashinferCutlassMoEMethod` (no bias, optional
    SwiGLU scalars, no padding)

The SM100 trtllm-gen path also lives under ``MoeRunnerBackend.FLASHINFER_MXFP4``
but is intentionally left in the legacy bypass path for now; migrating it is a
follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput


@dataclass
class FlashInferMxfp4CutlassMoeQuantInfo(MoeQuantInfo):
    """Quantization payload for the SM90 cutlass W4A16 MXFP4 MoE path.

    Weights and scales are pre-interleaved at load time via
    ``interleave_moe_{weights,scales}_for_sm90_mixed_gemm``; this dataclass
    only carries references plus the per-call routing/topology fields.
    """

    # Pre-interleaved weights (uint8, packed FP4)
    w13_weight: torch.Tensor  # [E, 2*N, K/2]
    w2_weight: torch.Tensor  # [E, K, N/2]

    # Pre-interleaved E8M0 block scales (uint8; viewed as int32 at call time)
    w13_weight_scale: torch.Tensor  # [E, 2*N, K/32]
    w2_weight_scale: torch.Tensor  # [E, K, N/32]

    # Per-expert bias. GPT-OSS has both; DSv4 leaves both None.
    w13_bias: Optional[torch.Tensor] = None  # bf16 [E, 2*N]
    w2_bias: Optional[torch.Tensor] = None  # bf16 [E, K]

    # Per-expert SwiGLU scalars (fp32 [E]). Either all three are present
    # (clamped SwiGLU) or all three are None (kernel default SwiGLU).
    swiglu_alpha: Optional[torch.Tensor] = None
    swiglu_beta: Optional[torch.Tensor] = None
    swiglu_limit: Optional[torch.Tensor] = None

    # TP/EP topology (forwarded to the FlashInfer kernel)
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    moe_ep_size: int = 1
    moe_ep_rank: int = 0

    # GPT-OSS pads its input hidden dim up to the (pre-padded) loaded weight
    # width and trims the output back. DSv4 leaves this as ``None`` (no pad).
    padded_hidden: Optional[int] = None


def _flashinfer_cutlass_fused_moe():
    """Lazy import — keeps non-flashinfer wheels importable."""
    if not is_flashinfer_available():
        raise RuntimeError(
            "flashinfer_mxfp4 runner backend requires flashinfer to be installed."
        )
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    return cutlass_fused_moe, ActivationType


@register_fused_func("none", "flashinfer_mxfp4")
def fused_experts_none_to_flashinfer_mxfp4(
    dispatch_output: "StandardDispatchOutput",
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> "StandardCombineInput":
    """SM90 W4A16 MXFP4 fused expert forward pass.

    Mirrors the legacy ``Mxfp4MoEMethod._apply_sm90_cutlass`` and DSv4's
    ``Mxfp4FlashinferCutlassMoEMethod.apply`` exactly; difference vs those is
    that all per-layer state arrives via ``quant_info`` rather than via the
    layer module, so this function is layer-agnostic.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert isinstance(
        quant_info, FlashInferMxfp4CutlassMoeQuantInfo
    ), f"Unexpected quant_info type for flashinfer_mxfp4: {type(quant_info)}"

    flashinfer_cutlass_fused_moe, ActivationType = _flashinfer_cutlass_fused_moe()

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    # Under ``--moe-runner-backend flashinfer_mxfp4`` topk may be in bypassed
    # form (the SM100 trtllm-gen path does routing internally). The cutlass
    # SM90 path needs explicit topk_ids / topk_weights; materialize here.
    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard()
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    # GPT-OSS: pad input hidden dim up to the loaded weight width. DSv4
    # leaves padded_hidden as None (or equal to origin_hidden), no pad.
    origin_hidden = x.shape[-1]
    padded_hidden = quant_info.padded_hidden
    do_pad = padded_hidden is not None and padded_hidden != origin_hidden
    if do_pad:
        x = torch.nn.functional.pad(
            x,
            (0, padded_hidden - origin_hidden),
            mode="constant",
            value=0.0,
        )

    out_hidden = padded_hidden if do_pad else origin_hidden
    output_dtype = torch.bfloat16
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        out = torch.empty(x.shape[0], out_hidden, dtype=output_dtype, device=x.device)

    flashinfer_cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=quant_info.w13_weight,
        fc2_expert_weights=quant_info.w2_weight,
        output_dtype=output_dtype,
        quant_scales=[
            quant_info.w13_weight_scale.view(torch.int32),
            quant_info.w2_weight_scale.view(torch.int32),
        ],
        fc1_expert_biases=quant_info.w13_bias,
        fc2_expert_biases=quant_info.w2_bias,
        swiglu_alpha=quant_info.swiglu_alpha,
        swiglu_beta=quant_info.swiglu_beta,
        swiglu_limit=quant_info.swiglu_limit,
        tp_size=quant_info.moe_tp_size,
        tp_rank=quant_info.moe_tp_rank,
        ep_size=quant_info.moe_ep_size,
        ep_rank=quant_info.moe_ep_rank,
        use_w4_group_scaling=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=next_power_of_2(x.shape[0]),
        output=out,
    )

    if do_pad:
        out = out[:, :origin_hidden].contiguous()

    return StandardCombineInput(hidden_states=out)
