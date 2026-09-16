from __future__ import annotations

"""
MoE runner backend for FlashInfer's CuTe DSL contiguous grouped FP8 GEMM
(``flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous``, SM100 / SM103).

It serves 128x128 block-quantized FP8 experts (Qwen3-*-FP8, DeepSeek-V3-style
checkpoints) on datacenter Blackwell without DeepGEMM. Per MoE layer:

    scatter tokens into the contiguous expert-major layout
    -> gate/up grouped GEMM -> fused SiLU-and-mul + per-token-group quant
    -> down grouped GEMM -> weighted gather back into token order.

The kernel takes plain fp32 scales (per-row 1x128 for activations, 128x128
for weights), so the checkpoint's ``weight_scale_inv`` is consumed as loaded;
unlike the ``deep_gemm`` runner on SM100 no UE8M0 requantization happens.

Layout contract of the kernel: ``m_indices`` sorted, every internal expert
boundary 128-row aligned, no ``-1`` padding rows. The scatter pads every
expert to a multiple of 128 rows and stamps the pad rows with their expert's
index; the gather never reads them. The total row count is the graph-static
bound the ``deep_gemm`` compact layout uses, so the path is CUDA-graph
capturable.

Every active expert therefore costs at least one 128-row tile and the slack
tiles are computed too, so the path favors prefill and large batches; at
decode batch sizes the Triton fused MoE does less work per step.

Standard dispatch only (``--moe-a2a-backend none``); TP and EP work through
the dispatcher's local expert ids. Enable with
``--moe-runner-backend flashinfer_cutedsl_fp8``.
"""

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo, register_fused_func
from sglang.srt.runtime_context import get_exec, get_platform
from sglang.srt.utils import ceil_div, dispose_tensor

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

# Kernel constraint: 128-row M tiles, 1x128 activation and 128x128 weight scales.
BLOCK_SIZE = 128


@functools.cache
def has_flashinfer_cutedsl_fp8_group_gemm() -> bool:
    try:
        from flashinfer.gemm import (  # noqa: F401
            group_gemm_fp8_nt_groupwise_contiguous,
        )
    except ImportError:
        return False
    return True


def check_flashinfer_cutedsl_fp8_supported() -> None:
    """Fail at startup when this machine cannot run the backend."""
    if not get_platform().is_sm100:
        raise ValueError(
            "--moe-runner-backend flashinfer_cutedsl_fp8 requires an SM100/SM103 "
            "(datacenter Blackwell) GPU; the FlashInfer CuTe DSL contiguous "
            "grouped FP8 GEMM is not built for other architectures."
        )
    if not has_flashinfer_cutedsl_fp8_group_gemm():
        raise ValueError(
            "--moe-runner-backend flashinfer_cutedsl_fp8 requires a FlashInfer "
            "build that exports flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous "
            "(FlashInfer > 0.6.18) together with nvidia-cutlass-dsl."
        )


# MoeQuantInfo is a non-frozen ABC dataclass hierarchy; msgspec.Struct cannot subclass it.
@dataclass
class FlashInferCuteDslFp8MoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor  # [E, 2I, H] float8_e4m3fn
    w2_weight: torch.Tensor  # [E, H, I] float8_e4m3fn
    w13_weight_scale_inv: torch.Tensor  # [E, 2I / 128, H / 128] float32
    w2_weight_scale_inv: torch.Tensor  # [E, H / 128, I / 128] float32
    block_shape: List[int]


def _compact_all_tokens(num_assignments: int, num_experts: int) -> int:
    # Padding each expert to 128 rows adds at most 127 rows per non-empty
    # expert; the maximum over all routings keeps the buffer shape graph-static.
    max_nonempty_experts = min(num_assignments, num_experts)
    return BLOCK_SIZE * (
        max_nonempty_experts + (num_assignments - max_nonempty_experts) // BLOCK_SIZE
    )


def _check_supported(
    quant_info: MoeQuantInfo, runner_config: MoeRunnerConfig
) -> FlashInferCuteDslFp8MoeQuantInfo:
    if not isinstance(quant_info, FlashInferCuteDslFp8MoeQuantInfo):
        raise ValueError(
            "The flashinfer_cutedsl_fp8 MoE runner backend only supports "
            "128x128 block-quantized FP8 MoE models (Fp8MoEMethod); got quant "
            f"info {type(quant_info).__name__}."
        )
    if quant_info.block_shape != [BLOCK_SIZE, BLOCK_SIZE]:
        raise ValueError(
            "The flashinfer_cutedsl_fp8 MoE runner backend requires 128x128 "
            f"weight block quantization, got {quant_info.block_shape}."
        )
    for name, weight, scale in (
        ("w13", quant_info.w13_weight, quant_info.w13_weight_scale_inv),
        ("w2", quant_info.w2_weight, quant_info.w2_weight_scale_inv),
    ):
        if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
            raise ValueError(
                f"flashinfer_cutedsl_fp8 expects float8_e4m3fn {name}_weight with "
                f"float32 block scales, got {weight.dtype} / {scale.dtype}."
            )
        _, n, k = weight.shape
        if n % BLOCK_SIZE or k % BLOCK_SIZE:
            raise ValueError(
                "flashinfer_cutedsl_fp8 requires every expert GEMM dimension to "
                f"be a multiple of 128, got {name}_weight shape {tuple(weight.shape)} "
                "(check hidden_size and the per-partition intermediate size)."
            )
    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise ValueError(
            "The flashinfer_cutedsl_fp8 MoE runner backend only supports the gated "
            f"silu activation, got activation={runner_config.activation}, "
            f"is_gated={runner_config.is_gated}."
        )
    if (
        runner_config.gemm1_alpha is not None
        or runner_config.gemm1_clamp_limit is not None
        or runner_config.swiglu_limit is not None
    ):
        raise ValueError(
            "The flashinfer_cutedsl_fp8 MoE runner backend runs a plain "
            "SiLU-and-mul; it does not support gemm1_alpha / gemm1_clamp_limit / "
            "swiglu_limit."
        )
    if runner_config.apply_router_weight_on_input or runner_config.no_combine:
        raise ValueError(
            "The flashinfer_cutedsl_fp8 MoE runner backend does not support "
            "apply_router_weight_on_input or no_combine."
        )
    return quant_info


def _scatter_to_contiguous_layout(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expert-major rows padded to 128 per expert; returns (a, a_scale, m_indices, src2dst)."""
    from sglang.kernels.ops.moe.ep_moe_kernels import (
        ep_scatter,
        fused_moe_dispatch_index,
    )
    from sglang.kernels.ops.moe.triton_pad_expert_counts import pad_expert_counts

    device = x_q.device
    hidden_size = x_q.shape[1]
    all_tokens = _compact_all_tokens(topk_ids.numel(), num_experts)
    tokens_per_expert, unused_masked_dst = fused_moe_dispatch_index(
        topk_ids, num_local_experts=num_experts, m_max=1
    )
    dispose_tensor(unused_masked_dst)
    # The trailing expert absorbs the slack so the total stays graph-static.
    padded_tokens_per_expert = pad_expert_counts(
        tokens_per_expert, block_e=BLOCK_SIZE, all_tokens=all_tokens
    )

    # Pad rows are never gathered; zero them only for batch-invariant output.
    buffer_init = (
        torch.zeros
        if get_exec().deterministic.enable_deterministic_inference
        else torch.empty
    )
    a = buffer_init((all_tokens, hidden_size), dtype=x_q.dtype, device=device)
    a_scale = buffer_init(
        (all_tokens, ceil_div(hidden_size, BLOCK_SIZE)),
        dtype=torch.float32,
        device=device,
    )
    m_indices = torch.empty(all_tokens, dtype=torch.int32, device=device)
    src2dst = torch.empty_like(topk_ids, dtype=torch.int32)
    expert_start_loc = torch.empty(num_experts, dtype=torch.int32, device=device)
    # Passing the padded counts as the valid counts stamps pad rows with their
    # expert index; the kernel rejects the -1 rows ep_scatter would write.
    ep_scatter(
        recv_x=x_q,
        recv_x_scale=x_s,
        recv_topk=topk_ids,
        num_recv_tokens_per_expert=padded_tokens_per_expert,
        num_valid_tokens_per_expert=padded_tokens_per_expert,
        expert_start_loc=expert_start_loc,
        output_tensor=a,
        output_tensor_scale=a_scale,
        m_indices=m_indices,
        output_index=src2dst,
        scale_ue8m0=False,
        quant_block_size=BLOCK_SIZE,
        expert_alignment=BLOCK_SIZE,
    )
    return a, a_scale, m_indices, src2dst


def _run_grouped_mlp(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    m_indices: torch.Tensor,
    quant_info: FlashInferCuteDslFp8MoeQuantInfo,
) -> torch.Tensor:
    """gate/up grouped GEMM -> SiLU-and-mul + quant -> down grouped GEMM."""
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise_contiguous

    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    # (M, H) x (E, 2I, H)^T -> (M, 2I)
    gateup_output = group_gemm_fp8_nt_groupwise_contiguous(
        a=a,
        b=quant_info.w13_weight,
        a_scale=a_scale,
        b_scale=quant_info.w13_weight_scale_inv,
        m_indices=m_indices,
    )
    dispose_tensor(a)
    dispose_tensor(a_scale)

    # (M, 2I) -> (M, I) fp8 with (M, I / 128) fp32 scales
    down_input, down_input_scale = sglang_per_token_group_quant_fp8(
        gateup_output, group_size=BLOCK_SIZE, fuse_silu_and_mul=True
    )
    dispose_tensor(gateup_output)

    # (M, I) x (E, H, I)^T -> (M, H)
    down_output = group_gemm_fp8_nt_groupwise_contiguous(
        a=down_input,
        b=quant_info.w2_weight,
        a_scale=down_input_scale,
        b_scale=quant_info.w2_weight_scale_inv,
        m_indices=m_indices,
    )
    dispose_tensor(down_input)
    dispose_tensor(down_input_scale)
    return down_output


@register_fused_func("none", "flashinfer_cutedsl_fp8")
def fused_experts_none_to_flashinfer_cutedsl_fp8(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    quant_info = _check_supported(quant_info, runner_config)

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    assert hidden_states.dtype == torch.bfloat16, (
        "The flashinfer_cutedsl_fp8 MoE runner backend only supports bf16 hidden "
        f"states, got {hidden_states.dtype}."
    )
    num_tokens, hidden_size = hidden_states.shape
    num_experts = quant_info.w13_weight.shape[0]
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)

    # Only the combined output enters the NCCL symmetric pool; intermediates
    # stay on the default allocator to bound pool occupancy.
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        output = torch.empty_like(hidden_states)
    if num_tokens == 0:
        return StandardCombineInput(hidden_states=output)

    x_q, x_s = sglang_per_token_group_quant_fp8(hidden_states, group_size=BLOCK_SIZE)
    a, a_scale, m_indices, src2dst = _scatter_to_contiguous_layout(
        x_q=x_q, x_s=x_s, topk_ids=topk_ids, num_experts=num_experts
    )
    dispose_tensor(x_q)
    dispose_tensor(x_s)

    down_output = _run_grouped_mlp(
        a=a, a_scale=a_scale, m_indices=m_indices, quant_info=quant_info
    )

    # Rows routed to non-local experts (topk id -1 under EP) are skipped.
    post_reorder_deepgemm(
        down_output=down_output,
        output=output,
        src2dst=src2dst,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk=topk_ids.shape[1],
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
    )
    dispose_tensor(down_output)
    return StandardCombineInput(hidden_states=output)
