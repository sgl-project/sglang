"""SM120-specific DeepGEMM MoE path.

Consumer Blackwell's DeepGEMM contiguous grouped GEMM takes standard-layout
activations, so SM120 uses a contiguous scatter/gather permute instead of the
masked layout the other architectures use. Kept out of ``deep_gemm.py`` so the
shared runner keeps a single code path.

Entry points, all no-ops off SM120:
  * ``use_swizzle`` — layout choice for the shared runner
  * ``maybe_pre_permute`` — contiguous scatter, or ``None`` to fall through
  * ``maybe_post_permute`` — matching gather, or ``None`` to fall through
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.utils import ceil_div, dispose_tensor, is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmRunnerInput
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

_is_sm120 = is_sm120_supported()

# Above this the standard path uses the contiguous grouped GEMM; the masked
# layout's [num_local_experts, capacity, k] transients OOM at prefill sizes.
_STANDARD_CONTIG_MIN_TOKENS = 1024


def is_supported() -> bool:
    """True when this build can serve the SM120 standard-layout MoE path.

    Off SM120 the shared masked path is correct, so this is vacuously true;
    on SM120 it reports whether the contiguous implementation is present.
    """
    return True


def use_swizzle() -> bool:
    """SM120's contiguous GEMM consumes standard-layout activations only."""
    return not _is_sm120


def allows_masked_standard_layout() -> bool:
    """SM120 cannot serve the masked-standard layout for DSV4 shapes: its varlen
    activation kernel requires ``D // 8 >= num_experts`` (512 // 8 = 64 < 256),
    so keep upstream's memory-budget heuristic off this path.
    """
    return not _is_sm120


def _eligible(hidden_states, quant_info, runner_config) -> bool:
    return (
        _is_sm120
        and hidden_states.shape[0] >= _STANDARD_CONTIG_MIN_TOKENS
        and quant_info.w13_weight.dtype != torch.bfloat16
        # Under EP the standard dispatcher maps non-local experts to -1,
        # which the contiguous path does not handle; keep the masked path.
        and runner_config.num_local_experts == runner_config.num_experts
    )


def maybe_pre_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_info,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> Optional[DeepGemmRunnerInput]:
    """Contiguous scatter for SM120; ``None`` means use the shared masked path."""
    if not _eligible(hidden_states, quant_info, runner_config):
        return None
    return _pre_permute_standard_contig(
        hidden_states, topk_ids, topk_weights, runner_config, running_state
    )


def maybe_post_permute(
    runner_output,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> Optional[StandardCombineInput]:
    """Gather matching ``maybe_pre_permute``; ``None`` means fall through."""
    if not running_state.get("contig_mode"):
        return None

    from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    gather_out = torch.empty(
        running_state["hidden_states_shape"],
        device=running_state["hidden_states_device"],
        dtype=running_state["hidden_states_dtype"],
    )
    ep_gather(
        runner_output.hidden_states,
        running_state["topk_ids"],
        running_state["topk_weights"],
        running_state["output_index"],
        gather_out,
    )
    dispose_tensor(runner_output.hidden_states)
    if runner_config.routed_scaling_factor is not None:
        gather_out *= runner_config.routed_scaling_factor
    return StandardCombineInput(hidden_states=gather_out)


def _pre_permute_standard_contig(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepGemmRunnerInput:
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmRunnerInput

    num_tokens, K = hidden_states.shape
    num_experts = runner_config.num_local_experts
    device = hidden_states.device

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    running_state["hidden_states_shape"] = hidden_states.shape
    running_state["hidden_states_device"] = device
    running_state["hidden_states_dtype"] = hidden_states.dtype
    running_state["contig_mode"] = True

    ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
    q, q_scale = sglang_per_token_group_quant_fp8(
        hidden_states,
        128,
        column_major_scales=ue8m0,
        scale_tma_aligned=ue8m0,
        scale_ue8m0=ue8m0,
    )
    dispose_tensor(hidden_states)

    # ep_scatter fills expert slots in blocks of 128; size from a static upper
    # bound and accumulate counts on-device to avoid a GPU->CPU sync.
    flat_ids = topk_ids.flatten().to(torch.int64)
    counts = torch.zeros(num_experts, device=device, dtype=torch.int32)
    counts.index_add_(0, flat_ids.clamp_min(0), (flat_ids >= 0).to(torch.int32))
    counts_aligned = (counts + 127) // 128 * 128
    all_tokens = ceil_div(num_tokens * runner_config.top_k, 128) * 128 + (
        num_experts * 128
    )
    running_state["all_tokens"] = all_tokens

    # Pad slots (m_indices == -1) are skipped by the grouped GEMM, so the
    # buffer needs no zero fill.
    input_tensor = torch.empty((all_tokens, K), device=device, dtype=q.dtype)
    if ue8m0:
        input_tensor_scale = torch.zeros(
            (ceil_div(K // 128, 4), all_tokens), device=device, dtype=torch.int
        ).transpose(0, 1)
    else:
        input_tensor_scale = torch.empty(
            (all_tokens, K // 128), device=device, dtype=torch.float32
        )
    m_indices = torch.full((all_tokens,), -1, device=device, dtype=torch.int32)
    output_index = torch.empty_like(topk_ids)
    expert_start_loc = torch.empty_like(counts_aligned)

    ep_scatter(
        q,
        q_scale,
        topk_ids,
        counts_aligned,
        counts,
        expert_start_loc,
        input_tensor,
        input_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=ue8m0,
    )
    dispose_tensor(q)
    if q_scale is not None:
        dispose_tensor(q_scale)

    running_state["output_index"] = output_index

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices,
    )
