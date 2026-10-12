"""Marlin MoE runner core with hook support for LoRA injection.

Uses Marlin int4/int8 kernels for the base MoE projections.
LoRA deltas are injected via hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import DispatchMoeRunnerCore, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        moe_sum_reduce_triton,
    )
    from sglang.kernels.ops.moe.moe_wna16_marlin import (
        moe_wna16_marlin_gemm,
        mxfp4_moe_decode_gemm,
    )
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        gated_marlin_activation,
        get_scalar_type,
        is_mxfp4_marlin_weights,
        marlin_moe_use_atomic_add,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        moe_align_block_size,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace


# Decode GEMMs (moe_block_size 8) of MXFP4 weights skip Marlin's issue-bound
# dequant, and the intermediate padding of layers whose padding is checked zero.
_MXFP4_DECODE_GEMM = envs.SGLANG_OPT_MXFP4_MOE_DECODE_GEMM.get()
# (warps_n, warps_k, stages) of the gate_up and the down GEMM, tuned on GB300 at
# hidden 3584 and intermediate 192 per rank
_MXFP4_DECODE_GATE_UP_CFG = (1, 8, 4)
_MXFP4_DECODE_DOWN_CFG = (4, 1, 4)
_zero_padding: dict[int, bool] = {}


def _mxfp4_decode_cfg_fits(
    cfg: tuple[int, int, int], size_n: int, k_valid: int
) -> bool:
    warps_n, warps_k, stages = cfg
    return size_n % (64 * warps_n) == 0 and k_valid % (16 * warps_k * stages) == 0


def _use_mxfp4_decode_path(
    block_size_m: int,
    is_mxfp4: bool,
    hidden_states: torch.Tensor,
    quant_info,
    hidden: int,
    inter: int,
) -> bool:
    return (
        _MXFP4_DECODE_GEMM
        and block_size_m == 8
        and is_mxfp4
        and hidden_states.dtype == torch.bfloat16
        and quant_info.w13_g_idx is None
        and quant_info.w2_g_idx is None
        and quant_info.is_k_full
        and _mxfp4_decode_cfg_fits(_MXFP4_DECODE_GATE_UP_CFG, 2 * inter, hidden)
        and _mxfp4_decode_cfg_fits(_MXFP4_DECODE_DOWN_CFG, hidden, inter)
    )


def _marlin_padding_is_zero(
    quant_info, hidden: int, inter_real: int, inter: int
) -> bool:
    # Marlin keeps each 64-column tile of w13 (int32 columns 128 t .. 128 t + 127
    # of a k16 row), its scales' 64-column groups and the k16 rows of w2
    # together, so the padded columns inter_real .. inter - 1 of each half of
    # w13 are whole tiles and the padded K rows of w2 whole k16 rows / scale
    # groups.
    num_experts = quant_info.w13_qweight.shape[0]
    w13 = quant_info.w13_qweight.view(torch.int32).reshape(
        num_experts, hidden // 16, -1
    )
    w2 = quant_info.w2_qweight.view(torch.int32).reshape(num_experts, inter // 16, -1)
    s13 = quant_info.w13_scales.view(torch.uint8)
    s2 = quant_info.w2_scales.view(torch.uint8)
    pads, pad_scales = [w2[:, inter_real // 16 :]], [s2[:, inter_real // 32 :]]
    for half in range(2):
        t0, t1 = (half * inter + inter_real) // 64, (half + 1) * inter // 64
        pads.append(w13[:, :, 128 * t0 : 128 * t1])
        pad_scales.append(s13[:, :, 64 * t0 : 64 * t1])
    # zero FP4 codes give zero products unless the E8M0 scale is NaN (255)
    return not any(bool(p.any()) for p in pads) and not any(
        bool((s == 255).any()) for s in pad_scales
    )


def _mxfp4_decode_inter_valid(quant_info, hidden: int, inter: int, inter_real) -> int:
    """Intermediate size the decode GEMMs read: inter_real once the layer's
    Marlin padding is known to be zero, otherwise the padded size."""
    if (
        inter_real is None
        or not 0 < inter_real < inter
        or inter_real % 64
        or not _mxfp4_decode_cfg_fits(_MXFP4_DECODE_DOWN_CFG, hidden, inter_real)
    ):
        return inter
    key = quant_info.w13_qweight.data_ptr()
    zero = _zero_padding.get(key)
    if zero is None:
        if torch.cuda.is_current_stream_capturing():
            return inter  # checked by the eager warmup forward before capture
        zero = _marlin_padding_is_zero(quant_info, hidden, inter_real, inter)
        _zero_padding[key] = zero
    return inter_real if zero else inter


class MarlinLoraRunnerCore(DispatchMoeRunnerCore):
    """
    MoE runner using Marlin kernels for base projections, with hooks for LoRA.

    Pipeline:
      1. moe_wna16_marlin_gemm (gate_up)
      1.5. hooks.after_gate_up
      2. silu_and_mul
      3. moe_wna16_marlin_gemm (down)
      3.5. hooks.after_down
      4. moe_sum_reduce
    """

    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN

    def run_from_dispatch(
        self,
        dispatch_output: StandardDispatchOutput,
        quant_info: MarlinMoeQuantInfo,
        runner_config: MoeRunnerConfig,
        hooks=None,
    ) -> StandardCombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        assert hooks is not None, "hooks must be provided for MarlinLoraRunnerCore"

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        assert runner_config.is_gated, (
            "MarlinLoraRunnerCore only supports gated experts"
        )
        assert torch.cuda.get_device_capability(hidden_states.device)[0] >= 9, (
            "MarlinLoraRunnerCore requires CUDA compute capability >= 9"
        )
        routed_scaling_factor = runner_config.routed_scaling_factor

        M, K = hidden_states.shape
        E = quant_info.w13_qweight.shape[0]
        N = quant_info.w2_qweight.shape[1] * 16
        topk = topk_ids.shape[1]
        num_bits = quant_info.weight_bits
        is_mxfp4 = is_mxfp4_marlin_weights(
            num_bits,
            quant_info.w13_scales,
            quant_info.w2_scales,
            quant_info.w13_qzeros,
            quant_info.w2_qzeros,
        )
        use_atomic_add = marlin_moe_use_atomic_add(hidden_states, is_mxfp4)

        for block_size_m in [8, 16, 32, 48, 64]:
            if M * topk / E / block_size_m < 0.9:
                break

        # Under EP the dispatcher already localized topk_ids (-1 = non-local); align
        # over the global expert count like fused_marlin_moe, not the local E.
        align_num_experts = (
            quant_info.global_num_experts if quant_info.expert_map is not None else E
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, align_num_experts
        )

        use_mxfp4_decode = _use_mxfp4_decode_path(
            block_size_m, is_mxfp4, hidden_states, quant_info, K, N
        )
        is_ep = quant_info.expert_map is not None
        if use_mxfp4_decode:
            inter_valid = _mxfp4_decode_inter_valid(
                quant_info, K, N, runner_config.intermediate_size_per_partition
            )

        # Per-call workspace like fused_experts_none_to_marlin: a shared buffer aliases
        # inter-block locks across in-flight kernels/graphs and deadlocks capture.
        # The decode GEMM never shares an output between threadblocks, so it
        # takes no locks.
        workspace = (
            None
            if use_mxfp4_decode
            else marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)
        )

        # Pass scales + global scale so fp4-marlin weights (the ModelOpt NVFP4
        # W4A16 fallback) resolve to float4_e2m1f instead of uint4b8.
        scalar_type1 = get_scalar_type(
            num_bits,
            quant_info.w13_qzeros is not None,
            quant_info.w13_scales,
            quant_info.w13_global_scale,
        )
        scalar_type2 = get_scalar_type(
            num_bits,
            quant_info.w2_qzeros is not None,
            quant_info.w2_scales,
            quant_info.w2_global_scale,
        )

        # Stage 1: Gate/Up (Marlin)
        intermediate_cache1 = torch.empty(
            (M * topk, 2 * N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        if use_mxfp4_decode:
            warps_n, warps_k, stages = _MXFP4_DECODE_GATE_UP_CFG
            intermediate_cache1 = mxfp4_moe_decode_gemm(
                hidden_states,
                intermediate_cache1,
                quant_info.w13_qweight,
                quant_info.w13_bias,
                quant_info.w13_scales,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights,
                top_k=topk,
                mul_topk_weights=False,
                is_ep=is_ep,
                size_m=M,
                size_n=2 * N,
                size_k=K,
                warps_n=warps_n,
                warps_k=warps_k,
                stages=stages,
                n_half_valid=inter_valid,
            )
        else:
            intermediate_cache1 = moe_wna16_marlin_gemm(
                hidden_states,
                intermediate_cache1,
                quant_info.w13_qweight,
                quant_info.w13_bias,
                quant_info.w13_scales,
                quant_info.w13_global_scale,
                quant_info.w13_qzeros,
                quant_info.w13_g_idx,
                quant_info.w13_g_idx_sort_indices,
                workspace,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights,
                moe_block_size=block_size_m,
                top_k=topk,
                mul_topk_weights=False,
                is_ep=quant_info.expert_map is not None,
                b_q_type=scalar_type1,
                size_m=M,
                size_n=2 * N,
                size_k=K,
                is_k_full=quant_info.is_k_full,
                use_atomic_add=use_atomic_add,
                use_fp32_reduce=True,
                is_zp_float=False,
            )

        # Hook: after gate_up
        if hooks.after_gate_up:
            intermediate_cache1_3d = intermediate_cache1.view(M, topk, 2 * N)
            hooks.after_gate_up(
                hidden_states, intermediate_cache1_3d, topk_weights, topk_ids
            )

        # Stage 2: Activation
        intermediate_cache2 = torch.empty(
            (M * topk, N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        gated_marlin_activation(
            runner_config.activation,
            runner_config.gemm1_alpha,
            runner_config.gemm1_clamp_limit,
            intermediate_cache1.view(-1, 2 * N),
            intermediate_cache2,
        )

        # Stage 3: Down (Marlin)
        intermediate_cache3 = torch.empty(
            (M * topk, K), device=hidden_states.device, dtype=hidden_states.dtype
        )
        if quant_info.expert_map is not None:
            intermediate_cache3.zero_()

        if use_mxfp4_decode:
            warps_n, warps_k, stages = _MXFP4_DECODE_DOWN_CFG
            intermediate_cache3 = mxfp4_moe_decode_gemm(
                intermediate_cache2,
                intermediate_cache3,
                quant_info.w2_qweight,
                quant_info.w2_bias,
                quant_info.w2_scales,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights,
                top_k=1,
                mul_topk_weights=True,
                is_ep=is_ep,
                size_m=M * topk,
                size_n=K,
                size_k=N,
                warps_n=warps_n,
                warps_k=warps_k,
                stages=stages,
                k_valid=inter_valid,
            )
        else:
            intermediate_cache3 = moe_wna16_marlin_gemm(
                intermediate_cache2,
                intermediate_cache3,
                quant_info.w2_qweight,
                quant_info.w2_bias,
                quant_info.w2_scales,
                quant_info.w2_global_scale,
                quant_info.w2_qzeros,
                quant_info.w2_g_idx,
                quant_info.w2_g_idx_sort_indices,
                workspace,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights,
                moe_block_size=block_size_m,
                top_k=1,
                mul_topk_weights=True,
                is_ep=quant_info.expert_map is not None,
                b_q_type=scalar_type2,
                size_m=M * topk,
                size_n=K,
                size_k=N,
                is_k_full=quant_info.is_k_full,
                use_atomic_add=use_atomic_add,
                use_fp32_reduce=True,
                is_zp_float=False,
            )
        intermediate_cache3 = intermediate_cache3.view(M, topk, K)

        # Hook: after down
        if hooks.after_down:
            hooks.after_down(
                intermediate_cache2, intermediate_cache3, topk_weights, topk_ids
            )

        # Stage 4: Reduction. Never alias hidden_states even under inplace: the sink
        # forward still reads it (stock fused_experts_none_to_marlin does the same).
        output = torch.empty_like(hidden_states)
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0
        # NOTE: fusion opportunity here
        moe_sum_reduce_triton(intermediate_cache3, output, routed_scaling_factor)

        return StandardCombineInput(hidden_states=output)
