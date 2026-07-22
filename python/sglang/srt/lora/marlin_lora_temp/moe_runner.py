"""Experimental Marlin MoE-LoRA execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.lora.marlin_lora_temp.policy import (
    use_post_reduce_down_delta,
)
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import silu_and_mul

    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        _align_block_size_jit as moe_align_block_size,
    )
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        get_scalar_type,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        moe_sum_reduce_triton,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.lora.marlin_lora_temp.activation import silu_and_mul_add_delta
    from sglang.srt.lora.marlin_lora_temp.direct_decode import (
        direct_decode_down_shrink,
        direct_decode_gate_expand,
    )
    from sglang.srt.lora.marlin_lora_temp.shared_outer import (
        fused_base_mapped_shared_lora_reduce,
        fused_base_shared_lora_reduce,
        fused_base_shared_lora_reduce_config,
        weighted_topk_rank_sum,
    )

# Keep side-stream events alive through CUDA graph capture.
_MARLIN_LORA_OVERLAP_EVENTS: list = []


def _use_shared_outer_factorization(
    lora_info, num_tokens: int, router_topk: int
) -> bool:
    """Whether the homogeneous shared-outer adapter can avoid top-k repetition."""

    rank = lora_info.max_lora_rank
    return bool(
        num_tokens > 0
        and router_topk > 1
        and lora_info.experts_shared_outer_loras
        and 0 < rank <= 64
        and lora_info.gate_up_lora_a_weights.shape[:2] == (1, 1)
        and lora_info.gate_up_lora_a_weights.shape[2] == 2 * rank
        and lora_info.gate_up_lora_b_weights.shape[0] == 1
        and lora_info.gate_up_lora_b_weights.shape[-1] == rank
        and lora_info.down_lora_a_weights.shape[0] == 1
        and lora_info.down_lora_a_weights.shape[2] == rank
        and lora_info.gate_up_lora_b_weights.shape[1]
        == lora_info.down_lora_a_weights.shape[1]
        and lora_info.gate_up_lora_b_weights.shape[2]
        == 2 * lora_info.down_lora_a_weights.shape[3]
        and lora_info.down_lora_b_weights.shape[:2] == (1, 1)
        and lora_info.down_lora_b_weights.shape[2]
        == lora_info.gate_up_lora_a_weights.shape[3]
        and lora_info.down_lora_b_weights.shape[-1] == rank
    )


def _use_multi_shared_outer_decode_factorization(
    lora_info,
    hidden_states: torch.Tensor,
    *,
    num_tokens: int,
    hidden_size: int,
    router_topk: int,
    num_experts: int,
    intermediate_size: int,
    ep_active: bool,
) -> bool:
    """Select the B200 direct-decode factorization for two to four slots."""

    slots = lora_info.gate_up_lora_a_weights.shape[0]
    if not (
        hidden_states.is_cuda
        and torch.cuda.get_device_capability(hidden_states.device) == (10, 0)
        and hidden_states.dtype == torch.bfloat16
        and not ep_active
        and slots >= 2
        and 0 < num_tokens <= 32
        and hidden_size == 6144
        and router_topk == 6
        and num_experts == 256
        and intermediate_size in (384, 768)
        and lora_info.max_lora_rank == 32
        and lora_info.experts_shared_outer_loras
    ):
        return False
    rank = lora_info.max_lora_rank
    return bool(
        lora_info.gate_up_lora_a_weights.shape[1:] == (1, 2 * rank, hidden_size)
        and lora_info.gate_up_lora_b_weights.shape
        == (slots, num_experts, 2 * intermediate_size, rank)
        and lora_info.down_lora_a_weights.shape
        == (slots, num_experts, rank, intermediate_size)
        and lora_info.down_lora_b_weights.shape == (slots, 1, hidden_size, rank)
    )


def _use_multi_shared_outer_prefill_factorization(
    lora_info,
    *,
    num_tokens: int,
    hidden_size: int,
    router_topk: int,
    num_experts: int,
    intermediate_size: int,
    ep_active: bool,
) -> bool:
    """Select top-k-collapsed shared-outer GEMMs for multi-slot prefill."""

    slots = lora_info.gate_up_lora_a_weights.shape[0]
    rank = lora_info.max_lora_rank
    if not (
        slots >= 2
        and (num_tokens > 32 or ep_active)
        and router_topk > 1
        and lora_info.experts_shared_outer_loras
        and 0 < rank <= 64
    ):
        return False
    return bool(
        lora_info.gate_up_lora_a_weights.shape == (slots, 1, 2 * rank, hidden_size)
        and lora_info.gate_up_lora_b_weights.shape
        == (slots, num_experts, 2 * intermediate_size, rank)
        and lora_info.down_lora_a_weights.shape
        == (slots, num_experts, rank, intermediate_size)
        and lora_info.down_lora_b_weights.shape == (slots, 1, hidden_size, rank)
    )


def _weighted_rank_sum_block_m(num_tokens: int) -> int:
    """Launch geometry for the small rank-sum reduction."""

    return 2 if num_tokens >= 2048 else 1


def _use_fused_shared_outer_tail(
    lora_info,
    hidden_states: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    router_topk: int,
) -> bool:
    """Restrict the fused tail to its supported geometry."""

    if not hidden_states.is_cuda:
        return False
    return bool(
        0 < num_tokens <= 512
        and hidden_size == 6144
        and router_topk == 6
        and lora_info.max_lora_rank == 32
        and hidden_states.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(hidden_states.device) == (10, 0)
    )


def _use_direct_decode_kernels(
    lora_info,
    *,
    factored_shared_outer: bool,
    fused_shared_outer_tail: bool,
    ep_active: bool,
    num_tokens: int,
    num_experts: int,
    intermediate_size: int,
) -> bool:
    """Use no-sort kernels for supported Inkling decode shapes."""

    return bool(
        factored_shared_outer
        and fused_shared_outer_tail
        and not ep_active
        and 0 < num_tokens <= 32
        and num_experts == 256
        and intermediate_size in (384, 768)
        and lora_info.gate_up_lora_b_weights.shape[1] == num_experts
        and lora_info.down_lora_a_weights.shape[1] == num_experts
    )


def fused_experts_experimental_sgl_marlin_lora(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
) -> StandardCombineInput:
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.lora.trtllm_lora_temp import (
        get_lora_side_stream,
        is_two_stream_active,
    )
    from sglang.srt.lora.trtllm_lora_temp.environ import experimental_lora_enabled
    from sglang.srt.model_executor.runner import get_is_capture_mode

    if not isinstance(dispatch_output, StandardDispatchOutput):
        raise TypeError("experimental_sgl_marlin requires the standard MoE dispatcher")

    if not (lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0):
        raise ValueError(
            "experimental_sgl_marlin LoRA requires --lora-use-virtual-experts"
        )

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids
    if (
        topk_ids.ndim != 2
        or topk_weights.shape != topk_ids.shape
        or topk_ids.dtype != torch.int32
        or topk_weights.dtype != torch.float32
    ):
        raise ValueError(
            "experimental_sgl_marlin requires contiguous int32 ids and FP32 weights"
        )
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("experimental_sgl_marlin requires contiguous top-k tensors")

    assert runner_config.activation == "silu", "Only SiLU activation is supported."
    routed_scaling_factor = runner_config.routed_scaling_factor
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    M, K = hidden_states.shape
    E = quant_info.w13_qweight.shape[0]
    N = quant_info.w2_qweight.shape[1] * 16
    topk = topk_ids.shape[1]
    num_bits = quant_info.weight_bits
    global_experts = runner_config.num_experts or E
    local_experts = runner_config.num_local_experts or E
    ep_active = local_experts < global_experts
    if ep_active:
        assert E == local_experts, (
            f"Marlin has {E} local experts but runner_config declares "
            f"{local_experts}"
        )
        assert (
            lora_info.gate_up_lora_b_weights.shape[1]
            == lora_info.down_lora_a_weights.shape[1]
            == E
        ), "EP requires locally sharded per-expert LoRA weights"

    # In eager mode skip the LoRA stages when no adapter is live; during capture
    # always record them; inactive pool slots contain zero weights.
    run_lora = get_is_capture_mode() or lora_info.has_active_lora
    single_shared_outer = run_lora and _use_shared_outer_factorization(
        lora_info, M, topk
    )
    multi_shared_outer = run_lora and _use_multi_shared_outer_decode_factorization(
        lora_info,
        hidden_states,
        num_tokens=M,
        hidden_size=K,
        router_topk=topk,
        num_experts=E,
        intermediate_size=N,
        ep_active=ep_active,
    )
    multi_prefill_shared_outer = (
        run_lora
        and _use_multi_shared_outer_prefill_factorization(
            lora_info,
            num_tokens=M,
            hidden_size=K,
            router_topk=topk,
            num_experts=E,
            intermediate_size=N,
            ep_active=ep_active,
        )
    )
    factored_shared_outer = (
        single_shared_outer or multi_shared_outer or multi_prefill_shared_outer
    )
    fused_shared_outer_tail = (
        factored_shared_outer
        and not multi_prefill_shared_outer
        and _use_fused_shared_outer_tail(lora_info, hidden_states, M, K, topk)
    )
    direct_decode = _use_direct_decode_kernels(
        lora_info,
        factored_shared_outer=factored_shared_outer,
        fused_shared_outer_tail=fused_shared_outer_tail,
        ep_active=ep_active,
        num_tokens=M,
        num_experts=E,
        intermediate_size=N,
    )
    post_reduce_down = use_post_reduce_down_delta(
        run_lora=run_lora,
        routed_scaling_factor=routed_scaling_factor,
        num_tokens=M,
    )
    two_stream = (
        run_lora and experimental_lora_enabled() and is_two_stream_active(hidden_states)
    )
    staged_down_shrink = two_stream and (factored_shared_outer or post_reduce_down)

    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    gate_up_delta = None
    lora_event = None
    down_rank_sum = None
    routing_cache: dict = {}
    # Collapsed top-k-1 routing needs a distinct graph-replay cache.
    collapsed_routing_cache: dict = {}
    collapsed_topk_ids = (
        lora_info.token_lora_mapping.view(M, 1) if multi_prefill_shared_outer else None
    )
    collapsed_topk_weights = topk_weights[:, :1] if multi_prefill_shared_outer else None
    if run_lora:
        use_direct_expand = (
            lora_info.max_lora_rank <= 64
            or lora_info.gate_up_lora_a_weights.shape[2]
            != lora_info.gate_up_lora_b_weights.shape[-1]
        )
        # Allocate and prewarm on main; side streams launch kernels only.
        gate_up_delta = hidden_states.new_empty((M, topk, 2 * N))
        if not direct_decode:
            merged_experts_fused_moe_lora_add(
                output=gate_up_delta,
                hidden_states=hidden_states,
                lora_a=lora_info.gate_up_lora_a_weights,
                lora_b=lora_info.gate_up_lora_b_weights,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                token_lora_mapping=lora_info.token_lora_mapping,
                mul_routed_weight=False,
                experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
                experts_shared_outer_loras_b=False,
                routing_cache=routing_cache,
                stage="routing",
                prewarm_a_routing=not factored_shared_outer,
                local_expert_offset=0,
                local_num_experts=E,
            )
        if multi_prefill_shared_outer:
            assert collapsed_topk_ids is not None
            assert collapsed_topk_weights is not None
            # Prewarm the selected shared-A route on main.
            merged_experts_fused_moe_lora_add(
                output=gate_up_delta,
                hidden_states=hidden_states,
                lora_a=lora_info.gate_up_lora_a_weights,
                lora_b=lora_info.gate_up_lora_b_weights,
                topk_ids=collapsed_topk_ids,
                topk_weights=collapsed_topk_weights,
                token_lora_mapping=lora_info.token_lora_mapping,
                mul_routed_weight=False,
                experts_shared_outer_loras_a=True,
                experts_shared_outer_loras_b=False,
                routing_cache=collapsed_routing_cache,
                stage="routing",
                prewarm_a_routing=True,
                prewarm_b_routing=False,
                local_expert_offset=0,
                local_num_experts=E,
            )
        gate_up_intermediate = hidden_states.new_empty(
            (
                (
                    lora_info.gate_up_lora_a_weights.shape[0],
                    M,
                    lora_info.gate_up_lora_a_weights.shape[2],
                )
                if multi_shared_outer
                else (
                    (M, lora_info.gate_up_lora_a_weights.shape[2])
                    if single_shared_outer or multi_prefill_shared_outer
                    else (M, topk, lora_info.gate_up_lora_a_weights.shape[2])
                )
            )
        )
        # Prewarm down routing and allocate its intermediate on main.
        if not direct_decode:
            merged_experts_fused_moe_lora_add(
                output=gate_up_delta,
                hidden_states=hidden_states,
                lora_a=lora_info.down_lora_a_weights,
                lora_b=lora_info.down_lora_b_weights,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                token_lora_mapping=lora_info.token_lora_mapping,
                mul_routed_weight=True,
                experts_shared_outer_loras_a=False,
                experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
                routing_cache=routing_cache,
                stage="routing",
                prewarm_b_routing=not factored_shared_outer,
                local_expert_offset=0,
                local_num_experts=E,
            )
        if multi_prefill_shared_outer:
            assert collapsed_topk_ids is not None
            assert collapsed_topk_weights is not None
            # The collapsed shared-A/B stages reuse the same top-k-1 cache.
            merged_experts_fused_moe_lora_add(
                output=gate_up_delta,
                hidden_states=hidden_states,
                lora_a=lora_info.down_lora_a_weights,
                lora_b=lora_info.down_lora_b_weights,
                topk_ids=collapsed_topk_ids,
                topk_weights=collapsed_topk_weights,
                token_lora_mapping=lora_info.token_lora_mapping,
                mul_routed_weight=False,
                experts_shared_outer_loras_a=False,
                experts_shared_outer_loras_b=True,
                routing_cache=collapsed_routing_cache,
                stage="routing",
                prewarm_a_routing=False,
                prewarm_b_routing=True,
                local_expert_offset=0,
                local_num_experts=E,
            )
        down_intermediate = (
            hidden_states.new_empty((M, topk, lora_info.down_lora_a_weights.shape[2]))
            if staged_down_shrink or factored_shared_outer
            else None
        )
        if factored_shared_outer and not fused_shared_outer_tail:
            down_rank_sum = hidden_states.new_empty(
                (M, lora_info.down_lora_a_weights.shape[2])
            )

        def _run_gate_up_delta():
            if factored_shared_outer:
                if multi_shared_outer:
                    torch.matmul(
                        hidden_states,
                        lora_info.gate_up_lora_a_weights[:, 0].transpose(1, 2),
                        out=gate_up_intermediate,
                    )
                elif multi_prefill_shared_outer:
                    assert collapsed_topk_ids is not None
                    assert collapsed_topk_weights is not None
                    # Collapsed routing writes one selected [2R] vector per token.
                    merged_experts_fused_moe_lora_add(
                        output=gate_up_delta,
                        hidden_states=hidden_states,
                        lora_a=lora_info.gate_up_lora_a_weights,
                        lora_b=lora_info.gate_up_lora_b_weights,
                        topk_ids=collapsed_topk_ids,
                        topk_weights=collapsed_topk_weights,
                        token_lora_mapping=lora_info.token_lora_mapping,
                        mul_routed_weight=False,
                        experts_shared_outer_loras_a=True,
                        experts_shared_outer_loras_b=False,
                        routing_cache=collapsed_routing_cache,
                        stage="shrink",
                        prewarm_b_routing=False,
                        intermediate_buffer=gate_up_intermediate,
                        local_expert_offset=0,
                        local_num_experts=E,
                    )
                else:
                    torch.mm(
                        hidden_states,
                        lora_info.gate_up_lora_a_weights[0, 0].T,
                        out=gate_up_intermediate,
                    )
                if direct_decode:
                    direct_decode_gate_expand(
                        gate_up_intermediate,
                        lora_info.gate_up_lora_b_weights,
                        topk_ids,
                        lora_info.token_lora_mapping,
                        gate_up_delta,
                    )
                    return
                merged_experts_fused_moe_lora_add(
                    output=gate_up_delta,
                    hidden_states=hidden_states,
                    lora_a=lora_info.gate_up_lora_a_weights,
                    lora_b=lora_info.gate_up_lora_b_weights,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    token_lora_mapping=lora_info.token_lora_mapping,
                    mul_routed_weight=False,
                    experts_shared_outer_loras_a=True,
                    experts_shared_outer_loras_b=False,
                    routing_cache=routing_cache,
                    fuse_add_to_output=False,
                    use_direct_expand_add=True,
                    local_expert_offset=0,
                    local_num_experts=E,
                    stage="expand",
                    intermediate_buffer=gate_up_intermediate,
                    broadcast_intermediate=True,
                )
                return
            merged_experts_fused_moe_lora_add(
                output=gate_up_delta,
                hidden_states=hidden_states,
                lora_a=lora_info.gate_up_lora_a_weights,
                lora_b=lora_info.gate_up_lora_b_weights,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                token_lora_mapping=lora_info.token_lora_mapping,
                mul_routed_weight=False,
                experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
                experts_shared_outer_loras_b=False,
                routing_cache=routing_cache,
                fuse_add_to_output=False,
                use_direct_expand_add=use_direct_expand,
                local_expert_offset=0,
                local_num_experts=E,
                intermediate_buffer=gate_up_intermediate,
            )

        if two_stream:
            lora_event = torch.cuda.Event()
            side_stream = get_lora_side_stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                _run_gate_up_delta()
                lora_event.record()
            if torch.cuda.is_current_stream_capturing():
                _MARLIN_LORA_OVERLAP_EVENTS.append(lora_event)
        else:
            _run_gate_up_delta()

    # Start alignment after the side-stream fork. EP IDs are already localized;
    # the wrapper owns Marlin's extra sentinel bucket.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, E
    )

    # A cached workspace can outlive its CUDA graph-private pool.
    workspace = marlin_make_workspace(hidden_states.device, max_blocks_per_sm=4)

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

    # Stage 1: gate_up (marlin) — concurrent with the side-stream delta above.
    intermediate_cache1 = torch.empty(
        (M * topk, 2 * N), device=hidden_states.device, dtype=hidden_states.dtype
    )
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
        is_ep=ep_active,
        b_q_type=scalar_type1,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=quant_info.is_k_full,
        use_atomic_add=True,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    # Stage 2: activation, with the gate_up delta folded in.
    intermediate_cache2 = torch.empty(
        (M * topk, N), device=hidden_states.device, dtype=hidden_states.dtype
    )
    down_shrink_done = None
    if run_lora:
        if lora_event is not None:
            torch.cuda.current_stream().wait_event(lora_event)
        silu_and_mul_add_delta(
            intermediate_cache1.view(-1, 2 * N),
            gate_up_delta.view(-1, 2 * N),
            intermediate_cache2,
        )
        if staged_down_shrink:
            # Overlap down shrink with the Marlin down GEMM and base reduction.
            act_done = torch.cuda.Event()
            act_done.record()
            down_shrink_done = torch.cuda.Event()
            side_stream = get_lora_side_stream()
            side_stream.wait_event(act_done)
            assert down_intermediate is not None
            with torch.cuda.stream(side_stream):
                if factored_shared_outer and not direct_decode:
                    # Buffers were allocated on main for stable graph ownership.
                    down_intermediate.zero_()
                # Reuse an output tensor because shrink does not consume it.
                if direct_decode:
                    direct_decode_down_shrink(
                        intermediate_cache2,
                        lora_info.down_lora_a_weights,
                        topk_ids,
                        lora_info.token_lora_mapping,
                        down_intermediate,
                    )
                else:
                    merged_experts_fused_moe_lora_add(
                        output=intermediate_cache2,
                        hidden_states=intermediate_cache2,
                        lora_a=lora_info.down_lora_a_weights,
                        lora_b=lora_info.down_lora_b_weights,
                        topk_ids=topk_ids,
                        topk_weights=topk_weights,
                        token_lora_mapping=lora_info.token_lora_mapping,
                        mul_routed_weight=True,
                        experts_shared_outer_loras_a=False,
                        experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
                        routing_cache=routing_cache,
                        stage="shrink",
                        prewarm_b_routing=not factored_shared_outer,
                        intermediate_buffer=down_intermediate,
                        local_expert_offset=0,
                        local_num_experts=E,
                        zero_intermediate=(
                            ep_active
                            and lora_info.experts_shared_outer_loras
                            and not factored_shared_outer
                        ),
                    )
                if factored_shared_outer and not fused_shared_outer_tail:
                    assert down_rank_sum is not None
                    weighted_topk_rank_sum(
                        down_intermediate,
                        topk_weights,
                        down_rank_sum,
                        routed_scaling_factor,
                        block_m=_weighted_rank_sum_block_m(M),
                    )
                down_shrink_done.record()
            if torch.cuda.is_current_stream_capturing():
                _MARLIN_LORA_OVERLAP_EVENTS.append(act_done)
                _MARLIN_LORA_OVERLAP_EVENTS.append(down_shrink_done)
    else:
        silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)

    # Stage 3: down (marlin).
    intermediate_cache3 = torch.empty(
        (M * topk, K), device=hidden_states.device, dtype=hidden_states.dtype
    )
    if ep_active:
        intermediate_cache3.zero_()

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
        is_ep=ep_active,
        b_q_type=scalar_type2,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=quant_info.is_k_full,
        use_atomic_add=True,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    intermediate_cache3 = intermediate_cache3.view(M, topk, K)

    if factored_shared_outer:
        assert down_intermediate is not None
        if down_shrink_done is None:
            if direct_decode:
                direct_decode_down_shrink(
                    intermediate_cache2,
                    lora_info.down_lora_a_weights,
                    topk_ids,
                    lora_info.token_lora_mapping,
                    down_intermediate,
                )
            else:
                down_intermediate.zero_()
                merged_experts_fused_moe_lora_add(
                    output=intermediate_cache3,
                    hidden_states=intermediate_cache2,
                    lora_a=lora_info.down_lora_a_weights,
                    lora_b=lora_info.down_lora_b_weights,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    token_lora_mapping=lora_info.token_lora_mapping,
                    mul_routed_weight=True,
                    experts_shared_outer_loras_a=False,
                    experts_shared_outer_loras_b=True,
                    routing_cache=routing_cache,
                    stage="shrink",
                    prewarm_b_routing=False,
                    intermediate_buffer=down_intermediate,
                    local_expert_offset=0,
                    local_num_experts=E,
                )

    # The post-reduce delta wins for decode; larger batches and non-unit routing
    # scales keep the stock pre-reduce path.
    if run_lora and not post_reduce_down and not factored_shared_outer:
        merged_experts_fused_moe_lora_add(
            output=intermediate_cache3,
            hidden_states=intermediate_cache2,
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=lora_info.token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=routing_cache,
            local_expert_offset=0,
            local_num_experts=E,
            zero_intermediate=(ep_active and lora_info.experts_shared_outer_loras),
        )

    # Never alias hidden_states: the shared sink still reads it. The fused B200
    # tail wins through M=512; larger batches keep the standard reducer.
    output = torch.empty_like(hidden_states)
    if fused_shared_outer_tail:
        assert down_intermediate is not None
        if down_shrink_done is not None:
            torch.cuda.current_stream().wait_event(down_shrink_done)
        if multi_shared_outer:
            fused_base_mapped_shared_lora_reduce(
                intermediate_cache3,
                down_intermediate,
                topk_weights,
                lora_info.down_lora_b_weights,
                lora_info.token_lora_mapping,
                output,
                routed_scaling_factor,
                block_k=64,
            )
        else:
            tail_block_m, tail_block_k = fused_base_shared_lora_reduce_config(M)
            fused_base_shared_lora_reduce(
                intermediate_cache3,
                down_intermediate,
                topk_weights,
                lora_info.down_lora_b_weights[0, 0],
                output,
                routed_scaling_factor,
                block_m=tail_block_m,
                block_k=tail_block_k,
            )
    elif routed_scaling_factor == 1.0 and M <= 512:
        torch.sum(intermediate_cache3, dim=1, out=output)
    else:
        moe_sum_reduce_triton(intermediate_cache3, output, routed_scaling_factor)

    if factored_shared_outer and not fused_shared_outer_tail:
        assert down_rank_sum is not None
        assert down_intermediate is not None
        if down_shrink_done is None:
            weighted_topk_rank_sum(
                down_intermediate,
                topk_weights,
                down_rank_sum,
                routed_scaling_factor,
                block_m=_weighted_rank_sum_block_m(M),
            )
        else:
            # Wait only at the shared-B consumer.
            torch.cuda.current_stream().wait_event(down_shrink_done)
        if multi_prefill_shared_outer:
            assert collapsed_topk_ids is not None
            assert collapsed_topk_weights is not None
            # Route the weighted rank sum once per token; -1 rows stay base-only.
            merged_experts_fused_moe_lora_add(
                output=output,
                hidden_states=down_rank_sum,
                lora_a=lora_info.down_lora_a_weights,
                lora_b=lora_info.down_lora_b_weights,
                topk_ids=collapsed_topk_ids,
                topk_weights=collapsed_topk_weights,
                token_lora_mapping=lora_info.token_lora_mapping,
                mul_routed_weight=False,
                experts_shared_outer_loras_a=False,
                experts_shared_outer_loras_b=True,
                routing_cache=collapsed_routing_cache,
                fuse_add_to_output=True,
                use_direct_expand_add=False,
                local_expert_offset=0,
                local_num_experts=E,
                stage="expand",
                intermediate_buffer=down_rank_sum,
            )
        else:
            output.addmm_(down_rank_sum, lora_info.down_lora_b_weights[0, 0].T)
    elif post_reduce_down and not factored_shared_outer:
        if down_shrink_done is not None:
            torch.cuda.current_stream().wait_event(down_shrink_done)
        merged_experts_fused_moe_lora_add(
            output=output,
            hidden_states=intermediate_cache2,
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=lora_info.token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=0,
            local_num_experts=E,
            stage="expand" if down_shrink_done is not None else "all",
            intermediate_buffer=(
                down_intermediate if down_shrink_done is not None else None
            ),
            zero_intermediate=(
                ep_active
                and lora_info.experts_shared_outer_loras
                and down_shrink_done is None
            ),
        )

    return StandardCombineInput(hidden_states=output)
