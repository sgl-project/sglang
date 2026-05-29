"""sgl_flashinfer_trtllm MoE LoRA dispatch (original single-stream).

This is the LoRA-enabled fused-experts path added by the trtllm-lora work — it
was originally a function in ``layers/moe/moe_runner/flashinfer_trtllm.py`` and
is now hosted here so that file holds only a re-export. The function name
remains ``fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora`` for
import-site stability.

When ``SGLANG_LORA_TWO_STREAM=1`` is set, this is the function the
``install_two_stream_overrides()`` monkey-patch swaps for the side-stream
version in :mod:`sglang.srt.lora.trtllm_moe.moe_overlap`. Otherwise it runs as
the active path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.utils.common import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmFp8MoeQuantInfo,
    )
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


def fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora(
    dispatch_output: "StandardDispatchOutput",
    quant_info: "FlashInferTrtllmFp8MoeQuantInfo",
    runner_config: "MoeRunnerConfig",
    lora_info,
) -> "StandardCombineInput":
    from flashinfer.fused_moe import Fp8QuantizationType

    from sglang.jit_kernel.flashinfer_trtllm_moe import (
        trtllm_fp8_block_scale_moe_lora_finalize,
        trtllm_fp8_block_scale_routed_moe_lora,
    )
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        _pack_topk_for_flashinfer_routed,
        fused_experts_none_to_flashinfer_trtllm_fp8,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.lora.lora_moe_runners import build_lora_hooks
    from sglang.srt.lora.triton_ops import merged_experts_fused_moe_lora_add
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    assert runner_config.activation == "silu" and runner_config.is_gated, (
        "sgl_flashinfer_trtllm LoRA currently supports the gated SwiGLU FP8 "
        "Qwen path only."
    )
    assert quant_info.block_quant and not quant_info.use_mxfp8, (
        "sgl_flashinfer_trtllm LoRA currently supports DeepSeekFp8 block-quant "
        "checkpoints only."
    )
    assert quant_info.weight_block_k is not None
    assert quant_info.w13_weight_scale_inv is not None
    assert quant_info.w2_weight_scale_inv is not None

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    if not get_is_capture_mode() and not lora_info.has_active_lora:
        return fused_experts_none_to_flashinfer_trtllm_fp8(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
            use_sgl_kernel=True,
        )

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    if use_virtual_lora_store:
        hooks = None
        token_lora_mapping = lora_info.token_lora_mapping
        fused_lora_routing_cache: dict = {}
    else:
        hooks = build_lora_hooks(hidden_states, lora_info, topk_ids)
        token_lora_mapping = None
        fused_lora_routing_cache = {}

    # Fuse the per-token scale transpose into the quant kernel (column-major scales) so the
    # `.t()` is a free view -> drops the standalone ~2us transpose+copy. Byte/shape-identical.
    a_q, a_sf = per_token_group_quant_fp8(
        hidden_states, quant_info.weight_block_k, column_major_scales=True
    )
    a_sf_t = a_sf.t()

    # EP-aware LoRA: under MoE EP each rank computes the delta only for the experts it
    # owns (passed via local_expert_offset/local_num_experts below). gate_up_delta stays
    # new_empty even though non-owned [token, k] slots are then left unwritten -- the
    # trtllm MoE is itself EP-aware, so those slots never feed the all-reduced output.
    gate_up_delta_shape = (
        hidden_states.shape[0],
        runner_config.top_k,
        quant_info.w13_weight.shape[1],
    )
    gate_up_delta = (
        hidden_states.new_empty(gate_up_delta_shape)
        if use_virtual_lora_store
        else hidden_states.new_zeros(gate_up_delta_shape)
    )
    if use_virtual_lora_store:
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
        )
    elif hooks.after_gate_up is not None:
        hooks.after_gate_up(hidden_states, gate_up_delta, topk_weights, topk_ids)

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, quant_info.intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    packed_topk_ids = _pack_topk_for_flashinfer_routed(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    direct_down_output = None
    if use_virtual_lora_store:
        with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
            direct_down_output = torch.empty(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

    moe_result = trtllm_fp8_block_scale_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=a_q,
        hidden_states_scale=a_sf_t,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=quant_info.w13_weight_scale_inv,
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=quant_info.w2_weight_scale_inv,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        num_experts=quant_info.global_num_experts,
        top_k=runner_config.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=quant_info.intermediate_size,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        routing_method_type=(
            RoutingMethodType.TopK
            if quant_info.routing_method_type == RoutingMethodType.DeepSeekV3
            else quant_info.routing_method_type
        ),
        use_shuffled_weight=False,
        do_finalize=use_virtual_lora_store,
        output=(
            direct_down_output
            if direct_down_output is not None
            else torch.empty_like(hidden_states)
        ),
        tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        activation_type=quant_info.activation_type,
    )
    if use_virtual_lora_store:
        output = moe_result
        merged_experts_fused_moe_lora_add(
            output=output,
            hidden_states=activation_lora_input.view(-1, quant_info.intermediate_size),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
        )
        return StandardCombineInput(hidden_states=output)

    gemm2_output, expert_weights, expanded_idx_to_permuted_idx = moe_result

    down_delta_shape = (
        hidden_states.shape[0],
        runner_config.top_k,
        hidden_states.shape[1],
    )
    down_delta = (
        hidden_states.new_empty(down_delta_shape)
        if use_virtual_lora_store
        else hidden_states.new_zeros(down_delta_shape)
    )
    if use_virtual_lora_store:
        merged_experts_fused_moe_lora_add(
            output=down_delta,
            hidden_states=activation_lora_input.view(-1, quant_info.intermediate_size),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
        )
    elif hooks.after_down is not None:
        hooks.after_down(
            activation_lora_input.view(-1, quant_info.intermediate_size),
            down_delta,
            topk_weights,
            topk_ids,
        )

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    output = trtllm_fp8_block_scale_moe_lora_finalize(
        gemm2_output=gemm2_output,
        expert_weights=expert_weights,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        down_lora_delta=down_delta,
        output=output,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
    )

    return StandardCombineInput(hidden_states=output)
