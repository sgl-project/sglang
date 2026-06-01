import functools
from typing import List, Optional, Union

import torch


@functools.cache
def get_sgl_trtllm_moe_sm100_module():
    import flashinfer.fused_moe.core as fi_core

    from sglang.jit_kernel.flashinfer_trtllm_moe.jit import (
        gen_sgl_trtllm_gen_fused_moe_sm100_module,
    )

    original_gen = fi_core.gen_trtllm_gen_fused_moe_sm100_module
    fi_core.gen_trtllm_gen_fused_moe_sm100_module = (
        gen_sgl_trtllm_gen_fused_moe_sm100_module
    )
    try:
        fi_core.get_trtllm_moe_sm100_module.cache_clear()
        return fi_core.get_trtllm_moe_sm100_module()
    finally:
        fi_core.gen_trtllm_gen_fused_moe_sm100_module = original_gen
        fi_core.get_trtllm_moe_sm100_module.cache_clear()


@functools.cache
def get_sgl_trtllm_moe_sm100_raw_module():
    from flashinfer.fused_moe.core import setup_cubin_loader

    from sglang.jit_kernel.flashinfer_trtllm_moe.jit import (
        gen_sgl_trtllm_gen_fused_moe_sm100_module,
    )

    module = gen_sgl_trtllm_gen_fused_moe_sm100_module()
    moe_op = module.build_and_load()
    setup_cubin_loader(str(module.get_library_path()))
    return moe_op


def _validate_routing_replay_out(
    routing_replay_out: Optional[torch.Tensor], top_k: int
) -> None:
    if routing_replay_out is None:
        return
    assert routing_replay_out.dim() == 2
    assert routing_replay_out.shape[1] == top_k
    assert routing_replay_out.dtype == torch.int16
    assert routing_replay_out.is_cuda
    assert routing_replay_out.is_contiguous()


def trtllm_fp8_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type=None,
    activation_type: Optional[int] = None,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    from flashinfer.fused_moe.core import ActivationType, Fp8QuantizationType

    _validate_routing_replay_out(routing_replay_out, top_k)

    if fp8_quantization_type is None:
        fp8_quantization_type = Fp8QuantizationType.DeepSeekFp8
    if activation_type is None:
        activation_type = ActivationType.Swiglu.value

    output = torch.empty(
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
    )
    result = get_sgl_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(
        routing_logits,
        None,
        None,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        do_finalize,
        enable_pdl,
        tune_max_num_tokens,
        fp8_quantization_type,
        activation_type,
        norm_topk_prob,
        routing_replay_out,
    )

    return result[0] if do_finalize else result


def trtllm_fp8_block_scale_routed_moe(
    topk_ids: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type=None,
    activation_type: Optional[int] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    from flashinfer.fused_moe.core import ActivationType, Fp8QuantizationType

    if fp8_quantization_type is None:
        fp8_quantization_type = Fp8QuantizationType.DeepSeekFp8
    if activation_type is None:
        activation_type = ActivationType.Swiglu.value
    if output is None:
        output = torch.empty(
            hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
        )

    result = get_sgl_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(
        None,
        topk_ids,
        None,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        do_finalize,
        enable_pdl,
        tune_max_num_tokens,
        fp8_quantization_type,
        activation_type,
        True,
    )

    return result[0] if do_finalize else result


def trtllm_fp8_block_scale_routed_moe_lora(
    topk_ids: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gate_up_lora_delta: torch.Tensor,
    activation_lora_input: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type=None,
    activation_type: Optional[int] = None,
    lora_ready_event: int = 0,
) -> Union[List[torch.Tensor], torch.Tensor]:
    from flashinfer.fused_moe.core import ActivationType, Fp8QuantizationType
    from flashinfer.utils import device_support_pdl

    if fp8_quantization_type is None:
        fp8_quantization_type = Fp8QuantizationType.DeepSeekFp8
    if activation_type is None:
        activation_type = ActivationType.Swiglu.value
    if enable_pdl is None:
        enable_pdl = device_support_pdl(hidden_states.device)
    if output is None:
        output = torch.empty(
            hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
        )

    assert gate_up_lora_delta.is_contiguous()
    assert activation_lora_input.is_contiguous()
    empty_expert_weights = torch.empty(
        (0,), dtype=torch.bfloat16, device=hidden_states.device
    )

    result = get_sgl_trtllm_moe_sm100_raw_module().sgl_trtllm_fp8_block_scale_moe_lora(
        None,
        topk_ids,
        empty_expert_weights,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        do_finalize,
        enable_pdl,
        [-1, -1],
        fp8_quantization_type,
        activation_type,
        True,
        None,
        gate_up_lora_delta,
        activation_lora_input,
        lora_ready_event,
    )

    return output if do_finalize else result


def trtllm_fp8_block_scale_moe_lora_finalize(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    down_lora_delta: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: Optional[float],
) -> torch.Tensor:
    get_sgl_trtllm_moe_sm100_raw_module().sgl_trtllm_fp8_block_scale_moe_lora_finalize(
        gemm2_output,
        expert_weights,
        expanded_idx_to_permuted_idx,
        down_lora_delta,
        output,
        routed_scaling_factor,
    )
    return output


def trtllm_fp4_block_scale_routed_moe_lora(
    topk_ids: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    gate_up_lora_delta: torch.Tensor,
    activation_lora_input: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    output: Optional[torch.Tensor] = None,
    act_type: Optional[int] = None,
    norm_topk_prob: bool = True,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """NVFP4 sibling of :func:`trtllm_fp8_block_scale_routed_moe_lora`.

    Decomposed (unfused-activation) MoE-LoRA on the flashinfer-trtllm backend:
    routing -> gather -> gate_up grouped GEMM (raw 2*inter) -> activation that
    adds ``gate_up_lora_delta`` pre-SwiGLU and writes ``activation_lora_input``
    -> NvFP4 quant -> down grouped GEMM -> finalize. The down-LoRA is merged into
    the output afterwards by the dispatch (virtual-experts) or via the finalize.

    ``hidden_states`` is bf16 ``[num_tokens, hidden]`` (path 3: the op permutes then
    NvFP4-quantizes it internally, ``hidden_states_scale=None``) or, legacy, packed NvFP4
    (uint8 ``[num_tokens, hidden//2]``) with ``hidden_states_scale`` the fp8-e4m3 block
    scale ``[num_tokens, hidden//16]``.
    """
    from flashinfer.fused_moe.core import ActivationType
    from flashinfer.utils import device_support_pdl

    if act_type is None:
        act_type = ActivationType.Swiglu.value
    if enable_pdl is None:
        enable_pdl = device_support_pdl(hidden_states.device)

    num_tokens = hidden_states.shape[0]
    hidden_size = (
        hidden_states.shape[1] * 2
        if hidden_states.dtype == torch.uint8
        else hidden_states.shape[1]
    )
    if output is None:
        output = torch.empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device=hidden_states.device
        )

    assert gate_up_lora_delta.is_contiguous()
    assert activation_lora_input.is_contiguous()
    empty_expert_weights = torch.empty(
        (0,), dtype=torch.bfloat16, device=hidden_states.device
    )

    result = get_sgl_trtllm_moe_sm100_raw_module().sgl_trtllm_fp4_block_scale_moe_lora(
        None,  # routing_logits (precomputed routing via packed topk_ids)
        topk_ids,  # expert_indices (packed: (expert_id<<16)|weight_bf16.view(int16))
        empty_expert_weights,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        None,  # gemm1_bias
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        gemm2_weights,
        gemm2_weights_scale,
        None,  # gemm2_bias
        output1_scales_scalar,
        output1_scales_gate_scalar,
        output2_scales_scalar,
        None,  # per_token_scales (the down-GEMM act scale is produced internally)
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize,
        enable_pdl,
        act_type,
        output,
        [-1, -1],  # config_index (autotuner)
        norm_topk_prob,
        None,  # routing_replay_out
        gate_up_lora_delta,
        activation_lora_input,
    )

    return output if do_finalize else result


def trtllm_fp4_block_scale_moe_lora_finalize(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    down_lora_delta: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: Optional[float],
) -> torch.Tensor:
    """NvFP4 analog of :func:`trtllm_fp8_block_scale_moe_lora_finalize` — combines
    the permuted (bf16) GEMM2 output by expert weight and merges the down-LoRA
    delta into the per-token output (non-virtual-experts hook path)."""
    get_sgl_trtllm_moe_sm100_raw_module().sgl_trtllm_fp4_block_scale_moe_lora_finalize(
        gemm2_output,
        expert_weights,
        expanded_idx_to_permuted_idx,
        down_lora_delta,
        output,
        routed_scaling_factor,
    )
    return output
