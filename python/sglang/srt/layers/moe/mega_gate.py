"""DeepSeek-V4.1 MegaGate dispatch and SGLang routing bookkeeping."""

from __future__ import annotations

import torch

from sglang.kernels.ops.moe.mega_gate import bf16_mega_gate, is_mega_gate_available
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _mask_topk_ids_padded_region,
    _post_process_topk_ids,
    _zero_topk_weights_padded_region,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import is_sm100_supported


def mega_gate_supported(moe, dispatch_info) -> bool:
    """Static admission: server config, gate weight layout, and routing mode."""
    if (
        not envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.get()
        or not is_sm100_supported()
        or not is_mega_gate_available()
        or moe.config.model_type != "deepseek_v41"
        or moe.topk.enable_waterfill
        or get_exec().deterministic.enable_deterministic_inference
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
        or (
            dispatch_info is not None
            and dispatch_info.ep_dispatch_algorithm in ("lp", "fake")
        )
    ):
        return False
    weight = moe.gate.weight
    num_experts, hidden = weight.shape
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or hidden % 256 != 0
        or not 0 < num_experts <= 512
        or num_experts % 4 != 0
        or not 0 < moe.config.num_experts_per_tok <= min(32, num_experts)
    ):
        return False
    for bias in (
        moe.gate.e_score_correction_bias,
        moe.gate.e_score_correction_bias_vl if not moe.is_hash else None,
    ):
        if bias is not None and (
            bias.dtype != torch.float32 or not bias.is_contiguous()
        ):
            return False
    if moe.is_hash:
        return moe.topk.score_func == "sqrtsoftplus"
    config = moe.topk.topk_config
    return (
        config.scoring_func == "sqrtsoftplus"
        and config.renormalize
        and not (config.top_k == 1 and moe.gate.e_score_correction_bias_vl is not None)
        and not config.use_grouped_topk
        and config.custom_routing_function is None
        and not config.torch_native
    )


def should_use_mega_gate(moe, hidden_states, dispatch_info) -> bool:
    if moe._mega_gate_supported is None:
        moe._mega_gate_supported = mega_gate_supported(moe, dispatch_info)
    return (
        moe._mega_gate_supported
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
    )


def run_mega_gate(
    moe, hidden_states, input_ids, num_token_non_padded, valid_mask, dispatch_info
):
    if moe.is_hash:
        config = TopKConfig(
            top_k=moe.topk.topk,
            num_fused_shared_experts=moe.topk.num_fused_shared_experts,
            routed_scaling_factor=moe.topk.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=moe.topk.apply_routed_scaling_factor_on_output,
            allow_routed_experts_capture=False,
        )
    else:
        config = moe.topk.topk_config
    num_tokens = hidden_states.shape[0]

    hash_ids = None
    image_mask = None
    image_bias = None
    if moe.is_hash:
        if input_ids is None:
            raise ValueError("DeepSeek-V4.1 hash routing requires input_ids.")
        safe_ids = (
            torch.where(valid_mask, input_ids, 0)
            if valid_mask is not None
            else input_ids
        )
        hash_ids = moe.topk.tid2eid[safe_ids].to(torch.int64)
    elif moe.gate.e_score_correction_bias_vl is not None and input_ids is not None:
        image_bias = moe.gate.e_score_correction_bias_vl
        image_mask = input_ids == moe.config.image_token_id

    weights, ids = bf16_mega_gate(
        hidden_states,
        moe.gate.weight,
        moe.config.num_experts_per_tok,
        routed_scaling_factor=(
            config.routed_scaling_factor
            if config.apply_routed_scaling_factor_on_output
            else 1.0
        ),
        ep_rank=get_parallel().moe_ep_rank,
        bias=moe.gate.e_score_correction_bias,
        image_bias=image_bias,
        image_token_mask=image_mask,
        valid_token_mask=valid_mask,
        hash_topk_ids=hash_ids,
    )
    num_shared = config.num_fused_shared_experts
    if num_shared:
        shared_ids = moe.gate.weight.shape[0] + torch.arange(
            num_shared, dtype=ids.dtype, device=ids.device
        )
        ids = torch.cat([ids, shared_ids.expand(num_tokens, -1)], dim=-1)
        shared_weight = (
            1.0
            if config.apply_routed_scaling_factor_on_output
            else 1.0 / config.routed_scaling_factor
        )
        weights = torch.cat(
            [weights, weights.new_full((num_tokens, num_shared), shared_weight)], dim=-1
        )

    logits_shape = hidden_states.new_empty((0, moe.gate.weight.shape[0]))
    ids, weights, recorder_ids = _post_process_topk_ids(
        ids,
        weights,
        config,
        logits_shape,
        moe.layer_id,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=dispatch_info,
        padded_rows_masked=True,
    )
    if num_shared and config.apply_routed_scaling_factor_on_output:
        weights[:, -num_shared:] = 1.0
    if num_token_non_padded is not None:
        _mask_topk_ids_padded_region(ids, num_token_non_padded)
        _zero_topk_weights_padded_region(weights, num_token_non_padded)
        if recorder_ids is not None:
            _mask_topk_ids_padded_region(recorder_ids, num_token_non_padded)
    if recorder_ids is not None:
        get_global_expert_distribution_recorder().on_select_experts(
            topk_ids=recorder_ids
        )
    return weights, ids
