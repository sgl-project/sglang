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


def should_use_mega_gate(moe, hidden_states, device_sm, dispatch_info) -> bool:
    if (
        not envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.get()
        or device_sm // 10 != 10
        or getattr(moe.config, "model_type", None) != "deepseek_v41"
        # vLLM's Flash measurements favor the unfused router at <=16 rows.
        # Keep the conservative threshold for target and draft expert counts.
        or not 16 < hidden_states.shape[0] <= (1 << 20)
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_contiguous()
        or getattr(moe.topk, "enable_waterfill", False)
        or get_exec().deterministic.enable_deterministic_inference
        or envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        or envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    ):
        return False
    weight = moe.gate.weight
    num_experts, hidden = weight.shape
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or hidden != hidden_states.shape[1]
        or hidden % 256 != 0
        or not 0 < num_experts <= 512
        or num_experts % 4 != 0
        or not 0 < moe.config.num_experts_per_tok <= min(32, num_experts)
    ):
        return False
    for bias in (
        moe.gate.e_score_correction_bias,
        moe.gate.e_score_correction_bias_vl,
    ):
        if bias is not None and (
            bias.dtype != torch.float32 or not bias.is_contiguous()
        ):
            return False
    if dispatch_info is not None and dispatch_info.ep_dispatch_algorithm in (
        "lp",
        "fake",
    ):
        # Preserve the LP solver's collectives and fake routing's logit override.
        return False
    if moe.is_hash:
        if moe.topk.score_func != "sqrtsoftplus":
            return False
    else:
        config = moe.topk.topk_config
        if (
            config.scoring_func != "sqrtsoftplus"
            or not config.renormalize
            or config.top_k <= 1
            or config.use_grouped_topk
            or config.custom_routing_function is not None
            or config.torch_native
        ):
            return False
    return is_mega_gate_available()


def run_mega_gate(moe, hidden_states, input_ids, num_token_non_padded, dispatch_info):
    if moe.is_hash:
        config = TopKConfig(
            top_k=moe.topk.topk,
            num_fused_shared_experts=moe.topk.num_fused_shared_experts,
            routed_scaling_factor=moe.topk.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=moe.topk.apply_routed_scaling_factor_on_output,
        )
    else:
        config = moe.topk.topk_config
    num_tokens = hidden_states.shape[0]
    valid_mask = None
    if num_token_non_padded is not None:
        valid_mask = (
            torch.arange(num_tokens, device=hidden_states.device) < num_token_non_padded
        )

    hash_ids = None
    image_mask = None
    image_bias = None
    if moe.is_hash:
        # Match HashTopK: image IDs still select their checkpoint hash routes.
        # Padding IDs may be outside the vocabulary, so sanitize before lookup.
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
        image_mask = (input_ids == moe.config.image_token_id).contiguous()

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

    # MegaGate emits logical routed IDs. Reuse SGLang's EPLB, capture and
    # home-rank shared-slot mapping; only the expert count is needed from logits.
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
    # Shared-slot remapping may overwrite weights in padded rows. Mask last,
    # before recording, including when every row on this DP rank is padding.
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
