import torch
import torch.nn.functional as F

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.moe.topk import (
    _RENORMALIZE_SUM_EPSILON,
    StandardTopKOutput,
    StandardTopKOutputPacked,
    _mask_topk_ids_padded_region,
    _post_process_topk_ids,
    _zero_topk_weights_padded_region,
)
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    has_per_rank_fused_shared_slots,
)
from sglang.srt.utils import is_cuda


def _scale_fused_shared_weights(weights, num_fused_shared_experts, scaling_factor):
    # Standard EP replicates the fused shared expert on every rank and all-reduces,
    # so the shared columns carry a 1/ep_size factor.
    if num_fused_shared_experts and scaling_factor is not None:
        weights[:, -num_fused_shared_experts:] *= scaling_factor
    return weights


def vision_topk(
    moe,
    logits,
    input_ids,
    num_token_non_padded=None,
    expert_location_dispatch_info=None,
):
    config = moe.topk.topk_config
    num_fused_shared_experts = config.num_fused_shared_experts
    use_mega_moe = get_moe_a2a_backend().is_megamoe()
    if num_fused_shared_experts and not use_mega_moe:
        # This path bypasses _post_process_topk_ids, which appends the per-rank slots.
        assert not has_per_rank_fused_shared_slots(num_fused_shared_experts), (
            "VL routing does not support per-rank fused shared slots"
        )
    if is_cuda():
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
        from sglang.srt.layers.moe.utils import get_moe_runner_backend

        # Same admission as _fused_gate_emits_packed_ids: the shared-expert slots
        # rescaled below would rewrite weights after the router.
        packed_topk = None
        if (
            num_fused_shared_experts == 0
            and not use_mega_moe
            and get_moe_runner_backend().is_flashinfer_mxfp4()
        ):
            packed_topk = torch.empty(
                (logits.shape[0], config.top_k), dtype=torch.int32, device=logits.device
            )

        weights, indices = moe_fused_gate(
            logits,
            moe.gate.e_score_correction_bias,
            topk=config.top_k,
            scoring_func="sqrtsoftplus",
            num_fused_shared_experts=num_fused_shared_experts,
            bias_alt=moe.gate.e_score_correction_bias_vl,
            input_ids=input_ids,
            bias_alt_token_id=moe.config.image_token_id,
            renormalize=config.renormalize and config.top_k > 1,
            renormalize_epsilon=_RENORMALIZE_SUM_EPSILON,
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=config.apply_routed_scaling_factor_on_output,
            num_token_non_padded=num_token_non_padded,
            packed_out=packed_topk,
            sqrtsoftplus_log1p=True,
        )
        weights, indices = _finish_vision_topk(
            moe,
            logits,
            weights,
            indices,
            num_token_non_padded,
            expert_location_dispatch_info,
            padded_rows_masked=True,
        )
        if packed_topk is not None:
            return StandardTopKOutputPacked(weights, indices, logits, packed_topk)
        return StandardTopKOutput(weights, indices, logits)
    scores = F.softplus(logits.float()).sqrt()
    if input_ids is None:
        bias = moe.gate.e_score_correction_bias
    else:
        bias = torch.where(
            (input_ids == moe.config.image_token_id)[:, None],
            moe.gate.e_score_correction_bias_vl,
            moe.gate.e_score_correction_bias,
        )
    # The shared slots appended below use the same layout as biased_grouped_topk_gpu.
    topk_routed = config.top_k - num_fused_shared_experts
    indices = (scores + bias).topk(topk_routed, dim=-1).indices
    weights = scores.gather(-1, indices)
    routed_sum = weights.sum(-1, keepdim=True, dtype=torch.float32)
    if num_fused_shared_experts:
        shared_ids = logits.shape[-1] + torch.arange(
            num_fused_shared_experts, device=indices.device, dtype=indices.dtype
        )
        indices = torch.cat(
            [indices, shared_ids.expand(indices.shape[0], -1)],
            dim=-1,
        )
        weights = F.pad(weights, (0, num_fused_shared_experts))
        weights[:, topk_routed:] = routed_sum / config.routed_scaling_factor
    if config.renormalize and config.top_k > 1:
        weights = weights / (routed_sum + _RENORMALIZE_SUM_EPSILON)
    if config.apply_routed_scaling_factor_on_output:
        weights = weights * config.routed_scaling_factor
    weights, indices = weights.float(), indices.int()
    weights, indices = _finish_vision_topk(
        moe,
        logits,
        weights,
        indices,
        num_token_non_padded,
        expert_location_dispatch_info,
    )
    return StandardTopKOutput(weights, indices, logits)


def _finish_vision_topk(
    moe,
    logits,
    weights,
    indices,
    num_token_non_padded,
    expert_location_dispatch_info,
    padded_rows_masked=False,
):
    config = moe.topk.topk_config
    use_mega_moe = get_moe_a2a_backend().is_megamoe()
    if use_mega_moe:
        # Use the same EPLB mapping, recording and home-rank shared slots as
        # ordinary top-k before handing physical expert IDs to MegaMoE.
        indices, weights, recorder_ids = _post_process_topk_ids(
            indices,
            weights,
            config,
            logits,
            moe.layer_id,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
            padded_rows_masked=padded_rows_masked,
        )
        if (
            config.num_fused_shared_experts
            and config.apply_routed_scaling_factor_on_output
        ):
            # MegaMoE applies no post-expert scale when it is already in top-k.
            weights[:, -config.num_fused_shared_experts :] = 1.0
        if recorder_ids is not None:
            get_global_expert_distribution_recorder().on_select_experts(
                topk_ids=recorder_ids
            )
    else:
        weights = _scale_fused_shared_weights(
            weights,
            config.num_fused_shared_experts,
            config.fused_shared_experts_scaling_factor,
        )
    # Shared-slot remapping can overwrite padded weights; mask last so idle
    # and partially padded DP ranks never dispatch a phantom shared expert.
    if num_token_non_padded is not None and (
        not padded_rows_masked or (use_mega_moe and config.num_fused_shared_experts)
    ):
        _mask_topk_ids_padded_region(indices, num_token_non_padded)
        _zero_topk_weights_padded_region(weights, num_token_non_padded)
    return weights, indices
