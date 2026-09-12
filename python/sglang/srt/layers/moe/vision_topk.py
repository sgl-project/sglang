"""Modality-dependent MoE routing for DeepSeek-V4.1."""

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.topk import (
    _RENORMALIZE_SUM_EPSILON,
    StandardTopKOutput,
    _mask_topk_ids_padded_region,
    _zero_topk_weights_padded_region,
)
from sglang.srt.layers.moe.utils import has_per_rank_fused_shared_slots
from sglang.srt.utils import is_cuda


def _scale_fused_shared_weights(weights, num_fused_shared_experts, scaling_factor):
    # Standard EP replicates the fused shared expert on every rank and
    # all-reduces the outputs, so the shared columns carry a 1/ep_size factor
    # (applied by _post_process_topk_ids on the paths that go through it).
    if num_fused_shared_experts and scaling_factor is not None:
        weights[:, -num_fused_shared_experts:] *= scaling_factor
    return weights


def vision_topk(moe, logits, input_ids, num_token_non_padded=None):
    config = moe.topk.topk_config
    num_fused_shared_experts = config.num_fused_shared_experts
    if num_fused_shared_experts:
        # The per-rank shared-slot layout is appended by _post_process_topk_ids,
        # which this routing path bypasses.
        assert not has_per_rank_fused_shared_slots(num_fused_shared_experts), (
            "VL routing does not support per-rank fused shared slots"
        )
    if is_cuda():
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

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
        )
        weights = _scale_fused_shared_weights(
            weights,
            num_fused_shared_experts,
            config.fused_shared_experts_scaling_factor,
        )
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
    # topk for routed experts only; the shared expert slots are appended below
    # with the same layout as biased_grouped_topk_gpu.
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
    weights = _scale_fused_shared_weights(
        weights, num_fused_shared_experts, config.fused_shared_experts_scaling_factor
    )
    weights, indices = weights.float(), indices.int()
    if num_token_non_padded is not None:
        _mask_topk_ids_padded_region(indices, num_token_non_padded)
        _zero_topk_weights_padded_region(weights, num_token_non_padded)
    return StandardTopKOutput(weights, indices, logits)
