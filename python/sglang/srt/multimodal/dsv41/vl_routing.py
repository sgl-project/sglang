import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    _mask_topk_ids_padded_region,
    _zero_topk_weights_padded_region,
)
from sglang.srt.utils import is_cuda


def vision_topk(moe, logits, input_ids, num_token_non_padded=None):
    config = moe.topk.topk_config
    if is_cuda():
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

        weights, indices = moe_fused_gate(
            logits,
            moe.gate.e_score_correction_bias,
            topk=config.top_k,
            scoring_func="sqrtsoftplus",
            bias_alt=moe.gate.e_score_correction_bias_vl,
            input_ids=input_ids,
            bias_alt_token_id=moe.config.image_token_id,
            renormalize=config.renormalize and config.top_k > 1,
            renormalize_epsilon=1e-20,
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=config.apply_routed_scaling_factor_on_output,
            num_token_non_padded=num_token_non_padded,
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
    indices = (scores + bias).topk(config.top_k, dim=-1).indices
    weights = scores.gather(-1, indices)
    if config.renormalize and config.top_k > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    if config.apply_routed_scaling_factor_on_output:
        weights = weights * config.routed_scaling_factor
    weights, indices = weights.float(), indices.int()
    if num_token_non_padded is not None:
        _mask_topk_ids_padded_region(indices, num_token_non_padded)
        _zero_topk_weights_padded_region(weights, num_token_non_padded)
    return StandardTopKOutput(weights, indices, logits)
