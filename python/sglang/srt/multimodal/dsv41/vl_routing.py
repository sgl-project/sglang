"""Reference V4.1 modality-dependent expert selection for eager MoE."""

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.topk import StandardTopKOutput


def vision_topk(moe, logits, input_ids):
    config = moe.topk.topk_config
    scores = F.softplus(logits.float()).sqrt()
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
    return StandardTopKOutput(weights.float(), indices.int(), logits)
