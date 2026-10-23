import torch
import torch.nn.functional as F

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import transform_select_experts_inputs
from sglang.srt.layers.moe.topk import (
    _RENORMALIZE_SUM_EPSILON,
    StandardTopKOutput,
    StandardTopKOutputPacked,
    _mask_topk_ids_padded_region,
    _post_process_topk_ids,
    _zero_topk_weights_padded_region,
    apply_routing_simulation,
    routing_simulation_enabled,
)
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
)
from sglang.srt.utils import is_cuda


def vision_topk(
    moe,
    logits,
    input_ids,
    num_token_non_padded=None,
    expert_location_dispatch_info=None,
):
    bound, dispatch_info = moe.topk.routing_inputs()
    if num_token_non_padded is None:
        num_token_non_padded = bound
    if expert_location_dispatch_info is None:
        expert_location_dispatch_info = dispatch_info
    config = moe.topk.topk_config
    logits, correction_bias = transform_select_experts_inputs(
        logits, moe.gate.e_score_correction_bias, expert_location_dispatch_info
    )
    correction_bias_vl = moe.gate.e_score_correction_bias_vl
    if (
        expert_location_dispatch_info is not None
        and expert_location_dispatch_info.ep_dispatch_algorithm == "fake"
    ):
        correction_bias_vl = torch.zeros_like(correction_bias_vl)
    num_shared_topk_slots = config.num_shared_topk_slots
    use_mega_moe = get_moe_a2a_backend().is_megamoe()
    if is_cuda():
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

        # Same admission as _fused_gate_emits_packed_ids: the shared-expert slots
        # rescaled below would rewrite weights after the router.
        packed_topk = None
        if (
            num_shared_topk_slots == 0
            and expert_location_dispatch_info is None
            and not use_mega_moe
            and not routing_simulation_enabled()
            and config.runner_backend is not None
            and config.runner_backend.is_flashinfer_mxfp4()
        ):
            packed_topk = torch.empty(
                (logits.shape[0], config.top_k), dtype=torch.int32, device=logits.device
            )

        weights, indices = moe_fused_gate(
            logits,
            correction_bias,
            topk=config.top_k,
            scoring_func="sqrtsoftplus",
            num_fused_shared_experts=num_shared_topk_slots,
            bias_alt=correction_bias_vl,
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
        bias = correction_bias
    else:
        bias = torch.where(
            (input_ids == moe.config.image_token_id)[:, None],
            correction_bias_vl,
            correction_bias,
        )
    # The shared slots appended below use the same layout as biased_grouped_topk_gpu.
    topk_routed = config.top_k - num_shared_topk_slots
    indices = (scores + bias).topk(topk_routed, dim=-1).indices
    weights = scores.gather(-1, indices)
    routed_sum = weights.sum(-1, keepdim=True, dtype=torch.float32)
    if num_shared_topk_slots:
        shared_ids = logits.shape[-1] + torch.arange(
            num_shared_topk_slots, device=indices.device, dtype=indices.dtype
        )
        indices = torch.cat(
            [indices, shared_ids.expand(indices.shape[0], -1)],
            dim=-1,
        )
        weights = F.pad(weights, (0, num_shared_topk_slots))
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
    if routing_simulation_enabled():
        apply_routing_simulation(
            indices,
            weights,
            logits.shape[1],
            num_routed_topk=config.top_k - config.num_shared_topk_slots,
            top_k=config.top_k,
            layer_id=moe.layer_id,
        )
        padded_rows_masked = False
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
    if recorder_ids is not None:
        get_global_expert_distribution_recorder().on_select_experts(
            topk_ids=recorder_ids
        )
    # Shared-slot remapping can overwrite padded weights; mask last so idle
    # and partially padded DP ranks never dispatch a phantom shared expert.
    if num_token_non_padded is not None and (
        not padded_rows_masked or (use_mega_moe and config.num_shared_topk_slots)
    ):
        _mask_topk_ids_padded_region(indices, num_token_non_padded)
        _zero_topk_weights_padded_region(weights, num_token_non_padded)
    return weights, indices
