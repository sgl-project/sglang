from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import NamedTuple, Optional

import torch

from sglang.kernels.ops.moe.mega_gate import bf16_mega_gate
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _post_process_topk_ids,
    apply_routing_simulation,
    routing_simulation_enabled,
)
from sglang.srt.models.deepseek_common.utils import _device_sm
from sglang.srt.runtime_context import get_exec, get_forward, get_parallel
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

# DeepGEMM bf16_mega_gate scores one token block per launch.
_MEGA_GATE_MAX_ROWS = 2048
# Per-forward cache of the gate's row masks, installed by mega_gate_forward_scope.
_ROW_MASKS_ATTR = "_mega_gate_row_masks"


class MegaGateRouting(NamedTuple):
    topk_weights: Optional[torch.Tensor]
    topk_ids: Optional[torch.Tensor]
    routed_scaling_factor: float
    in_buffer: bool


def mega_gate_supported(moe) -> bool:
    if (
        _device_sm // 10 != 10
        or moe.config.model_type != "deepseek_v41"
        or moe.topk.enable_waterfill
        or moe.num_fused_shared_experts
        or get_exec().deterministic.enable_deterministic_inference
    ):
        return False
    weight = moe.gate.weight
    num_experts, hidden = weight.shape
    top_k = moe.config.num_experts_per_tok
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or hidden % 256 != 0
        or not 0 < num_experts <= 512
        or num_experts % 4 != 0
        or not 0 < top_k <= min(32, num_experts)
    ):
        return False
    bias_vl = None if moe.is_hash else moe.gate.e_score_correction_bias_vl
    for bias in (moe.gate.e_score_correction_bias, bias_vl):
        if bias is not None and (
            bias.dtype != torch.float32
            or not bias.is_contiguous()
            or bias.shape != (num_experts,)
        ):
            return False
    if moe.is_hash:
        return (
            moe.topk.score_func == "sqrtsoftplus"
            and moe.topk.num_shared_topk_slots == 0
            and moe.topk.topk == top_k
        )
    config = moe.topk.topk_config
    return (
        config.scoring_func == "sqrtsoftplus"
        and config.sqrtsoftplus_log1p
        and config.renormalize
        and config.num_shared_topk_slots == 0
        and config.top_k == top_k
        and not (top_k == 1 and bias_vl is not None)
        and not config.use_grouped_topk
        and config.custom_routing_function is None
        and not config.torch_native
    )


def should_use_mega_gate(moe, hidden_states: torch.Tensor, forward_batch) -> bool:
    if not envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.get() or forward_batch is None:
        return False
    # Prefill graph capture clears is_extend_in_batch, so gate on the mode itself.
    mode = forward_batch.forward_mode
    return (
        (mode.is_decode() or mode.is_target_verify())
        and moe.mega_gate_supported
        and 0 < hidden_states.shape[0] <= _MEGA_GATE_MAX_ROWS
        and not get_forward().sp_active
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
    )


def _captures_routed_experts(moe) -> bool:
    return (
        not moe.is_hash
        and moe.topk.topk_config.allow_routed_experts_capture
        and get_global_experts_capturer() is not None
    )


@functools.lru_cache
def _ones(num_experts: int, device: torch.device) -> torch.Tensor:
    return torch.ones(num_experts, dtype=torch.int32, device=device)


def _physical_dispatch_maps(info) -> tuple[torch.Tensor, torch.Tensor]:
    if info.ep_dispatch_algorithm == "static":
        rank_map = info.partial_logical_to_rank_dispatch_physical_map
        return rank_map.to(torch.int32)[:, None], _ones(len(rank_map), rank_map.device)
    return (
        info.partial_logical_to_all_physical_map.to(torch.int32),
        info.partial_logical_to_all_physical_map_num_valid.to(torch.int32),
    )


@contextmanager
def mega_gate_forward_scope(forward_batch):
    """One model forward: the gate's row masks are built once and shared by its
    MoE layers. Each forward (eager, warmup or graph capture) builds its own."""
    if forward_batch is None or not envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.get():
        yield
        return
    try:
        setattr(forward_batch, _ROW_MASKS_ATTR, {})
    except (AttributeError, TypeError):
        yield
        return
    try:
        yield
    finally:
        try:
            delattr(forward_batch, _ROW_MASKS_ATTR)
        except AttributeError:
            pass


def _gate_row_masks(moe, num_tokens, bound, image_input_ids, device, forward_batch):
    """(valid-row mask, image-token mask) for the gate, built once per forward."""
    cache = getattr(forward_batch, _ROW_MASKS_ATTR, None)
    key = (
        num_tokens,
        id(bound) if bound is not None else None,
        id(image_input_ids) if image_input_ids is not None else None,
    )
    if isinstance(cache, dict):
        hit = cache.get(key)
        # The cached inputs are held, so their ids cannot be reused this forward.
        if hit is not None and hit[0] is bound and hit[1] is image_input_ids:
            return hit[2], hit[3]
    valid = None
    if bound is not None:
        valid = (
            torch.arange(num_tokens, device=device, dtype=torch.int32)
            < bound.reshape(-1)[:1]
        )
    image = None
    if image_input_ids is not None:
        image = (image_input_ids[:num_tokens] == moe.config.image_token_id).contiguous()
    if isinstance(cache, dict):
        cache[key] = (bound, image_input_ids, valid, image)
    return valid, image


def run_mega_gate(
    moe, hidden_states, input_ids, buf=None, forward_batch=None
) -> Optional[MegaGateRouting]:
    bound, dispatch_info = moe.topk.routing_inputs()
    if dispatch_info is not None and dispatch_info.ep_dispatch_algorithm in (
        "lp",
        "fake",
    ):
        return None
    if moe.is_hash and input_ids is None:
        return None
    num_tokens = hidden_states.shape[0]
    use_image = (
        not moe.is_hash
        and moe.gate.e_score_correction_bias_vl is not None
        and input_ids is not None
    )
    valid, image = _gate_row_masks(
        moe,
        num_tokens,
        bound,
        input_ids if use_image else None,
        hidden_states.device,
        forward_batch,
    )
    hash_ids = None
    if moe.is_hash:
        token_ids = input_ids[:num_tokens]
        if valid is not None:
            token_ids = torch.where(valid, token_ids, 0)
        hash_ids = moe.topk.tid2eid[token_ids].to(torch.int64)

    routed_scaling_factor = float(moe.routed_scaling_factor)
    fused_in_topk = moe.routed_scaling_factor_fused_in_topk
    # The benchmark override writes logical ids, so with EPLB it must run before the remap.
    in_buffer = (
        buf is not None
        and not _captures_routed_experts(moe)
        and not (dispatch_info is not None and routing_simulation_enabled())
    )
    to_physical_map = logical_count = None
    if in_buffer and dispatch_info is not None:
        # NOTE(kpham-sgl): MegaMoE reads physical expert slots from this buffer, so
        # under EPLB the gate must map logical ids in-kernel to stay on this path.
        to_physical_map, logical_count = _physical_dispatch_maps(dispatch_info)
    weights, ids = bf16_mega_gate(
        hidden_states,
        moe.gate.weight,
        moe.config.num_experts_per_tok,
        routed_scaling_factor=(
            routed_scaling_factor if in_buffer or fused_in_topk else 1.0
        ),
        ep_rank=get_parallel().moe_ep_rank,
        bias=moe.gate.e_score_correction_bias,
        image_bias=moe.gate.e_score_correction_bias_vl if image is not None else None,
        image_token_mask=image,
        valid_token_mask=valid,
        hash_topk_ids=hash_ids,
        to_physical_map=to_physical_map,
        logical_count=logical_count,
        out=(
            (buf.topk_idx[:num_tokens], buf.topk_weights[:num_tokens])
            if in_buffer
            else None
        ),
    )
    if not moe.is_hash and routing_simulation_enabled():
        top_k = ids.shape[1]
        apply_routing_simulation(
            ids,
            weights,
            moe.gate.weight.shape[0],
            num_routed_topk=top_k,
            top_k=top_k,
            layer_id=moe.layer_id,
            num_token_non_padded=bound,
            weight_scale=(
                routed_scaling_factor if in_buffer and not fused_in_topk else 1.0
            ),
        )
    if in_buffer:
        get_global_expert_distribution_recorder().on_select_experts(
            topk_ids=buf.topk_idx[:num_tokens]
        )
        return MegaGateRouting(None, None, 1.0, True)

    if moe.is_hash:
        config = TopKConfig(
            top_k=moe.topk.topk,
            routed_scaling_factor=moe.topk.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=moe.topk.apply_routed_scaling_factor_on_output,
            allow_routed_experts_capture=False,
        )
    else:
        config = moe.topk.topk_config
    ids, weights, recorder_ids = _post_process_topk_ids(
        ids,
        weights,
        config,
        hidden_states.new_empty((0, moe.gate.weight.shape[0])),
        moe.layer_id,
        num_token_non_padded=bound,
        expert_location_dispatch_info=dispatch_info,
        padded_rows_masked=True,
    )
    if recorder_ids is not None:
        get_global_expert_distribution_recorder().on_select_experts(
            topk_ids=recorder_ids
        )
    return MegaGateRouting(
        weights, ids, 1.0 if fused_in_topk else routed_scaling_factor, False
    )
