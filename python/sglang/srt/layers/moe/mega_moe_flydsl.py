# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Aiter MegaMoEV2 adapter for the SGLang MegaMoE hooks."""

from __future__ import annotations

import functools
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

from sglang.kernels.ops.moe.fill_padded_rows import _fill_padded_rows
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_is_extend_in_batch,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_MORI_SHMEM_READY = False
_MEGA_MOE_INSTANCE: dict = {}


def _import_aiter_megamoe():
    try:
        from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
    except ImportError as exc:
        raise ImportError(
            "Aiter MegaMoE requires a package containing MegaMoEV2 and the "
            f"A16W4 shuffle operators. Original import error: {exc}"
        ) from exc

    return SimpleNamespace(
        op=MegaMoEV2,
        shuffle_weight=shuffle_weight_a16w4,
        shuffle_scale=shuffle_scale_a16w4,
    )


def _mtpr() -> int:
    mtpr = int(envs.SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR.get())
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(
            f"SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR={mtpr} must be a positive power of two"
        )
    return mtpr


def _decode_mtpr() -> int:
    decode_mtpr = int(envs.SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR.get())
    if decode_mtpr == 0:
        return 0
    mtpr = _mtpr()
    if decode_mtpr < 0 or decode_mtpr > mtpr or decode_mtpr & (decode_mtpr - 1):
        raise ValueError(
            f"SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR={decode_mtpr} must be 0 (off) "
            "or a power of two no larger than "
            f"SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR={mtpr}"
        )
    if decode_mtpr == mtpr:
        _warn_decode_mtpr_merged(mtpr)
        return 0
    return decode_mtpr


@functools.cache
def _warn_decode_mtpr_merged(mtpr: int) -> None:
    logger.warning(
        "SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR equals "
        "SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=%d; running one MegaMoE capacity",
        mtpr,
    )


def _capacities() -> tuple[int, ...]:
    decode_mtpr = _decode_mtpr()
    return (decode_mtpr, _mtpr()) if decode_mtpr else (_mtpr(),)


def _is_decode_step_mode(forward_batch: ForwardBatch | None) -> bool:
    # IDLE also runs in prefill steps; _select_mtpr tells the two apart.
    if forward_batch is None:
        return False
    mode = forward_batch.forward_mode
    return (
        mode.is_decode_or_idle() or mode.is_target_verify() or mode.is_draft_extend_v2()
    )


def _select_mtpr(forward_batch: ForwardBatch | None) -> int:
    """The capacity this forward runs with; every EP rank must pick the same one.

    Local token counts, graph buckets and graph-vs-eager can differ across
    ranks (DP counts are gathered only under require_mlp_tp_gather), so only
    the step kind decides.
    """
    mtpr = _mtpr()
    decode_mtpr = _decode_mtpr()
    if not decode_mtpr or not _is_decode_step_mode(forward_batch):
        return mtpr
    if get_is_capture_mode():
        # Decode-step graphs replay only when no DP rank extends
        # (can_run_decode_cuda_graph is min-reduced over the group).
        return decode_mtpr
    # is_extend_in_batch is max-reduced over the DP group.
    return mtpr if get_is_extend_in_batch() else decode_mtpr


def _ep_rank_world():
    parallel = get_parallel()
    return parallel.moe_ep_rank, parallel.moe_ep_size


def _ensure_mori_shmem() -> None:
    global _MORI_SHMEM_READY
    if _MORI_SHMEM_READY:
        return

    import mori.shmem

    group_name = "megamoe_aiter"
    cpu_group = get_parallel().moe_ep_group.cpu_group
    try:
        torch._C._distributed_c10d._register_process_group(group_name, cpu_group)
    except Exception as exc:
        if "already registered" not in str(exc):
            raise
    mori.shmem.shmem_torch_process_group_init(group_name)
    _MORI_SHMEM_READY = True


def build_mega_moe_experts_weights(layer) -> None:
    if getattr(layer, "_mega_moe_weights_built", False):
        return

    backend = _import_aiter_megamoe()

    def scale_param(name):
        scale = getattr(layer, name, None)
        if scale is None:
            scale = getattr(layer, name + "_inv", None)
        if scale is None:
            raise AttributeError(f"Aiter MegaMoE weight scale {name} is missing")
        return scale

    def shuffle_existing(weight, scale, *, gate_up):
        experts = weight.shape[0]
        shuffled_weight = backend.shuffle_weight(weight, 16, gate_up).contiguous()
        shuffled_scale = backend.shuffle_scale(
            scale.view(-1, scale.shape[-1]), experts, gate_up
        ).contiguous()
        return (
            shuffled_weight.view(torch.uint8).contiguous(),
            shuffled_scale.view(torch.uint8).contiguous(),
        )

    w13_scale = scale_param("w13_weight_scale")
    w2_scale = scale_param("w2_weight_scale")
    w13_shape, w13_dtype = layer.w13_weight.shape, layer.w13_weight.dtype
    w2_shape, w2_dtype = layer.w2_weight.shape, layer.w2_weight.dtype
    w13_scale_shape, w13_scale_dtype = w13_scale.shape, w13_scale.dtype
    w2_scale_shape, w2_scale_dtype = w2_scale.shape, w2_scale.dtype
    layer._mega_w1, layer._mega_w1_scale = shuffle_existing(
        layer.w13_weight.data, w13_scale.data, gate_up=True
    )
    layer._mega_w2, layer._mega_w2_scale = shuffle_existing(
        layer.w2_weight.data, w2_scale.data, gate_up=False
    )

    layer.w13_weight.data = layer._mega_w1.view(w13_dtype).view(w13_shape)
    w13_scale.data = layer._mega_w1_scale.view(w13_scale_dtype).view(w13_scale_shape)
    layer.w2_weight.data = layer._mega_w2.view(w2_dtype).view(w2_shape)
    w2_scale.data = layer._mega_w2_scale.view(w2_scale_dtype).view(w2_scale_shape)
    layer._mega_moe_weights_built = True


def _get_or_build_mega_moe(
    layer,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    quant: str,
    swiglu_limit: float,
    mtpr: int,
):
    rank, world = _ep_rank_world()
    shape_key = (rank, world, model_dim, inter_dim, experts, topk, quant, swiglu_limit)
    mega = _MEGA_MOE_INSTANCE.get((*shape_key, mtpr))
    if mega is not None:
        return mega
    _ensure_mori_shmem()
    backend = _import_aiter_megamoe()
    # Construction synchronizes the device and barriers every EP rank, so all
    # capacities are built together on the first (eager) call, in the same order
    # on every rank: a capacity first used later could be inside graph capture.
    for capacity in _capacities():
        key = (*shape_key, capacity)
        if key in _MEGA_MOE_INSTANCE:
            continue
        _MEGA_MOE_INSTANCE[key] = backend.op(
            rank=rank,
            world_size=world,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant=quant,
            w1=layer._mega_w1,
            w1_scale=layer._mega_w1_scale,
            w2=layer._mega_w2,
            w2_scale=layer._mega_w2_scale,
            max_tok_per_rank=capacity,
            swiglu_limit=swiglu_limit,
            small_m_mtprs=(
                (128, 256)
                if (world, model_dim, inter_dim, experts, topk, quant, swiglu_limit)
                == (8, 6144, 2048, 384, 8, "a8w4", 0.0)
                else (256, 512)
            ),
        )
        logger.info("Aiter MegaMoE: built capacity max_tok_per_rank=%d", capacity)
    return _MEGA_MOE_INSTANCE[(*shape_key, mtpr)]


def _swap_layer_weights(mega, layer) -> None:
    mega.bind_layer(
        getattr(layer, "layer_id", 0),
        layer._mega_w1,
        layer._mega_w1_scale,
        layer._mega_w2,
        layer._mega_w2_scale,
    )


def _rank_consistent_max_tokens(hidden_states: torch.Tensor) -> int:
    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens and not is_dsa_enable_prefill_cp():
        return max(global_num_tokens)
    return hidden_states.shape[0]


def should_use_mega_moe(moe: Any, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if getattr(moe, "_use_target_mega_moe", False):
        return _rank_consistent_max_tokens(hidden_states) <= 256
    if get_is_capture_mode():
        return True
    return _rank_consistent_max_tokens(hidden_states) <= _mtpr()


def forward_mega_moe(
    moe: Any,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch | None = None,
    input_ids_global: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    shared_forward = getattr(moe, "_forward_shared_experts", None)
    overlap = shared_forward is not None and _should_overlap_shared_and_routed(
        moe, num_tokens
    )
    if overlap:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = shared_forward(hidden_states)
        stream_context = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = (
            shared_forward(hidden_states) if shared_forward is not None else None
        )
        stream_context = nullcontext()

    with stream_context:
        output = _run_mega_routed(moe, hidden_states, forward_batch, input_ids_global)
    if overlap:
        current_stream.wait_stream(moe.alt_stream)
    if shared_output is not None:
        output.add_(shared_output)
    return output


def _should_overlap_shared_and_routed(
    moe: Any,
    num_tokens: int,
) -> bool:
    if envs.SGLANG_AITER_MEGA_RANK_SYNC.get():
        return False
    return (
        getattr(moe, "alt_stream", None) is not None
        and getattr(moe, "num_fused_shared_experts", 0) == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )


def _run_mega_routed(
    moe: Any,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch | None,
    input_ids_global: torch.Tensor | None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    hidden_size = moe.config.hidden_size
    topk = moe.config.num_experts_per_tok + getattr(moe, "num_fused_shared_experts", 0)
    experts = moe.experts.num_experts
    if num_tokens:
        router_forward = getattr(moe, "_mega_moe_router_logits", None)
        router_logits = (
            router_forward(hidden_states, forward_batch)
            if router_forward is not None
            else moe.gate(hidden_states, forward_batch=forward_batch)
        )
        topk_kwargs = (
            {"input_ids": input_ids_global} if getattr(moe, "is_hash", False) else {}
        )
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            **topk_kwargs,
        )
        x_in = hidden_states
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights.to(torch.float32)
        num_token_non_padded, _ = moe.topk.routing_inputs()
        if num_token_non_padded is not None:
            _fill_padded_rows(
                topk_ids,
                num_token_non_padded,
                experts,
            )
    else:
        rank_sync = envs.SGLANG_AITER_MEGA_RANK_SYNC.get()
        x_in = (
            hidden_states.new_ones((1, hidden_size))
            if rank_sync
            else hidden_states.new_zeros((1, hidden_size))
        )
        if rank_sync:
            ep_rank, ep_world = _ep_rank_world()
            expert_base = ep_rank * (experts // ep_world)
            topk_ids = (
                expert_base
                + torch.arange(topk, device=hidden_states.device, dtype=torch.int32)
            ).unsqueeze(0)
            topk_weights = hidden_states.new_full(
                (1, topk), 1.0 / topk, dtype=torch.float32
            )
        else:
            topk_ids = torch.full(
                (1, topk),
                experts,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            topk_weights = hidden_states.new_zeros((1, topk), dtype=torch.float32)

    selected_mtpr = _select_mtpr(forward_batch)
    x_in = x_in.contiguous()
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    assert x_in.shape[0] <= selected_mtpr, (
        f"Aiter MegaMoE local tokens {x_in.shape[0]} exceed MTPR {selected_mtpr}"
    )
    mega = _get_or_build_mega_moe(
        moe.experts,
        model_dim=hidden_size,
        inter_dim=moe.config.moe_intermediate_size,
        experts=experts,
        topk=topk,
        quant=envs.SGLANG_AMD_FLYDSL_MEGA_QUANT.get() or "a8w4",
        swiglu_limit=float(getattr(moe.config, "swiglu_limit", 0.0) or 0.0),
        mtpr=selected_mtpr,
    )
    _swap_layer_weights(mega, moe.experts)

    if getattr(moe, "_use_target_mega_moe", False):
        output = mega.forward_small_m(
            x_in,
            topk_weights,
            topk_ids,
            slice_output=False,
            max_tokens=_rank_consistent_max_tokens(hidden_states),
        )[:num_tokens]
    else:
        output = mega.forward(x_in, topk_weights, topk_ids, slice_output=False)[:num_tokens]

    if not getattr(moe, "routed_scaling_factor_fused_in_topk", True):
        output.mul_(getattr(moe, "routed_scaling_factor", 1.0))
    return output
