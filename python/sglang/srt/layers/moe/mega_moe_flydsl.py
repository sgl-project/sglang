# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Aiter MegaMoEV2 adapter for the SGLang MegaMoE hooks."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.moe.fill_padded_rows import _fill_padded_rows
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


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


def _sync_tokens(
    forward_batch: ForwardBatch | None,
    *,
    local_tokens: int,
    mtpr: int,
) -> int:
    if not envs.SGLANG_AITER_MEGA_RANK_SYNC.get():
        return local_tokens

    if forward_batch is not None and (
        forward_batch.forward_mode.is_idle()
        or (
            (original_mode := getattr(forward_batch, "_original_forward_mode", None))
            is not None
            and original_mode.is_idle()
        )
    ):
        return local_tokens

    if forward_batch is not None and forward_batch.is_extend_in_batch:
        sync_tokens = mtpr
    else:
        sync_tokens = (
            forward_batch.mega_moe_sync_tokens if forward_batch is not None else None
        )
    if sync_tokens is None:
        global_tokens = get_dp_global_num_tokens()
        sync_tokens = max(global_tokens) if global_tokens else local_tokens

    sync_tokens = max(int(sync_tokens), local_tokens)
    if sync_tokens > mtpr:
        raise ValueError(f"Aiter MegaMoE sync_tokens={sync_tokens} exceeds MTPR={mtpr}")
    return sync_tokens


def _forward_with_sync_config(
    mega,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    config_tokens: int,
):
    required = (
        "_select_config",
        "quantize",
        "_run_fused_stage1",
        "_run_stage2",
    )
    missing = [name for name in required if not hasattr(mega, name)]
    if missing:
        raise RuntimeError(
            "Aiter MegaMoE config contract changed; missing " + ", ".join(missing)
        )

    run_tokens = int(x.shape[0])
    config = mega._select_config(config_tokens)
    x_q, scales = mega.quantize(x)
    mega._run_fused_stage1(
        x_q,
        topk_weights,
        scales,
        topk_ids,
        stream=None,
        config=config.stage1,
    )
    return mega._run_stage2(run_tokens, None, False, config)


def _ep_rank_world():
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    group = get_moe_ep_group().device_group
    return torch.distributed.get_rank(group), torch.distributed.get_world_size(group)


def _ensure_mori_shmem() -> None:
    global _MORI_SHMEM_READY
    if _MORI_SHMEM_READY:
        return

    import mori.shmem

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    group_name = "megamoe_aiter"
    cpu_group = get_moe_ep_group().cpu_group
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
    layer._mega_w1, layer._mega_w1_scale = shuffle_existing(
        layer.w13_weight.data, w13_scale.data, gate_up=True
    )
    layer._mega_w2, layer._mega_w2_scale = shuffle_existing(
        layer.w2_weight.data, w2_scale.data, gate_up=False
    )

    experts = layer.w13_weight.shape[0]
    layer.w13_weight.data = layer._mega_w1.view(experts, -1)
    w13_scale.data = layer._mega_w1_scale.view(experts, -1)
    layer.w2_weight.data = layer._mega_w2.view(experts, -1)
    w2_scale.data = layer._mega_w2_scale.view(experts, -1)
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
    _ensure_mori_shmem()
    backend = _import_aiter_megamoe()
    rank, world = _ep_rank_world()
    key = (
        rank,
        world,
        model_dim,
        inter_dim,
        experts,
        topk,
        quant,
        swiglu_limit,
        mtpr,
    )
    mega = _MEGA_MOE_INSTANCE.get(key)
    if mega is None:
        mega = backend.op(
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
            max_tok_per_rank=mtpr,
            swiglu_limit=swiglu_limit,
        )
        _MEGA_MOE_INSTANCE[key] = mega
    return mega


def _swap_layer_weights(mega, layer) -> None:
    mega._s1_w1 = layer._mega_w1
    mega._s1_w1_scale = layer._mega_w1_scale
    mega.w2 = layer._mega_w2
    mega.w2_scale = layer._mega_w2_scale


def should_use_mega_moe(moe: DeepseekV2MoE, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if get_is_capture_mode():
        return True
    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens and not is_dsa_enable_prefill_cp():
        max_tokens = max(global_num_tokens)
    else:
        max_tokens = hidden_states.shape[0]
    return max_tokens <= _mtpr()


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch | None = None,
    input_ids_global: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    overlap = _should_overlap_shared_and_routed(moe, num_tokens)
    if overlap:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        stream_context = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        stream_context = nullcontext()

    with stream_context:
        output = _run_mega_routed(moe, hidden_states, forward_batch, input_ids_global)
    if overlap:
        current_stream.wait_stream(moe.alt_stream)
    if shared_output is not None:
        output.add_(shared_output)
    return output


def _should_overlap_shared_and_routed(
    moe: DeepseekV2MoE,
    num_tokens: int,
) -> bool:
    if envs.SGLANG_AITER_MEGA_RANK_SYNC.get():
        return False
    return (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )


def _run_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch | None,
    input_ids_global: torch.Tensor | None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    hidden_size = moe.config.hidden_size
    topk = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    experts = moe.experts.num_experts
    if num_tokens:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id
            ),
            **topk_kwargs,
        )
        x_in = hidden_states
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights.to(torch.float32)
        if forward_batch is not None and forward_batch.num_token_non_padded is not None:
            _fill_padded_rows(
                topk_ids,
                forward_batch.num_token_non_padded,
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

    selected_mtpr = _mtpr()
    sync_tokens = _sync_tokens(
        forward_batch,
        local_tokens=int(x_in.shape[0]),
        mtpr=selected_mtpr,
    )
    x_in = x_in.contiguous()
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    assert (
        x_in.shape[0] <= selected_mtpr
    ), f"Aiter MegaMoE local tokens {x_in.shape[0]} exceed MTPR {selected_mtpr}"
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

    if envs.SGLANG_AITER_MEGA_RANK_SYNC.get():
        output = _forward_with_sync_config(
            mega,
            x_in,
            topk_weights,
            topk_ids,
            config_tokens=sync_tokens,
        )[:num_tokens]
    else:
        output = mega.forward(x_in, topk_weights, topk_ids, slice_output=False)[
            :num_tokens
        ]

    from sglang.srt.models.deepseek_common.utils import _use_aiter

    if not (moe.experts.should_fuse_routed_scaling_factor_in_topk or _use_aiter):
        output.mul_(moe.routed_scaling_factor)
    return output
