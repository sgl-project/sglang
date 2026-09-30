# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""aiter MegaMoEV2 (gfx950) for Kimi-K3 routed experts on ROCm.

Replaces MoRI dispatch -> aiter fused_moe -> MoRI combine with aiter's fused
kernel pair (dispatch + GEMM1 + SiTUv2, GEMM2 + combine) over MoRI shmem. It runs
under ``--moe-a2a-backend mori`` so the K3 SP-MoE / EP plumbing is unchanged;
only the routed-expert call is swapped. Enabled with
``SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1``.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var, is_hip

_MORI_SHMEM_READY = False
_MEGA_MOE_INSTANCE: dict = {}


def use_aiter_mega_moe() -> bool:
    return is_hip() and get_bool_env_var("SGLANG_AMD_USE_FLYDSL_MEGA_MOE")


def _mtpr() -> int:
    mtpr = int(os.environ.get("SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR", "8192"))
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(
            f"SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR={mtpr} must be a positive power of two"
        )
    return mtpr


def _ensure_mori_shmem() -> None:
    # Same group name and first-registration rule as moriep.init_mori_op, so
    # whichever of the two runs first owns the one shmem init.
    global _MORI_SHMEM_READY
    if _MORI_SHMEM_READY:
        return
    import mori

    group_name = "mori"
    cpu_group = get_parallel().moe_ep_group.cpu_group
    try:
        torch._C._distributed_c10d._register_process_group(group_name, cpu_group)
    except Exception as exc:
        if "already registered" not in str(exc):
            raise
    else:
        mori.shmem.shmem_torch_process_group_init(group_name)
    _MORI_SHMEM_READY = True


def _layer_weights(experts):
    """Reuse the aiter A8W4 SiTU layout Mxfp4MoEMethod already produced
    (shuffle_weight_a16w4 / shuffle_scale_a16w4, gate_up=True for w13)."""
    cached = getattr(experts, "_aiter_mega_weights", None)
    if cached is not None:
        return cached
    method = experts.quant_method
    if getattr(method, "hidden_pad", 0) or getattr(method, "intermediate_pad", 0):
        raise ValueError(
            "aiter MegaMoE needs unpadded expert weights, got "
            f"hidden_pad={method.hidden_pad} intermediate_pad={method.intermediate_pad}"
        )
    if not getattr(experts.w13_weight, "is_shuffled", False):
        raise ValueError("aiter MegaMoE needs the aiter-shuffled MXFP4 expert layout")
    cached = tuple(
        t.data.contiguous().view(torch.uint8)
        for t in (
            experts.w13_weight,
            experts.w13_weight_scale,
            experts.w2_weight,
            experts.w2_weight_scale,
        )
    )
    experts._aiter_mega_weights = cached
    return cached


def _get_mega_moe(experts, weights, *, model_dim, inter_dim, topk, situ_beta,
                  situ_linear_beta):
    _ensure_mori_shmem()
    from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

    parallel = get_parallel()
    rank, world = parallel.moe_ep_rank, parallel.moe_ep_size
    quant = os.environ.get("SGLANG_AMD_FLYDSL_MEGA_QUANT") or "a8w4"
    mtpr = _mtpr()
    key = (rank, world, model_dim, inter_dim, experts.num_experts, topk, quant,
           situ_beta, situ_linear_beta, mtpr)
    mega = _MEGA_MOE_INSTANCE.get(key)
    if mega is None:
        w1, w1_scale, w2, w2_scale = weights
        mega = MegaMoEV2(
            rank=rank,
            world_size=world,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts.num_experts,
            topk=topk,
            quant=quant,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=mtpr,
            act="situv2",
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
        _MEGA_MOE_INSTANCE[key] = mega
    return mega


def forward_routed_experts(experts, routed_input, topk_output, *, model_dim,
                           inter_dim, topk, situ_beta, situ_linear_beta):
    """Semantically ``experts(routed_input, topk_output)`` on the MoRI a2a path:
    local rows in, combined local rows out."""
    weights = _layer_weights(experts)
    num_tokens = routed_input.shape[0]
    if num_tokens:
        x = routed_input.to(torch.bfloat16).contiguous()
        topk_ids = topk_output.topk_ids.to(torch.int32).contiguous()
        topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()
    else:
        # Every EP rank has to enter the collective kernels; one row routed to
        # the out-of-range expert id contributes nothing.
        x = routed_input.new_zeros((1, model_dim), dtype=torch.bfloat16)
        topk_ids = torch.full(
            (1, topk), experts.num_experts, dtype=torch.int32, device=x.device
        )
        topk_weights = torch.zeros((1, topk), dtype=torch.float32, device=x.device)

    mega = _get_mega_moe(
        experts,
        weights,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    # One MegaMoEV2 (and its shmem buffers) serves every layer; only the
    # expert weights differ per layer.
    mega._s1_w1, mega._s1_w1_scale, mega.w2, mega.w2_scale = weights
    return mega.forward(x, topk_weights, topk_ids, slice_output=False)[:num_tokens]
