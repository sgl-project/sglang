"""Muse router/pooler shape guards and model integration."""

import logging

import torch

from sglang.srt.rocm_muse_spark.muse_router_preproject import router_preproject

_LOGGED = set()


def _note(name, m):
    if name not in _LOGGED:
        logging.getLogger(__name__).info("Muse Day0 fast path: %s, rows=%d", name, m)
        _LOGGED.add(name)


def can_share_router_preproject(moe, x):
    op = moe._routing_op()
    return (
        0 < x.shape[0] <= 64
        and x.shape[1] == 8192
        and x.is_contiguous()
        and x.dtype == torch.bfloat16
        and moe.router_in_fp32
        and moe.router.weight.dtype == torch.float32
        and moe.router.weight.shape == (8192, 256)
        and moe.router.weight.is_contiguous()
        and moe.top_k == 8
        and moe.num_experts == 256
        and moe.pre_expert_proj is not None
        and moe.pre_expert_proj.bias is None
        and moe.pre_expert_proj.weight.shape == (4096, 8192)
        and moe.pre_expert_proj.weight.dtype == torch.bfloat16
        and moe.pre_expert_proj.weight.is_contiguous()
        and moe.pre_expert_norm is not None
        and moe.pre_expert_norm.weight is not None
        and op.router_score_normalization == "sigmoid"
        and op.num_valid_tokens is None
        and op.local_expert_mapping is None
        and op.sol_topk_out is None
    )


def share_router_preproject(moe, x):
    _note(
        "shared router/preproject epilogue"
        if x.shape[0] <= 16
        else "shared router/norm with hipBLASLt pooler",
        x.shape[0],
    )
    return router_preproject(
        x,
        moe.router.weight,
        moe.pre_expert_proj.weight,
        moe.expert_bias,
        moe.pre_expert_norm._get_effective_weight(),
        float(moe.router_multiplier),
        float(moe.route_scale),
        float(moe.pre_expert_norm.eps),
    )
