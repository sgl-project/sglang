"""Muse Spark MoE routing front on ROCm: one kernel sequence for the FP32 router and the BF16 pooler.

At decode / verify widths (up to 64 rows) the router logits, the sigmoid top-8 routing, the pooler projection
and the pre-expert RMSNorm share one epilogue (``kernels/ops/moe/router_pooler_topk``) instead of separate
router GEMM, routing, pooler GEMM and norm launches (``SGLANG_MUSE_SHARED_ROUTER_POOLER``). Verify batches of
8 / 16 / 24 rows run the router and the pooler as one GEMM over a packed weight prepared after loading
(``SGLANG_MUSE_JOINT_ROUTER_POOLER``).
"""

from __future__ import annotations

import logging

import torch

from sglang.kernels.ops.moe.router_pooler_topk import (
    FP8_JOINT_MAX_M,
    FP8_JOINT_MIN_M,
    joint_router_pooler_applies,
    joint_router_pooler_fp8_applies,
    joint_router_pooler_topk,
    joint_router_pooler_topk_fp8,
    pack_joint_router_pooler_weight,
    quantize_joint_pooler,
    router_pooler_topk,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

SHARED_ROUTER_POOLER = envs.SGLANG_MUSE_SHARED_ROUTER_POOLER.get()
JOINT_ROUTER_POOLER = envs.SGLANG_MUSE_JOINT_ROUTER_POOLER.get()
# The joint form's pooler tiles run on an FP8 copy of the pooler when the boundary kernel hands over the FP8
# activation (same switch as the FP8 dense GEMMs).
JOINT_POOLER_FP8 = JOINT_ROUTER_POOLER and envs.SGLANG_MUSE_DENSE_FP8.get()
_JOINT_ROWS = (8, 16, 24)
_logged: set = set()


def can_share_router_pooler(moe, x: torch.Tensor) -> bool:
    """The Muse Spark TP8 routing front: 8192 hidden, 256 experts top-8 sigmoid, 4096-wide pooler + norm."""
    return 0 < x.shape[0] <= 64 and _routing_front_matches(moe, x)


def can_route_fp8_joint(moe, x: torch.Tensor, x_fp8) -> bool:
    """The FP8 joint form's wider admission: the same routing front for up to ``FP8_JOINT_MAX_M`` rows when the
    boundary kernel handed over the FP8 activation and the FP8 pooler copy exists."""
    return (
        JOINT_POOLER_FP8
        and x_fp8 is not None
        and FP8_JOINT_MIN_M <= x.shape[0] <= FP8_JOINT_MAX_M
        and moe.router_pooler_fp8_weight is not None
        and _routing_front_matches(moe, x)
    )


def _routing_front_matches(moe, x: torch.Tensor) -> bool:
    op = moe._routing_op()
    return (
        x.shape[1] == 8192
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


def route_with_shared_pooler(moe, x: torch.Tensor, quant_out=None, x_fp8=None):
    """(top-8 scores, expert ids, normalized expert input, MXFP4 input or None) for ``x``; requires
    ``can_share_router_pooler``. ``quant_out`` receives the compact A4 input on the shared (not joint) path.
    ``x_fp8`` = (per-row FP8 copy of ``x``, scales) from the boundary kernel for the FP8 pooler tiles."""
    m = x.shape[0]
    norm_weight = moe.pre_expert_norm._get_effective_weight()
    if (
        JOINT_POOLER_FP8
        and x_fp8 is not None
        and joint_router_pooler_fp8_applies(
            x_fp8[0], moe.router_pooler_packed_weight, moe.router_pooler_fp8_weight
        )
    ):
        key = ("joint_fp8", moe.layer_idx, m)
        if key not in _logged:
            _logged.add(key)
            logger.info(
                "Muse joint router/pooler (FP8 pooler) active: layer=%d rows=%d",
                moe.layer_idx,
                m,
            )
        return *joint_router_pooler_topk_fp8(
            x_fp8[0],
            x_fp8[1],
            x,
            moe.router_pooler_packed_weight,
            (moe.router_pooler_fp8_weight, moe.router_pooler_fp8_scale),
            moe.expert_bias,
            norm_weight,
            float(moe.router_multiplier),
            float(moe.route_scale),
            float(moe.pre_expert_norm.eps),
            quant_out=quant_out,
        ), quant_out
    if JOINT_ROUTER_POOLER and (
        m in _JOINT_ROWS
        or (
            envs.SGLANG_MUSE_DENSE_TILED.get()
            and joint_router_pooler_applies(x, moe.router_pooler_packed_weight)
        )
    ):
        packed = moe.router_pooler_packed_weight
        assert moe.pre_expert_proj.weight.data_ptr() == packed.data_ptr()
        key = ("joint", moe.layer_idx, m)
        if key not in _logged:
            _logged.add(key)
            logger.info(
                "Muse joint router/pooler active: layer=%d rows=%d", moe.layer_idx, m
            )
        return *joint_router_pooler_topk(
            x,
            packed,
            moe.expert_bias,
            norm_weight,
            float(moe.router_multiplier),
            float(moe.route_scale),
            float(moe.pre_expert_norm.eps),
            quant_out=quant_out,
        ), quant_out
    key = "shared" if m <= 16 else "shared_completed"
    if key not in _logged:
        _logged.add(key)
        logger.info(
            "Muse fast path: %s, rows=%d",
            (
                "shared router/preproject epilogue"
                if m <= 16
                else "shared router/norm with hipBLASLt pooler"
            ),
            m,
        )
    return *router_pooler_topk(
        x,
        moe.router.weight,
        moe.pre_expert_proj.weight,
        moe.expert_bias,
        norm_weight,
        float(moe.router_multiplier),
        float(moe.route_scale),
        float(moe.pre_expert_norm.eps),
        pooler_tiled=getattr(moe.pre_expert_proj, "small_m_tiled_weight", None),
        pooler_rowmajor=envs.SGLANG_MUSE_DENSE_TILED.get(),
        quant_out=quant_out,
    ), quant_out


@torch.no_grad()
def prepare_joint_router_pooler(moe) -> None:
    """Pack ``[pooler; router high; router low]`` once after loading; the pooler weight aliases the prefix."""
    router, pooler = moe.router.weight, moe.pre_expert_proj.weight
    assert router.is_cuda and pooler.is_cuda
    assert router.shape == (8192, 256) and router.dtype == torch.float32
    assert pooler.shape == (4096, 8192) and pooler.dtype == torch.bfloat16
    packed = getattr(moe, "router_pooler_packed_weight", None)
    if packed is None:
        # Runs in post_load_weights, before warmup or graph capture. The pooler prefix holds exactly the
        # original BF16 values and stays available to the plain nn.Linear path at larger M.
        packed = pack_joint_router_pooler_weight(router, pooler)
        pooler.data = packed[:4096]
        moe.register_buffer("router_pooler_packed_weight", packed, persistent=False)
        fp8_weight, fp8_scale = (
            quantize_joint_pooler(packed) if JOINT_POOLER_FP8 else (None, None)
        )
        moe.register_buffer("router_pooler_fp8_weight", fp8_weight, persistent=False)
        moe.register_buffer("router_pooler_fp8_scale", fp8_scale, persistent=False)
    else:
        # Later checkpoint writes already land in the aliased prefix; refresh only the router parts,
        # without moving any captured addresses.
        assert pooler.data_ptr() == packed.data_ptr()
        high = router.T.to(torch.bfloat16)
        packed[4096:4352].copy_(high)
        packed[4352:].copy_((router.T - high.float()).to(torch.bfloat16))
        if moe.router_pooler_fp8_weight is not None:
            fp8_weight, fp8_scale = quantize_joint_pooler(packed)
            moe.router_pooler_fp8_weight.copy_(fp8_weight)
            moe.router_pooler_fp8_scale.copy_(fp8_scale)
    assert pooler.is_contiguous() and pooler.data_ptr() == packed.data_ptr()
