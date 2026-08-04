# SPDX-License-Identifier: Apache-2.0
"""gfx942 (CDNA3) route for MXFP4 MoE, through aiter's *Triton* kernels.

Why this file exists
--------------------
`AiterRunnerCore.run` calls `aiter.fused_moe`, the FlyDSL/asm dispatcher.  On
gfx942 that aborts inside the activation quantiser::

    [AITER] csrc/kernels/quant_kernels.cu:1984
    fused_dynamic_mx_quant_moe_sort_hip: not support output type: fp4x2

i.e. aiter has no gfx942 a4w4 path.  But aiter *also* ships pure-Triton MXFP4
MoE GEMMs (`aiter/ops/triton/moe/moe_op_mxfp4*.py`) built on `tl.dot_scaled`,
which upcasts MXFP4 to bf16 in registers onto the bf16 MFMA that CDNA3 has had
since day one.  This module routes gfx942 MXFP4 MoE to them.

Two things the installed aiter build must carry:

* `arch_info.is_fp4_avail()` must include `gfx942` (the allowlist was never a
  capability probe).
* `aiter/ops/triton/configs/moe/gfx942-MOE-MX_FP4.json` must exist.  Without it
  `get_optimal_moe_config` warns and returns a 256x256x64 tile that does not
  compile on CDNA3 (64 KB LDS vs CDNA4's 160 KB).

Scope guard: everything here is behind `use_triton_mxfp4_moe()`, which is False
on any arch but gfx942 and on any build where the Triton kernels do not import.
No other MoE path changes.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_ENV = "SGLANG_AITER_MXFP4_TRITON"


@functools.lru_cache(maxsize=1)
def _arch() -> str:
    try:
        name = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001
        return ""
    return name.split(":")[0]


@functools.lru_cache(maxsize=1)
def _kernels_importable() -> bool:
    try:
        import aiter.ops.triton.moe.moe_align_block_size  # noqa: F401
        import aiter.ops.triton.moe.moe_op_mxfp4  # noqa: F401

        # The SiTU variant lives here rather than in aiter: it is an ordinary
        # Triton kernel needing only aiter's public helpers, so aiter requires
        # no source modification. See mxfp4_situ_fused.py, which also carries
        # the `N // 2` fix for the EP zero-fill bug still present upstream in
        # aiter/ops/triton/_triton_kernels/moe/moe_op_mxfp4_silu_fused.py.
        from sglang.srt.layers.moe.moe_runner import mxfp4_situ_fused  # noqa: F401

        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("aiter Triton MXFP4 MoE kernels unavailable: %s", exc)
        return False


@functools.lru_cache(maxsize=1)
def use_triton_mxfp4_moe() -> bool:
    """True iff MXFP4 MoE should go through aiter's Triton kernels.

    `SGLANG_AITER_MXFP4_TRITON=0/1` forces it either way; otherwise it is
    auto-enabled only on gfx942, where the FlyDSL fp4x2 path does not exist.
    """
    forced = os.environ.get(_ENV)
    if forced is not None:
        return forced == "1"
    return _arch() == "gfx942" and _kernels_importable()


# --------------------------------------------------------------------------
# global -> local expert id
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def _global_to_local_table(
    num_global: int, num_local: int, ep_rank: int, device: str
) -> torch.Tensor:
    """[num_global] int32: local id, or -1 if the expert is on another EP rank.

    Same convention as sglang's own `StandardDispatcher.local_expert_mapping`.
    """
    t = torch.full((num_global,), -1, dtype=torch.int32, device=device)
    lo = ep_rank * num_local
    hi = min(lo + num_local, num_global)
    if hi > lo:
        t[lo:hi] = torch.arange(hi - lo, dtype=torch.int32, device=device)
    return t


@functools.lru_cache(maxsize=1)
def _topk_reduce_kernel():
    """Build the fused top-k reduce kernel once.

    Cached, not defined inside `_topk_reduce`: a `@triton.jit` decorator inside
    the caller would construct a fresh JITFunction on every MoE layer of every
    forward, re-running the decorator's source inspection each time. Kept out of
    module scope so importing this file does not hard-require triton.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        down_ptr, ids_ptr, out_ptr, H, TOPK: tl.constexpr, BLOCK_H: tl.constexpr
    ):
        t = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = offs < H
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            lid = tl.load(ids_ptr + t * TOPK + k)
            # mask the LOAD, not the result: `down` is `empty`, so an unowned
            # row may hold NaN and `0 * NaN` would poison the sum.
            v = tl.load(
                down_ptr + (t * TOPK + k) * H + offs, mask=hmask & (lid >= 0), other=0.0
            )
            acc += v.to(tl.float32)
        tl.store(out_ptr + t * H + offs, acc.to(out_ptr.dtype.element_ty), mask=hmask)

    return _kernel


def _topk_reduce(down: torch.Tensor, local_ids: torch.Tensor, H: int) -> torch.Tensor:
    """Sum the top-k expert outputs for each token, skipping unowned rows.

    Replaces this pair::

        down = torch.zeros((T * topk, 1, H), ...)   # fill kernel
        ...
        return down.view(T, topk, H).sum(dim=1)     # reduce kernel

    The `zeros` existed only so the reduce could safely read rows the GEMM never
    wrote: rows whose expert lives on another EP rank are dropped by the align
    step.  Masking those out of the load removes the need to pre-zero at all, so
    `down` can be `empty` and the two kernels collapse into one.

    The masked load is what makes `empty` safe -- an unwritten row may hold NaN,
    and multiplying by a 0/1 mask after the fact would propagate it.
    `tl.load(..., mask=..., other=0.0)` never materialises the value.
    """
    import triton

    T, topk = local_ids.shape
    out = torch.empty((T, H), dtype=down.dtype, device=down.device)
    BLOCK_H = 512
    _topk_reduce_kernel()[(T, triton.cdiv(H, BLOCK_H))](
        down,
        local_ids,
        out,
        H,
        TOPK=topk,
        BLOCK_H=BLOCK_H,
        num_warps=4,
    )
    return out


def _align(
    local_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`ignore_invalid_expert=True` DROPS the -1 rows rather than bucketing
    them, so at EP16 the grouped GEMM stops padding 15/16 of the rows into
    blocks that only write zeros. The caller must therefore not assume every
    row of `down` is written -- see the `torch.empty` note in
    fused_moe_mxfp4_triton."""
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(
        local_ids, block_size, num_experts, ignore_invalid_expert=True
    )


@functools.lru_cache(maxsize=8)
def _unit_scales(num_experts: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The Triton MXFP4 kernels carry a second fp32 scale on top of the e8m0
    microscales.  MXFP4 checkpoints have none, so it is 1 -- cache it rather
    than re-materialising two tensors every layer of every forward."""
    return (
        torch.ones(1, dtype=torch.float32, device=device),
        torch.ones(num_experts, dtype=torch.float32, device=device),
    )


# --------------------------------------------------------------------------
# the MoE itself
# --------------------------------------------------------------------------
def fused_moe_mxfp4_triton(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    num_global_experts: Optional[int] = None,
    ep_rank: int = 0,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """MXFP4 MoE = gate/up GEMM (fused activation) + down GEMM + top-k reduce.

    Layout expected (exactly what `Mxfp4MoEMethod.create_weights` allocates,
    minus aiter's `shuffle_weight` preshuffle, which the Triton kernels do not
    consume):

        w13_weight [E, 2*I, H/2] uint8   e2m1, first I rows gate, next I up
        w13_scale  [E, 2*I, H/32] uint8  e8m0
        w2_weight  [E, H,   I/2] uint8
        w2_scale   [E, H,   I/32] uint8
    """
    import triton.language as tl  # noqa: F401  (torch_to_triton_dtype values)
    from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4
    from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config
    from aiter.ops.triton.utils.types import torch_to_triton_dtype

    from sglang.srt.layers.moe.moe_runner.mxfp4_situ_fused import (
        fused_moe_mxfp4_act,
    )

    assert activation in ("silu", "situ"), activation

    w13 = w13_weight.view(torch.uint8)
    w2 = w2_weight.view(torch.uint8)
    E, N13, _ = w13.shape
    H = w2.shape[1]
    inter = N13 // 2
    T = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    dev = hidden_states.device
    dt = hidden_states.dtype

    # ---- int32 address guard -------------------------------------------
    # The aiter Triton MoE kernels index their operands as
    # `ptr + stride * offs_token[:, None]` with `offs_token` loaded as int32
    # from sorted_token_ids, so the flat element offset must stay under 2^31.
    # The binding term is the intermediate: (T * topk) rows x `inter` columns.
    # Past that the kernel walks off the end of the buffer -- observed as
    # "Memory access fault ... Write access to a read-only page". One unchunked
    # long prefill is exactly that shape. Split the token dimension rather than
    # patch the kernels; the
    # expert weights and the routing are unaffected by the split.
    max_rows = (2**31 - 1) // max(inter, H, hidden_states.shape[1])
    max_T = max(1, max_rows // max(topk, 1))
    if T > max_T:
        chunks = [
            fused_moe_mxfp4_triton(
                hidden_states[i : i + max_T],
                w13_weight,
                w2_weight,
                w13_scale,
                w2_scale,
                topk_weights[i : i + max_T],
                topk_ids[i : i + max_T],
                activation=activation,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                num_global_experts=num_global_experts,
                ep_rank=ep_rank,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            for i in range(0, T, max_T)
        ]
        return torch.cat(chunks, dim=0)

    config = get_optimal_moe_config(dt, use_mxfp4=True, M=T)
    bm = config["BLOCK_SIZE_M"]

    # global -> local ids; experts owned by another EP rank map to -1, which
    # _align drops.
    if num_global_experts is not None and num_global_experts != E:
        table = _global_to_local_table(num_global_experts, E, ep_rank, str(dev))
        local_ids = table[topk_ids.to(torch.long)]
    else:
        local_ids = topk_ids.to(torch.int32)

    sorted_token_ids, expert_ids, num_tokens_post_padded = _align(local_ids, bm, E)

    a_scale, b_scale = _unit_scales(E, str(dev))
    ct = torch_to_triton_dtype[dt]

    # ---- 1. gate/up GEMM, activation fused into the epilogue
    inter_states = torch.empty((T * topk, inter), dtype=dt, device=dev)
    fused_moe_mxfp4_act(
        hidden_states,
        w13,
        inter_states,
        a_scale,
        b_scale,
        None,
        w13_scale,
        topk_weights,
        local_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,  # mul_routed_weight
        topk,
        False,  # swizzle_mx_a  -- preshuffled scales are CDNA4-only
        False,  # swizzle_mx_b
        config,
        ct,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )

    # ---- 2. down GEMM; rows are already (token, expert) pairs -> top_k=1,
    #         and the routed weight folds in here.
    #
    # `empty` is safe only because the fused reduce masks out the rows the
    # align dropped (expert owned by another EP rank), which the GEMM never
    # writes. A `.sum()` over them would read uninitialised memory.
    down = torch.empty((T * topk, 1, H), dtype=dt, device=dev)
    fused_moe_mxfp4(
        inter_states,
        w2,
        down,
        a_scale,
        b_scale,
        None,
        w2_scale,
        topk_weights,
        local_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,  # mul_routed_weight
        1,
        False,
        False,
        config,
        ct,
    )

    # ---- 3. top-k reduce (fused with the zero-fill step 2 used to need)
    return _topk_reduce(down, local_ids, H)
