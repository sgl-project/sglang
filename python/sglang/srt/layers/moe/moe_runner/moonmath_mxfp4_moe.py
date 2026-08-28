# SPDX-License-Identifier: Apache-2.0
"""gfx942 (CDNA3) MXFP4 MoE through the moonmath_amd HIP GEMMs.

`aiter.fused_moe` has no gfx942 a4w4 path and aborts in the activation
quantiser, so MXFP4 MoE does not run on CDNA3 at all. This module routes the two
grouped GEMMs of an MXFP4 MoE layer to hand-written gfx942 kernels that
dequantize MXFP4 to bf16 in registers with `v_perm_b32` and feed the bf16 MFMA
directly. Against a tuned aiter Triton MXFP4 baseline on MI300X at Kimi-K3 TP8
shapes, the kernels themselves run 1.37x geomean on gate/up and 1.46x on down;
`benchmark/bench_moe.py` in moonmath-ai/amd-kernels reproduces that.

The kernels read a repacked weight layout that no other route understands (see
`repack_moonmath_mxfp4_weights`), so the choice is made once at weight load
rather than per forward. `use_moonmath_mxfp4_moe` is that decision, and it turns
down anything the compiled kernels do not cover -- a non-SituGLU activation, a
bias, a routing scale on the input, or a K the tiles are not built for -- which
leaves those layers exactly where they are today.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# down is compiled for K/128 in {3, 4, 6}; it stages its whole A tile in LDS for
# the n sweep, which is what puts a ceiling on K in the first place.
_DOWN_K_CHOICES = (384, 512, 768)

# Both kernels address their operands as a u32 byte offset into a buffer
# resource, so one operand must stay under 4 GiB. Half of that, in elements of a
# 2-byte dtype, is the token-chunk bound in `fused_moe_mxfp4_moonmath` -- a long
# unchunked prefill is the only shape that reaches it.
_MAX_ELEMS_PER_OPERAND = 1 << 30

_REDUCE_BLOCK_H = 512


@functools.lru_cache(maxsize=1)
def _gpu_arch() -> str:
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as exc:  # noqa: BLE001
        logger.debug("no HIP device to probe for the moonmath MXFP4 route: %s", exc)
        return ""


@functools.lru_cache(maxsize=1)
def _package_available() -> bool:
    try:
        from moonmath_amd import (  # noqa: F401
            mxfp4_moe_down,
            mxfp4_moe_gateup,
            mxfp4_moe_gateup_supports_k,
            repack_mxfp4,
        )

        return True
    except ImportError as exc:
        logger.debug("moonmath_amd MXFP4 MoE kernels unavailable: %s", exc)
        return False


@functools.lru_cache(maxsize=1)
def _route_available() -> bool:
    """Arch, package and opt-out gate, independent of any layer's shapes."""
    return (
        envs.SGLANG_USE_MOONMATH_MXFP4_MOE.get()
        and _gpu_arch() == "gfx942"
        and _package_available()
    )


def use_moonmath_mxfp4_moe(
    *,
    hidden_size: int,
    intermediate_size: int,
    activation: Optional[str],
    apply_router_weight_on_input: bool,
    has_bias: bool,
) -> bool:
    """Whether this layer's MXFP4 MoE should use the moonmath kernels.

    Called at weight load, since the answer decides the weight layout and cannot
    be revisited afterwards. Every condition is a property of the compiled
    kernels, not a tuning threshold. `intermediate_size` is the padded one the
    weights were allocated at, not the model's.
    """
    if not _route_available():
        return False
    # The gate/up kernel's only fused epilogue is the Kimi-K3 SituGLU, and it has
    # neither a bias term nor a slot for a per-row routing scale.
    if activation != "situ" or has_bias or apply_router_weight_on_input:
        return False
    # Ask the kernel rather than restating its rule: gate/up balances its wait
    # rungs over a whole slab pair, so K has to cover one.
    import moonmath_amd as ma

    if not ma.mxfp4_moe_gateup_supports_k(hidden_size):
        return False
    return intermediate_size in _DOWN_K_CHOICES


def repack_moonmath_mxfp4_weights(layer: torch.nn.Module) -> None:
    """Rewrite this layer's MXFP4 weights into the layout the kernels read.

        w13  [E, 2I, H/2] uint8  ->  [E, H/32, 2I, 16] uint8
        w2   [E, H,  I/2] uint8  ->  [E, I/32, H,  16] uint8
        scales transpose the same way, without the trailing 16

    Both transforms are pure permutations, so they change no value and cost
    nothing at run time. They are also required: the kernels do not raise on a
    stock layout, they compute wrong numbers. Expects w13 de-interleaved (all
    gate rows, then all up rows), which is what the aiter branch has already
    produced by this point -- the SiTU epilogue reads column n as gate and
    column n + I as up.
    """
    import moonmath_amd as ma
    from torch.nn.parameter import Parameter

    def packed(tensor: torch.Tensor, repack) -> torch.nn.Parameter:
        return Parameter(repack(tensor.data.view(torch.uint8)), requires_grad=False)

    # Rebound one at a time, so the transient peak is one extra copy of the
    # largest weight rather than one of the whole layer.
    layer.w13_weight = packed(layer.w13_weight, ma.repack_mxfp4)
    layer.w13_weight_scale = packed(layer.w13_weight_scale, ma.repack_mxfp4_scales)
    layer.w2_weight = packed(layer.w2_weight, ma.repack_mxfp4)
    layer.w2_weight_scale = packed(layer.w2_weight_scale, ma.repack_mxfp4_scales)

    # aiter's dispatcher reads these to decide whether it may select its
    # preshuffle-on kernels; nothing here is in that layout.
    layer.w13_weight.is_shuffled = False
    layer.w2_weight.is_shuffled = False


@functools.lru_cache(maxsize=4)
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=8)
def _local_expert_table(
    num_global: int, num_local: int, ep_rank: int, device: str
) -> torch.Tensor:
    """[num_global] int32: local expert id, or -1 if it lives on another EP rank.

    The standard dispatcher skips this translation for the aiter runner -- it
    builds an expert mask and leaves topk_ids global -- so the route has to do
    it. Same convention as `StandardDispatcher.local_expert_mapping`.
    """
    table = torch.full((num_global,), -1, dtype=torch.int32, device=device)
    lo = ep_rank * num_local
    hi = min(lo + num_local, num_global)
    if hi > lo:
        table[lo:hi] = torch.arange(hi - lo, dtype=torch.int32, device=device)
    return table


def _align(local_ids: torch.Tensor, block_size: int, num_experts: int):
    """Sort rows by expert and pad each expert up to `block_size`.

    `ignore_invalid_expert=True` DROPS the -1 rows rather than bucketing them,
    so at EP16 the GEMMs stop padding 15/16 of the rows into blocks that only
    write zeros. The caller must therefore not assume every row of the output
    is written -- see the `torch.empty` note in `_gate_up`.

    Imported here rather than at module scope: moe_align_block_size pulls in
    triton, and this file is imported by the quant method on every arch.
    """
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(
        local_ids, block_size, num_experts, ignore_invalid_expert=True
    )


def _plan_shape(
    *, rows: int, num_local_experts: int, num_global_experts: int
) -> Tuple[int, int]:
    """Host-side `(rows, active_experts)` estimate for the tile-height planners.

    The planners want rows per expert, and the true active-expert count is only
    on the device -- reading it would sync and would not survive graph capture.
    The expected number of distinct experts among n independent draws gets both
    endpoints right (a handful of decode rows land on almost as many distinct
    experts; a prefill lands on all of them), and this only picks a tile height,
    so a few percent of error moves nothing.
    """
    if num_local_experts < num_global_experts:
        # Rows routed to another EP rank are dropped by the align step, so this
        # rank packs its tiles from a proportionally smaller pool.
        rows = rows * num_local_experts / num_global_experts
    e = max(num_local_experts, 1)
    active = e * (1.0 - (1.0 - 1.0 / e) ** rows)
    return max(int(rows), 1), max(round(active), 1)


@functools.lru_cache(maxsize=1)
def _topk_reduce_kernel():
    """Built once and cached; a `@triton.jit` inside the caller would rebuild a
    JITFunction on every MoE layer of every forward. Kept out of module scope so
    importing this file does not hard-require triton."""
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        down_ptr,
        ids_ptr,
        weight_ptr,
        out_ptr,
        H,
        TOPK: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        t = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = offs < H
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            lid = tl.load(ids_ptr + t * TOPK + k)
            w = tl.load(weight_ptr + t * TOPK + k).to(tl.float32)
            # Mask the LOAD, not the result: `down` is `empty`, so a row whose
            # expert lives on another EP rank may hold NaN and `0 * NaN` would
            # poison the sum.
            v = tl.load(
                down_ptr + (t * TOPK + k) * H + offs, mask=hmask & (lid >= 0), other=0.0
            )
            acc += v.to(tl.float32) * w
        tl.store(out_ptr + t * H + offs, acc.to(out_ptr.dtype.element_ty), mask=hmask)

    return _kernel


def _weighted_topk_reduce(
    *,
    down: torch.Tensor,
    local_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden: int,
) -> torch.Tensor:
    """Weighted sum of a token's top-k expert outputs, skipping dropped rows.

    The routing weight is applied here rather than in the down GEMM's epilogue:
    that kernel reads it as bf16, while this reduce already holds every value in
    an fp32 accumulator, so folding it in here keeps the weight at its loaded
    precision for free.
    """
    import triton

    num_tokens, topk = local_ids.shape
    out = torch.empty((num_tokens, hidden), dtype=down.dtype, device=down.device)
    grid = (num_tokens, triton.cdiv(hidden, _REDUCE_BLOCK_H))
    _topk_reduce_kernel()[grid](
        down,
        local_ids,
        topk_weights,
        out,
        hidden,
        TOPK=topk,
        BLOCK_H=_REDUCE_BLOCK_H,
        num_warps=4,
    )
    return out


def _gate_up(
    *,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    align,
    block_m: int,
    rows: int,
    top_k: int,
    inter: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    """Gate/up GEMM with the SituGLU fused into the epilogue -> [rows, inter].

    `empty` is safe here: the only rows left unwritten are the ones the align
    dropped, and the down GEMM walks the same dropped-row list, so nothing
    reads them.
    """
    import moonmath_amd as ma

    sorted_token_ids, expert_ids, num_tokens_post_padded = align
    out = torch.empty(
        (rows, inter), dtype=hidden_states.dtype, device=hidden_states.device
    )
    ma.mxfp4_moe_gateup(
        A=hidden_states,
        B=weight,
        Bs=scale,
        C=out,
        topk_weights=None,  # unused without mul_routed_weight
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        num_m_blocks=expert_ids.numel(),
        block_m=block_m,
        num_valid_tokens=rows,
        top_k=top_k,
        epilogue=ma.EPI_SITU,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    return out


def _down(
    *,
    inter_states: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    align,
    block_m: int,
    rows: int,
    hidden: int,
    plan_rows: int,
    plan_experts: int,
) -> torch.Tensor:
    """Down GEMM -> [rows, hidden]. Rows are already (token, expert) pairs, so
    its `top_k` is 1 and its A index is the sorted slot itself."""
    import moonmath_amd as ma

    sorted_token_ids, expert_ids, num_tokens_post_padded = align
    num_m_blocks = expert_ids.numel()
    nt = ma.mxfp4_moe_down_nt(plan_rows, plan_experts)
    n_steps = ma.mxfp4_moe_down_n_steps(
        num_m_blocks=num_m_blocks,
        N=hidden,
        num_cus=_num_cus(inter_states.device.index),
        block_m=block_m,
        nt=nt,
    )
    out = torch.empty(
        (rows, hidden), dtype=inter_states.dtype, device=inter_states.device
    )
    ma.mxfp4_moe_down(
        A=inter_states,
        B=weight,
        Bs=scale,
        C=out,
        topk_weights=None,  # the routing weight is folded into the reduce
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        num_m_blocks=num_m_blocks,
        block_m=block_m,
        n_steps=n_steps,
        nt=nt,
        num_valid_tokens=rows,
        top_k=1,
    )
    return out


def fused_moe_mxfp4_moonmath(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    num_global_experts: Optional[int] = None,
    ep_rank: int = 0,
) -> torch.Tensor:
    """MXFP4 MoE = gate/up GEMM (SituGLU fused) + down GEMM + weighted reduce.

    The weights and their scales must already be in the repacked layout that
    `repack_moonmath_mxfp4_weights` produces.
    """
    import moonmath_amd as ma

    w13 = w13_weight.view(torch.uint8)  # [E, H/32, 2I, 16]
    w2 = w2_weight.view(torch.uint8)  # [E, I/32, H,  16]
    num_local_experts = w13.shape[0]
    inter = w13.shape[2] // 2
    hidden = w2.shape[2]
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]

    assert (
        hidden_states.dtype == torch.bfloat16
    ), f"moonmath MXFP4 MoE needs bf16 activations, got {hidden_states.dtype}"

    max_tokens = _max_tokens_per_call(inter=inter, hidden=hidden, top_k=top_k)
    if num_tokens > max_tokens:
        return torch.cat(
            [
                fused_moe_mxfp4_moonmath(
                    hidden_states[i : i + max_tokens],
                    w13_weight,
                    w2_weight,
                    w13_scale,
                    w2_scale,
                    topk_weights[i : i + max_tokens],
                    topk_ids[i : i + max_tokens],
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                    num_global_experts=num_global_experts,
                    ep_rank=ep_rank,
                )
                for i in range(0, num_tokens, max_tokens)
            ],
            dim=0,
        )

    # The gate/up kernel takes an arbitrary A row stride but assumes k is
    # contiguous.
    if hidden_states.stride(1) != 1:
        hidden_states = hidden_states.contiguous()

    local_ids = _local_expert_ids(
        topk_ids=topk_ids,
        num_local_experts=num_local_experts,
        num_global_experts=num_global_experts,
        ep_rank=ep_rank,
    )
    rows = num_tokens * top_k
    plan_rows, plan_experts = _plan_shape(
        rows=rows,
        num_local_experts=num_local_experts,
        num_global_experts=num_global_experts or num_local_experts,
    )

    # The two GEMMs have different tile-height menus (16/32/48 against
    # 16/32/64), so each needs an alignment padded to its own block size. They
    # coincide below ~29 rows per expert, which is every decode step; reuse the
    # one align there rather than pay for a second.
    bm_gate_up = ma.mxfp4_moe_gateup_block_m(plan_rows, plan_experts)
    bm_down = ma.mxfp4_moe_down_block_m(plan_rows, plan_experts, inter)
    align_gate_up = _align(local_ids, bm_gate_up, num_local_experts)
    align_down = (
        align_gate_up
        if bm_down == bm_gate_up
        else _align(local_ids, bm_down, num_local_experts)
    )

    inter_states = _gate_up(
        hidden_states=hidden_states,
        weight=w13,
        scale=w13_scale,
        align=align_gate_up,
        block_m=bm_gate_up,
        rows=rows,
        top_k=top_k,
        inter=inter,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    down = _down(
        inter_states=inter_states,
        weight=w2,
        scale=w2_scale,
        align=align_down,
        block_m=bm_down,
        rows=rows,
        hidden=hidden,
        plan_rows=plan_rows,
        plan_experts=plan_experts,
    )
    return _weighted_topk_reduce(
        down=down, local_ids=local_ids, topk_weights=topk_weights, hidden=hidden
    )


def _max_tokens_per_call(*, inter: int, hidden: int, top_k: int) -> int:
    """Token count at which an operand would outgrow the kernels' u32 offsets.

    Splitting the token dimension is enough to stay under it: the expert weights
    and the routing are untouched by the split.
    """
    return max(1, (_MAX_ELEMS_PER_OPERAND // max(inter, hidden)) // max(top_k, 1))


def _local_expert_ids(
    *,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    num_global_experts: Optional[int],
    ep_rank: int,
) -> torch.Tensor:
    """Global expert ids -> local ones, with another rank's experts mapped to
    -1, which `_align` drops."""
    if num_global_experts is None or num_global_experts == num_local_experts:
        return topk_ids.to(torch.int32)
    table = _local_expert_table(
        num_global=num_global_experts,
        num_local=num_local_experts,
        ep_rank=ep_rank,
        device=str(topk_ids.device),
    )
    return table[topk_ids.to(torch.long)]
