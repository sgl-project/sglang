"""CUDA JIT wrappers for AttnRes score and combine kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
    override_jit_cuda_arch,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_BLOCK_H: int = 1024  # H-chunk size for score (thread count) and combine (chunk width)
_MAX_ROWS: int = 16  # next_pow2(8+1), K3 has <=8 snapshots

_DIM: int = 7168  # K3 hidden size, template parameter of the fused/chain kernels
_MAX_BANK_ROWS: int = 8  # K3 has <= 8 snapshots, upper bound of the nvb dispatch tables


def _make_name(*args):
    return "kimi_k3_attn_res_" + "_".join(str(a) for a in args)


@cache_once
def _jit_score_module() -> Module:
    """Compile and cache the JIT AttnRes score kernel."""
    args = make_cpp_args(_BLOCK_H, is_arch_support_pdl())
    return load_jit(
        _make_name("score"),
        *args,
        cuda_files=["kimi_k3/attn_res_score.cuh"],
        cuda_wrappers=[("run", f"AttnResScoreKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


@cache_once
def _jit_combine_module() -> Module:
    """Compile and cache the JIT AttnRes combine kernel."""
    args = make_cpp_args(_BLOCK_H, _MAX_ROWS, is_arch_support_pdl())
    return load_jit(
        _make_name("combine"),
        *args,
        cuda_files=["kimi_k3/attn_res_combine.cuh"],
        cuda_wrappers=[("run", f"AttnResCombineKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


@cache_once
def _jit_chain_module() -> Module:
    """Compile and cache the optimized score/merge/norm chain kernels."""
    args = make_cpp_args(_DIM, _MAX_BANK_ROWS, is_arch_support_pdl())
    return load_jit(
        _make_name("chain"),
        *args,
        cuda_files=["kimi_k3/attn_res/chain_reg.cuh"],
        cuda_wrappers=[("run", f"AttnResChain<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


@cache_once
def _jit_fused_tma_module(
    chunk_rows: int, occupancy: int, consumer_regs: int
) -> Module:
    """Compile and cache the warp-specialized TMA aggregation kernel (per-row
    bulk copies into chunk slots; chunk_rows / occupancy / consumer_regs are
    tuning knobs). The smem ring is frozen at 2 chunk slots and PDL is always
    on: the kernel targets SM100+, where both are unconditional wins."""
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            "attn_res_fused_tma requires SM100+ (tcgen05, cp.async.bulk)"
        )
    args = make_cpp_args(
        _DIM,
        _MAX_BANK_ROWS,
        chunk_rows,
        occupancy,
        consumer_regs,
    )
    with override_jit_cuda_arch(major, minor, suffix="a"):
        return load_jit(
            _make_name("fused_tma"),
            *args,
            cuda_files=["kimi_k3/attn_res/fused_tma.cuh"],
            cuda_wrappers=[("run", f"AttnResFusedTmaKernel<{args}>::run")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )


@cache_once
def _jit_score_fused_add_module() -> Module:
    """Compile and cache the JIT AttnRes score+residual-add kernel."""
    args = make_cpp_args(_BLOCK_H, is_arch_support_pdl())
    return load_jit(
        _make_name("score_fused_add"),
        *args,
        cuda_files=["kimi_k3/attn_res_score_fused_add.cuh"],
        cuda_wrappers=[("run", f"AttnResScoreFusedAddKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def attn_res_chain(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
) -> None:
    """Optimized aggregation chain, one host call launching three kernels:

    - score: per-row dot(v, cw) * rrms(v), widest vectors x unroll 2; only
      the prefix-row CTA PDL-waits on the previous kernel.
    - merge: softmax(scores) -> weighted row sum, row count dispatched
      through a compile-time kernel table on nvb; rows prefetched before the
      PDL wait. Also writes per-H-chunk sum(mixed^2) partials.
    - norm: mixed * rsqrt(sum(partials)/H + eps) * ow — no reduction at all.

    The scores / partials / mixed workspace is allocated C++-side in a single
    allocation. Restrictions: H == 7168, nvb in [1, 8].

    Parameters
    ----------
    prefix_sum : [T, H] bf16
    bank       : [T, NB, H] bf16 (rows 0..nvb-1 are aggregated)
    cw         : [H] bf16 — precomputed score_norm_weight * proj_weight
    ow         : [H] bf16 — output RMSNorm weight
    out        : [T, H] bf16 output buffer
    nvb        : number of valid bank rows (1..8)
    eps        : RMSNorm epsilon (shared by score and output norms)
    """
    _jit_chain_module().run(prefix_sum, bank, cw, ow, out, nvb, eps)


# Benchmarked-best (chunk_rows, occupancy, consumer_regs) per nvb
# (GB200/GB300-class, H=7168). nvb=1 is latency-bound per token, so 2 CTAs/SM
# (occupancy=2, which excludes setmaxnreg: 4*Nc + 2*Np > 512) wins large T by
# ~17%; nvb=4/8 fill 5-row chunks exactly ((nvb+1) % 5 == 0 or covers it in
# 1-2 chunks); everything else is fastest on the balanced 4-row chunk. All
# entries use the setmaxnreg producer/consumer split. consumer_regs sits on
# the 200..232 performance plateau (below 200 the consumer loop starves); we
# take 200, not the 232 budget cap, so each SMSP keeps 2K registers free for
# PDL-overlapped neighbor kernels instead of allocating them idle.
_TMA_BEST_CONFIG: dict[int, tuple[int, int, int]] = {
    1: (2, 2, 0),
    2: (4, 1, 200),
    3: (4, 1, 200),
    4: (5, 1, 200),
    5: (3, 1, 200),
    6: (4, 1, 200),
    7: (4, 1, 200),
    8: (5, 1, 200),
}


def attn_res_fused_tma(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    write_prefix: bool = False,
    chunk_rows: int | None = None,
    occupancy: int | None = None,
    consumer_regs: int | None = None,
) -> None:
    """Warp-specialized TMA aggregation (score -> online softmax -> weighted
    combine -> fused output RMSNorm), one persistent CTA per SM: a producer
    warp (group) fetches rows with one bulk copy each into chunk slots of
    `chunk_rows` rows (one barrier pair per chunk, double-buffered ring of 2
    slots); the 8 consumer warps score and fold one chunk per rendezvous,
    with cw / ow staged in TMEM.

    Restrictions: H == 7168, nvb in [1, 8], SM100a+. Knobs left as None take
    the per-nvb benchmarked best from _TMA_BEST_CONFIG (each combination
    compiles its own module; occupancy > 1 requires the smem ring to fit
    that many copies per SM).

    Parameters
    ----------
    prefix_sum : [T, H] bf16
    bank       : [T, NB, H] bf16 (rows 0..nvb-1 are aggregated)
    cw         : [H] bf16 — precomputed score_norm_weight * proj_weight
    ow         : [H] bf16 — output RMSNorm weight
    out        : [T, H] bf16 output buffer
    nvb        : number of valid bank rows (1..8)
    eps        : RMSNorm epsilon (shared by score and output norms)
    write_prefix : also snapshot the prefix row into bank[:, nvb, :]
                 (bit-exact copy, fused into the score pass which already
                 has the row in registers); requires NB > nvb
    """
    if not 1 <= nvb <= _MAX_BANK_ROWS:
        raise ValueError(
            f"attn_res_fused_tma: nvb must be in [1, {_MAX_BANK_ROWS}], got {nvb}"
        )
    best = _TMA_BEST_CONFIG[nvb]
    if best[1] > 1 and prefix_sum.shape[0] < 128:
        # occupancy=2 only pays off once there are enough tokens to fill both
        # CTAs per SM; below that its tighter register budget just costs ~10%.
        best = (4, 1, 200)
    config = (
        chunk_rows if chunk_rows is not None else best[0],
        occupancy if occupancy is not None else best[1],
        consumer_regs if consumer_regs is not None else best[2],
    )
    _jit_fused_tma_module(*config).run(
        prefix_sum, bank, cw, ow, out, nvb, eps, write_prefix
    )


def attn_res_score_fused_add(
    prefix_a: torch.Tensor,
    prefix_b: torch.Tensor,
    prefix_out: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    scores: torch.Tensor,
    nvb: int,
    eps: float,
) -> None:
    """attn_res_score with the upstream residual add fused: the prefix row is
    computed as bf16(prefix_a + prefix_b) on the fly, written to prefix_out
    (bit-identical to the standalone add), and scored."""
    _jit_score_fused_add_module().run(
        prefix_a, prefix_b, prefix_out, bank, cw, scores, nvb, eps
    )


def attn_res_score(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    scores: torch.Tensor,
    nvb: int,
    eps: float,
) -> None:
    """Launch the JIT AttnRes score kernel.

    Parameters
    ----------
    prefix_sum : [T, H] bf16
    bank       : [T, NB, H] bf16
    cw         : [H] fp32 — precomputed norm_weight * proj_weight
    scores     : [T, MAX_ROWS] fp32 output buffer
    nvb        : number of valid bank rows (0..8)
    eps        : RMSNorm epsilon
    """
    module = _jit_score_module()
    module.run(prefix_sum, bank, cw, scores, nvb, eps)


def attn_res_combine(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    scores: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
) -> None:
    """Launch the JIT AttnRes combine kernel.

    Parameters
    ----------
    prefix_sum : [T, H] bf16
    bank       : [T, NB, H] bf16
    scores     : [T, MAX_ROWS] fp32 — output from attn_res_score
    out        : [T, H] bf16 output buffer
    nvb        : number of valid bank rows (0..8)
    """
    module = _jit_combine_module()
    module.run(prefix_sum, bank, scores, out, nvb)
