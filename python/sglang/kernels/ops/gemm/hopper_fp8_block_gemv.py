from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Hopper small-M blockwise-FP8 GEMV (see csrc/gemm/hopper_fp8_block_gemv.cuh).
#
# Decode-time dense projections of a blockwise-FP8 model (128x128 weight
# blocks, 1x128 activation groups) run at M = tokens-in-flight, i.e. 1..8 for
# bs <= 2 with a few draft tokens. DeepGEMM's TMA/WGMMA pipeline moves those
# few-MB weights at ~1 TB/s on H20 (18 us for a 21 MB projection that streams
# in 6 us). Streaming the weights straight into mma.m16n8k32 e4m3 fragments,
# one warp per 16-row tile for all 8 token columns, runs near the bandwidth
# floor at these M. M <= 16 (two n8 column groups) covers bs <= 4 verify.

_MAX_M = 16
_MAX_K = 4096  # 16 x K bytes of activation slab in shared memory
_MAX_SIDE_N = 8  # side projection columns, one extra block each

# Occupancy inputs for the config search. Registers per thread is what ncu
# reports for the shipped variants (126 for <8, 2, 4, 2>); the shared-memory
# budget is the "Shared Memory Configuration Size" ncu reports on H20.
_REGS_PER_SM = 64 * 1024
_REGS_PER_THREAD = 128
_SMEM_PER_SM = 135 * 1024
# Scale blocks a warp needs to own before its share of the split-K reduction and
# its two barriers are worth the extra K parallelism.
_MIN_KB_PER_WARP = 4
# Warps a block keeps after a halving. Below this the block has too little
# internal parallelism to cover the load latency (1536x1536: k_split 4 -> 2
# leaves 2 warps and costs 4.70 -> 5.21 us).
_MIN_WARPS = 4


@functools.lru_cache(maxsize=1)
def _sm_count() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def _resident_blocks(max_m: int, row_tiles: int, k_split: int, k: int, nkb: int) -> int:
    """How many blocks of this shape the whole GPU holds at once."""
    warps = row_tiles * k_split
    smem = (
        max_m * (k + 16) + max_m * nkb * 4 + k_split * row_tiles * 32 * (max_m // 2) * 4
    )
    per_sm = min(_REGS_PER_SM // (32 * warps * _REGS_PER_THREAD), _SMEM_PER_SM // smem)
    return _sm_count() * max(1, per_sm)


def _config(m: int, n: int, k: int) -> tuple[int, int, int, int]:
    """(max_m, row_tiles_per_block, k_split_warps, k_unroll)."""
    max_m = 8 if m <= 8 else 16
    tiles = (n + 15) // 16
    nkb = k // 128
    row_tiles = 2 if tiles >= 256 else 1
    # Split K across warps until the grid covers the SMs a few times over;
    # each warp keeps >= 2 scale blocks so the unrolled loads still pipeline.
    k_split = 1
    for cand in (8, 4, 2):
        if nkb % cand == 0 and nkb // cand >= 2 and cand * row_tiles <= 8:
            k_split = cand
            break
    # Two ways that greedy split overshoots, both fixed by halving it (k_split is
    # a power of two, so nkb stays divisible). Halve only while the block keeps
    # enough warps to hide the weight-load latency.
    #   - too little K per warp: the reduction stops paying for itself;
    #   - one wave too many: a wider block fits fewer times per SM, so the grid
    #     spills past the resident-block count and pays a whole tail wave. ncu on
    #     5120x4096 <8, 2, 4, 2>: 160 blocks against 156 slots, SM Active only 57%
    #     of Elapsed. Halving takes 12.80 -> 11.99 us (2560x4096: 9.18 -> 8.12,
    #     4096x1024: 4.77 -> 4.41).
    if k_split > 1 and row_tiles * (k_split // 2) >= _MIN_WARPS:
        blocks = (n + 16 * row_tiles - 1) // (16 * row_tiles)
        wide = _resident_blocks(max_m, row_tiles, k_split, k, nkb)
        narrow = _resident_blocks(max_m, row_tiles, k_split // 2, k, nkb)
        if nkb // k_split < _MIN_KB_PER_WARP or (blocks > wide and blocks <= narrow):
            k_split //= 2
    unroll = 2 if (nkb // k_split) % 2 == 0 else 1
    return max_m, row_tiles, k_split, unroll


@cache_once
def _jit_hopper_fp8_block_gemv_module(
    max_m: int, row_tiles: int, k_split: int, unroll: int
) -> Module:
    args = make_cpp_args(max_m, row_tiles, k_split, unroll)
    return load_jit(
        "hopper_fp8_block_gemv",
        *args,
        cuda_files=["gemm/hopper_fp8_block_gemv.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::HopperFp8BlockGemvKernel<{args}>::run"),
            ("run_side", f"sglang::HopperFp8BlockGemvKernel<{args}>::run_side"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def use_hopper_fp8_block_gemv(
    m: int, n: int, k: int, block_size: Sequence[int]
) -> bool:
    if not (
        1 <= m <= _MAX_M
        and tuple(block_size) == (128, 128)
        and k % 128 == 0
        and 128 <= k <= _MAX_K
        and n >= 16
    ):
        return False
    # Two column groups (M > 8) double the MMA and slab traffic; measured on
    # H20 they still beat DeepGEMM only on the narrow projections (N <= 1536:
    # 1024x4096 9.4 -> 7.0 us), and lose on the wide ones (5120x4096 18.5 -> 23 us).
    return m <= 8 or n <= 1536


def use_hopper_fp8_block_gemv_side(side_w: torch.Tensor, k: int) -> bool:
    return (
        side_w.dtype == torch.bfloat16
        and side_w.is_contiguous()
        and side_w.ndim == 2
        and 1 <= side_w.shape[0] <= _MAX_SIDE_N
        and side_w.shape[1] == k
    )


def hopper_fp8_block_gemv(
    xq: torch.Tensor,
    xs: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    x_bf16: Optional[torch.Tensor] = None,
    side_w: Optional[torch.Tensor] = None,
):
    """out[M, N] (bf16) = xq[M, K] @ wq[N, K]^T with 1x128 / 128x128 fp8 block scales.

    xq / wq are float8_e4m3fn, xs is fp32 [M, K/128] (row-major), ws is fp32
    [ceil(N/128), K/128]; all contiguous.

    With ``x_bf16`` (the unquantized [M, K] activation) and ``side_w`` (bf16
    [S, K], S <= 8) one extra block also computes side_out[M, S] = x @ side_w^T
    (fp32 accumulation, bf16 output) and the call returns ``(out, side_out)``.
    """
    m, k = xq.shape
    n = wq.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    module = _jit_hopper_fp8_block_gemv_module(*_config(m, n, k))
    if side_w is None:
        module.run(xq, xs, wq, ws, out)
        return out
    side_out = torch.empty((m, side_w.shape[0]), dtype=torch.bfloat16, device=xq.device)
    module.run_side(xq, xs, wq, ws, out, x_bf16, side_w, side_out)
    return out, side_out
