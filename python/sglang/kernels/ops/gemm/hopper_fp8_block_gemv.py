from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MAX_M = 16
_MAX_K = 4096  # 16 x K bytes of activation slab in shared memory
_MAX_SIDE_N = 8  # side projection columns, one extra block each

# Occupancy inputs measured with ncu on H20 for the shipped variants.
_REGS_PER_SM = 64 * 1024
_REGS_PER_THREAD = 128
_SMEM_PER_SM = 135 * 1024
# Below this many scale blocks per warp, the split-K reduction costs more than it saves.
_MIN_KB_PER_WARP = 4
# Fewer warps per block no longer hide the weight-load latency.
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
    # Halve when a warp owns too little K or the wider block spills the grid into
    # a tail wave (5120x4096 on H20: 12.80 -> 11.99 us).
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
    # Past M=8 a second column group doubles MMA work; it wins on H20 only for N <= 1536.
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
