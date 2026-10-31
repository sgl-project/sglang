"""Tiny BF16-activation GEMM for skinny ``x[m, k] @ w[n, k].T`` with a handful of rows.

Two kernel templates behind one entry point (``kernels/jit/csrc/gemm/tiny_gemm/``):

- SIMT (``simt.cuh``, CUDA and ROCm): CUDA-core dot products; K / 16 lanes cover a
  row of w and reduce it with shuffles (and shared memory past one warp), a CTA owns
  a few rows of w.
- MMA (``mma.cuh``, CUDA sm90+, BF16 weights): ``mma.sync`` on the tensor cores, K
  split over a thread-block cluster and its warps, reduced through DSMEM. FP32
  weights always take the SIMT kernel (exact FP32 products).

Small batches take the SIMT kernel and larger ones MMA, from a batch size given by a
cost model per shape (``_mma_min_m``); ROCm and other archs only have the SIMT kernel.
Each kernel's schedule is fixed per shape, so a row's result depends on the batch
only through that switch. See :func:`tiny_gemm_bf16`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, TypeAlias, Union

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.jit.utils.occupancy import get_max_active_clusters
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# NOTE: CUDA constraints
_WARP_THREADS: int = 32
_MAX_BLOCK_THREADS: int = 1024
# Mirrors SIMTTrait in simt.cuh: a lane owns 16 elements of x, so K / 16 lanes cover
# one row of w.
_SIMT_ELEMS_PER_LANE: int = 16
# Rows of x per SIMT CTA; larger batches copy the weight tile over more CTAs.
_SIMT_MAX_M_TILE: int = 16
# Mirrors MMATrait in mma.cuh: a CTA owns 16 rows of w, a warp works on 64-element
# chunks of K, a token tile is 8 rows of x.
_MMA_TILE_N: int = 16
_MMA_CHUNK_K: int = 64

_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)


class SIMTSchedule(NamedTuple):
    n_tile: int  # rows of w per CTA
    max_m_tile: int  # rows of x per CTA
    rows_per_lane: int  # rows of w per lane group of K / 16 lanes


class MMASchedule(NamedTuple):
    num_warps: int
    cluster_size: int
    max_m_tiles: int  # token tiles of 8 rows per CTA, at most


Schedule: TypeAlias = Union[SIMTSchedule, MMASchedule]

# MMA from the m where SIMT's per-row cost, K * (4 + N_TILE) / 128 cycles (x re-read from
# L2 + FMAs), outgrows MMA's fixed DSMEM reduction (~300 cycles). Fitted on B200; m = 1
# always takes SIMT.
_MMA_FIXED_CYCLES: int = 300
_NEVER: int = 1 << 30

# Per-shape tuning (B200 / H200 measurements), overriding the rules above and in
# _simt_schedule.
# K <= 512 leaves MMA little K to split, outside the cost model.
_TUNED_MMA_MIN_M = {
    (1536, 128): 8,
    (128, 512): _NEVER,
}
# 128x512 never takes MMA, and its 128 KB weight is cheap to copy: 4 rows of x per CTA
# beat the default 16 at every m (up to 2x at m = 16).
_TUNED_SIMT_SCHEDULES = {
    (128, 512): SIMTSchedule(n_tile=4, max_m_tile=4, rows_per_lane=1),
}


@cache_once
def _get_num_sm() -> int:
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).multi_processor_count


@cache_once
def _arch_family() -> str:
    if is_hip_runtime():
        return "hip"
    return {9: "sm90", 10: "sm100"}.get(get_jit_cuda_arch().major, "other")


def _simt_lanes(k: int) -> int:
    return k // _SIMT_ELEMS_PER_LANE


def _supports_simt(n: int, k: int) -> bool:
    lanes = _simt_lanes(k)
    if k % _SIMT_ELEMS_PER_LANE or lanes == 0:
        return False
    subwarp = _WARP_THREADS % lanes == 0 and any(
        n % d == 0 and (d * lanes) % _WARP_THREADS == 0 for d in range(1, n + 1)
    )
    cta = lanes % _WARP_THREADS == 0 and lanes <= _MAX_BLOCK_THREADS
    return subwarp or cta


def _supports_mma(n: int, k: int, weight_dtype: torch.dtype) -> bool:
    return (
        weight_dtype == torch.bfloat16
        and _arch_family() in ("sm90", "sm100")
        and n % _MMA_TILE_N == 0
        and k % _MMA_CHUNK_K == 0
    )


@cache_once
def _simt_schedule(n: int, k: int) -> SIMTSchedule:
    """A tuned CTA shape, else as few rows of w per CTA as fit one wave with up to
    16 rows of x."""
    if (n, k) in _TUNED_SIMT_SCHEDULES:
        return _TUNED_SIMT_SCHEDULES[(n, k)]
    lanes = _simt_lanes(k)
    if lanes <= _WARP_THREADS:
        # One row of w per lane group, N_TILE groups per CTA.
        candidates = [
            d
            for d in range(1, min(n, _MAX_BLOCK_THREADS // lanes) + 1)
            if n % d == 0 and d * lanes % _WARP_THREADS == 0
        ]
    else:
        # One lane group (a multiple of 32 lanes) covers the CTA and loops over N_TILE
        # rows; one thread finishes one output.
        candidates = [
            d for d in range(1, min(n, lanes // _SIMT_MAX_M_TILE) + 1) if n % d == 0
        ]
    n_tile = next((d for d in candidates if n // d <= _get_num_sm()), candidates[-1])
    rows_per_lane = 1 if lanes <= _WARP_THREADS else n_tile
    return SIMTSchedule(n_tile, _SIMT_MAX_M_TILE, rows_per_lane)


@cache_once
def _mma_schedule(n: int, k: int) -> MMASchedule:
    """Warps per CTA, CTAs per cluster and token tiles per CTA for an [n, k] weight."""
    num_tiles = n // _MMA_TILE_N
    num_chunks = k // _MMA_CHUNK_K

    # About two chunks per warp, then an even K split, then one CTA per SM, then the
    # most CTAs, then fewer warps (B200 measurements). Clusters of 3 to 8 keep two
    # CTAs per SM to stay co-resident, so at most 8 warps; clusters of 1 or 2 fit one
    # wave at 16.
    def fits(cluster: int, warps: int) -> bool:
        # All clusters co-resident at the CTAs per SM the launcher plans for (mirrors
        # MMATrait::kWaveCTAsPerSM): a cluster's CTAs share one GPC, so this is the
        # driver's cluster occupancy, not the SM count.
        if cluster == 1:
            return True
        small_slice = _MMA_TILE_N * k // cluster * torch.bfloat16.itemsize <= 32 * 1024
        ctas_per_sm = 2 if small_slice and warps <= 8 else 1
        return num_tiles <= get_max_active_clusters(cluster, ctas_per_sm)

    def score(schedule: tuple[int, int]) -> tuple:
        cluster, warps = schedule
        splits = cluster * warps
        max_chunks = -(-num_chunks // splits)
        num_ctas = num_tiles * cluster
        return (
            -abs(max_chunks - 2),
            num_chunks % splits == 0,
            num_ctas <= _get_num_sm(),
            num_ctas,
            -warps,
        )

    cluster, warps = max(
        (
            (cluster, warps)
            for cluster in range(1, min(8, num_chunks) + 1)
            for warps in range(1, (16 if cluster <= 2 else 8) + 1)
            if cluster * warps <= num_chunks and fits(cluster, warps)
        ),
        key=score,
    )
    # Token tiles per CTA: x and the accumulators of 8 tiles fit next to up to 128
    # bytes of weight registers per lane (two chunks of bf16).
    max_chunks = -(-num_chunks // (warps * cluster))
    weight_bytes = max_chunks * 2 * 16 * torch.bfloat16.itemsize
    return MMASchedule(warps, cluster, 8 if weight_bytes <= 128 else 4)


@cache_once
def _mma_min_m(n: int, k: int, weight_dtype: torch.dtype) -> int:
    """The smallest m that takes the MMA kernel on this device."""
    simt_ok = _supports_simt(n, k)
    mma_ok = _supports_mma(n, k, weight_dtype)
    if not (simt_ok or mma_ok):
        raise ValueError(f"tiny_gemm: unsupported shape N={n}, K={k}")
    if not mma_ok:
        return _NEVER
    if not simt_ok:
        return 1
    if (n, k) in _TUNED_MMA_MIN_M:
        return _TUNED_MMA_MIN_M[(n, k)]
    per_row = k * (4 + _simt_schedule(n, k).n_tile) / 128
    return max(2, round(1 + _MMA_FIXED_CYCLES / per_row))


@cache_once
def _jit_module(
    n: int,
    k: int,
    weight_dtype: torch.dtype,
    out_dtype: torch.dtype,
    schedule: Schedule,
) -> Module:
    if weight_dtype not in _WEIGHT_DTYPES:
        raise ValueError("tiny_gemm requires BF16 or FP32 weights")
    # Template arguments: N, K, [Weight,] Out, the schedule's fields in order, kUsePDL;
    # the MMA kernel is BF16-only.
    pdl = is_arch_support_pdl()
    if isinstance(schedule, MMASchedule):
        name, kernel = "mma", "TinyMMAKernel"
        args = make_cpp_args(n, k, out_dtype, *schedule, pdl)
    else:
        name, kernel = "simt", "TinySIMTKernel"
        args = make_cpp_args(n, k, weight_dtype, out_dtype, *schedule, pdl)
    return load_jit(
        f"tiny_gemm_{name}",
        *args,
        cuda_files=[f"gemm/tiny_gemm/{name}.cuh"],
        cuda_wrappers=[("run", f"{kernel}<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _get_module(
    n: int,
    k: int,
    weight_dtype: torch.dtype,
    out_dtype: torch.dtype,
    use_mma: bool,
) -> Module:
    schedule = _mma_schedule(n, k) if use_mma else _simt_schedule(n, k)
    return _jit_module(n, k, weight_dtype, out_dtype, schedule)


def _run(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    n, k = w.shape
    use_mma = x.shape[0] >= _mma_min_m(n, k, w.dtype)
    _get_module(n, k, w.dtype, out.dtype, use_mma).run(x, w, out)


@register_custom_op(op_name="tiny_gemm_bf16", mutates_args=["out"])
def _tiny_gemm_custom_op(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _run(x, w, out)


@cache_once
def can_use_tiny_gemm(n: int, k: int) -> bool:
    """Whether :func:`tiny_gemm_bf16` can serve ``[m, k] @ [n, k].T`` (any m).
    Callers fall back to a general GEMM when this is False."""
    return _supports_simt(n, k) or _supports_mma(n, k, torch.bfloat16)


@debug_kernel_api
def tiny_gemm_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Equal to `torch.nn.functional.linear(x, w)`, accumulated in FP32.
    Call :func:`can_use_tiny_gemm` first: shapes outside the supported set raise
    rather than falling back.

    :param x: Shape [m, k], must be bf16_t
    :param w: Shape [n, k], BF16 or FP32; never downcast
    """
    n = w.size(0)
    if out is None:
        out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
        out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
    else:
        assert out_dtype is None or out_dtype == out.dtype
    _tiny_gemm_custom_op(x, w, out)
    return out
