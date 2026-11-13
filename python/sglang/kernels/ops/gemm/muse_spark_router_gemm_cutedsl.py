"""Muse Spark router logits on SM10x: serving tactics and entry points of the TF32 cluster split-K GEMM.

The kernel is in ``muse_spark_router_gemm_cutedsl_kernel``. This module imports no CuTe DSL, so
models can import the eligibility check on platforms without ``cutlass``; the kernel module loads
on the first launch.
"""

from __future__ import annotations

import functools

import msgspec
import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import is_arch_support_pdl

# Tactic space. cta_m = experts per CTA, cta_n = tokens per CTA, split_k = cluster
# size along K (16 is the non-portable cluster maximum), cta_k = K per pipeline stage.
_MMA_M = (64, 128)
_MMA_N = (8, 16, 32, 64, 128)
_SPLIT_K = (2, 4, 8, 16)
_CTA_K = (32, 64, 128)
# Pipeline depth: as deep as shared memory allows, capped at _MAX_STAGES by default;
# the SM107 serve table raises the cap (_SM107_STAGES) so shared memory binds there too.
_MIN_STAGES = 2
_MAX_STAGES = 12

_FP32_BYTES = 4
_AB_BUFFER_ALIGN_BYTES = 1024  # 128 B swizzle atoms of sA / sB
_MAILBOX_ALIGN_BYTES = 128
_MBARRIER_BYTES = 8


class RouterTactic(msgspec.Struct, frozen=True):
    """One specialization: expert tile, token tile, cluster split-K, stages, K tile."""

    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int
    cta_k: int = 64
    # Programmatic dependent launch (when the device supports it): the weight
    # loads start before the previous kernel finishes, and the next kernel can
    # launch as soon as this one has issued its weight loads.
    pdl: bool = True


# =====================================================================
# Host side: SMEM sizing, tactics, entry points.
# =====================================================================
def _smem_bytes(tactic: RouterTactic, stages: int) -> int:
    """Upper bound of the kernel's shared memory (allocation order and alignment)."""

    def align(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    # SharedStorage: ab pipeline (2 per stage), acc pipeline (2), mbar_reduce, TMEM slot.
    cursor = (2 * stages + 3) * _MBARRIER_BYTES + 4
    stage_bytes = tactic.cta_k * _FP32_BYTES * stages
    cursor = align(cursor, _AB_BUFFER_ALIGN_BYTES) + tactic.mma_m * stage_bytes  # sA
    cursor = align(cursor, _AB_BUFFER_ALIGN_BYTES) + tactic.mma_n * stage_bytes  # sB
    tile = tactic.mma_m * tactic.mma_n
    cursor = align(cursor, _MAILBOX_ALIGN_BYTES) + tile * _FP32_BYTES  # sStage
    slots = (tactic.split_k - 1) * (tile // tactic.split_k) * _FP32_BYTES
    return align(cursor, _MAILBOX_ALIGN_BYTES) + slots  # sSlots


def _max_stages(
    tactic: RouterTactic, smem_capacity: int, limit: int = _MAX_STAGES
) -> int:
    return next(
        (
            s
            for s in range(limit, _MIN_STAGES - 1, -1)
            if _smem_bytes(tactic, s) <= smem_capacity
        ),
        0,
    )


def default_tactic(
    m: int,
    e: int,
    k: int,
    split_k: int = 0,
    mma_m: int = 64,
    stages: int = 0,
    cta_k: int = 64,
    mma_n: int = 0,
    pdl: bool = True,
    smem_capacity: int | None = None,
    max_stages_limit: int = _MAX_STAGES,
) -> RouterTactic:
    smem_capacity = smem_capacity or current_arch().smem_bytes
    # Token tile: the smallest covering M, or an explicit one (larger M runs
    # several token tiles).
    mma_n = mma_n or next((n for n in _MMA_N if n >= m), _MMA_N[-1])
    # Largest split whose sStage + sSlots still leave room for 4 pipeline stages.
    for split in (split_k,) if split_k else reversed(_SPLIT_K):
        tactic = RouterTactic(mma_m, mma_n, split, _MIN_STAGES, cta_k, pdl)
        max_stages = _max_stages(tactic, smem_capacity, max_stages_limit)
        if max_stages >= (_MIN_STAGES if split_k else 4) and (k // cta_k) % split == 0:
            break
    tactic = msgspec.structs.replace(
        tactic, ab_stages=min(stages, max_stages) if stages else max_stages
    )
    validate(tactic, m, e, k, smem_capacity)
    return tactic


def validate(
    tactic: RouterTactic, m: int, e: int, k: int, smem_capacity: int | None = None
) -> None:
    """Reject a tactic that cannot serve ``x[m, k] @ w[e, k]^T``."""
    smem_capacity = smem_capacity or current_arch().smem_bytes
    if tactic.mma_m not in _MMA_M or tactic.mma_n not in _MMA_N:
        raise ValueError(f"unsupported tile {tactic}")
    if tactic.split_k not in _SPLIT_K or tactic.cta_k not in _CTA_K:
        raise ValueError(f"unsupported split_k / cta_k in {tactic}")
    if m < 1 or e % tactic.mma_m:
        raise ValueError(f"M={m}, E={e} do not fit {tactic}")
    if k % tactic.cta_k or (k // tactic.cta_k) % tactic.split_k:
        raise ValueError(f"K={k} does not split into {tactic.split_k} x {tactic.cta_k}")
    if (
        tactic.ab_stages < _MIN_STAGES
        or _smem_bytes(tactic, tactic.ab_stages) > smem_capacity
    ):
        raise ValueError(f"tactic {tactic} does not fit in {smem_capacity} B of SMEM")
    tile = tactic.mma_m * tactic.mma_n
    if (tile // tactic.split_k) % 4:
        raise ValueError(f"tile {tactic} gives a rank segment of partial 16 B pieces")


def router_logits_tf32(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    tactic: RouterTactic | None = None,
) -> torch.Tensor:
    """``x[M, H]`` fp32 @ ``w[E, H]^T`` fp32 -> fp32 ``[M, E]`` (TF32 MMA, fp32 accumulate)."""
    m, h = x.shape
    e, k = w.shape
    assert k == h and x.dtype == w.dtype == torch.float32
    assert x.is_contiguous() and w.is_contiguous()
    smem_capacity = current_arch().smem_bytes
    if tactic is None:
        tactic = serve_tactic(m, e, k, smem_capacity, _capability(x.device.index))
    validate(tactic, m, e, k, smem_capacity)
    if out is None:
        out = torch.empty((m, e), dtype=torch.float32, device=x.device)
    # The CuTe DSL kernel module loads on the first launch, not when models import this one.
    from sglang.kernels.ops.gemm import muse_spark_router_gemm_cutedsl_kernel

    muse_spark_router_gemm_cutedsl_kernel.run(
        tactic, x, w, out, tactic.pdl and is_arch_support_pdl()
    )
    return out


# Serving tactics per device, keyed by compute capability: (max M, overrides for
# default_tactic) in increasing M. A device without a table keeps cuBLAS; add or
# replace one with register_serve_table().
#
# SM103 (GB300): picked in context, not in isolation: TP4 / TP8 decode MoE front (projection
# slice -> gather + norm + NVFP4 quant on the main stream, router + top-8 routing on the side stream),
# CUDA graph.
#
# SM107: picked in TP4 / TP8 serving (DFlash decode, nsys, expert start after the attention
# boundary, routing metadata on the router stream), not in isolation: beside the pooler GEMM a
# 16-CTA cluster waits for SMs, which a standalone run does not show. M ranges where no tactic
# beat cuBLAS at both TP sizes keep cuBLAS.
#
# A None entry keeps cuBLAS for its M range.
_SM107_STAGES = 24
_SERVE_TABLES = {
    (10, 3): (
        (16, dict(split_k=8, cta_k=128)),
        (64, dict(split_k=4, cta_k=64, mma_n=32)),
        (192, dict(split_k=8, cta_k=64, mma_n=64)),
        (256, dict(split_k=4, cta_k=64, mma_n=32)),
        (512, dict(split_k=4, cta_k=64, mma_n=64, pdl=False)),
        (1024, dict(split_k=4, cta_k=32, mma_m=128, mma_n=64, pdl=False)),
    ),
    (10, 7): (
        (
            32,
            dict(split_k=8, cta_k=32, mma_n=64, max_stages_limit=_SM107_STAGES),
        ),
        (
            64,
            dict(split_k=16, cta_k=128, mma_n=32, max_stages_limit=_SM107_STAGES),
        ),
        (
            128,
            dict(split_k=16, cta_k=32, mma_n=64, max_stages_limit=_SM107_STAGES),
        ),
        (256, None),
        (
            512,
            dict(
                split_k=4,
                cta_k=32,
                mma_m=128,
                mma_n=64,
                max_stages_limit=_SM107_STAGES,
            ),
        ),
        (1024, None),
    ),
}
SERVE_MAX_TOKENS = 1024


def register_serve_table(capability: tuple[int, int], table) -> None:
    """Add or replace the serving tactics of a compute capability: a sequence of
    (max M, default_tactic overrides or None for cuBLAS) in increasing M, covering M
    up to SERVE_MAX_TOKENS."""
    _SERVE_TABLES[tuple(capability)] = tuple(table)
    serve_tactic.cache_clear()


@functools.lru_cache(maxsize=None)
def _capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def _serve_overrides(m: int, capability: tuple) -> dict | None:
    for max_m, overrides in _SERVE_TABLES[capability]:
        if m <= max_m:
            return overrides
    raise ValueError(f"M={m} is above the serving table of {capability}")


@functools.lru_cache(maxsize=None)
def serve_tactic(
    m: int, e: int, k: int, smem_capacity: int = 0, capability: tuple = (10, 3)
) -> RouterTactic:
    smem_capacity = smem_capacity or current_arch().smem_bytes
    overrides = _serve_overrides(m, capability)
    if overrides is None:
        raise ValueError(f"M={m} runs cuBLAS on {capability}")
    return default_tactic(m, e, k, smem_capacity=smem_capacity, **overrides)


def router_tf32_eligible(x: torch.Tensor, w: torch.Tensor) -> bool:
    """FP32 router logits for 2 <= M <= SERVE_MAX_TOKENS on a device with a serving
    table (M = 1 keeps the GEMV), only with TF32 matmuls allowed: under "highest"
    cuBLAS keeps the logits FP32."""
    return (
        x.is_cuda
        and torch.get_float32_matmul_precision() != "highest"
        and x.dim() == 2
        and 2 <= x.shape[0] <= SERVE_MAX_TOKENS
        and _capability(x.device.index) in _SERVE_TABLES
        and _serve_overrides(x.shape[0], _capability(x.device.index)) is not None
        and x.dtype == torch.float32
        and w.dtype == torch.float32
        and x.shape[1] == w.shape[1]
        and x.shape[1] % 128 == 0
        and w.shape[0] % 128 == 0
        and x.is_contiguous()
        and w.is_contiguous()
        and x.data_ptr() % 32 == 0
        and w.data_ptr() % 32 == 0
    )


def router_tf32(
    x: torch.Tensor, w: torch.Tensor, tactic: RouterTactic | None = None
) -> torch.Tensor:
    """Router logits with the device's serving tactic, or an explicit one."""
    return router_logits_tf32(x, w, tactic=tactic)
