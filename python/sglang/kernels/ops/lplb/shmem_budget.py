"""Shared-memory budget accounting for the fused IPM kernel.

Fused layout (fp32), one block per LP, all state in shared memory::

    A        NC * NV       constraint matrix     (resident)
    c        NV            cost vector           (resident)
    x        NV            IPM state             (resident)
    ata      NC * NC       KKT matrix / Cholesky factor
    rhs      NC            ax2c, then delta
    d        NV            aliased with r = A.T @ delta

    S_elems = NC*NV + NC*NC + 3*NV + NC

Dynamic shared-memory cap per block (with opt-in via
``cudaFuncAttributeMaxDynamicSharedMemorySize``):

    A100   SM_80   164 KB   practical 160 KB
    H100   SM_90   227 KB   practical 223 KB
    H200   SM_90   227 KB
    H20    SM_90   227 KB
    B200   SM_100  228 KB   practical 224 KB
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# Per-block slack reserved for cuBLASDx workspace and CUDA runtime state.
_RUNTIME_PAD_BYTES = 256

# fp32
_BYTES_PER_ELEM = 4

# Static fallback for torch builds without `shared_memory_per_block_optin`.
GPU_BUDGETS_BYTES: dict[str, int] = {
    "a100": 160 * 1024,
    "h100": 223 * 1024,
    "h200": 223 * 1024,
    "h20": 223 * 1024,
    "b200": 224 * 1024,
}

# H100/H200/H20 share a budget, so SM major 9 covers all three.
_SM_MAJOR_TO_GPU_KEY = {8: "a100", 9: "h100", 10: "b200"}

# Reserve the same 4 KiB margin used by the static practical budgets.
_PRACTICAL_MARGIN_BYTES = 4096


def _fallback_gpu_key_for_device(index: int) -> str:
    major, _ = torch.cuda.get_device_capability(index)
    key = _SM_MAJOR_TO_GPU_KEY.get(major)
    if key is None:
        raise RuntimeError(
            f"LPLB shmem budget: unrecognized SM major {major} for device "
            f"{index}; this torch build does not expose "
            "shared_memory_per_block_optin, so no safe shared-memory budget "
            "can be derived. Upgrade torch or add an explicit architecture budget."
        )
    return key


def _canonicalize_device_index(device: torch.device | int | str | None) -> int:
    """Resolve index-less CUDA devices before using them as cache keys."""
    if device is None:
        return torch.cuda.current_device()
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.lru_cache(maxsize=None)
def _budget_bytes_for_device(index: int) -> int:
    """Return the safe dynamic shared-memory budget for a CUDA device.

    Uses the live device property when available and a fail-closed static
    fallback for older torch builds.
    """
    optin_bytes = getattr(
        torch.cuda.get_device_properties(index), "shared_memory_per_block_optin", None
    )
    if optin_bytes is None:
        logger.warning(
            "LPLB shmem budget: torch build lacks "
            f"shared_memory_per_block_optin (device {index}); falling back "
            "to the static per-GPU table."
        )
        return GPU_BUDGETS_BYTES[_fallback_gpu_key_for_device(index)]
    return optin_bytes - _PRACTICAL_MARGIN_BYTES


def budget_bytes_for_device(device: torch.device | int | str | None = None) -> int:
    """Return the safe shared-memory budget for `device`."""
    return _budget_bytes_for_device(_canonicalize_device_index(device))


@dataclass(frozen=True)
class ShmemBreakdown:
    nc: int
    nv: int
    a_bytes: int
    c_bytes: int
    x_bytes: int
    ata_bytes: int
    rhs_bytes: int
    d_bytes: int
    pad_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.a_bytes
            + self.c_bytes
            + self.x_bytes
            + self.ata_bytes
            + self.rhs_bytes
            + self.d_bytes
            + self.pad_bytes
        )

    def as_kib(self) -> float:
        return self.total_bytes / 1024.0


def shmem_bytes(nc: int, nv: int, bytes_per_elem: int = _BYTES_PER_ELEM) -> int:
    """Exact byte count for the fused layout with the given (NC, NV)."""
    return bytes_per_elem * (nc * nv + nc * nc + 3 * nv + nc) + _RUNTIME_PAD_BYTES


def breakdown(
    nc: int, nv: int, bytes_per_elem: int = _BYTES_PER_ELEM
) -> ShmemBreakdown:
    """Per-array byte breakdown — useful for debugging shmem pressure."""
    b = bytes_per_elem
    return ShmemBreakdown(
        nc=nc,
        nv=nv,
        a_bytes=b * nc * nv,
        c_bytes=b * nv,
        x_bytes=b * nv,
        ata_bytes=b * nc * nc,
        rhs_bytes=b * nc,
        d_bytes=b * nv,
        pad_bytes=_RUNTIME_PAD_BYTES,
    )


def gpu_budget_bytes(gpu: str) -> int:
    """Look up a named static fallback budget."""
    key = gpu.lower()
    if key not in GPU_BUDGETS_BYTES:
        raise ValueError(
            f"unknown gpu '{gpu}', expected one of {sorted(GPU_BUDGETS_BYTES)}"
        )
    return GPU_BUDGETS_BYTES[key]


def fits(nc: int, nv: int, budget_bytes: int) -> bool:
    return shmem_bytes(nc, nv) <= budget_bytes


def assert_fits(nc: int, nv: int, budget_bytes: int) -> None:
    """Raise if the fused kernel will not fit in `budget_bytes` of shared memory."""
    used = shmem_bytes(nc, nv)
    if used > budget_bytes:
        raise ValueError(
            f"fused IPM kernel needs {used / 1024:.1f} KiB of shared memory for "
            f"NC={nc}, NV={nv}, but the device budget is {budget_bytes / 1024:.1f} "
            f"KiB/block. Either reduce problem size or switch to a tiled design."
        )


def max_nc_for_nv(nv: int, budget_bytes: int) -> int:
    """Largest NC that fits for a given NV. Solves
        4 * (NC^2 + (NV+1)*NC + 3*NV) + pad <= budget_bytes
    via the quadratic formula (monotone in NC). Returns 0 if even NC=1 overflows.
    """
    b = _BYTES_PER_ELEM
    # budget_bytes - pad >= b * (NC^2 + (NV+1)*NC + 3*NV)
    rhs = (budget_bytes - _RUNTIME_PAD_BYTES) / b - 3 * nv
    if rhs <= 0:
        return 0
    # NC^2 + (NV+1)*NC - rhs <= 0
    import math

    disc = (nv + 1) ** 2 + 4 * rhs
    nc_max = int((-(nv + 1) + math.sqrt(disc)) / 2.0)
    while nc_max > 0 and shmem_bytes(nc_max, nv) > budget_bytes:
        nc_max -= 1
    return max(nc_max, 0)


def report(nc: int, nv: int, budget_bytes: int) -> str:
    """Human-readable summary — used by kernels on init for logging."""
    bd = breakdown(nc, nv)
    status = "FITS" if bd.total_bytes <= budget_bytes else "OVER BUDGET"
    return (
        f"[shmem] NC={nc} NV={nv} | "
        f"A={bd.a_bytes / 1024:.1f}K "
        f"ata={bd.ata_bytes / 1024:.1f}K "
        f"rest={(bd.c_bytes + bd.x_bytes + bd.rhs_bytes + bd.d_bytes) / 1024:.1f}K | "
        f"total={bd.total_bytes / 1024:.1f}K / {budget_bytes / 1024:.1f}K  {status}"
    )
