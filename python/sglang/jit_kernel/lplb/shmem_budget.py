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
    H100   SM_90   227 KB   practical 223 KB  <- default target
    H200   SM_90   227 KB
    H20    SM_90   227 KB
    B200   SM_100  228 KB   practical 224 KB
    SM120           99 KB   (consumer Blackwell -- not in the table below;
                             this is exactly why the budget is read live off
                             the device, see ``_budget_bytes_for_device``)
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

# Practical per-block dynamic shmem caps (bytes) -- FALLBACK ONLY. Production
# code reads the budget live off the device via `_budget_bytes_for_device`,
# so this table can't silently go stale for an arch it has no entry for (as
# happened for SM120: absent here, `_gpu_key_for_device` fell back to
# "h100"'s 223 KiB against a real 99 KiB cap, so a launch that passed the
# pre-flight check still crashed). Kept for old torch builds that predate
# `shared_memory_per_block_optin`, and as a documented reference table.
GPU_BUDGETS_BYTES: dict[str, int] = {
    "a100": 160 * 1024,  # unreachable today: LPLB gates on SM >= 9 (Hopper+)
    "h100": 223 * 1024,
    "h200": 223 * 1024,
    "h20": 223 * 1024,
    "b200": 224 * 1024,
}

# H100/H200/H20 share a budget, so SM major 9 covers all three.
_SM_MAJOR_TO_GPU_KEY = {8: "a100", 9: "h100", 10: "b200"}

# `shared_memory_per_block_optin` minus this margin reproduces
# `GPU_BUDGETS_BYTES`'s existing "practical" values (e.g. H100:
# 232448 - 4096 = 223 KiB), confirmed against live device properties.
_PRACTICAL_MARGIN_BYTES = 4096


# Not on the torch.compile trace path (unlike `cache_once`'s users in
# jit_kernel/utils.py), so plain `lru_cache` is fine here — and tests need
# its `.cache_clear()` hook, which `cache_once` doesn't expose.
@functools.lru_cache(maxsize=None)
def _gpu_key_for_device(device) -> str:
    major, _ = torch.cuda.get_device_capability(device)
    key = _SM_MAJOR_TO_GPU_KEY.get(major)
    if key is None:
        logger.warning(
            f"LPLB shmem budget: unrecognized SM major {major} (device "
            f"{device}), falling back to 'h100'."
        )
        key = "h100"
    return key


def _canonicalize_device_index(device: torch.device | int | str | None) -> int:
    """Resolve `device` to a concrete CUDA index, shared by every lookup
    below that caches on the index: a generic ``"cuda"`` / index-less
    ``torch.device`` is NOT cached as given, since it tracks the *current*
    device and a later ``torch.cuda.set_device()`` would otherwise read a
    stale cached entry for it.
    """
    if device is None:
        return torch.cuda.current_device()
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.lru_cache(maxsize=None)
def _budget_bytes_for_device(index: int) -> int:
    """Shmem budget in bytes for device `index`, read directly from device
    properties -- the primary path, immune to a static table missing an
    arch's entry. Falls back to `GPU_BUDGETS_BYTES` via `_gpu_key_for_device`
    if the running torch build predates `shared_memory_per_block_optin`.
    """
    optin = getattr(
        torch.cuda.get_device_properties(index), "shared_memory_per_block_optin", None
    )
    if optin is None:
        logger.warning(
            "LPLB shmem budget: torch build lacks "
            f"shared_memory_per_block_optin (device {index}); falling back "
            "to the static per-GPU table."
        )
        return GPU_BUDGETS_BYTES[_gpu_key_for_device(index)]
    return optin - _PRACTICAL_MARGIN_BYTES


def budget_bytes_for_device(device: torch.device | int | str | None = None) -> int:
    """Shmem budget in bytes for `device` (default: current device). See
    `_budget_bytes_for_device` for the live-property/fallback split.
    """
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
    """Look up a named budget in the static `GPU_BUDGETS_BYTES` table --
    fallback/documentation path; production code should get its cap from
    `budget_bytes_for_device` instead (see its module-docstring rationale).
    """
    key = gpu.lower()
    if key not in GPU_BUDGETS_BYTES:
        raise ValueError(
            f"unknown gpu '{gpu}', expected one of {sorted(GPU_BUDGETS_BYTES)}"
        )
    return GPU_BUDGETS_BYTES[key]


# Default cap for callers that don't pass one explicitly (e.g. quick checks
# from a REPL); production call sites always pass the device's actual
# `budget_bytes_for_device(...)`.
_DEFAULT_BUDGET_BYTES = GPU_BUDGETS_BYTES["h100"]


def fits(nc: int, nv: int, budget_bytes: int = _DEFAULT_BUDGET_BYTES) -> bool:
    return shmem_bytes(nc, nv) <= budget_bytes


def assert_fits(nc: int, nv: int, budget_bytes: int = _DEFAULT_BUDGET_BYTES) -> None:
    """Raise if the fused kernel will not fit in `budget_bytes` of shared memory."""
    used = shmem_bytes(nc, nv)
    if used > budget_bytes:
        raise ValueError(
            f"fused IPM kernel needs {used/1024:.1f} KiB of shared memory for "
            f"NC={nc}, NV={nv}, but the device budget is {budget_bytes/1024:.1f} "
            f"KiB/block. Either reduce problem size or switch to a tiled design."
        )


def max_nc_for_nv(nv: int, budget_bytes: int = _DEFAULT_BUDGET_BYTES) -> int:
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


def report(nc: int, nv: int, budget_bytes: int = _DEFAULT_BUDGET_BYTES) -> str:
    """Human-readable summary — used by kernels on init for logging."""
    bd = breakdown(nc, nv)
    status = "FITS" if bd.total_bytes <= budget_bytes else "OVER BUDGET"
    return (
        f"[shmem] NC={nc} NV={nv} | "
        f"A={bd.a_bytes/1024:.1f}K "
        f"ata={bd.ata_bytes/1024:.1f}K "
        f"rest={(bd.c_bytes+bd.x_bytes+bd.rhs_bytes+bd.d_bytes)/1024:.1f}K | "
        f"total={bd.total_bytes/1024:.1f}K / {budget_bytes/1024:.1f}K  {status}"
    )
