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
"""

from __future__ import annotations

# Per-block slack reserved for cuBLASDx workspace and CUDA runtime state.
_RUNTIME_PAD_BYTES = 256

# fp32
_BYTES_PER_ELEM = 4

# Practical per-block dynamic shmem cap on H100 (bytes), the default target.
_H100_BUDGET_BYTES = 223 * 1024


def shmem_bytes(nc: int, nv: int, bytes_per_elem: int = _BYTES_PER_ELEM) -> int:
    """Exact byte count for the fused layout with the given (NC, NV)."""
    return bytes_per_elem * (nc * nv + nc * nc + 3 * nv + nc) + _RUNTIME_PAD_BYTES


def assert_fits(nc: int, nv: int) -> None:
    """Raise if the fused kernel will not fit on H100."""
    used = shmem_bytes(nc, nv)
    if used > _H100_BUDGET_BYTES:
        raise ValueError(
            f"fused IPM kernel needs {used / 1024:.1f} KiB of shared memory for "
            f"NC={nc}, NV={nv}, but h100 allows "
            f"{_H100_BUDGET_BYTES / 1024:.1f} KiB/block. "
            f"Either reduce problem size or switch to a tiled design."
        )
