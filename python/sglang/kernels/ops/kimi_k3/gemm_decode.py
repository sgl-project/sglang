"""Decode-shape GEMM for gfx942/gfx950 (MI300X / MI355X).

Wraps gemm_decode.hip: bf16 GEMM optimised for small-M decode batches
(M <= 512) using MFMA intrinsics and LDS-staged cooperative loading.
The kernel is ~2x faster than hipBLASLt at typical decode shapes.

Usage::

    from sglang.kernels.ops.kimi_k3.gemm_decode import gemm_decode, is_available

    if is_available():
        out = gemm_decode(x, weight)   # out = x @ weight.T, bf16
    else:
        out = torch.mm(x, weight.t())
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_lib: Optional[ctypes.CDLL] = None
_fn:  Optional[object]       = None
_AVAILABLE = False


def _build_and_load() -> bool:
    global _lib, _fn, _AVAILABLE
    if _fn is not None:
        return _AVAILABLE

    csrc = Path(__file__).parent.parent.parent / "jit" / "csrc" / "kimi_k3" / "gemm_decode.hip"
    if not csrc.exists():
        return False

    so = csrc.with_suffix(".so")
    if not so.exists() or so.stat().st_mtime < csrc.stat().st_mtime:
        try:
            _compile(csrc, so)
        except Exception as e:
            logger.debug("gemm_decode compile failed: %s", e)
            return False

    try:
        _lib = ctypes.CDLL(str(so))
        fn = _lib.sglang_gemm_decode
        fn.restype  = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # B
            ctypes.c_void_p,  # C
            ctypes.c_int,     # M
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_void_p,  # stream
        ]
        _fn = fn
        _AVAILABLE = True
        return True
    except Exception as e:
        logger.debug("gemm_decode load failed: %s", e)
        return False


def _compile(src: Path, dst: Path) -> None:
    import subprocess, shutil

    hipcc = shutil.which("hipcc")
    if hipcc is None:
        raise FileNotFoundError("hipcc not found")

    gpu = os.environ.get("GPU_TARGETS", os.environ.get("HIP_VISIBLE_DEVICES", ""))
    arch_flag = f"--offload-arch={gpu}" if gpu and "," not in gpu else "--offload-arch=native"

    aiter_inc = Path("/sgl-workspace/aiter/csrc/include")
    inc = [f"-I{aiter_inc}"] if aiter_inc.exists() else []

    cmd = [
        hipcc, "-x", "hip", arch_flag,
        "-O3", "-std=c++20",
        "-mllvm", "-amdgpu-early-inline-all=true",
        "-mllvm", "-amdgpu-function-calls=false",
        *inc,
        str(src), "-shared", "-fPIC", "-o", str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])


def is_available() -> bool:
    return _build_and_load()


def gemm_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Compute x @ weight.T using the MFMA decode kernel.

    Returns None if the kernel is unavailable or x does not satisfy the
    decode-shape contract (2-D, bf16, M <= 512, N <= 4096, contiguous).
    The N <= 4096 guard prevents routing large-N draft-model layers (e.g.
    N=7168 in DSpark) through a tile-fixed kernel that produces wrong output
    at untested output widths.
    """
    if (
        not _AVAILABLE
        or _fn is None
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[0] > 512
        or weight.shape[0] > 4096
    ):
        return None

    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    err = _fn(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_void_p(0),
    )
    return out if err == 0 else None
