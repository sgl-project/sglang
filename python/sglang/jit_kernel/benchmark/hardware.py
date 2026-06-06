"""Measured hardware roofline ceilings for benchmark %-of-peak columns.

Peaks are measured empirically on the current device (datasheet numbers are
sparsity-inflated and overstate the achievable ceiling) and cached per
(device_name, dtype) for the process lifetime.
"""

from typing import Dict, Tuple

import torch

_BW_CACHE: Dict[str, float] = {}
_FLOPS_CACHE: Dict[Tuple[str, torch.dtype], float] = {}

_BW_BYTES = 256 * 1024 * 1024  # 256 MiB per buffer
_GEMM_M = 8192
_WARMUP = 5
_ITERS = 30


def _device_name() -> str:
    return torch.cuda.get_device_name(torch.cuda.current_device())


def _time_loop(fn, iters: int) -> float:
    for _ in range(_WARMUP):
        fn()
    torch.cuda.synchronize()
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    tic.record()
    for _ in range(iters):
        fn()
    toc.record()
    torch.cuda.synchronize()
    return tic.elapsed_time(toc) / 1000 / iters  # seconds per iter


def peak_bandwidth_gbps() -> float:
    """Achievable HBM bandwidth (GB/s), measured via a large copy (read + write)."""
    key = _device_name()
    if key not in _BW_CACHE:
        n = _BW_BYTES // 4
        src = torch.empty(n, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        secs = _time_loop(lambda: dst.copy_(src), _ITERS)
        _BW_CACHE[key] = (src.nbytes + dst.nbytes) / (1024**3) / secs
    return _BW_CACHE[key]


def peak_tflops(dtype: torch.dtype = torch.bfloat16) -> float:
    """Achievable compute (TFLOP/s) for `dtype`, measured via a large square GEMM.

    Returns NaN for dtypes that do not support a plain ``@`` matmul (e.g. fp8).
    """
    key = (_device_name(), dtype)
    if key not in _FLOPS_CACHE:
        m = _GEMM_M
        try:
            a = torch.randn(m, m, dtype=dtype, device="cuda")
            b = torch.randn(m, m, dtype=dtype, device="cuda")
            secs = _time_loop(lambda: torch.mm(a, b), _ITERS)
            _FLOPS_CACHE[key] = (2 * m * m * m) / secs / 1e12
        except (RuntimeError, TypeError):
            _FLOPS_CACHE[key] = float("nan")
    return _FLOPS_CACHE[key]
