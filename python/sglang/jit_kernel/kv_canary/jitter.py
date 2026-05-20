"""Spin-wait kernel host wrapper for the kv-canary timing-jitter fuzzer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def spin_wait_step(*, cycles: torch.Tensor) -> None:
    """Launch one spin-wait kernel that busy-loops for ``cycles[0]`` device clock ticks.

    Args:
        cycles: shape ``[1]``, int64, on CUDA. The kernel reads the cycle count from this device tensor
            every launch (or every cuda-graph replay), so the host can re-randomize between replays
            via ``cycles.copy_(...)`` without re-capturing the graph. ``cycles[0] <= 0`` returns
            immediately after the load.
    """
    if cycles.dtype != torch.int64:
        raise ValueError(f"kv-canary jitter: cycles must be int64, got {cycles.dtype}")
    if cycles.device.type != "cuda":
        raise ValueError(
            f"kv-canary jitter: cycles must live on CUDA, got {cycles.device}"
        )
    if cycles.shape != (1,):
        raise ValueError(
            f"kv-canary jitter: cycles must have shape [1], got {tuple(cycles.shape)}"
        )
    if not cycles.is_contiguous():
        raise ValueError("kv-canary jitter: cycles must be contiguous")

    module = _jit_jitter_module()
    module.spin_wait_step_cuda(cycles)


@cache_once
def _jit_jitter_module() -> "Module":
    return load_jit(
        "kv_canary_jitter",
        cuda_files=["kv_canary/jitter.cuh"],
        cuda_wrappers=[
            ("spin_wait_step_cuda", "canary::spin_wait_step_cuda"),
        ],
    )
