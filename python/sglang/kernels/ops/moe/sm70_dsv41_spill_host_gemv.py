"""SM70 host MXFP4 GEMV for spilled decode experts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_STARTED = False
_START_FAILED = False


@cache_once
def _module() -> Module:
    cap = torch.cuda.get_device_capability()
    if cap[0] != 7:
        raise RuntimeError(
            f"sm70_dsv41 spill_host_gemv requires SM70 (Volta); got SM{cap[0]}{cap[1]}"
        )
    return load_jit(
        "sm70_dsv41_spill_host_gemv",
        cpp_files=["sm70_dsv41_host_mxfp4_gemv.h"],
        cuda_files=["sm70_dsv41_spill_host_gemv.cuh"],
        cuda_wrappers=[
            ("host_gemv_start", "sm70_dsv41::host_gemv_start"),
            ("host_gemv_stop", "sm70_dsv41::host_gemv_stop"),
            ("host_mxfp4_moe_expert", "sm70_dsv41::host_mxfp4_moe_expert"),
            ("spill_request", "sm70_dsv41::spill_request"),
            ("spill_join", "sm70_dsv41::spill_join"),
        ],
        extra_include_paths=[str(KERNEL_PATH / "csrc")],
        extra_cflags=[
            "-O3",
            "-mavx2",
            "-mfma",
            "-mf16c",
            "-pthread",
            "-march=native",
            "-funroll-loops",
        ],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lpthread"],
    )


def sm70_dsv41_host_gemv_enabled() -> bool:
    return bool(envs.SGLANG_DSV41_HOST_GEMV.get())


def sm70_dsv41_host_gemv_available() -> bool:
    if not sm70_dsv41_host_gemv_enabled() or not torch.cuda.is_available():
        return False
    if _START_FAILED:
        return False
    try:
        return torch.cuda.get_device_capability()[0] == 7
    except (AssertionError, RuntimeError):
        return False


def host_gemv_start(n_threads: Optional[int] = None) -> bool:
    """Start the per-process CPU worker pool and mapped mailbox. Idempotent."""
    global _STARTED, _START_FAILED
    if _STARTED:
        return True
    if _START_FAILED:
        return False
    if n_threads is None:
        n_threads = int(envs.SGLANG_DSV41_HOST_GEMV_THREADS.get() or 4)
    n_threads = max(1, min(int(n_threads), 16))
    try:
        _module().host_gemv_start(
            torch.tensor([n_threads], dtype=torch.int32, device="cpu")
        )
        _STARTED = True
        return True
    except Exception:
        _START_FAILED = True
        return False


def host_gemv_stop() -> None:
    global _STARTED
    if not _STARTED:
        return
    _module().host_gemv_stop(torch.tensor([0], dtype=torch.int32, device="cpu"))
    _STARTED = False


def host_mxfp4_moe_expert(
    x: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    weight: float = 1.0,
    y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CPU GEMV of one marlin_v100 packed expert. All tensors on CPU."""
    if y is None:
        y = torch.empty(x.shape[-1], dtype=torch.float16, device="cpu")
    w = torch.tensor([float(weight)], dtype=torch.float32, device="cpu")
    _module().host_mxfp4_moe_expert(
        x.contiguous(),
        w13.contiguous(),
        s13.contiguous().view(torch.uint8),
        w2.contiguous(),
        s2.contiguous().view(torch.uint8),
        y,
        w,
    )
    return y


def spill_request(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden: torch.Tensor,
    map_table: torch.Tensor,
    host_map: torch.Tensor,
    bases: torch.Tensor,
) -> None:
    """Remap kept ids in place and post spilled hits to the host mailbox."""
    _module().spill_request(
        topk_ids,
        topk_weights,
        hidden,
        map_table,
        host_map,
        bases,
    )


def spill_join(output: torch.Tensor) -> None:
    """Wait for the CPU GEMV and add y into ``output`` in place."""
    _module().spill_join(output)
