"""JIT build of sgl-kernel's ROCm quick all-reduce `qr_*` calls, for INT4 and INT8 at TP4.

The two-shot kernel (kernels/jit/csrc/distributed/quick_all_reduce/) is restructured for latency
but keeps sgl-kernel's buffer sizes, flags and arithmetic, so on bf16 cast to fp16 its outputs
match bitwise. Asked for INT4, it runs the all-reduces below 64 MiB in INT8. A group it does not
serve (another world size or level, no bf16 -> fp16 cast) gets sgl-kernel's quick all-reduce.
"""

import os
from typing import Optional

import sgl_kernel.allreduce as sgl_kernel_qr
import torch

from sglang.kernels.jit.utils import cache_once, load_jit

_ENTRY_POINTS = (
    "init",
    "get_handle",
    "open_handles",
    "all_reduce",
    "destroy",
    "handle_bytes",
)

# Grid cap, also sizing the flags; measured best on MI350X TP4 INT4.
_MAX_BLOCKS = 256

# Max bytes per all-reduce when the caller passes none: sgl-kernel's qr_max_size().
_DEFAULT_MAX_SIZE = 1 << 31

# sgl-kernel's QuickReduceQuantLevel
_INT8, _INT4 = 1, 3
# The latency-bound decode and short-extend sizes, where INT8 (about 1/16 of INT4's error) still
# beats the bf16 custom AR on MI350X TP4 (23.8 vs 31.3 us at 3.1 MB); prefill keeps INT4.
_INT8_BELOW_BYTES = 64 << 20


@cache_once
def _module():
    return load_jit(
        "quick_all_reduce_hip",
        _MAX_BLOCKS,
        cuda_files=["distributed/quick_all_reduce_hip.cuh"],
        cuda_wrappers=[
            (name, f"quick_all_reduce_hip::QuickAllReduce::{name}")
            for name in _ENTRY_POINTS
        ],
        # sgl-kernel's build defines
        extra_cuda_cflags=[
            f"-DQR_MAX_BLOCKS={_MAX_BLOCKS}",
            "-D__HIP_NO_HALF_OPERATORS__=1",
            "-D__HIP_NO_HALF_CONVERSIONS__=1",
        ],
    )


# Handles init_custom_qr took from sgl-kernel, so the other calls route them back there; the
# handles are process-wide, like the communicators that own them.
_SGL_KERNEL_HANDLES: set[int] = set()


def jit_serves(world_size: int) -> bool:
    """Whether the JIT build serves a `world_size` group at the quantization level and
    bf16 -> fp16 cast that `QuickAllReduce` reads from the environment."""
    return (
        world_size == 4
        and os.environ.get("ROCM_QUICK_REDUCE_QUANTIZATION", "NONE") in ("INT4", "INT8")
        and int(os.environ.get("ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", 1)) != 0
    )


def init_custom_qr(
    rank: int, world_size: int, qr_max_size: Optional[int] = None
) -> int:
    if jit_serves(world_size):
        return _module().init(world_size, rank, qr_max_size or 0)
    fa = sgl_kernel_qr.init_custom_qr(world_size, rank, qr_max_size)
    _SGL_KERNEL_HANDLES.add(fa)
    return fa


def qr_get_handle(fa: int) -> torch.Tensor:
    if fa in _SGL_KERNEL_HANDLES:
        return sgl_kernel_qr.qr_get_handle(fa)
    out = torch.empty(_module().handle_bytes(), dtype=torch.uint8)
    _module().get_handle(fa, out)
    return out


def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None:
    if fa in _SGL_KERNEL_HANDLES:
        sgl_kernel_qr.qr_open_handles(fa, handles)
        return
    _module().open_handles(fa, torch.stack(handles))


def _flat(t: torch.Tensor) -> torch.Tensor:
    # QuickAllReduce admits weakly contiguous tensors; sgl-kernel reduces their raw memory.
    return t.as_strided((t.numel(),), (1,))


def qr_all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    quant_level: int,
    cast_bf2half: bool,
) -> None:
    if fa in _SGL_KERNEL_HANDLES:
        sgl_kernel_qr.qr_all_reduce(fa, inp, out, quant_level, cast_bf2half)
        return
    if quant_level == _INT4 and inp.numel() * inp.element_size() < _INT8_BELOW_BYTES:
        quant_level = _INT8
    _module().all_reduce(fa, _flat(inp), _flat(out), quant_level, cast_bf2half)


def qr_destroy(fa: int) -> None:
    if fa in _SGL_KERNEL_HANDLES:
        _SGL_KERNEL_HANDLES.discard(fa)
        sgl_kernel_qr.qr_destroy(fa)
        return
    _module().destroy(fa)


def qr_max_size() -> int:
    return _DEFAULT_MAX_SIZE
