# SPDX-License-Identifier: Apache-2.0
"""aiter's asm block-scale bpreshuffle GEMM (fp8 x fp8 -> bf16), launched from its code
objects into an output its producer already zeroed.

The asm kernels run split-K by accumulating the splits atomically into the bf16 output, so
aiter's launcher zero-fills it with hipMemsetAsync first: one more graph node per call. A
producer that writes the zeros itself launches the same kernel here instead.
"""

from __future__ import annotations

import csv
import ctypes
import os
from functools import lru_cache
from typing import Dict, Tuple

import torch

from sglang.kernels.jit.utils.hipcc import HipKernel, check_hip, hip_lib

# Every field of aiter's KernelArgs (asm_a8w8_blockscale_bpreshuffle.cu) is padded to 16 bytes.
_PTR_FIELDS = ("c", "a", "b", "a_scale", "b_scale", "bias")
_INT_FIELDS = (
    "m",
    "n",
    "k",
    "lda",
    "ldb",
    "ldc",
    "ks",
    "scale_m",
    "scale_n",
    "scale_k",
)


class _KernelArgs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        f
        for name in _PTR_FIELDS
        for f in ((name, ctypes.c_void_p), (f"_pad_{name}", ctypes.c_uint8 * 8))
    ] + [
        f
        for name in _INT_FIELDS
        for f in ((name, ctypes.c_uint32), (f"_pad_{name}", ctypes.c_uint8 * 12))
    ]


assert ctypes.sizeof(_KernelArgs) == 256, (
    "kernarg ABI drift: keep _KernelArgs in step with aiter's KernelArgs"
)
_SCALE_BLOCK = 128


@lru_cache(maxsize=None)
def _kernel_table() -> Dict[str, Tuple[int, int, str]]:
    """Kernel symbol -> (tile_m, tile_n, code object path) of aiter's split-K kernels."""
    from aiter.jit.core import get_asm_dir

    folder = os.path.join(get_asm_dir(), "fp8gemm_blockscale")
    with open(os.path.join(folder, "fp8gemm_bf16_blockscale.csv")) as f:
        return {
            row["knl_name"]: (
                int(row["tile_m"]),
                int(row["tile_n"]),
                os.path.join(folder, row["co_name"]),
            )
            for row in csv.DictReader(f)
            if int(row["splitK"]) and int(row["bpreshuffle"])
        }


@lru_cache(maxsize=None)
def _kernel(device: int, kernel_name: str) -> HipKernel:
    module = ctypes.c_void_p()
    path = _kernel_table()[kernel_name][2]
    with torch.cuda.device(device):
        check_hip(
            hip_lib().hipModuleLoad(ctypes.byref(module), path.encode()),
            f"hipModuleLoad {path}",
        )
        return HipKernel(module, kernel_name)


def asm_split_k_supported(kernel_name: str, n: int, k: int, split_k: int) -> bool:
    entry = _kernel_table().get(kernel_name)
    return entry is not None and n % entry[1] == 0 and k % (_SCALE_BLOCK * split_k) == 0


def gemm_a8w8_blockscale_bpreshuffle_asm_prezeroed(
    xq: torch.Tensor,
    wq: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    kernel_name: str,
    split_k: int,
    bias: torch.Tensor,
) -> torch.Tensor:
    """aiter's gemm_a8w8_blockscale_bpreshuffle_asm into ``out`` that holds zeros, without the
    memset; ``bias`` is the fp32 [1, n] bias (aiter's zero bias buffer for none). The caller
    checks asm_split_k_supported()."""
    m, k = xq.shape
    n = wq.shape[0]
    tile_m, tile_n, _ = _kernel_table()[kernel_name]
    args = _KernelArgs(
        c=out.data_ptr(),
        a=xq.data_ptr(),
        b=wq.data_ptr(),
        a_scale=x_scale.data_ptr(),
        b_scale=w_scale.data_ptr(),
        bias=bias.data_ptr(),
        m=m,
        n=n,
        k=k,
        lda=xq.stride(0),
        ldb=wq.stride(0),
        ldc=out.stride(0) * out.element_size(),
        ks=split_k,
        scale_m=m,
        scale_n=(n + _SCALE_BLOCK - 1) // _SCALE_BLOCK,
        scale_k=(k + _SCALE_BLOCK - 1) // _SCALE_BLOCK,
    )
    _kernel(xq.device.index, kernel_name).launch(
        (n // tile_n * split_k, (m + tile_m - 1) // tile_m),
        args,
        torch.cuda.current_stream(xq.device).cuda_stream,
    )
    return out
