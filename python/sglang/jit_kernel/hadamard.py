from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import KERNEL_PATH, cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_hadamard_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    hadamard_include_dir = (KERNEL_PATH / "csrc" / "fast-hadamard-transform").resolve()
    return load_jit(
        "hadamard",
        *args,
        cuda_files=["fast-hadamard-transform/hadamard_jit.cuh"],
        cuda_wrappers=[
            ("hadamard_transform", f"HadamardKernel<{args}>::run"),
            ("hadamard_transform_12n", f"Hadamard12NKernel<{args}>::run"),
            ("hadamard_transform_20n", f"Hadamard20NKernel<{args}>::run"),
            ("hadamard_transform_28n", f"Hadamard28NKernel<{args}>::run"),
            ("hadamard_transform_40n", f"Hadamard40NKernel<{args}>::run"),
        ],
        extra_include_paths=[str(hadamard_include_dir)],
    )


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("hadamard_transform only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)
    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    if dim_og % 8 != 0:
        x = torch.nn.functional.pad(x, (0, 8 - dim_og % 8))
    dim = x.size(1)

    out = torch.empty_like(x)
    module = _jit_hadamard_module(x.dtype)
    module.hadamard_transform(x, out, scale)

    if dim_og % 8 != 0:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)


def hadamard_transform_12n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("hadamard_transform_12n only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)
    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    pad_multiple = 4 * 12
    if dim_og % pad_multiple != 0:
        x = torch.nn.functional.pad(x, (0, pad_multiple - dim_og % pad_multiple))

    out = torch.empty_like(x)
    module = _jit_hadamard_module(x.dtype)
    module.hadamard_transform_12n(x, out, scale)

    if dim_og % pad_multiple != 0:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)


def hadamard_transform_20n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("hadamard_transform_20n only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)
    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    pad_multiple = 4 * 20
    if dim_og % pad_multiple != 0:
        x = torch.nn.functional.pad(x, (0, pad_multiple - dim_og % pad_multiple))

    out = torch.empty_like(x)
    module = _jit_hadamard_module(x.dtype)
    module.hadamard_transform_20n(x, out, scale)

    if dim_og % pad_multiple != 0:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)


def hadamard_transform_28n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("hadamard_transform_28n only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)
    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    pad_multiple = 4 * 28
    if dim_og % pad_multiple != 0:
        x = torch.nn.functional.pad(x, (0, pad_multiple - dim_og % pad_multiple))

    out = torch.empty_like(x)
    module = _jit_hadamard_module(x.dtype)
    module.hadamard_transform_28n(x, out, scale)

    if dim_og % pad_multiple != 0:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)


def hadamard_transform_40n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("hadamard_transform_40n only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)
    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    pad_multiple = 4 * 40
    if dim_og % pad_multiple != 0:
        x = torch.nn.functional.pad(x, (0, pad_multiple - dim_og % pad_multiple))

    out = torch.empty_like(x)
    module = _jit_hadamard_module(x.dtype)
    module.hadamard_transform_40n(x, out, scale)

    if dim_og % pad_multiple != 0:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)
