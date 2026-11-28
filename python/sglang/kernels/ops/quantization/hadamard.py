from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import KERNEL_PATH, cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_hadamard_module(dtype: torch.dtype, multiple: int, log_n: int) -> Module:
    """One kernel: rows of up to ``multiple << log_n`` elements; unsupported sizes
    fail the build (static_assert in hadamard_jit.cuh)."""
    args = make_cpp_args(multiple, log_n, dtype)
    hadamard_include_dir = (KERNEL_PATH / "csrc" / "fast-hadamard-transform").resolve()
    return load_jit(
        "hadamard",
        *args,
        cuda_files=["fast-hadamard-transform/hadamard_jit.cuh"],
        cuda_wrappers=[("hadamard_transform", f"HadamardKernel<{args}>::run")],
        extra_include_paths=[str(hadamard_include_dir)],
    )


@cache_once
def precompile_hadamard(
    dtype: torch.dtype, dim: int, multiple: int, pad_multiple: int
) -> Module:
    """The module serving rows of ``dim`` elements (``multiple`` x 2^k Hadamard)."""
    padded_dim = (dim + pad_multiple - 1) // pad_multiple * pad_multiple
    # dim / multiple rounded up to a power of two; the kernel zero-fills the rest.
    log_n = (padded_dim // multiple - 1).bit_length()
    return _jit_hadamard_module(dtype, multiple, log_n)


def _hadamard_transform_impl(
    x: torch.Tensor,
    scale: float,
    multiple: int,
    pad_multiple: int,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("hadamard_transform only supports CUDA tensors")

    shapes_og = x.size()
    dim_og = x.size(-1)
    x = x.reshape(-1, dim_og)
    if x.stride(-1) != 1:
        x = x.contiguous()

    needs_pad = dim_og % pad_multiple != 0
    if needs_pad:
        x = torch.nn.functional.pad(x, (0, pad_multiple - dim_og % pad_multiple))

    module = precompile_hadamard(x.dtype, dim_og, multiple, pad_multiple)
    out = torch.empty_like(x)
    module.hadamard_transform(x, out, scale)

    if needs_pad:
        out = out[:, :dim_og]
    return out.reshape(shapes_og)


def _hadamard_transform_fake_impl(
    x: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(fake_impl=_hadamard_transform_fake_impl)
def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard_transform_impl(x, scale, 1, 8)


def hadamard_transform_12n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard_transform_impl(x, scale, 12, 4 * 12)


def hadamard_transform_20n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard_transform_impl(x, scale, 20, 4 * 20)


def hadamard_transform_28n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard_transform_impl(x, scale, 28, 4 * 28)


def hadamard_transform_40n(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _hadamard_transform_impl(x, scale, 40, 4 * 40)
