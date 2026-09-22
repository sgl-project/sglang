"""AOT FP32 DeepSelect Top-K for supported NVIDIA CUDA architectures."""

from __future__ import annotations

import torch

from . import deepselect_ops as _deepselect_ops  # noqa: F401

_INPUT_ALIGNMENT_BYTES = 1024
_OUTPUT_ALIGNMENT_BYTES = 32


def get_deepselect_supported_architectures() -> tuple[int, ...]:
    """Return CUDA compute capabilities compiled into the AOT extension."""
    return tuple(_deepselect_ops.get_supported_architectures())


def is_deepselect_supported(device=None) -> bool:
    """Return whether the AOT extension contains code for a CUDA device."""
    if torch.version.cuda is None or not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError, ValueError):
        return False
    return major * 10 + minor in get_deepselect_supported_architectures()


def _aligned_empty(rows: int, cols: int, alignment_bytes: int, dtype, device):
    alignment = alignment_bytes // dtype.itemsize
    stride = (cols + alignment - 1) // alignment * alignment
    return torch.empty((rows, stride), dtype=dtype, device=device)[:, :cols]


def deepselect_topk_fp32(
    input: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unsorted FP32 values and int32 indices for each input row."""
    if input.dtype != torch.float32 or input.dim() != 2:
        raise ValueError("input must be a 2D float32 tensor")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if not is_deepselect_supported(input.device):
        major, minor = torch.cuda.get_device_capability(input.device)
        raise RuntimeError(
            f"deepselect_topk_fp32 was not compiled for SM{major}{minor}; "
            f"compiled architectures: {get_deepselect_supported_architectures()}"
        )
    if not 0 < topk <= min(4096, input.shape[1]):
        raise ValueError("topk must be in [1, min(4096, input.shape[1])]")

    if (
        input.stride(1) != 1
        or input.stride(0) * input.element_size() % _INPUT_ALIGNMENT_BYTES
    ):
        aligned = _aligned_empty(
            input.shape[0],
            input.shape[1],
            _INPUT_ALIGNMENT_BYTES,
            input.dtype,
            input.device,
        )
        aligned.copy_(input)
        input = aligned

    values = _aligned_empty(
        input.shape[0], topk, _OUTPUT_ALIGNMENT_BYTES, input.dtype, input.device
    )
    indices = _aligned_empty(
        input.shape[0], topk, _OUTPUT_ALIGNMENT_BYTES, torch.int32, input.device
    )
    if input.shape[0] == 0:
        return values, indices
    torch.ops.sgl_kernel.deepselect_topk_fp32(input, values, indices, topk)
    return values, indices
