"""AOT FP32 DeepSelect Top-K for NVIDIA Hopper (SM90)."""

from __future__ import annotations

import torch

from . import deepselect_ops as _deepselect_ops  # noqa: F401

_INPUT_ALIGNMENT_BYTES = 1024
_OUTPUT_ALIGNMENT_BYTES = 32


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
