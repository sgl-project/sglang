"""AOT DeepSelect Top-K for supported NVIDIA CUDA architectures."""

from __future__ import annotations

import functools
from typing import Optional

import torch

from . import deepselect_ops as _deepselect_ops  # noqa: F401

_INPUT_ALIGNMENT_BYTES = 1024
_OUTPUT_ALIGNMENT_BYTES = 32


@functools.lru_cache(maxsize=1)
def get_stride_requirement() -> tuple[int, int]:
    """Return the input and output row-stride requirements in bytes."""
    return _INPUT_ALIGNMENT_BYTES, _OUTPUT_ALIGNMENT_BYTES


def get_deepselect_supported_architectures() -> tuple[int, ...]:
    """Return CUDA compute capabilities compiled into the AOT extension."""
    return tuple(_deepselect_ops.get_supported_architectures())


def is_deepselect_supported(device=None) -> bool:
    """Return whether the AOT extension contains code for a CUDA device."""
    if torch.version.cuda is None or not torch.cuda.is_available():
        return False
    if isinstance(device, int):
        device_index = device
    else:
        try:
            normalized_device = (
                torch.device("cuda", torch.cuda.current_device())
                if device is None
                else torch.device(device)
            )
        except (RuntimeError, TypeError, ValueError):
            return False
        if normalized_device.type != "cuda":
            return False
        device_index = (
            torch.cuda.current_device()
            if normalized_device.index is None
            else normalized_device.index
        )
    try:
        return bool(_deepselect_ops.is_supported_device(device_index))
    except RuntimeError:
        return False


def _aligned_empty(rows: int, cols: int, alignment_bytes: int, dtype, device):
    alignment = alignment_bytes // dtype.itemsize
    stride = (cols + alignment - 1) // alignment * alignment
    return torch.empty((rows, stride), dtype=dtype, device=device)[:, :cols]


def topk(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    begin: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int64,
    sorted_index: bool = False,
    hint: Optional[torch.Tensor] = None,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Select the largest values from every input row.

    This follows the public DeepSelect ``topk`` interface. ``end`` contains the
    per-row exclusive valid length and is the ``topk_lengths`` input used by
    variable-length decode.
    """
    if hint is not None:
        raise ValueError("hint is not supported currently")

    values = (
        _aligned_empty(
            input.shape[0],
            topk,
            _OUTPUT_ALIGNMENT_BYTES,
            input.dtype,
            input.device,
        )
        if return_value
        else None
    )
    if output_idx is None:
        output_idx = _aligned_empty(
            input.shape[0],
            topk,
            _OUTPUT_ALIGNMENT_BYTES,
            indices_type,
            input.device,
        )
    elif output_idx.dtype != indices_type:
        raise ValueError("output_idx dtype must match indices_type")

    torch.ops.sgl_kernel.deepselect_topk(
        input,
        topk,
        begin,
        end,
        sorted,
        sorted_index,
        values,
        output_idx,
        output_idx_offset,
        idx_oob_fill_value,
        value_oob_fill_value,
        return_value,
        abort_when_nan_found,
    )
    return values, output_idx


deepselect_topk = topk
