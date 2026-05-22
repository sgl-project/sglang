# Copied and adapted from: https://github.com/vllm-project/vllm-metal
# SPDX-License-Identifier: Apache-2.0
"""Tensor bridge between MLX and PyTorch.

Provides zero-copy conversion when possible using Apple Silicon's unified memory.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    import mlx.core as mx

logger = logging.getLogger(__name__)

_MLX_AVAILABLE: bool = False
try:
    import mlx.core as mx  # noqa: F811

    _MLX_AVAILABLE = True
except ImportError:
    pass


def is_mlx_available() -> bool:
    """Return True when the ``mlx`` package can be imported."""
    return _MLX_AVAILABLE


@lru_cache(maxsize=1)
def use_mlx() -> bool:
    """Return True when the user opted-in via ``SGLANG_USE_MLX=1`` **and** MLX is importable."""
    return bool(envs.SGLANG_USE_MLX.get()) and _MLX_AVAILABLE


# MPS has a 4GB (2^32 bytes) limit for MPSTemporaryNDArray allocations.
# Metal may allocate multiple temporary buffers internally, so we use a
# conservative threshold of 1GB to avoid hitting the limit.
# See: https://github.com/anthropics/vllm-metal/issues/43
_MPS_SAFE_SIZE_BYTES = 1 << 30  # 1GB

# MLX to PyTorch dtype mapping
# TODO(perf): float64 is CPU-only in MLX (see ml-explore/mlx#1843).
# When the target device is GPU/MPS we should auto-downcast float64 → float32
# to avoid a runtime error; when the target is CPU we can keep float64.
# For now float64 is omitted from the mapping so it hits the ValueError
# fallback in mlx_to_torch().
MLX_TO_TORCH_DTYPE = (
    {
        mx.float32: torch.float32,
        mx.float16: torch.float16,
        mx.bfloat16: torch.bfloat16,
        mx.int32: torch.int32,
        mx.int64: torch.int64,
        mx.int16: torch.int16,
        mx.int8: torch.int8,
        mx.uint8: torch.uint8,
        mx.bool_: torch.bool,
    }
    if _MLX_AVAILABLE
    else {}
)

# PyTorch to MLX dtype mapping
TORCH_TO_MLX_DTYPE = {v: k for k, v in MLX_TO_TORCH_DTYPE.items()}


def get_torch_device() -> torch.device:
    """Get the PyTorch device for Metal/MPS.

    Returns:
        torch.device for MPS if available, else CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_tensor_size_bytes(array: mx.array) -> int:
    """Calculate the size of an MLX array in bytes.

    Args:
        array: MLX array

    Returns:
        Size in bytes
    """
    return array.size * array.dtype.size


def _is_safe_for_mps(array: mx.array) -> bool:
    """Check if an array is safe to transfer to MPS without hitting size limits.

    MPS has a 4GB limit for MPSTemporaryNDArray, but Metal may allocate
    multiple temporary buffers internally. We use a conservative threshold.

    Args:
        array: MLX array to check

    Returns:
        True if safe to transfer to MPS, False if should stay on CPU
    """
    return _get_tensor_size_bytes(array) < _MPS_SAFE_SIZE_BYTES


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array.

    Uses numpy as an intermediate to enable zero-copy on unified memory.

    Args:
        tensor: PyTorch tensor (can be on any device)

    Returns:
        MLX array with the same data
    """
    # Move to CPU if on MPS for numpy conversion
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()

    tensor = tensor.detach()

    # Note: numpy does not support bfloat16.
    if tensor.dtype == torch.bfloat16:
        return mx.array(tensor)

    return mx.array(tensor.numpy())


# TODO(perf): accept a list/batch of arrays and convert them in one pass
# to reduce the Python ↔ MLX round-trip overhead.
def mlx_to_torch(
    array: mx.array,
    device: torch.device | Literal["mps", "cpu"] | None = None,
    already_contiguous: bool = False,
) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor.

    Uses numpy as an intermediate to enable zero-copy on unified memory.

    Args:
        array: MLX array
        device: Target PyTorch device (default: MPS if available)
        already_contiguous: Skip contiguity check if array is known contiguous

    Returns:
        PyTorch tensor with the same data
    """
    if device is None:
        device = get_torch_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # Use memoryview for zero-copy conversion (bypasses numpy for bfloat16)
    # reference: https://github.com/ml-explore/mlx/issues/403
    torch_dtype = MLX_TO_TORCH_DTYPE.get(array.dtype)
    if torch_dtype is not None:
        if already_contiguous:
            # Fast path: skip contiguity check, single eval
            mx.eval(array)
            buffer = memoryview(array)
        else:
            # MLX views / non-contiguous arrays expose a non-contiguous buffer (or
            # sometimes no usable buffer), which `torch.frombuffer` can't consume.
            # Make contiguous first, then eval once
            array = mx.contiguous(array)
            mx.eval(array)
            buffer = memoryview(array)

        tensor = torch.frombuffer(buffer, dtype=torch_dtype).reshape(array.shape)
    else:
        # Fallback to numpy path for unsupported dtypes
        raise ValueError(f"Unsupported MLX dtype: {array.dtype}")

    # Move to target device, but check for MPS size limits first
    if device.type == "mps":
        if _is_safe_for_mps(array):
            tensor = tensor.to(device)
        else:
            # Large tensor - keep on CPU to avoid MPS 4GB limit crash
            # See: https://github.com/anthropics/vllm-metal/issues/43
            logger.debug(
                "Tensor too large for MPS (%d bytes > %d limit), keeping on CPU",
                _get_tensor_size_bytes(array),
                _MPS_SAFE_SIZE_BYTES,
            )
    elif device.type != "cpu":
        tensor = tensor.to(device)

    return tensor


def sync_mlx() -> None:
    """Synchronize MLX operations.

    Call this before converting MLX arrays to ensure all operations complete.
    """
    # Prefer an explicit MLX barrier when available; otherwise force evaluation.
    # `mx.eval([])` is a no-op, so we evaluate a tiny scalar as a safe fallback.
    try:
        mx.synchronize()
    except (AttributeError, TypeError):
        mx.eval(mx.array(0, dtype=mx.int32))


def sync_torch() -> None:
    """Synchronize PyTorch MPS operations.

    Call this before converting PyTorch tensors to ensure all operations complete.
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


__all__ = [
    "is_mlx_available",
    "use_mlx",
    "mlx_to_torch",
    "torch_to_mlx",
    "get_torch_device",
]
