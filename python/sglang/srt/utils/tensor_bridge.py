# Copied and adapted from: https://github.com/vllm-project/vllm-metal
# SPDX-License-Identifier: Apache-2.0
"""Tensor bridge between MLX and PyTorch on Apple silicon.

The MLX backend requires MLX >= 0.32 and PyTorch >= 2.13, so Metal tensors can
cross the framework boundary through DLPack without a CPU copy. PyTorch CPU
inputs still copy into MLX-owned storage.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    import mlx.core as mx


@lru_cache(maxsize=1)
def _mlx_core():
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError("The MLX tensor bridge requires MLX >= 0.32.0") from None
    return mx


def _get_torch_device() -> torch.device:
    """Get the PyTorch device for Metal/MPS.

    Returns:
        torch.device for MPS if available, else CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array.

    MPS tensors share their Metal allocation through DLPack. The producer is
    synchronized before handing the allocation to MLX because MPS and MLX do
    not share stream state. CPU tensors are copied into MLX-owned storage.

    Args:
        tensor: PyTorch CPU or MPS tensor.

    Returns:
        MLX array with the same data
    """
    mx = _mlx_core()
    tensor = tensor.detach()

    if tensor.device.type == "mps":
        torch.mps.synchronize()
        return mx.asarray(tensor, copy=False)
    if tensor.device.type == "cpu":
        return mx.array(tensor)
    raise ValueError(
        f"The MLX tensor bridge supports CPU and MPS tensors, got {tensor.device}"
    )


def mlx_to_torch(
    array: mx.array,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor.

    MLX arrays share their unified-memory allocation with PyTorch through
    DLPack, including explicit CPU views. MLX is evaluated before the handoff
    because the frameworks do not share stream state. Other target devices copy.

    Args:
        array: MLX array
        device: Target PyTorch device (default: MPS if available)

    Returns:
        PyTorch tensor with the same data
    """
    mx = _mlx_core()
    target_device = _get_torch_device() if device is None else torch.device(device)

    mx.eval(array)
    if target_device.type == "cpu":
        # MLX owns CPU-accessible unified memory, so request a CPU DLPack view
        # explicitly instead of importing on MPS and copying back to CPU.
        dlpack = array.__dlpack__(dl_device=(1, 0), copy=False)
        tensor = torch.utils.dlpack.from_dlpack(dlpack)
    elif target_device.type == "mps":
        tensor = torch.utils.dlpack.from_dlpack(array)
    else:
        raise ValueError(
            f"The MLX tensor bridge supports CPU and MPS targets, got {target_device}"
        )
    return tensor


__all__ = [
    "mlx_to_torch",
    "torch_to_mlx",
]
