# Copied and adapted from: https://github.com/vllm-project/vllm-metal
# SPDX-License-Identifier: Apache-2.0
"""Tensor bridge between MLX and PyTorch on Apple silicon.

The MLX backend requires MLX >= 0.32 and PyTorch >= 2.13.  Ordinary
``torch_to_mlx`` conversion creates an independent MLX allocation.  The
zero-copy ``mlx_call`` helper is available for a complete MLX operation and
keeps all borrowed DLPack inputs alive until the result has been evaluated.
This lifetime boundary matters because MLX may donate a borrowed input buffer
to a lazy operation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Literal

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


def _torch_to_mlx(
    tensor: torch.Tensor,
    *,
    copy: bool,
    synchronize: bool = True,
) -> mx.array:
    """Convert one tensor, optionally borrowing its MPS allocation."""
    mx = _mlx_core()
    tensor = tensor.detach()

    if tensor.device.type == "mps":
        if synchronize:
            # Torch and MLX do not share stream state on Metal.
            torch.mps.synchronize()
        return mx.asarray(tensor, copy=copy)
    if tensor.device.type == "cpu":
        # CPU tensors always get MLX-owned storage.  In particular, do not
        # expose a NumPy/memoryview alias whose lifetime is controlled by the
        # caller.
        return mx.array(tensor)
    raise ValueError(
        f"The MLX tensor bridge supports CPU and MPS tensors, got {tensor.device}"
    )


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert a PyTorch tensor to an independent MLX array.

    MPS inputs are copied inside the unified Metal device.  Use ``mlx_call``
    when a zero-copy Torch-to-MLX-to-Torch operation is required; it owns the
    borrowed MLX arrays for the complete lazy operation.

    Args:
        tensor: PyTorch CPU or MPS tensor.

    Returns:
        MLX array with the same data
    """
    array = _torch_to_mlx(tensor, copy=True)
    if tensor.device.type == "mps":
        # Complete the ownership transfer before the caller may mutate or
        # release the Torch source.
        _mlx_core().eval(array)
    return array


def mlx_call(
    operation: Callable[..., mx.array],
    *tensors: torch.Tensor,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> torch.Tensor:
    """Run one MLX operation with zero-copy Torch MPS inputs.

    The imported MLX arrays remain strongly referenced until
    :func:`mlx_to_torch` evaluates and exports ``operation``'s result.  Keep
    the operation inside this call; returning a lazy MLX result for later use
    would outlive the borrow scope.
    """
    detached = tuple(tensor.detach() for tensor in tensors)
    if any(tensor.device.type == "mps" for tensor in detached):
        torch.mps.synchronize()
    borrowed = tuple(
        _torch_to_mlx(tensor, copy=False, synchronize=False) for tensor in detached
    )
    result = operation(*borrowed)
    output = mlx_to_torch(result, device=device)
    # Keep the imported MLX objects, not just their Torch owners, alive through
    # lazy result evaluation and DLPack export.
    _ = borrowed
    return output


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
    "mlx_call",
    "mlx_to_torch",
    "torch_to_mlx",
]
