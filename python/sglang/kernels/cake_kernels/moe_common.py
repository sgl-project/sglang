"""Private helpers shared by the ``moe_*`` Cake adapters.

Importing this module must not import ``torch`` or ``flashinfer``; every
helper that touches a device does so lazily inside the function.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

from sglang.kernels.cake_kernels._support import (
    device_capability,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch


@lru_cache(maxsize=None)
def device_sm_count(device_index: int) -> int:
    """Cached physical SM count of ``cuda:device_index``."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def current_cuda_index(device: Optional[torch.device] = None) -> Optional[int]:
    """Resolve a CUDA device index without raising; ``None`` when unavailable."""
    import torch

    if torch.version.cuda is None or not torch.cuda.is_available():
        return None
    if device is None:
        return torch.cuda.current_device()
    device = torch.device(device)
    if device.type != "cuda":
        return None
    return device.index if device.index is not None else torch.cuda.current_device()


def cuda_device_in(
    device_index: Optional[int], archs: Iterable[Tuple[int, int]]
) -> bool:
    """``True`` when ``device_index`` is a CUDA device of one of ``archs``."""
    return device_index is not None and device_capability(device_index) in tuple(archs)


def contiguous_cuda(
    tensor: torch.Tensor,
    *,
    shape: Optional[Tuple[int, ...]] = None,
    dtype=None,
    ndim: Optional[int] = None,
) -> bool:
    """Shape/dtype/contiguity check mirroring FlashInfer's ``_require_tensor``."""
    import torch

    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
        return False
    if not tensor.is_contiguous():
        return False
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        return False
    if ndim is not None and tensor.ndim != ndim:
        return False
    if dtype is not None:
        allowed = dtype if isinstance(dtype, tuple) else (dtype,)
        if tensor.dtype not in allowed:
            return False
    return True


def same_device(*tensors: torch.Tensor) -> bool:
    devices = {t.device for t in tensors}
    return len(devices) == 1


def modules_available(*names: str) -> bool:
    """``flashinfer_module_available`` that never raises."""
    try:
        return flashinfer_module_available(*names)
    except Exception:  # pragma: no cover - defensive, find_spec quirks
        return False
