"""Shared, import-light helpers for the Cake adapters.

Every adapter in this package performs the same two checks before forwarding
to FlashInfer: the current device must be one of the architectures the Cake
kernel was built for, and the FlashInfer installation must actually ship the
module that implements it. Older FlashInfer releases (for example the
``0.7.0.post1`` pin) lack most Cake entry points; in that case the adapter
reports ``False`` and the caller keeps its existing backend.

Importing this module must not import ``torch`` or ``flashinfer``.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import Iterable, Tuple

SM100 = (10, 0)
SM103 = (10, 3)
SM120 = (12, 0)
SM121 = (12, 1)
SM90 = (9, 0)

BLACKWELL_DATACENTER: Tuple[Tuple[int, int], ...] = (SM100, SM103)
BLACKWELL_ALL: Tuple[Tuple[int, int], ...] = (SM100, SM103, SM120, SM121)


@lru_cache(maxsize=None)
def flashinfer_module_available(*module_names: str) -> bool:
    """Return ``True`` when every named FlashInfer module can be imported.

    Uses ``importlib.util.find_spec`` so the check neither imports FlashInfer
    nor triggers a JIT build. Names are fully qualified, for example
    ``"flashinfer.jit.cake_blackwell_softmax"``.
    """
    if find_spec("flashinfer") is None:
        return False
    try:
        return all(find_spec(name) is not None for name in module_names)
    except (ImportError, ValueError):
        # find_spec raises when a parent package is missing or is not a package.
        return False


@lru_cache(maxsize=None)
def device_capability(device_index: int) -> Tuple[int, int]:
    """Cached ``torch.cuda.get_device_capability`` for one device index."""
    import torch

    return tuple(torch.cuda.get_device_capability(device_index))  # type: ignore[return-value]


def device_in(device_index: int, archs: Iterable[Tuple[int, int]]) -> bool:
    """``True`` when the device's compute capability is one of ``archs``."""
    return device_capability(device_index) in tuple(archs)


def cuda_tensor_on(tensor, archs: Iterable[Tuple[int, int]]) -> bool:
    """``True`` when ``tensor`` lives on a CUDA device of one of ``archs``.

    ``torch.version.cuda`` is checked so ROCm builds never qualify.
    """
    import torch

    return (
        tensor.is_cuda
        and torch.version.cuda is not None
        and device_in(tensor.device.index, archs)
    )


def table_view_ok(
    tensor, *, rows_min: int, cols: int, dtype, device, align_elements: int = 8
) -> bool:
    """2-D ``[rows >= rows_min, cols]`` view the Cake table kernels read in place.

    Unit last stride, a row pitch that is a multiple of ``align_elements``
    (16 bytes for BF16) and a 16-byte-aligned data pointer; column chunks of a
    wider projection (``stride(0) > cols``) pass.
    """
    import torch

    return (
        isinstance(tensor, torch.Tensor)
        and tensor.ndim == 2
        and tensor.dtype == dtype
        and tensor.device == device
        and int(tensor.shape[0]) >= rows_min
        and int(tensor.shape[1]) == cols
        and tensor.stride(1) == 1
        and tensor.stride(0) % align_elements == 0
        and tensor.data_ptr() % (align_elements * tensor.element_size()) == 0
    )


def thd_view_ok(tensor, *, shape, dtype, device, align_elements: int = 8) -> bool:
    """Token-major ``[T, H, D]`` view the Cake varlen attention reads in place.

    Unit last stride, a head stride that is a multiple of ``align_elements`` of
    at least ``D``, a token stride that is a multiple of ``align_elements`` of
    at least ``H * head stride`` and a 16-byte-aligned data pointer: the column
    chunks of a fused ``[T, 3 * H * D]`` projection, the kind slices of a
    ``[T, H, 3, D]`` pack and contiguous tensors all pass.
    """
    import torch

    if not (
        isinstance(tensor, torch.Tensor)
        and tensor.ndim == 3
        and tuple(tensor.shape) == tuple(shape)
        and tensor.dtype == dtype
        and tensor.device == device
    ):
        return False
    heads, head_dim = int(tensor.shape[1]), int(tensor.shape[2])
    row_stride, head_stride, elem_stride = (int(s) for s in tensor.stride())
    return (
        elem_stride == 1
        and head_stride >= head_dim
        and head_stride % align_elements == 0
        and row_stride >= heads * head_stride
        and row_stride % align_elements == 0
        and tensor.data_ptr() % (align_elements * tensor.element_size()) == 0
    )
