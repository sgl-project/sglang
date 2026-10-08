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
