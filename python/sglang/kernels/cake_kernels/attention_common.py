"""Import-light helpers shared by the Cake attention adapters.

Extends :mod:`sglang.kernels.cake_kernels._support` with the extra compute
capabilities the attention products are built for (SM107 for the DCP / DSA /
MSA routes, SM110 for the Thor XQA/GQA exports) and with the cached physical
SM count some routes pin (the Cake TRT-LLM-style MLA decode ships 148- and
152-SM builds only).

Importing this module must not import ``torch`` or ``flashinfer``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Tuple

from sglang.kernels.cake_kernels._support import (
    SM90,
    SM100,
    SM103,
    SM120,
    SM121,
    cuda_tensor_on,
    device_capability,
    device_in,
    flashinfer_module_available,
)

SM107 = (10, 7)
SM110 = (11, 0)

BLACKWELL_DC: Tuple[Tuple[int, int], ...] = (SM100, SM103)
BLACKWELL_DC_AND_SM107: Tuple[Tuple[int, int], ...] = (SM100, SM103, SM107)
BLACKWELL_CONSUMER: Tuple[Tuple[int, int], ...] = (SM120, SM121)

__all__ = [
    "SM90",
    "SM100",
    "SM103",
    "SM107",
    "SM110",
    "SM120",
    "SM121",
    "BLACKWELL_DC",
    "BLACKWELL_DC_AND_SM107",
    "BLACKWELL_CONSUMER",
    "cuda_tensor_on",
    "device_capability",
    "device_in",
    "device_sm_count",
    "flashinfer_module_available",
    "is_cuda_tensor",
    "same_cuda_device",
]


@lru_cache(maxsize=None)
def device_sm_count(device_index: int) -> int:
    """Cached ``multi_processor_count`` of one CUDA device."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def is_cuda_tensor(tensor) -> bool:
    """``True`` for a CUDA tensor on a CUDA (not ROCm) build of torch."""
    import torch

    return bool(tensor.is_cuda) and torch.version.cuda is not None


def same_cuda_device(*tensors) -> bool:
    """``True`` when every tensor is a CUDA tensor on the same device index."""
    if not tensors:
        return False
    first = tensors[0]
    if not is_cuda_tensor(first):
        return False
    return all(
        is_cuda_tensor(t) and t.device.index == first.device.index for t in tensors
    )


def archs_in(archs: Iterable[Tuple[int, int]], *tensors) -> bool:
    """``True`` when all tensors share one CUDA device of one of ``archs``."""
    return same_cuda_device(*tensors) and cuda_tensor_on(tensors[0], archs)
