# SPDX-License-Identifier: Apache-2.0
"""CUDA / Ascend NPU helpers for KT EP hybrid paths in ``kt_ep_wrapper``.

``torch_npu`` mocks ``torch.cuda.is_available`` as False on Ascend; branch on
``torch.Tensor.device.type`` instead of ``torch.cuda.is_available()`` alone.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Optional

import torch


def _accel_mod(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.cuda
    if device.type == "npu":
        import torch_npu  # noqa: F401

        return torch.npu
    raise TypeError(
        f"KT EP hybrid stream APIs require device type 'cuda' or 'npu', got {device.type!r}"
    )


def kt_device_synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize the current thread with work on ``device`` (or default accel)."""
    if device is None:
        if torch.cuda.is_available() and getattr(torch.version, "cuda", None):
            torch.cuda.synchronize()
            return
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        return
    _accel_mod(device).synchronize(device)


def kt_current_stream(device: torch.device) -> Any:
    return _accel_mod(device).current_stream(device)


def kt_current_stream_handle(device: torch.device) -> int:
    """Native stream handle for ``kt_kernel`` / ``submit_with_cuda_stream``."""
    stream = kt_current_stream(device)
    if device.type == "npu":
        return int(stream.npu_stream)
    return int(stream.cuda_stream)


def kt_new_stream(device: torch.device) -> Any:
    return _accel_mod(device).Stream(device=device)


def kt_new_event(device: torch.device) -> Any:
    return _accel_mod(device).Event()


def kt_stream_context(stream: Any, device: torch.device) -> AbstractContextManager:
    return _accel_mod(device).stream(stream)


def kt_maybe_cuda_host_register(
    cpu_tensor: torch.Tensor, nbytes: int, device: torch.device
) -> None:
    """Pin host memory for DMA when running on CUDA; no-op on NPU."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.cudart().cudaHostRegister(cpu_tensor.data_ptr(), nbytes, 0)
