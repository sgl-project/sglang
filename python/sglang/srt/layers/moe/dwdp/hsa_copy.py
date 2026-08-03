"""HSA SDMA copy engine used by ROCm DWDP backends."""

from __future__ import annotations

import logging
import os
from typing import Iterable

import torch

logger = logging.getLogger(__name__)


class HsaCopyUnavailableError(RuntimeError):
    pass


class HsaSdmaCopyEngine:
    """Submit tensor copies to pair-specific HSA SDMA engines.

    Completion tickets are host-waited because HIP events cannot represent an
    external HSA queue dependency. DWDP therefore keeps graph capture disabled.
    """

    def __init__(self):
        if torch.version.hip is None:
            raise HsaCopyUnavailableError("HSA copy requires ROCm PyTorch")
        try:
            import sgl_kernel  # noqa: F401
        except Exception as error:
            raise HsaCopyUnavailableError(
                f"failed to load sgl_kernel HSA copy operators: {error}"
            ) from error
        required = (
            "dwdp_hsa_copy_is_available",
            "dwdp_hsa_copy_engine_for_devices",
            "dwdp_hsa_copy_async",
            "dwdp_hsa_copy_wait",
            "dwdp_hsa_copy_destroy",
        )
        missing = [name for name in required if not hasattr(torch.ops.sgl_kernel, name)]
        if missing:
            raise HsaCopyUnavailableError(
                f"sgl_kernel has no HSA copy operators: {missing}"
            )
        if not torch.ops.sgl_kernel.dwdp_hsa_copy_is_available():
            raise HsaCopyUnavailableError("HSA SDMA copy API is unavailable")
        self._ops = torch.ops.sgl_kernel

    def engine_for_devices(self, destination_device: int, source_device: int) -> int:
        return int(
            self._ops.dwdp_hsa_copy_engine_for_devices(
                destination_device,
                source_device,
            )
        )

    @classmethod
    def create_or_none(cls):
        if os.environ.get("SGLANG_DWDP_DISABLE_HSA_COPY", "0") == "1":
            logger.info("ROCm DWDP HSA copy disabled by environment")
            return None
        try:
            return cls()
        except HsaCopyUnavailableError as error:
            logger.warning(
                "ROCm DWDP HSA copy unavailable; using HIP stream copy_: %s",
                error,
            )
            return None

    def submit(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        *,
        destination_device: int | None = None,
        source_device: int | None = None,
    ) -> int:
        if not destination.is_contiguous() or not source.is_contiguous():
            raise ValueError("HSA DWDP copies require contiguous tensors")
        if destination.numel() != source.numel():
            raise ValueError(
                f"HSA DWDP copy size mismatch: {destination.shape} vs {source.shape}"
            )
        if destination.element_size() != source.element_size():
            raise ValueError(
                f"HSA DWDP copy dtype width mismatch: {destination.dtype} vs "
                f"{source.dtype}"
            )
        destination_device = (
            destination.device.index
            if destination_device is None
            else destination_device
        )
        source_device = source.device.index if source_device is None else source_device
        if destination_device is None or source_device is None:
            raise ValueError("HSA DWDP copy requires explicit HIP device indices")
        return int(
            self._ops.dwdp_hsa_copy_async(
                destination,
                source,
                int(destination_device),
                int(source_device),
            )
        )

    def wait(self, ticket: int) -> None:
        value = int(self._ops.dwdp_hsa_copy_wait(ticket))
        self._ops.dwdp_hsa_copy_destroy(ticket)
        if value != 0:
            raise RuntimeError(
                f"HSA DWDP copy ticket {ticket} completed with signal value {value}"
            )

    def wait_all(self, tickets: Iterable[int]) -> None:
        first_error = None
        for ticket in tickets:
            try:
                self.wait(ticket)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
