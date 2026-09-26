"""Biren SUPA device operations for the SRT platform layer.

SUPA is the software stack for Biren (BR-series) accelerators.  It is exposed
to PyTorch as the ``supa`` PrivateUse1 backend once Biren's ``torch_br`` plugin
has been imported, so every access goes through :func:`_supa` to make sure the
plugin is loaded first.
"""

import logging
from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform

logger = logging.getLogger(__name__)


def _supa():
    """Return the ``torch.supa`` module, importing the Biren plugin if needed."""
    if not hasattr(torch, "supa"):
        import torch_br  # noqa: F401
    return torch.supa


class SupaDeviceMixin(DeviceMixin):
    """Biren SUPA implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.OOT
    device_name: str = "supa"
    device_type: str = "supa"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(_supa().get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        return float(_supa().max_memory_allocated(device))

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("supa", local_rank)

    def set_device(self, device: "torch.device") -> None:
        _supa().set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(_supa().get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        props = _supa().get_device_properties(device_id)
        # BR1xx device properties do not expose a UUID.
        return str(getattr(props, "uuid", f"supa:{device_id}"))

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        # torch.supa does not implement get_device_capability; BR1xx reports the
        # architecture as major.minor in the device properties (e.g. 10.4).
        props = _supa().get_device_properties(device_id)
        return DeviceCapability(int(props.major), int(props.minor))

    def empty_cache(self) -> None:
        _supa().empty_cache()

    def synchronize(self) -> None:
        _supa().synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        free, total = _supa().mem_get_info(device_id)
        return int(free), int(total)

    def is_pin_memory_available(self, device=None) -> bool:
        if device is not None and str(device) == "cpu":
            return False
        return True

    def get_torch_distributed_backend_str(self) -> str:
        # SCCL is Biren's collective communication library (libsuccl.so).
        return "sccl"

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            _supa().manual_seed_all(seed)


class SupaSRTPlatform(SupaDeviceMixin, SRTPlatform):
    """In-tree Biren SUPA SRT platform."""

    def supports_fp8(self) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False
