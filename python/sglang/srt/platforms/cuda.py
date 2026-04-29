"""CUDA device operations for the SRT platform layer."""

import importlib
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)

if TYPE_CHECKING:
    import torch


def _import_torch() -> Any:
    return importlib.import_module("torch")


class CudaDeviceMixin(DeviceMixin):
    """CUDA implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        torch = _import_torch()
        return int(torch.cuda.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        torch = _import_torch()
        return float(torch.cuda.max_memory_allocated(device))

    def get_device(self, local_rank: int) -> "torch.device":
        torch = _import_torch()
        return torch.device("cuda", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch = _import_torch()
        torch.cuda.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        torch = _import_torch()
        return str(torch.cuda.get_device_name(device_id))

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        torch = _import_torch()
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major, minor)

    def empty_cache(self) -> None:
        torch = _import_torch()
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        torch = _import_torch()
        torch.cuda.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        torch = _import_torch()
        return torch.cuda.mem_get_info(device_id)

    def get_torch_distributed_backend_str(self) -> str:
        return "nccl"
