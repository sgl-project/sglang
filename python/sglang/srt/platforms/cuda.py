"""CUDA device operations for the SRT platform layer."""

import logging
import os
from contextlib import contextmanager
from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform

logger = logging.getLogger(__name__)


class CudaDeviceMixin(DeviceMixin):
    """CUDA implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(torch.cuda.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        return float(torch.cuda.max_memory_allocated(device))

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("cuda", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch.cuda.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_properties(device_id).uuid)

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major, minor)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return torch.cuda.mem_get_info(device_id)

    def is_pin_memory_available(self, device=None) -> bool:
        if device is not None and str(device) == "cpu":
            return False
        return True

    @contextmanager
    def reindex_device_id(self, gpu_id: int):
        if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            logger.warning(
                "`CUDA_DEVICE_ORDER` is not set to `PCI_BUS_ID`. Please set "
                "`CUDA_DEVICE_ORDER=PCI_BUS_ID` to avoid unexpected behavior."
            )

        original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if original_cuda_visible_devices:
            cuda_visible_devices = original_cuda_visible_devices.split(",")
        else:
            cuda_visible_devices = []

        str_gpu_id = (
            cuda_visible_devices[gpu_id] if cuda_visible_devices else str(gpu_id)
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu_id

        logger.debug(f"Set CUDA_VISIBLE_DEVICES to {str_gpu_id}")

        yield 0

        if original_cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]

    def get_torch_distributed_backend_str(self) -> str:
        return "nccl"

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            torch.cuda.manual_seed_all(seed)


class CudaSRTPlatform(CudaDeviceMixin, SRTPlatform):
    """Default in-tree CUDA SRT platform."""

    def supports_fp8(self) -> bool:
        return True

    def support_cuda_graph(self) -> bool:
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        return True
