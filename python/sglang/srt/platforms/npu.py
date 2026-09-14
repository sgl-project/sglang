"""NPU (Huawei Ascend)device operations for the SRT platform layer."""

from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform


class NpuDeviceMixin(DeviceMixin):
    """NPU implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(torch.npu.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        return float(torch.npu.max_memory_allocated(device))

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("npu", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch.npu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(torch.npu.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        return str(torch.npu.get_device_properties(device_id).uuid)

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        # The return value of torch_npu.npu.get_device_capability() is configured
        # via the environment variable TORCH_NPU_DEVICE_CAPABILITY, which is only
        # used for compatibility with native PyTorch and does not represent the
        # actual capabilities of the NPU hardware
        return DeviceCapability(0, 0)

    def empty_cache(self) -> None:
        torch.npu.empty_cache()

    def synchronize(self) -> None:
        torch.npu.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return torch.npu.mem_get_info(device_id)

    def is_pin_memory_available(self, device=None) -> bool:
        if device is not None and str(device) == "cpu":
            return False
        return True

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            if hasattr(torch, "npu"):
                torch.npu.manual_seed_all(seed)


class NpuSRTPlatform(NpuDeviceMixin, SRTPlatform):
    """Default in-tree NPU SRT platform."""

    def get_default_attention_backend(self) -> str:
        return "ascend"

    def get_dispatch_key_name(self) -> str:
        return "npu"

    def supports_fp8(self) -> bool:
        # NPU quantization backends in hardware_backend/npu/quantization
        return True

    def support_cuda_graph(self) -> bool:
        # NPUGraphRunner in hardware_backend/npu/graph_runner
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        return False
