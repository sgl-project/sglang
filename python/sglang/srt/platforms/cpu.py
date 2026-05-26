"""CPU device operations for the SRT platform layer."""

import gc
import platform as _platform
from typing import Optional

import psutil
import torch

from sglang.srt.platforms.device_mixin import (
    CpuArchEnum,
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform


class CpuDeviceMixin(DeviceMixin):
    """CPU implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(psutil.virtual_memory().total)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        return float(psutil.Process().memory_info().rss)

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("cpu")

    def set_device(self, device: "torch.device") -> None:
        # True no-op. Do NOT call ``torch.set_default_device("cpu")`` — that
        # would flip the process-wide default and break code that constructs
        # ``torch.empty(..., device="cuda")`` elsewhere.
        pass

    def get_device_name(self, device_id: int = 0) -> str:
        arch = self.get_cpu_architecture()
        proc = _platform.processor() or _platform.machine() or "unknown"
        if arch == CpuArchEnum.ARM:
            return f"cpu (aarch64: {proc})"
        if arch == CpuArchEnum.X86:
            return f"cpu (x86_64: {proc})"
        return f"cpu ({proc})"

    def get_device_uuid(self, device_id: int = 0) -> str:
        return _platform.machine()

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        return None

    def empty_cache(self) -> None:
        # Trigger a GC pass. ``empty_cache`` is called at well-defined teardown
        # points (Scheduler.flush_cache, weight reload, etc.) where a brief
        # collection pause is acceptable. Beats the base ``pass`` no-op for
        # actual CPU memory pressure.
        gc.collect()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        vm = psutil.virtual_memory()
        return (vm.available, vm.total)

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"


class CpuSRTPlatform(CpuDeviceMixin, SRTPlatform):
    """Default in-tree CPU SRT platform."""

    def supports_fp8(self) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False

    def is_pin_memory_available(self) -> bool:
        return False
