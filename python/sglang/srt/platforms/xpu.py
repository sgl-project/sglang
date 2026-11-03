"""XPU device operations for the SRT platform layer."""

import logging
from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform

logger = logging.getLogger(__name__)

# The sysman process query counts then fills, so a process starting between the two
# calls fails one attempt but not the next.
_XPU_MEMORY_QUERY_ATTEMPTS = 2


def _device_memory_in_use(device_id: int) -> int:
    """Device memory in use on an XPU; Level Zero sysman is the only whole-device source, since torch misses DWDP's physical memory and mem_get_info is a stub."""
    # lazy: xpu_vmm_utils imports sglang.srt.utils.common, which imports this module
    from sglang.srt.utils.xpu_vmm_utils import (
        get_device_memory_in_use,
        has_physical_memory,
        local_physical_memory_bytes,
    )

    torch_in_use = torch.xpu.memory_allocated(device_id)
    if not has_physical_memory():
        return torch_in_use
    last_error: RuntimeError | OSError | None = None
    for _ in range(_XPU_MEMORY_QUERY_ATTEMPTS):
        try:
            return max(torch_in_use, get_device_memory_in_use(device_id))
        except (RuntimeError, OSError) as error:
            last_error = error
    logger.warning(
        f"Level Zero memory query failed for XPU {device_id} ({last_error}); "
        f"counting torch's allocator plus this process's own Level Zero memory"
    )
    return torch_in_use + local_physical_memory_bytes(device_id)


class XpuDeviceMixin(DeviceMixin):
    """XPU implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(torch.xpu.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        return float(torch.xpu.max_memory_allocated(device))

    def get_device(self, device_id: int = 0) -> "torch.device":
        return torch.device("xpu", device_id)

    def set_device(self, device: "torch.device") -> None:
        torch.xpu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(torch.xpu.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        return str(torch.xpu.get_device_properties(device_id).uuid)

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        # TODO: torch.xpu.get_device_capability
        device = torch.xpu.current_device()
        major, minor = torch.ops.sgl_kernel.query_device.default(device)
        return DeviceCapability(major, minor)

    def empty_cache(self) -> None:
        torch.xpu.empty_cache()

    def synchronize(self) -> None:
        torch.xpu.synchronize()

    def get_vmm_backend_class(self) -> type:
        # lazy: xpu_vmm_utils imports sglang.srt.utils.common, which imports this module
        from sglang.srt.utils.xpu_vmm_utils import XpuVmmBackend

        return XpuVmmBackend

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        # TODO: simple return of torch.xpu.mem_get_info
        """Return the available and total device memory in Bytes."""

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return 0, 0

        num_gpus = torch.xpu.device_count()
        if device_id < 0 or device_id >= num_gpus:
            raise ValueError(f"Invalid XPU device_id={device_id}. num_gpus={num_gpus}")

        current = torch.xpu.current_device()
        if current != device_id:
            logger.warning(
                "current device is not %s, but %s; this may cause useless memory allocation for torch XPU context.",
                device_id,
                current,
            )

        used_memory = _device_memory_in_use(device_id)
        total_gpu_memory = torch.xpu.get_device_properties(device_id).total_memory
        free_gpu_memory = total_gpu_memory - used_memory

        return free_gpu_memory, total_gpu_memory

    def is_pin_memory_available(self, device=None) -> bool:
        if device is not None and str(device) == "cpu":
            return False
        return True

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            torch.xpu.manual_seed_all(seed)


class XpuSRTPlatform(XpuDeviceMixin, SRTPlatform):
    """Default in-tree XPU SRT platform."""

    capabilities = PlatformCapabilities(
        supports_triton=True,
        graph_capture=True,
        piecewise_graph=True,
    )
