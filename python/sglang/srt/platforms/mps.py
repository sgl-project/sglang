"""Apple Metal/MPS device operations for the in-tree SRT platform layer.

PyTorch exposes Apple's GPU as ``mps`` rather than through the CUDA API.  Keep
the implementation deliberately small: MPS is a unified-memory, single-device
platform for now, and the SRT scheduler/allocator still uses the ordinary
Torch model runner and KV-pool implementations.  The platform object supplies
the device lifecycle hooks that were previously skipped because MPS fell
through to the base ``SRTPlatform``.
"""

from __future__ import annotations

import platform as _platform
from typing import Optional

import psutil
import torch

from sglang.srt.platforms.device_mixin import (
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform


def _recommended_working_set_size() -> int:
    getter = getattr(torch.mps, "recommended_max_memory", None)
    if not callable(getter):
        raise RuntimeError(
            "The SGLang MPS backend requires "
            "torch.mps.recommended_max_memory() from Torch 2.13.x"
        )
    total = int(getter())
    if total <= 0:
        raise RuntimeError(
            "torch.mps.recommended_max_memory() returned a non-positive Metal "
            "working-set limit; refusing to size the model and KV pools from "
            "raw system RAM"
        )
    return total


class MpsDeviceMixin(DeviceMixin):
    """Device operations backed by ``torch.backends.mps``."""

    _enum: PlatformEnum = PlatformEnum.MPS
    device_name: str = "mps"
    device_type: str = "mps"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        # Metal cannot safely keep all unified system RAM resident.  PyTorch's
        # value maps to MTLDevice.recommendedMaxWorkingSetSize and is the
        # authoritative capacity for model/KV-pool planning.
        return _recommended_working_set_size()

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        # Driver allocation includes cached MPS buffers and, unlike
        # current_allocated_memory(), accounts for the resident Metal working
        # set that reduces the safe device budget.
        return float(torch.mps.driver_allocated_memory())

    def get_device_count(self) -> int:
        # PyTorch currently exposes one process-wide MPS device.
        return 1

    def get_device_core_count(self, device_id: int = 0) -> int:
        # There is no public PyTorch API for the Apple GPU core count.
        return 0

    def get_device(self, local_rank: int = 0) -> torch.device:
        if local_rank not in (-1, 0):
            raise ValueError(f"MPS exposes one device; got local_rank={local_rank}")
        return torch.device("mps")

    def set_device(self, device: torch.device) -> None:
        if str(device).split(":", 1)[0] != "mps":
            raise ValueError(f"MPS platform cannot select device {device}")
        # MPS currently has no set_device/current_device API.

    def create_stream(self, device: Optional[torch.device] = None):
        # PyTorch MPS exposes one implicit command stream.  Localize the
        # compatibility object behind the platform boundary instead of making
        # ModelRunner call a CUDA-shaped API that upstream Torch does not have.
        from sglang._platform_stubs import Stream

        return Stream(device=device)

    def get_device_name(self, device_id: int = 0) -> str:
        return f"Apple MPS ({_platform.machine()})"

    def get_device_uuid(self, device_id: int = 0) -> str:
        # There is no stable public GPU UUID in the PyTorch MPS API.
        return "mps:0"

    def get_device_capability(self, device_id: int = 0):
        return None

    def empty_cache(self) -> None:
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def synchronize(self) -> None:
        synchronize = getattr(torch.mps, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        vm = psutil.virtual_memory()
        total = self.get_device_total_memory(device_id)
        metal_headroom = max(0, total - int(torch.mps.driver_allocated_memory()))
        # Unified memory has two independent ceilings: host memory available
        # to the process and the remaining Metal working set.  Respect the
        # tighter one so either kind of pressure shrinks the KV-pool budget.
        available = min(int(vm.available), metal_headroom)
        return available, total

    def is_pin_memory_available(self, device=None) -> bool:
        return False

    def get_torch_distributed_backend_str(self) -> str:
        # MPS tensors are not supported by NCCL/HCCL collectives.
        return "gloo"


class MpsSRTPlatform(MpsDeviceMixin, SRTPlatform):
    """Built-in SRT platform for Apple Silicon MPS."""

    def init_backend(self) -> None:
        # ServerArgs performs the same cached check early for user-facing
        # launch failures. Keep the platform boundary authoritative for
        # programmatic ModelRunner construction as well.
        from sglang.srt.hardware_backend.mps.runtime import validate_mps_runtime

        validate_mps_runtime()

    def get_default_attention_backend(self) -> str:
        return "torch_native"

    def get_compile_backend(self, mode: str | None = None) -> str:
        # Torch MPS does not yet have an SGLang graph runner.
        return "eager"

    def get_dispatch_key_name(self) -> str:
        # This foundation deliberately uses the normal Torch fused-op path.
        return "native"
