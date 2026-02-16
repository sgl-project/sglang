# SPDX-License-Identifier: Apache-2.0
"""
Minimal CPU Platform for SGLang.

This provides a lightweight CPU fallback that works even without
sglang.multimodal_gen installed. It is the last-resort platform
in the detection order.
"""

import platform
from functools import lru_cache
from typing import Any

import psutil
import torch

from sglang.platforms.interface import CpuArchEnum, Platform, PlatformEnum


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name = "CPU"
    device_type = "cpu"
    dispatch_key = "CPU"

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        """Get the CPU architecture."""
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64", "i386", "i686"):
            return CpuArchEnum.X86
        elif machine in ("arm64", "aarch64"):
            return CpuArchEnum.ARM
        else:
            return CpuArchEnum.UNSPECIFIED

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return platform.processor()

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return platform.machine()

    @classmethod
    @lru_cache(maxsize=1)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return True

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        """Get the local torch.device for CPU platform."""
        return torch.device("cpu")

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        # Returns 0.0 because the CPU platform has no dedicated device memory.
        # System RAM usage should be checked via psutil.virtual_memory() instead.
        return 0.0

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        total_free_memory = psutil.virtual_memory().available
        n_numa_node = 1
        free_memory = total_free_memory / n_numa_node

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_memory, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_memory = float(tensor.item())

        return free_memory / (1 << 30)
