# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/cpu.py

import platform
from functools import lru_cache
from typing import Any

import psutil
import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    CpuArchEnum,
    Platform,
    PlatformEnum,
)


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
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        # For CPU, we can't easily get memory usage without additional libraries
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
        # For simplicity, we assume 1 NUMA node for now in this platform abstraction
        # as get_cpu_ids_by_node is not available in multimodal_gen.runtime.utils
        n_numa_node = 1
        free_memory = total_free_memory / n_numa_node

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_memory, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_memory = float(tensor.item())

        return free_memory / (1 << 30)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator.CpuCommunicator"
