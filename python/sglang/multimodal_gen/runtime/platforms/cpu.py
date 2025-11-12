# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/cpu.py

import platform

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
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # This is a rough estimate for CPU memory
        # In practice, you might want to use psutil or similar
        return 0

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
    def get_device_communicator_cls(cls) -> str:
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator.CpuCommunicator"
