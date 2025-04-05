# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.2/vllm/platforms/xpu.py

import logging
from functools import lru_cache
from typing import Any

import torch

from sglang.srt.platforms.interface import Platform, PlatformEnum

logger = logging.getLogger(__name__)


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    torch_distributed_backend: str = "xccl"

    @classmethod
    def get_device_module(cls) -> Any:
        return torch.xpu

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_sku(cls, device_id: int = 0) -> str:
        return torch.xpu.get_device_name(device_id)

    @classmethod
    def get_device_count(cls) -> int:
        return torch.xpu.device_count()

    @classmethod
    @lru_cache(maxsize=1)
    def get_device_available_memory(
        cls, device_id: int = 0, distributed=False, empty_cache=True
    ) -> float:
        num_gpus = torch.xpu.device_count()
        assert device_id < num_gpus

        if torch.xpu.current_device() != device_id:
            print(
                f"WARNING: current device is not {device_id}, but {torch.xpu.current_device()}, ",
                "which may cause useless memory allocation for torch XPU context.",
            )

        if empty_cache:
            torch.xpu.empty_cache()
        used_memory = torch.xpu.memory_allocated()
        total_gpu_memory = torch.xpu.get_device_properties(device_id).total_memory
        free_gpu_memory = total_gpu_memory - used_memory

        if distributed:
            tensor = torch.tensor(
                free_gpu_memory, dtype=torch.float32, device=cls.get_device(device_id)
            )
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
            free_gpu_memory = tensor.item()

        return free_gpu_memory / (1 << 30)

    @classmethod
    def supports_overlap_scheduler(cls) -> bool:
        """
        Check if the current platform supports overlap scheduler
        """
        return True

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def device_support_bf16(cls) -> bool:
        device_name = cls.get_device_name().lower()
        if device_name.count("arc") > 0:
            return False
        elif device_name.count("data center gpu") > 0:
            return True
        else:
            logger.warning("Unknown device name %s, always use float16", device_name)
            return False
