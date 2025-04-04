# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.2/vllm/platforms/cpu.py

import logging
from functools import lru_cache
from typing import Any

import psutil
import torch

from sglang.srt.platforms.interface import Platform, PlatformEnum

logger = logging.getLogger(__name__)


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"
    torch_distributed_backend: str = "gloo"

    @classmethod
    @lru_cache(maxsize=16)
    def get_device(cls, device_id: int = 0) -> str:
        return "cpu"

    @classmethod
    def get_device_module(cls) -> Any:
        """Get `torch.device_module` like `torch.cuda` of current platform."""
        raise torch.cpu

    @classmethod
    def get_device_available_memory(
        cls, device_id: int = 0, distributed=False, empty_cache=True
    ) -> float:
        free_memory = psutil.virtual_memory().available
        if distributed:
            tensor = torch.tensor(
                free_memory, dtype=torch.float32, device=cls.get_device(device_id)
            )
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
            free_memory = tensor.item()

        return free_memory / (1 << 30)

    @classmethod
    def supports_overlap_scheduler(cls) -> bool:
        """
        Check if the current platform supports overlap scheduler
        """
        return True

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False
