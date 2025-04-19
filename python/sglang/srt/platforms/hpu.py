# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.2/vllm/platforms/hpu.py

import logging
import subprocess
from functools import lru_cache
from typing import Any, Optional

import torch

from sglang.srt.platforms.interface import Platform, PlatformEnum

logger = logging.getLogger(__name__)


class HpuPlatform(Platform):
    _enum = PlatformEnum.HPU
    device_name: str = "hpu"
    device_type: str = "hpu"
    torch_distributed_backend: str = "hccl"

    @classmethod
    def get_device_module(cls) -> Any:
        raise torch.hpu

    @classmethod
    def get_device_count(cls) -> int:
        torch.hpu.device_count()

    @classmethod
    @lru_cache(maxsize=1)
    def get_memory_capacity(cls) -> Optional[float]:
        try:
            # Run hl-smi and capture the output
            result = subprocess.run(
                ["hl-smi --query | grep 'Total'"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"hl-smi error: {result.stderr.strip()}")

            # Parse the output to extract memory values in MiB
            memory_values = [
                float(mem.split(" ")[-2]) for mem in result.stdout.strip().split("\n")
            ]

            if not memory_values:
                raise ValueError("No GPU memory values found.")

            # Return the minimum memory value
            return min(memory_values)

        except FileNotFoundError:
            raise RuntimeError(
                "hl-smi not found. Ensure Habana drivers are installed and accessible."
            )

    @classmethod
    def get_device_available_memory(
        cls, device_id: int = 0, distributed=False, empty_cache=True
    ) -> float:
        num_gpus = torch.hpu.device_count()
        assert device_id < num_gpus

        if torch.hpu.current_device() != device_id:
            print(
                f"WARNING: current device is not {device_id}, but {torch.hpu.current_device()}, ",
                "which may cause useless memory allocation for torch HPU context.",
            )

        free_gpu_memory, _ = torch.hpu.mem_get_info()

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
