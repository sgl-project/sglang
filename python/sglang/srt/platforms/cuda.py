# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.2/vllm/platforms/cuda.py

import logging
import re
import subprocess
from functools import lru_cache
from typing import Any, Optional

import torch

from sglang.srt.platforms.interface import DeviceCapability, Platform, PlatformEnum

logger = logging.getLogger(__name__)


class CudaPlatform(Platform):
    _enum: PlatformEnum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"
    torch_distributed_backend: str = "nccl"

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> Optional[DeviceCapability]:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_module(cls) -> Any:
        return torch.cuda

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_sku(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_core_count(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    @classmethod
    def get_device_count(cls) -> int:
        return torch.cuda.device_count()

    @classmethod
    @lru_cache(maxsize=1)
    def get_memory_capacity(cls) -> Optional[float]:
        try:
            # Run nvidia-smi and capture the output
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi error: {result.stderr.strip()}")

            # Parse the output to extract memory values
            memory_values = [
                float(mem)
                for mem in result.stdout.strip().split("\n")
                if re.match(r"^\d+(\.\d+)?$", mem.strip())
            ]

            if not memory_values:
                raise ValueError("No GPU memory values found.")

            # Return the minimum memory value
            return min(memory_values)

        except FileNotFoundError:
            raise RuntimeError(
                "nvidia-smi not found. Ensure NVIDIA drivers are installed and accessible."
            )

    @classmethod
    def get_device_available_memory(
        cls, device_id: int = 0, distributed=False, empty_cache=True
    ) -> float:
        num_gpus = cls.get_device_count()
        assert device_id < num_gpus

        if torch.cuda.current_device() != device_id:
            print(
                f"WARNING: current device is not {device_id}, but {torch.cuda.current_device()}, ",
                "which may cause useless memory allocation for torch CUDA context.",
            )

        if empty_cache:
            torch.cuda.empty_cache()
        free_gpu_memory, _ = torch.cuda.mem_get_info(device_id)

        if distributed:
            tensor = torch.tensor(
                free_gpu_memory, dtype=torch.float32, device=cls.get_device(device_id)
            )
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
            free_gpu_memory = tensor.item()

        return free_gpu_memory / (1 << 30)

    @classmethod
    def supports_overlap_scheduler(cls) -> bool:
        return True

    @classmethod
    def seed_everything(cls, seed: Optional[int] = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            torch.cuda.manual_seed_all(seed)

    @classmethod
    def supports_fp8(cls) -> bool:
        return cls.has_device_capability(89)

    @classmethod
    def is_triton_avaliable(cls) -> bool:
        return True
