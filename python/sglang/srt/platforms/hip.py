# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.2/vllm/platforms/rocm.py

import logging
import subprocess
from functools import lru_cache
from typing import Any, Optional

import torch

from sglang.srt.platforms.interface import DeviceCapability, Platform, PlatformEnum

logger = logging.getLogger(__name__)


class HipPlatform(Platform):
    _enum: PlatformEnum = PlatformEnum.HIP
    device_name: str = "hip"
    device_type: str = "cuda"
    torch_distributed_backend: str = "nccl"

    supported_quantization: list[str] = [
        "awq",
        "gptq",
        "fp8",
        "compressed_tensors",
        "compressed-tensors",
        "fbgemm_fp8",
        "w8a8_fp8",
    ]

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
    def get_device_count(cls) -> int:
        return torch.cuda.device_count()

    @classmethod
    @lru_cache(maxsize=1)
    def get_memory_capacity(cls) -> Optional[float]:
        try:
            # Run rocm-smi and capture the output
            result = subprocess.run(
                [
                    "rocminfo | grep 'gfx' -A 100 | grep 'Pool 1' -A 5 | grep 'Size:' | awk '{print $2}'"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"rocm-smi error: {result.stderr.strip()}")

            # Parse the output to extract memory values in MiB
            memory_values = [
                float(mem.split("(")[0].strip()) / 1024
                for mem in result.stdout.strip().split("\n")
            ]

            if not memory_values:
                raise ValueError("No GPU memory values found.")

            # Return the minimum memory value
            return min(memory_values)

        except FileNotFoundError:
            raise RuntimeError(
                "rocm-smi not found. Ensure AMD ROCm drivers are installed and accessible."
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
    @lru_cache(maxsize=1)
    def supports_fp8(cls) -> bool:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx94", "gfx95", "gfx12"])

    @classmethod
    @lru_cache(maxsize=1)
    def is_fp8_fnuz(cls) -> bool:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        return "gfx94" in torch.cuda.get_device_properties(0).gcnArchName

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.is_fp8_fnuz():
            return torch.float8_e4m3fnuz
        else:
            return torch.float8_e4m3fn

    @classmethod
    def is_triton_avaliable(cls) -> bool:
        return True
