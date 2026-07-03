# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from functools import lru_cache
from typing import Any

import torch

from sglang._apple_silicon_memory import (
    apple_gpu_available_memory,
    apple_gpu_total_memory,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# SPDX-License-Identifier: Apache-2.0


logger = init_logger(__name__)


class MpsPlatform(Platform):
    _enum = PlatformEnum.MPS
    device_name: str = "mps"
    device_type: str = "mps"
    dispatch_key: str = "MPS"
    device_control_env_var: str = "MPS_VISIBLE_DEVICES"

    @classmethod
    @lru_cache(maxsize=1)
    def is_amp_supported(cls) -> bool:
        return False

    @classmethod
    @lru_cache(maxsize=1)
    def is_float64_supported(cls) -> bool:
        return False

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        return torch.device("mps")

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    @lru_cache(maxsize=1)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # Metal cannot use all of system RAM: cap at the recommended max
        # working set so callers do not overcommit the GPU budget (#21443).
        # lru_cache pins the first dispatch decision, which is safe: both
        # runtimes report the identical Metal working-set value.
        return apple_gpu_total_memory()

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable MPS "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used"
            )
            return False
        return True

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        return 0.0

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int | None = None,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:

        # Unified memory: cap at Metal's remaining working-set headroom
        # instead of treating all free system RAM as GPU-available (#21443).
        free_memory = apple_gpu_available_memory(empty_cache=empty_cache)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_memory, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_memory = float(tensor.item())

        return free_memory / (1 << 30)

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        # MPS supports SDPA (Scaled Dot-Product Attention) which is the most compatible
        logger.info("Using Torch SDPA backend for MPS.")
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        )

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # Use base communicator for MPS
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """Set the seed for MPS device."""
        if seed is not None:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # MPS doesn't have manual_seed_all like CUDA
            # The manual_seed above should be sufficient
