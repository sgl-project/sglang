# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from functools import lru_cache
from typing import Any

import psutil
import torch

from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# SPDX-License-Identifier: Apache-2.0


logger = init_logger(__name__)


def _recommended_working_set_size() -> int:
    """Return Metal's recommended GPU working-set limit in bytes."""
    getter = getattr(torch.mps, "recommended_max_memory", None)
    if not callable(getter):
        raise RuntimeError(
            "The SGLang MPS diffusion platform requires "
            "torch.mps.recommended_max_memory()"
        )
    total = int(getter())
    if total <= 0:
        raise RuntimeError(
            "torch.mps.recommended_max_memory() returned a non-positive Metal "
            "working-set limit; refusing to size GPU memory from raw system RAM"
        )
    return total


class MpsPlatform(Platform):
    _enum = PlatformEnum.MPS
    device_name: str = "mps"
    device_type: str = "mps"
    dispatch_key: str = "MPS"

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
    def set_device(cls, device: torch.device) -> None:
        pass

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "Apple Silicon MPS"

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return "mps"

    @classmethod
    @lru_cache(maxsize=1)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # Metal's working-set limit, not total system RAM, bounds GPU residency.
        return _recommended_working_set_size()

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        # Driver allocation includes resident cached MPS buffers.
        return float(torch.mps.driver_allocated_memory())

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int | None = None,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        if empty_cache:
            torch.mps.empty_cache()

        # Bound free memory by host RAM and Metal's remaining working-set headroom.
        total_gpu_memory = _recommended_working_set_size()
        metal_headroom = max(
            0, total_gpu_memory - int(torch.mps.driver_allocated_memory())
        )
        free_memory = min(psutil.virtual_memory().available, metal_headroom)

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
    def get_all_to_all_communicator_cls(cls) -> str:
        return (
            "sglang.multimodal_gen.runtime.distributed.device_communicators."
            "cpu_communicator.CpuCommunicator"
        )

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
