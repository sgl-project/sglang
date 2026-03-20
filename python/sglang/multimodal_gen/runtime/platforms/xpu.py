# SPDX-License-Identifier: Apache-2.0
# Intel XPU Platform for SGLang multimodal_gen runtime

import os
from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.platforms.interface import (
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def device_id_to_physical_device_id(device_id: int) -> int:
    if "ZE_AFFINITY_MASK" in os.environ:
        device_ids = os.environ["ZE_AFFINITY_MASK"].split(",")
        if device_ids == [""]:
            msg = (
                "ZE_AFFINITY_MASK is set to empty string, which means"
                " XPU support is disabled"
            )
            raise RuntimeError(msg)
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


class XPUPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED  # XPU not in PlatformEnum yet, use UNSPECIFIED
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    device_control_env_var: str = "ZE_AFFINITY_MASK"

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        return torch.device(f"xpu:{envs.LOCAL_RANK}")

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        # XPU doesn't have CUDA-style compute capability
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.xpu.get_device_name(device_id))

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.xpu.get_device_properties(device_id)
        return int(device_props.total_memory)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable XPU "
                "graph. Since enforce-eager is enabled, async output "
                "processor cannot be used"
            )
            return False
        return True

    @classmethod
    def is_full_nvlink(cls, physical_device_ids: list[int]) -> bool:
        # NVLink is NVIDIA-specific, not applicable to XPU
        return False

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        if empty_cache:
            torch.xpu.empty_cache()

        free_gpu_memory, _ = torch.xpu.mem_get_info(device_id)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="xpu")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

    @classmethod
    def get_attn_backend_cls_str(cls) -> str:
        # Use PyTorch's native SDPA which works on XPU
        return "SDPA"

    @classmethod
    def is_xpu(cls) -> bool:
        return True
