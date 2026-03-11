# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-ascend: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/platform.py

import os
from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def device_id_to_physical_device_id(device_id: int) -> int:
    if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
        if device_ids == [""]:
            msg = (
                "ASCEND_RT_VISIBLE_DEVICES is set to empty string, which means"
                " NPU support is disabled"
            )
            raise RuntimeError(msg)
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


class NPUPlatformBase(Platform):
    _enum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"
    dispatch_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        return torch.device(f"npu:{envs.LOCAL_RANK}")

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.npu.get_device_name(device_id))

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.npu.get_device_properties(device_id)
        return int(device_props.total_memory)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable NPU "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used"
            )
            return False
        return True

    @classmethod
    def is_full_nvlink(cls, physical_device_ids: list[int]) -> bool:
        logger.exception(
            "NVLink detection not possible, as context support was"
            " not found. Assuming no NVLink available."
        )
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
            torch.npu.empty_cache()

        free_gpu_memory, _ = torch.npu.mem_get_info(device_id)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="npu")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

    @classmethod
    def log_warnings(cls) -> None:
        pass

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.npu.reset_peak_memory_stats(device)
        return float(torch.npu.max_memory_allocated(device))

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        logger.info("Using Torch SDPA backend.")
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        )

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa

    @classmethod
    def enable_dit_layerwise_offload_for_wan_by_default(cls) -> bool:
        """The performance of the layerwise_offload feature depends on the device's memory size and the memory size occupied by the model. Use --dit-layerwise-offload True if it suitable for your case."""
        return False
