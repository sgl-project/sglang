# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-ascend: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/platform.py

import os
from enum import Enum
from functools import lru_cache
from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _check_sgl_kernel_npu_ops(backend: str, operator_names: tuple[str, ...]) -> None:
    import sgl_kernel_npu  # noqa: F401

    missing = [name for name in operator_names if not hasattr(torch.ops.npu, name)]
    if missing:
        operators = ", ".join(f"torch.ops.npu.{name}" for name in missing)
        platform = NPUPlatformBase.platform_generation().value
        raise ImportError(
            f"The installed sgl_kernel_npu is incompatible with {backend} "
            f"on {platform}: "
            f"missing registered operators: {operators}. "
            "Upgrade sgl-kernel-npu to a version providing these operators, "
            "built for your Ascend device and torch_npu/CANN versions."
        )


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


class NPUGenerationEnum(Enum):
    ASCEND_910 = "Ascend_910"
    ASCEND_950 = "Ascend_950"


class NPUPlatformBase(Platform):
    _enum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"
    dispatch_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"

    @classmethod
    def platform_generation(cls) -> NPUGenerationEnum:
        soc_version = torch.npu.get_soc_version()

        if soc_version >= 260:
            return NPUGenerationEnum.ASCEND_950

        return NPUGenerationEnum.ASCEND_910

    @classmethod
    @lru_cache(maxsize=1)
    def is_float64_supported(cls) -> bool:
        return False

    def tensor_on_device(self, t: torch.Tensor) -> bool:
        return t.is_npu

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        return torch.device(f"npu:{envs.LOCAL_RANK}")

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.npu.get_device_name(device_id))

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.npu.get_device_properties(device_id)
        return int(device_props.total_memory)

    @classmethod
    def inference_mode(cls):
        # npu kernels in diffusion paths may need tensor version counters
        return torch.no_grad()

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
        device_id: int | None = None,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        if empty_cache:
            torch.npu.empty_cache()

        if device_id is None:
            device_id = torch.npu.current_device()

        free_gpu_memory, _ = torch.npu.mem_get_info(device_id)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="npu")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

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
        if selected_backend == AttentionBackendEnum.FA:
            logger.info("Using Ascend Flash Attention backend.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa.AscendFABackend"

        elif selected_backend == AttentionBackendEnum.LASER_ATTN:
            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn import (  # noqa: F401
                    LaserAttentionBackend,
                )

                _check_sgl_kernel_npu_ops("laser_attn", ("laser_attn",))
                logger.info("Using Laser Attention backend")

                return "sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn.LaserAttentionBackend"
            except ImportError as e:
                logger.error(f"Failed to import Laser Attention backend: {e}")
                raise ImportError(
                    "Laser Attention backend is unavailable. "
                    "Laser Attention supports Ascend 910 (A2/A3) only. "
                    "Install sgl_kernel_npu built for your device; "
                    "on Ascend 950 (A5), select fa or rain_fusion_attn. "
                    f"Details: {e}"
                ) from e

        elif selected_backend == AttentionBackendEnum.BLOCK_SPARSE_ATTN:
            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.block_sparse_attn import (  # noqa: F401
                    BlockSparseAttentionBackend,
                )

                _check_sgl_kernel_npu_ops(
                    "block_sparse_attn",
                    ("sparse_block_estimate", "ada_block_sparse_attention"),
                )
                logger.info("Using Block Sparse Attention backend")

                return "sglang.multimodal_gen.runtime.layers.attention.backends.block_sparse_attn.BlockSparseAttentionBackend"
            except ImportError as e:
                logger.error(f"Failed to import Block Sparse Attention backend: {e}")
                raise ImportError(
                    "Block Sparse Attention backend is unavailable. "
                    "It supports Ascend 910 (A2/A3) only. "
                    "Install sgl_kernel_npu built for your device; "
                    "on Ascend 950 (A5), select fa or rain_fusion_attn. "
                    f"Details: {e}"
                ) from e

        elif selected_backend == AttentionBackendEnum.RAIN_FUSION_ATTN:
            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.rain_fusion_attn import (  # noqa: F401
                    RainFusionAttentionBackend,
                )

                if cls.platform_generation() == NPUGenerationEnum.ASCEND_910:
                    _check_sgl_kernel_npu_ops(
                        "rain_fusion_attn dense path", ("laser_attn",)
                    )
                logger.info("Using Rain Fusion Attention backend")

                return "sglang.multimodal_gen.runtime.layers.attention.backends.rain_fusion_attn.RainFusionAttentionBackend"
            except ImportError as e:
                logger.error(f"Failed to import Rain Fusion Attention backend: {e}")
                raise ImportError(
                    "Rain Fusion Attention backend is unavailable. "
                    "It requires torch_npu with npu_block_sparse_attention support; "
                    "on A2/A3, the dense path also requires sgl_kernel_npu. "
                    f"Details: {e}"
                ) from e

        logger.info("Using Torch SDPA backend.")
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        )

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa

    @classmethod
    def get_all_to_all_communicator_cls(cls) -> str:
        return (
            "sglang.multimodal_gen.runtime.distributed.device_communicators."
            "cpu_communicator.CpuCommunicator"
        )

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        """Whether automatic DiT layerwise offload is enabled on this platform."""
        return False
