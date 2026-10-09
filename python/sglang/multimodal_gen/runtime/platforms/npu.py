# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-ascend: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/platform.py

import os
from functools import lru_cache

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    MMPlatform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.platforms.npu import NPUDeviceMixin

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


class NPUPlatformBase(NPUDeviceMixin, MMPlatform):
    dispatch_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"

    @classmethod
    @lru_cache(maxsize=1)
    def is_float64_supported(cls) -> bool:
        return False

    def tensor_on_device(self, t: torch.Tensor) -> bool:
        return t.is_npu

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

                logger.info("Using Laser Attention backend")

                return "sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn.LaserAttentionBackend"
            except ImportError as e:
                logger.error(f"Failed to import Laser Attention backend: {e}")
                raise ImportError(
                    "Laser Attention backend is not installed. "
                    "It requires the `attentions` module which can be installed along with sgl_kernel_npu. "
                    "Manual installation from source is required. See https://github.com/sgl-project/sgl-kernel-npu."
                ) from e

        elif selected_backend == AttentionBackendEnum.BLOCK_SPARSE_ATTN:
            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.block_sparse_attn import (  # noqa: F401
                    BlockSparseAttentionBackend,
                )

                logger.info("Using Block Sparse Attention backend")

                return "sglang.multimodal_gen.runtime.layers.attention.backends.block_sparse_attn.BlockSparseAttentionBackend"
            except ImportError as e:
                logger.error(f"Failed to import Block Sparse Attention backend: {e}")
                raise ImportError(
                    "Block Sparse Attention backend is not installed. "
                    "It requires the `attentions` module which can be installed along with sgl_kernel_npu. "
                    "Manual installation from source is required. See https://github.com/sgl-project/sgl-kernel-npu."
                ) from e

        elif selected_backend == AttentionBackendEnum.RAIN_FUSION_ATTN:
            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.rain_fusion_attn import (  # noqa: F401
                    RainFusionAttentionBackend,
                )

                logger.info("Using Rain Fusion Attention backend")

                return "sglang.multimodal_gen.runtime.layers.attention.backends.rain_fusion_attn.RainFusionAttentionBackend"
            except ImportError as e:
                logger.error(f"Failed to import Rain Fusion Attention backend: {e}")
                raise ImportError(
                    "Rain Fusion Attention backend is not installed. "
                    "It requires the `attentions` module which can be installed along with sgl_kernel_npu. "
                    "Manual installation from source is required. See https://github.com/sgl-project/sgl-kernel-npu."
                ) from e

        logger.info("Using Torch SDPA backend.")
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        )

    def get_communicator_class(self) -> type:
        from sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )

        return CudaCommunicator

    def get_all_to_all_communicator_class(self) -> type:
        from sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator import (
            CpuCommunicator,
        )

        return CpuCommunicator

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        """Whether automatic DiT layerwise offload is enabled on this platform."""
        return False
