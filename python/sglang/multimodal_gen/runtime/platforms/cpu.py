# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/cpu.py

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    MMPlatform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.platforms.cpu import CpuDeviceMixin

logger = init_logger(__name__)


class CpuPlatform(CpuDeviceMixin, MMPlatform):
    dispatch_key = "CPU"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return True

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        if selected_backend not in (None, AttentionBackendEnum.TORCH_SDPA):
            logger.warning(
                "%s is not supported on CPU; falling back to Torch SDPA.",
                selected_backend,
            )

        logger.info("Using Torch SDPA backend for CPU.")
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        )

    def get_communicator_class(self) -> type:
        from sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator import (
            CpuCommunicator,
        )

        return CpuCommunicator

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        """Whether automatic DiT layerwise offload is enabled on this platform."""
        return False
