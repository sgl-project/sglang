# SPDX-License-Identifier: Apache-2.0
# Intel XPU Platform support for SGLang Diffusion

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    MMPlatform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.platforms.xpu import XpuDeviceMixin

logger = init_logger(__name__)


class XpuPlatform(XpuDeviceMixin, MMPlatform):
    """Platform implementation for Intel XPU (Data Center GPU Max, Arc, etc.)."""

    dispatch_key: str = "XPU"
    device_control_env_var: str = "ZE_AFFINITY_MASK"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """Check if async output is supported on Intel XPU."""
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, disable enforce-eager. "
                "Since enforce-eager is enabled, async output processor cannot be used"
            )
            return False
        return True

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        """Get the attention backend class string for Intel XPU.

        Defaults to XPU backend (requires fp16/bf16 and a supported head size),
        falling back to Torch SDPA if constraints are not met.
        """
        if selected_backend in (AttentionBackendEnum.FA, None):
            if dtype not in (torch.float16, torch.bfloat16):
                logger.info(
                    "XPU attention backend requires fp16/bf16 but got dtype=%s; falling back to Torch SDPA.",
                    dtype,
                )
                return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (  # noqa: F401
                    XPUAttentionBackend,
                )

                supported_sizes = XPUAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info(
                        "XPU attention backend does not support head_size=%d; falling back to Torch SDPA.",
                        head_size,
                    )
                    return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

                logger.info("Using XPU attention backend on Intel XPU.")
                return "sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend.XPUAttentionBackend"
            except Exception as e:
                logger.warning(
                    "Failed to import/use XPU attention backend (%s); falling back to Torch SDPA.",
                    e,
                )
                return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        if selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend for Intel XPU.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        if selected_backend in (
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.SAGE_ATTN_3,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN,
            AttentionBackendEnum.VMOBA_ATTN,
            AttentionBackendEnum.AITER,
        ):
            logger.warning(
                f"{selected_backend.name} is not supported on Intel XPU. "
                "Falling back to Torch SDPA backend."
            )
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        # Default fallback
        logger.info("Using Torch SDPA backend for Intel XPU (default).")
        return (
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        )

    def get_all_to_all_communicator_class(self) -> type:
        from sglang.multimodal_gen.runtime.distributed.device_communicators.cpu_communicator import (
            CpuCommunicator,
        )

        return CpuCommunicator
