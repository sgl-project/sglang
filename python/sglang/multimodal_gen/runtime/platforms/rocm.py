# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from rocm/vllm: https://github.com/ROCm/vllm/blob/v0.7.3%2Brocm/vllm/platforms/rocm.py
"""
This file is a platform abstraction for ROCm GPUs,
adjusted to match the structure and interface of `cuda.py`.
"""

import torch

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ROCm uses the same torch.cuda interface
class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"  # torch uses 'cuda' backend string
    dispatch_key: str = "CUDA"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA graph. "
                "Since enforce-eager is enabled, async output processor cannot be used"
            )
            return False
        return True

    @classmethod
    def log_warnings(cls) -> None:
        pass  # ROCm-specific warnings can be added here

    @classmethod
    def get_current_memory_usage(cls, device: torch.device | None = None) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.max_memory_allocated(device))

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        logger.info(
            "Trying SGL_DIFFUSION_ATTENTION_BACKEND=%s",
            envs.SGL_DIFFUSION_ATTENTION_BACKEND,
        )

        if selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        elif selected_backend in (AttentionBackendEnum.FA, None):
            pass

        elif selected_backend in (
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
        ):
            raise ValueError(
                f"{selected_backend.name} is not supported on {cls.device_name}."
            )
        elif selected_backend:
            raise ValueError(
                f"Invalid attention backend for {cls.device_name}: {selected_backend}"
            )

        target_backend = AttentionBackendEnum.FA
        if dtype not in (torch.float16, torch.bfloat16):
            logger.info(
                "Cannot use FlashAttention backend for dtype other than "
                "torch.float16 or torch.bfloat16."
            )
            target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.FA:
            try:
                import flash_attn  # noqa: F401

                from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (  # noqa: F401
                    FlashAttentionBackend,
                )

                supported_sizes = FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info(
                        "Cannot use FlashAttention-2 backend for head size %d.",
                        head_size,
                    )
                    target_backend = AttentionBackendEnum.TORCH_SDPA
            except ImportError:
                logger.info(
                    "Cannot use FlashAttention backend because the "
                    "flash_attn package is not found. "
                    "Make sure that flash_attn was built and installed "
                    "(on by default)."
                )
                target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")

            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        logger.info("Using Flash Attention backend.")

        return "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn.FlashAttentionBackend"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # works for ROCm too
