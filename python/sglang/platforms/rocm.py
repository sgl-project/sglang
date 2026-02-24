# SPDX-License-Identifier: Apache-2.0
"""
AMD ROCm (HIP) platform implementation.

This module provides the RocmPlatform class for AMD GPU support.
ROCm uses the HIP programming model and shares many APIs with CUDA.

Ops are registered as @cached_property with direct imports for:
- Lazy loading (import only on first access)
- IDE navigation (Ctrl+Click on import goes to implementation)
- Testability (property access validates import)
"""

from __future__ import annotations

import logging
import os
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any

import torch

from sglang.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _use_aiter() -> bool:
    """Check if AITER backend should be used for ROCm."""
    from sglang.srt.utils import get_bool_env_var

    return get_bool_env_var("SGLANG_USE_AITER")


class RocmPlatform(Platform):
    """
    AMD ROCm (HIP) platform implementation.

    ROCm uses the HIP programming model which provides CUDA-like APIs.
    torch.cuda APIs work on ROCm through the HIP compatibility layer.
    """

    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"  # ROCm uses torch.cuda APIs
    dispatch_key: str = "CUDA"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    def init_platform(self) -> None:
        """Initialize ROCm platform."""
        self.log_warnings()

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        """Get the local torch.device for the current process.

        ROCm uses CUDA APIs, so we use cuda device type with LOCAL_RANK.
        """
        import os

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")

    # =========================================================================
    # Device Capabilities
    # =========================================================================

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        """Get device compute capability (GFX version mapped to major.minor)."""
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get device UUID via torch.cuda properties."""
        props = torch.cuda.get_device_properties(device_id)
        return f"GPU-{device_id}-{props.name.replace(' ', '_')}-{props.total_memory}"

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return int(torch.cuda.get_device_properties(device_id).total_memory)

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """Return peak memory usage in bytes."""
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.max_memory_allocated(device))

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        """Return available memory in GiB."""
        if empty_cache:
            torch.cuda.empty_cache()

        free_gpu_memory, _ = torch.cuda.mem_get_info(device_id)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

    # =========================================================================
    # ROCm-specific capability checks
    # =========================================================================

    @classmethod
    @lru_cache(maxsize=1)
    def is_mi300x(cls) -> bool:
        """Check if running on AMD MI300X."""
        name = torch.cuda.get_device_name(0).lower()
        return "mi300x" in name or "instinct mi300x" in name

    @classmethod
    @lru_cache(maxsize=1)
    def is_mi250(cls) -> bool:
        """Check if running on AMD MI250."""
        name = torch.cuda.get_device_name(0).lower()
        return "mi250" in name

    # =========================================================================
    # Async Output Support
    # =========================================================================

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """ROCm supports async output when CUDA graphs are enabled."""
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA graph. "
                "Since enforce-eager is enabled, async output processor cannot be used."
            )
            return False
        return True

    # =========================================================================
    # Activation Ops - Lazy loaded via @cached_property
    # =========================================================================

    @cached_property
    def silu_and_mul(self):
        """SiLU activation fused with multiply (SwiGLU)."""
        from sgl_kernel import silu_and_mul

        return silu_and_mul

    @cached_property
    def gelu_and_mul(self):
        """GELU activation fused with element-wise multiply."""
        from sgl_kernel import gelu_and_mul

        return gelu_and_mul

    @cached_property
    def gelu_tanh_and_mul(self):
        """GELU (tanh approximation) activation fused with multiply."""
        from sgl_kernel import gelu_tanh_and_mul

        return gelu_tanh_and_mul

    @cached_property
    def gelu_quick(self):
        """QuickGELU activation (ROCm-specific optimized kernel)."""
        from sgl_kernel import gelu_quick

        return gelu_quick

    # =========================================================================
    # LayerNorm Ops - Uses aiter or vllm fallback
    # =========================================================================

    @cached_property
    def rmsnorm(self):
        """RMS Normalization kernel (aiter or vllm backend)."""
        if _use_aiter():
            from aiter import rmsnorm2d_fwd

            return rmsnorm2d_fwd
        else:
            from vllm._custom_ops import rms_norm

            return rms_norm

    @cached_property
    def fused_add_rmsnorm(self):
        """Fused add + RMS Normalization kernel (aiter or vllm backend)."""
        if _use_aiter():
            from aiter import rmsnorm2d_fwd_with_add

            return rmsnorm2d_fwd_with_add
        else:
            from vllm._custom_ops import fused_add_rms_norm

            return fused_add_rms_norm

    @cached_property
    def gemma_rmsnorm(self):
        """Gemma-style RMS Normalization - not optimized for ROCm."""
        raise NotImplementedError(
            "gemma_rmsnorm is not available on ROCm. "
            "Use forward_native fallback instead."
        )

    @cached_property
    def gemma_fused_add_rmsnorm(self):
        """Gemma-style fused add + RMS Normalization - not optimized for ROCm."""
        raise NotImplementedError(
            "gemma_fused_add_rmsnorm is not available on ROCm. "
            "Use forward_native fallback instead."
        )

    # =========================================================================
    # Attention Backend Selection
    # =========================================================================

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        """Get the attention backend class for ROCm."""
        if selected_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        elif selected_backend == AttentionBackendEnum.AITER:
            if dtype not in (torch.float16, torch.bfloat16):
                logger.warning(
                    "AITer backend works best with fp16/bf16 inputs but got dtype=%s. "
                    "Proceeding with AITer anyway.",
                    dtype,
                )
            logger.info("Using AITer backend on ROCm.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerBackend"

        elif selected_backend in (
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
        ):
            raise ValueError(
                f"{selected_backend.name} is not supported on {cls.device_name}."
            )
        elif selected_backend and selected_backend != AttentionBackendEnum.FA:
            raise ValueError(
                f"Invalid attention backend for {cls.device_name}: {selected_backend}"
            )

        # Default to FlashAttention if supported
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

                from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
                    FlashAttentionBackend,
                )

                supported_sizes = FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    logger.info(
                        "Cannot use FlashAttention backend for head size %d.",
                        head_size,
                    )
                    target_backend = AttentionBackendEnum.TORCH_SDPA
            except ImportError:
                logger.info(
                    "Cannot use FlashAttention backend because the "
                    "flash_attn package is not found."
                )
                target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        logger.info("Using Flash Attention backend.")
        return "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn.FlashAttentionBackend"

    # =========================================================================
    # Distributed Communication
    # =========================================================================

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Return the CUDA/ROCm device communicator class path (RCCL)."""
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator.CudaCommunicator"

    @cached_property
    def device_communicator(self) -> type:
        """Get ROCm device communicator (uses RCCL via CUDA communicator)."""
        from sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )

        return CudaCommunicator

    # =========================================================================
    # Server Args Post-processing
    # =========================================================================

    def postprocess_server_args(self, args: "ServerArgs") -> None:
        """Apply ROCm-specific defaults to server arguments.

        Note: attention_backend and sampling_backend defaults are handled by
        _handle_attention_backend_compatibility() and _handle_sampling_backend()
        in ServerArgs, which have complex logic accounting for MLA, aiter
        backend selection based on head count, etc.
        Do NOT set them here to avoid bypassing that logic.
        """
        pass

    # =========================================================================
    # Utilities
    # =========================================================================

    @classmethod
    def enable_dit_layerwise_offload_for_wan_by_default(cls) -> bool:
        """ROCm performs better without DIT layerwise offload on Wan."""
        return False

    @classmethod
    def log_warnings(cls) -> None:
        """Log ROCm-specific warnings."""
        # Check for common ROCm configuration issues
        if (
            "HIP_VISIBLE_DEVICES" in os.environ
            and "CUDA_VISIBLE_DEVICES" not in os.environ
        ):
            logger.warning(
                "HIP_VISIBLE_DEVICES is set but CUDA_VISIBLE_DEVICES is not. "
                "ROCm typically uses CUDA_VISIBLE_DEVICES for device selection."
            )
