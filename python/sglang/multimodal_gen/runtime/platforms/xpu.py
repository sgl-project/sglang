# SPDX-License-Identifier: Apache-2.0
# Intel XPU Platform support for SGLang Diffusion

import torch

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class XpuPlatform(Platform):
    """Platform implementation for Intel XPU (Data Center GPU Max, Arc, etc.)."""

    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    device_control_env_var: str = "ZE_AFFINITY_MASK"

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        """Intel XPU doesn't have CUDA-style compute capability."""
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of the Intel XPU device."""
        return torch.xpu.get_device_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get the UUID of the Intel XPU device."""
        props = torch.xpu.get_device_properties(device_id)
        return str(props.uuid)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total memory of the Intel XPU device in bytes."""
        props = torch.xpu.get_device_properties(device_id)
        return props.total_memory

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
    def log_warnings(cls) -> None:
        """Log any XPU-specific warnings."""
        pass

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """Get current memory usage on Intel XPU."""
        torch.xpu.reset_peak_memory_stats(device)
        return float(torch.xpu.max_memory_allocated(device))

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group=None,
    ) -> float:
        """Return the available device memory in GiB."""

        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return 0.0

        num_gpus = torch.xpu.device_count()
        if device_id < 0 or device_id >= num_gpus:
            raise ValueError(f"Invalid XPU device_id={device_id}. num_gpus={num_gpus}")

        current = torch.xpu.current_device()
        if current != device_id:
            logger.warning(
            "current device is not %s, but %s; this may cause useless memory allocation for torch XPU context.",
            device_id,
            current,
            )

        if empty_cache:
            torch.xpu.empty_cache()

        used_memory = float(torch.xpu.memory_allocated(device_id))
        total_gpu_memory = float(
            torch.xpu.get_device_properties(device_id).total_memory
        )

        free_gpu_memory = max(0.0, total_gpu_memory - used_memory)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(
                free_gpu_memory,
                dtype=torch.float32,
                device=torch.device("xpu", device_id),
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        """Get the attention backend class string for Intel XPU.

        Intel XPU supports PyTorch's native SDPA which is the most portable option.
        """
        if (
            selected_backend == AttentionBackendEnum.TORCH_SDPA
            or selected_backend is None
        ):
            logger.info("Using Torch SDPA backend for Intel XPU.")
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"

        if selected_backend == AttentionBackendEnum.FA:
            logger.warning(
                "Flash Attention is not natively supported on Intel XPU. "
                "Falling back to Torch SDPA backend."
            )
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

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Get device communicator class for Intel XPU distributed communication."""
        # Use base communicator for now; can be updated to use oneCCL-based communicator
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"
