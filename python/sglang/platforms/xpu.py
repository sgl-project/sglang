# SPDX-License-Identifier: Apache-2.0
"""
Intel XPU platform implementation.

This module provides the XpuPlatform class for Intel GPU support (Xe, Arc, Data Center GPUs).
Intel XPU uses the SYCL programming model exposed through PyTorch's torch.xpu API.

Ops are registered as @cached_property with direct imports for:
- Lazy loading (import only on first access)
- IDE navigation (Ctrl+Click on import goes to implementation)
- Testability (property access validates import)
"""

from __future__ import annotations

import logging
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any

import torch

from sglang.platforms.interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class XpuPlatform(Platform):
    """
    Intel XPU platform implementation.

    Supports Intel GPUs including:
    - Intel Arc (consumer GPUs)
    - Intel Data Center GPU Max Series (Ponte Vecchio)
    - Intel Data Center GPU Flex Series
    """

    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    device_control_env_var: str = "ONEAPI_DEVICE_SELECTOR"

    def init_platform(self) -> None:
        """Initialize XPU platform."""
        self.log_warnings()

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        """Get the local torch.device for the current process."""
        import os

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"xpu:{local_rank}")

    # =========================================================================
    # Device Capabilities
    # =========================================================================

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        """
        Get device compute capability.

        XPU version format is different from CUDA SM. We parse the version
        string to extract major.minor if available.
        """
        try:
            capability_info = torch.xpu.get_device_capability(device_id)
            version_str = capability_info.get("version", "")
            if not version_str:
                logger.debug(
                    "XPU device %d returned empty version string. "
                    "Device capability checks will be skipped.",
                    device_id,
                )
                return None
            parts = version_str.split(".")
            if len(parts) >= 2:
                major = int(parts[0])
                minor = int(parts[1]) if parts[1].isdigit() else 0
                # Ensure minor is in valid range [0, 9]
                if minor > 9:
                    logger.debug(
                        "XPU device %d minor version %d exceeds range [0-9], clamping to 9.",
                        device_id,
                        minor,
                    )
                    minor = 9
                return DeviceCapability(major=major, minor=minor)
            logger.warning(
                "XPU device %d returned unexpected version format: '%s'. "
                "Expected 'major.minor' format.",
                device_id,
                version_str,
            )
            return None
        except AttributeError as e:
            logger.warning(
                "XPU device capability API not available for device %d: %s. "
                "This may indicate an older PyTorch version or driver issue.",
                device_id,
                e,
            )
            return None
        except (ValueError, KeyError) as e:
            logger.warning(
                "Failed to parse XPU device capability for device %d: %s. "
                "Check driver and PyTorch XPU installation.",
                device_id,
                e,
            )
            return None

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.xpu.get_device_name(device_id))

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get device UUID from properties."""
        props = torch.xpu.get_device_properties(device_id)
        return f"XPU-{device_id}-{props.name.replace(' ', '_')}-{props.total_memory}"

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return int(torch.xpu.get_device_properties(device_id).total_memory)

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """Return current memory usage in bytes."""
        return float(torch.xpu.memory_allocated(device))

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
            torch.xpu.empty_cache()

        free_gpu_memory, total_memory = torch.xpu.mem_get_info(device_id)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="xpu")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

    # =========================================================================
    # XPU-specific capability checks
    # =========================================================================

    @classmethod
    @lru_cache(maxsize=8)
    def has_xmx_support(cls, device_id: int = 0) -> bool:
        """Check if XPU has XMX (Xe Matrix eXtensions) support."""
        try:
            return torch.xpu.get_device_properties(device_id).has_fp64
        except (AttributeError, RuntimeError):
            return False

    @classmethod
    @lru_cache(maxsize=1)
    def is_pvc(cls) -> bool:
        """Check if running on Intel Ponte Vecchio (Data Center GPU Max)."""
        name = torch.xpu.get_device_name(0).lower()
        return "max" in name or "ponte vecchio" in name or "pvc" in name

    @classmethod
    @lru_cache(maxsize=1)
    def is_arc(cls) -> bool:
        """Check if running on Intel Arc GPU."""
        name = torch.xpu.get_device_name(0).lower()
        return "arc" in name

    # =========================================================================
    # Async Output Support
    # =========================================================================

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """XPU async output support depends on graph compilation."""
        if enforce_eager:
            logger.warning(
                "Async output processing requires graph mode. "
                "Since enforce-eager is enabled, async output processor cannot be used."
            )
            return False
        return True

    # =========================================================================
    # Activation Ops - Uses sgl_kernel (same as CUDA)
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

    # =========================================================================
    # LayerNorm Ops - Uses sgl_kernel (same as CUDA)
    # =========================================================================

    @cached_property
    def rmsnorm(self):
        """RMS Normalization kernel."""
        from sgl_kernel import rmsnorm

        return rmsnorm

    @cached_property
    def fused_add_rmsnorm(self):
        """Fused add + RMS Normalization kernel."""
        from sgl_kernel import fused_add_rmsnorm

        return fused_add_rmsnorm

    @cached_property
    def gemma_rmsnorm(self):
        """Gemma-style RMS Normalization kernel."""
        from sgl_kernel import gemma_rmsnorm

        return gemma_rmsnorm

    @cached_property
    def gemma_fused_add_rmsnorm(self):
        """Gemma-style fused add + RMS Normalization kernel."""
        from sgl_kernel import gemma_fused_add_rmsnorm

        return gemma_fused_add_rmsnorm

    # =========================================================================
    # Distributed Communication
    # =========================================================================

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Return the XPU device communicator class path."""
        return "sglang.srt.distributed.device_communicators.xpu_communicator.XpuCommunicator"

    @cached_property
    def device_communicator(self) -> type:
        """Get XPU device communicator (uses oneCCL)."""
        from sglang.srt.distributed.device_communicators.xpu_communicator import (
            XpuCommunicator,
        )

        return XpuCommunicator

    @lru_cache(maxsize=1)
    def get_torch_distributed_backend_str(self) -> str:
        """XPU uses ccl (oneCCL) for distributed communication."""
        return "ccl"

    # =========================================================================
    # Server Args Post-processing
    # =========================================================================

    def postprocess_server_args(self, args: "ServerArgs") -> None:
        """Apply XPU-specific defaults to server arguments."""
        if args.attention_backend is None:
            args.attention_backend = "intel_xpu"
        if args.sampling_backend is None:
            args.sampling_backend = "pytorch"

    # =========================================================================
    # Utilities
    # =========================================================================

    @lru_cache(maxsize=1)
    def get_device(self, local_rank: int = 0) -> torch.device:
        """Get a torch.device for XPU."""
        return torch.device("xpu", local_rank)

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """Set random seeds for reproducibility on XPU."""
        import random

        import numpy as np

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.xpu.is_available():
                torch.xpu.manual_seed_all(seed)

    @classmethod
    def log_warnings(cls) -> None:
        """Log XPU-specific warnings."""
        # Check for common XPU configuration issues
        device_count = torch.xpu.device_count()
        if device_count > 1:
            logger.info(
                "Detected %d XPU devices. For multi-GPU, ensure ONEAPI_DEVICE_SELECTOR is set correctly.",
                device_count,
            )
