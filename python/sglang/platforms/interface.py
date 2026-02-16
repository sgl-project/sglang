# SPDX-License-Identifier: Apache-2.0
"""
Unified Platform Interface for SGLang.

This module defines the base Platform class and enumerations used for
platform abstraction across both SRT (LLM serving) and multimodal_gen
(diffusion models).

The platform interface provides:
- Platform detection and identification
- Device capability queries
- Memory management
- Backend selection (attention, sampling, MOE)
- Distributed communication configuration
- Server argument post-processing

Design Goals:
1. Lazy Loading - Don't import platform-specific modules until needed
2. IDE Navigation - Use @cached_property with direct imports for Ctrl+Click
3. Type Safety - Proper type hints for IDE autocomplete
4. Testability - Property access validates imports at test time
5. Performance - Use @lru_cache for frequently-called platform checks
"""

from __future__ import annotations

import enum
import logging
import random
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class PlatformEnum(enum.Enum):
    """Enumeration of supported platforms."""

    CUDA = enum.auto()
    ROCM = enum.auto()
    MUSA = enum.auto()
    NPU = enum.auto()
    XPU = enum.auto()
    HPU = enum.auto()
    CPU = enum.auto()
    MPS = enum.auto()
    TPU = enum.auto()
    OOT = enum.auto()  # Out-of-tree custom platform
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    """CPU architecture enumeration."""

    X86 = enum.auto()
    ARM = enum.auto()
    AARCH64 = enum.auto()
    UNSPECIFIED = enum.auto()


class DeviceCapability:
    """
    Device compute capability.

    Primary use: CUDA SM version (e.g., SM 8.0 for A100, SM 9.0 for H100).

    For non-CUDA platforms, this can represent platform-specific versioning
    or None may be returned by get_device_capability().

    Note: minor version must be in range [0, 9] for to_int() to work correctly.
    This constraint is validated at construction time.
    """

    __slots__ = ("major", "minor")

    def __new__(cls, major: int, minor: int) -> "DeviceCapability":
        if not isinstance(major, int) or not isinstance(minor, int):
            raise TypeError("major and minor must be integers")
        if major < 0:
            raise ValueError(f"major version must be non-negative, got {major}")
        if not (0 <= minor < 10):
            raise ValueError(f"minor version must be 0-9, got {minor}")
        instance = object.__new__(cls)
        object.__setattr__(instance, "major", major)
        object.__setattr__(instance, "minor", minor)
        return instance

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("DeviceCapability is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("DeviceCapability is immutable")

    def __repr__(self) -> str:
        return f"DeviceCapability(major={self.major}, minor={self.minor})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DeviceCapability):
            return self.major == other.major and self.minor == other.minor
        if isinstance(other, tuple) and len(other) == 2:
            return self.major == other[0] and self.minor == other[1]
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, DeviceCapability):
            return (self.major, self.minor) < (other.major, other.minor)
        if isinstance(other, tuple) and len(other) == 2:
            return (self.major, self.minor) < other
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, DeviceCapability):
            return (self.major, self.minor) <= (other.major, other.minor)
        if isinstance(other, tuple) and len(other) == 2:
            return (self.major, self.minor) <= other
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, DeviceCapability):
            return (self.major, self.minor) > (other.major, other.minor)
        if isinstance(other, tuple) and len(other) == 2:
            return (self.major, self.minor) > other
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, DeviceCapability):
            return (self.major, self.minor) >= (other.major, other.minor)
        if isinstance(other, tuple) and len(other) == 2:
            return (self.major, self.minor) >= other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.major, self.minor))

    def __iter__(self):
        """Allow unpacking: major, minor = capability"""
        return iter((self.major, self.minor))

    def as_version_str(self) -> str:
        """Return as version string, e.g., '8.0'."""
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express capability as integer for comparison.

        Example: SM 8.0 -> 80, SM 9.0 -> 90
        """
        return self.major * 10 + self.minor


class Platform:
    """
    Base class for all platform implementations.

    This unified interface consolidates platform-specific logic for both
    SRT (LLM serving) and multimodal_gen (diffusion models).

    Note: This is intentionally NOT an ABC (Abstract Base Class) because:
    1. Many methods have sensible default implementations
    2. Some methods raise NotImplementedError as a "soft" abstract - subclasses
       should override them if they can provide the functionality
    3. We want to allow instantiation of Platform for testing/mocking

    Subclasses MUST define:
    - _enum: PlatformEnum value identifying the platform
    - device_name: Human-readable name (e.g., "cuda", "rocm")
    - device_type: torch device type string

    Subclasses SHOULD override methods that raise NotImplementedError if they
    can provide the functionality (e.g., get_device_name, get_device_total_memory).

    ## Op Registration Pattern

    Ops are registered as @cached_property methods with direct imports inside.
    This provides:
    - Lazy loading: Import happens only on first access
    - IDE navigation: Ctrl+Click on import goes to actual implementation
    - Testability: Property access validates the import

    Example:
        @cached_property
        def silu_and_mul(self):
            '''SiLU fused with multiply.'''
            from sgl_kernel import silu_and_mul
            return silu_and_mul
    """

    # === Core Identity (must be defined by subclasses) ===
    _enum: ClassVar[PlatformEnum]
    device_name: ClassVar[str]
    device_type: ClassVar[str]

    # === Optional class attributes ===
    dispatch_key: ClassVar[str] = "CPU"
    simple_compile_backend: ClassVar[str] = "inductor"

    # Use tuple (immutable) instead of list to avoid shared mutable state
    supported_quantization: ClassVar[tuple[str, ...]] = ()

    # Dummy attribute for compatibility
    device: torch.device | None = None

    # =========================================================================
    # Platform Detection Methods
    # =========================================================================

    @lru_cache(maxsize=1)
    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    @lru_cache(maxsize=1)
    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    @lru_cache(maxsize=1)
    def is_hip(self) -> bool:
        """Alias for is_rocm (HIP is the ROCm programming model)."""
        return self._enum == PlatformEnum.ROCM

    @lru_cache(maxsize=1)
    def is_npu(self) -> bool:
        return self._enum == PlatformEnum.NPU

    @lru_cache(maxsize=1)
    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    @lru_cache(maxsize=1)
    def is_hpu(self) -> bool:
        return self._enum == PlatformEnum.HPU

    @lru_cache(maxsize=1)
    def is_musa(self) -> bool:
        return self._enum == PlatformEnum.MUSA

    @lru_cache(maxsize=1)
    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    @lru_cache(maxsize=1)
    def is_mps(self) -> bool:
        return self._enum == PlatformEnum.MPS

    @lru_cache(maxsize=1)
    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    @lru_cache(maxsize=1)
    def is_cuda_alike(self) -> bool:
        """Returns True for CUDA-compatible platforms (CUDA, ROCm)."""
        # Note: MUSA is NOT included here - it uses mccl, not nccl
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    # Static class methods for checks without instance
    @classmethod
    def is_cuda_static(cls) -> bool:
        return getattr(cls, "_enum", None) == PlatformEnum.CUDA

    @classmethod
    def is_rocm_static(cls) -> bool:
        return getattr(cls, "_enum", None) == PlatformEnum.ROCM

    # =========================================================================
    # Device Capabilities
    # =========================================================================

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        """
        Get device compute capability.

        Returns:
            DeviceCapability(major, minor) for CUDA/ROCm, or None for platforms
            where this concept doesn't apply.
        """
        return None

    @classmethod
    def has_device_capability(
        cls, capability: tuple[int, int] | int, device_id: int = 0
    ) -> bool:
        """Check if device meets minimum capability."""
        current = cls.get_device_capability(device_id)
        if current is None:
            return False
        if isinstance(capability, tuple):
            return current >= capability
        return current.to_int() >= capability

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get device model name (e.g., 'NVIDIA A100-SXM4-80GB')."""
        raise NotImplementedError

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get the uuid of a device, e.g. the PCI bus ID."""
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total device memory in bytes."""
        raise NotImplementedError

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        """Get available device memory in GiB."""
        raise NotImplementedError

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """Return the memory usage in bytes."""
        raise NotImplementedError

    @property
    def has_amx(self) -> bool:
        """Check for Intel AMX support (CPU only)."""
        return False

    @property
    def has_nvlink(self) -> bool:
        """Check for NVLink connectivity (NVIDIA CUDA only)."""
        return False

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        """Get the CPU architecture of the current platform."""
        return CpuArchEnum.UNSPECIFIED

    # =========================================================================
    # Platform Initialization
    # =========================================================================

    def init_platform(self) -> None:
        """
        One-time platform initialization.

        Called once after platform detection to perform setup like:
        - Monkey-patching PyTorch for compatibility
        - Loading vendor-specific libraries
        - Setting environment variables

        Override in subclasses for platform-specific initialization.
        """
        pass

    # =========================================================================
    # Server Argument Post-processing
    # =========================================================================

    def postprocess_server_args(self, args: "ServerArgs") -> None:
        """
        Apply platform-specific defaults to server arguments.

        Called during ServerArgs initialization to set backend defaults,
        enable/disable features, or apply other configuration changes.

        Override in subclasses to set platform-specific defaults.
        """
        pass

    # =========================================================================
    # Backend Selection
    # =========================================================================

    def get_default_attention_backend(self, model_config: Any = None) -> str:
        """Get the default attention backend for this platform."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.get_default_attention_backend() is not implemented. "
            "This method is reserved for future use in Phase 6 (Server Args Integration)."
        )

    def get_default_sampling_backend(self) -> str:
        """Get the default sampling backend for this platform."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.get_default_sampling_backend() is not implemented. "
            "This method is reserved for future use in Phase 6 (Server Args Integration)."
        )

    def get_default_moe_runner_backend(self) -> str:
        """Get the default MOE runner backend for this platform."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.get_default_moe_runner_backend() is not implemented. "
            "This method is reserved for future use in Phase 6 (Server Args Integration)."
        )

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: Any,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        """
        Get the attention backend class as a qualified string.

        Compatibility: Returns string path for multimodal_gen compatibility.
        """
        return ""

    # =========================================================================
    # Model and Quantization Verification
    # =========================================================================

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        """
        Verify whether the current platform supports the specified model
        architecture.
        """
        pass

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """Verify whether the quantization is supported by the current platform."""
        if cls.supported_quantization and quant not in cls.supported_quantization:
            raise ValueError(
                f"{quant} quantization is currently not supported in "
                f"{cls.device_name}."
            )

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """Check if the current platform supports async output."""
        raise NotImplementedError

    @classmethod
    def enable_dit_layerwise_offload_for_wan_by_default(cls) -> bool:
        """Whether to enable DIT layerwise offload by default."""
        return True

    # =========================================================================
    # Distributed Communication
    # =========================================================================

    @lru_cache(maxsize=1)
    def get_torch_distributed_backend_str(self) -> str:
        """
        Get the torch.distributed backend string.

        Returns:
            "mccl" for MUSA, "nccl" for CUDA/ROCm, "gloo" for CPU/MPS/others
        """
        # Check MUSA first - it's CUDA-like but uses mccl
        if self.is_musa():
            return "mccl"
        elif self.is_cuda_alike():
            return "nccl"
        else:
            return "gloo"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get the device communicator class as a qualified string.

        Returns string path for backward compatibility.
        """
        return "sglang.srt.distributed.device_communicators.base.DeviceCommunicatorBase"

    @cached_property
    def device_communicator(self) -> type:
        """
        Get the device communicator class (direct import, preferred API).

        Override in subclasses with direct imports for your platform.
        """
        from sglang.srt.distributed.device_communicators.base import (
            DeviceCommunicatorBase,
        )

        return DeviceCommunicatorBase

    # =========================================================================
    # Utilities
    # =========================================================================

    @lru_cache(maxsize=1)
    def get_device(self, local_rank: int = 0) -> torch.device:
        """Get a torch.device for this platform."""
        if self.is_cuda() or self.is_rocm():
            return torch.device("cuda", local_rank)
        elif self.is_musa():
            return torch.device("musa", local_rank)
        elif self.is_mps():
            return torch.device("mps")
        elif self.is_xpu():
            return torch.device("xpu", local_rank)
        elif self.is_hpu():
            return torch.device("hpu", local_rank)
        elif self.is_npu():
            return torch.device("npu", local_rank)
        else:
            return torch.device("cpu")

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """
        Set random seeds for reproducibility.

        Override in subclasses for platform-specific seeding.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # Only seed CUDA if available
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    @classmethod
    def inference_mode(cls):
        """
        A device-specific wrapper of `torch.inference_mode`.

        Some hardware backends (like TPU) don't support inference_mode
        and will fall back to torch.no_grad by overriding this method.
        """
        return torch.inference_mode(mode=True)


class UnspecifiedPlatform(Platform):
    """
    Placeholder platform for uninitialized state.

    Methods will raise RuntimeError if called before proper platform detection.
    """

    _enum = PlatformEnum.UNSPECIFIED
    device_name = "unspecified"
    device_type = ""

    def get_device(self, local_rank: int = 0) -> torch.device:
        raise RuntimeError(
            "Platform not initialized. Access current_platform to trigger detection."
        )

    def get_torch_distributed_backend_str(self) -> str:
        raise RuntimeError(
            "Platform not initialized. Access current_platform to trigger detection."
        )


class AttentionBackendEnum(enum.Enum):
    """Attention backend enumeration."""

    FA2 = enum.auto()
    FA = enum.auto()
    SLIDING_TILE_ATTN = enum.auto()
    TORCH_SDPA = enum.auto()
    SAGE_ATTN = enum.auto()
    SAGE_ATTN_3 = enum.auto()
    VIDEO_SPARSE_ATTN = enum.auto()
    SPARSE_VIDEO_GEN_2_ATTN = enum.auto()
    VMOBA_ATTN = enum.auto()
    AITER = enum.auto()
    SLA_ATTN = enum.auto()
    SAGE_SLA_ATTN = enum.auto()
    NO_ATTENTION = enum.auto()

    def __str__(self):
        return self.name.lower()

    @property
    def is_sparse(self) -> bool:
        return self in {
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN,
            AttentionBackendEnum.SPARSE_VIDEO_GEN_2_ATTN,
            AttentionBackendEnum.VMOBA_ATTN,
            AttentionBackendEnum.SLA_ATTN,
            AttentionBackendEnum.SAGE_SLA_ATTN,
        }


def resolve_obj_by_qualname(qualname: str):
    """
    Resolve an object by its qualified name.

    Args:
        qualname: Fully qualified name like 'sglang.platforms.cuda.CudaPlatform'

    Returns:
        The resolved object (class, function, etc.)

    Raises:
        ValueError: If qualname is invalid
        ImportError: If module cannot be imported
        AttributeError: If object not found in module
    """
    import importlib

    module_path, _, obj_name = qualname.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid qualname (no module path): {qualname}")
    if not obj_name:
        raise ValueError(f"Invalid qualname (no object name): {qualname}")

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{module_path}' for qualname '{qualname}': {e}"
        ) from e

    try:
        return getattr(module, obj_name)
    except AttributeError as e:
        raise AttributeError(
            f"Object '{obj_name}' not found in module '{module_path}': {e}"
        ) from e
