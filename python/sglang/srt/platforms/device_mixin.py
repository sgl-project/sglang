"""
Shared device abstraction for SGLang platforms.

DeviceMixin provides the common device identity queries and operations
shared between the SRT (LLM inference) and Multimodal (diffusion)
platform hierarchies.  Concrete per-device mixins (e.g. MyDeviceMixin)
implement the abstract operations; subsystem-specific platforms
(SRTPlatform, MMPlatform) inherit DeviceMixin and add their own methods.

Hierarchy example (OOT plugin)::

    DeviceMixin
    ├── MyDeviceMixin(DeviceMixin)        # vendor-specific device operations
    ├── SRTPlatform(DeviceMixin)          # + graph runner, KV pool, …
    │   └── MySRTPlatform(SRTPlatform, MyDeviceMixin)
    └── MMPlatform(DeviceMixin)           # + attention backend, VAE, …
        └── MyMMPlatform(MMPlatform, MyDeviceMixin)

Method status annotations:

- ``[Active]``  — SGLang core calls this method through ``current_platform``.
  OOT implementations take effect immediately.
- ``[Planned]`` — Reserved interface. SGLang core still uses hardcoded calls
  (e.g. ``torch.cuda.empty_cache()``). OOT implementations will NOT take
  effect until the core is migrated in a future PR.
"""

import enum
from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:
    import torch


class PlatformEnum(enum.Enum):
    """Enumeration of known platform types.

    Superset of both SRT and MM enums so that a single PlatformEnum can
    be shared across subsystems.
    """

    CUDA = enum.auto()
    ROCM = enum.auto()
    CPU = enum.auto()
    XPU = enum.auto()
    MUSA = enum.auto()
    NPU = enum.auto()
    TPU = enum.auto()
    MPS = enum.auto()
    OOT = enum.auto()  # Out-of-tree (external plugin)
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    """CPU architecture enumeration."""

    X86 = enum.auto()
    ARM = enum.auto()
    UNSPECIFIED = enum.auto()


class DeviceCapability(NamedTuple):
    """Device compute capability (major, minor).

    Uses NamedTuple for built-in comparison support:
    ``DeviceCapability(9, 0) >= DeviceCapability(8, 9)`` works naturally.
    """

    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """Express capability as ``<major><minor>`` (minor is single digit)."""
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


class DeviceMixin:
    """Mixin providing device identity queries and basic device operations.

    Class-level attributes (override in subclasses):
        _enum:       PlatformEnum identifying this platform.
        device_name: Human-readable short name (e.g. "cuda", "npu").
        device_type: ``torch.device`` type string (e.g. "cuda", "npu").
    """

    _enum: PlatformEnum = PlatformEnum.UNSPECIFIED
    device_name: str = "unknown"
    device_type: str = "cpu"

    # ------------------------------------------------------------------
    # Platform identity queries
    # ------------------------------------------------------------------

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    def is_musa(self) -> bool:
        return self._enum == PlatformEnum.MUSA

    def is_npu(self) -> bool:
        return self._enum == PlatformEnum.NPU

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_mps(self) -> bool:
        return self._enum == PlatformEnum.MPS

    def is_cuda_alike(self) -> bool:
        """True for CUDA, ROCm, or MUSA (all expose CUDA-like APIs)."""
        return self._enum in (
            PlatformEnum.CUDA,
            PlatformEnum.ROCM,
            PlatformEnum.MUSA,
        )

    def is_out_of_tree(self) -> bool:
        """True for externally-registered OOT platforms."""
        return self._enum == PlatformEnum.OOT

    # ------------------------------------------------------------------
    # Active methods — core calls these through current_platform.
    # OOT implementations take effect immediately.
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """[Active] Get total device memory in bytes."""
        raise NotImplementedError

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        """[Active] Get current peak memory usage in bytes."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Planned methods — reserved interface.  Core still uses hardcoded
    # calls (e.g. torch.cuda.*).  OOT implementations will NOT take
    # effect until the core is migrated in a future PR.
    # ------------------------------------------------------------------

    # ---- Device management ----

    def get_device(self, local_rank: int) -> "torch.device":
        """[Planned] Return ``torch.device`` for the given local rank."""
        raise NotImplementedError

    def set_device(self, device: "torch.device") -> None:
        """[Planned] Set the current device."""
        raise NotImplementedError

    def get_device_name(self, device_id: int = 0) -> str:
        """[Planned] Get human-readable device name."""
        raise NotImplementedError

    def get_device_uuid(self, device_id: int = 0) -> str:
        """[Planned] Get unique device identifier string."""
        raise NotImplementedError

    def get_device_capability(self, device_id: int = 0) -> Optional["DeviceCapability"]:
        """[Planned] Get device compute capability. None if N/A."""
        raise NotImplementedError

    def empty_cache(self) -> None:
        """[Planned] Release cached device memory. No-op for CPU-like platforms."""
        pass

    def synchronize(self) -> None:
        """[Planned] Synchronize device operations. No-op for CPU-like platforms."""
        pass

    # ---- Memory ----

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        """[Planned] Return ``(free_bytes, total_bytes)``."""
        raise NotImplementedError

    # ---- Distributed ----

    def get_torch_distributed_backend_str(self) -> str:
        """[Planned] Return the torch.distributed backend string (e.g. "nccl", "hccl")."""
        raise NotImplementedError

    def get_communicator_class(self) -> type | None:
        """[Planned] Return platform-specific communicator class, or None for default."""
        return None

    # ---- Misc ----

    @classmethod
    def inference_mode(cls):
        """[Planned] Return inference mode context manager."""
        import torch

        return torch.inference_mode(mode=True)

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """[Planned] Set random seeds for reproducibility across all libraries."""
        if seed is not None:
            import random

            import numpy as np
            import torch

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def verify_quantization(self, quant: str) -> None:
        """[Planned] Validate that a quantization method is supported. No-op by default."""
        pass

    @classmethod
    def get_cpu_architecture(cls) -> "CpuArchEnum":
        """[Planned] Detect CPU architecture."""
        import platform as _platform

        machine = _platform.machine().lower()
        if machine in ("x86_64", "amd64", "i386", "i686"):
            return CpuArchEnum.X86
        elif machine in ("arm64", "aarch64"):
            return CpuArchEnum.ARM
        return CpuArchEnum.UNSPECIFIED

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device_name})"
