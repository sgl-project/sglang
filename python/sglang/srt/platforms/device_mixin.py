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
import logging
import os
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from typing import NamedTuple, Optional

import numpy as np
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


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
    MLU = enum.auto()
    TPU = enum.auto()
    MPS = enum.auto()
    SAIL = enum.auto()
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


_DEVICE_TO_DISTRIBUTED_BACKEND: dict[str, str] = {
    "cuda": "nccl",
    "xpu": "xccl",
    "hpu": "hccl",
    "cpu": "gloo",
    "npu": "hccl" if not envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0 else "zbal",
    "musa": "mccl",
    "mlu": "cncl",
}


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

    def is_mlu(self) -> bool:
        return self._enum == PlatformEnum.MLU

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_mps(self) -> bool:
        return self._enum == PlatformEnum.MPS

    def is_sail(self) -> bool:
        return self._enum == PlatformEnum.SAIL

    def is_cuda_alike(self) -> bool:
        """True for CUDA, ROCm, MUSA, or SAIL (all expose CUDA-like APIs)."""
        return self._enum in (
            PlatformEnum.CUDA,
            PlatformEnum.ROCM,
            PlatformEnum.MUSA,
            PlatformEnum.SAIL,
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

    def memory_reserved(self, device_id: int = 0) -> int:
        """[Active] Bytes the caching allocator holds for this device.

        Routed through ``torch.accelerator`` so it follows whichever device
        torch was built for, instead of naming one vendor's namespace. Never
        defaulted to 0: a caller differencing this across a load reads 0 as
        "nothing was allocated" and would silently under-report. Override when
        a device's allocator is not reachable this way.
        """
        return int(torch.accelerator.memory_reserved(device_id))

    # ---- Weight cache ----

    def weight_cache_transports(self) -> tuple[str, ...]:
        """[Active] Weight-cache transport names this device can serve, best first.

        Empty means the weight cache daemon refuses to start here. A name must
        be registered in ``weight_cache.transport``. The default reproduces the
        platform gate the daemon applied before this was a capability, so a
        device opts in by overriding rather than by being added to a condition
        inside the daemon.
        """
        if self.is_cuda_alike():
            return ("vmm_fd", "torch_ipc")
        if self.is_npu():
            return ("torch_ipc",)
        return ()

    def validate_weight_cache_allocator(self) -> None:
        """[Active] Reject an allocator config whose memory cannot be exported.

        Called once by the daemon before it touches the device, so a
        configuration that would break the handle export fails with an
        actionable message instead of partway through it. The default carries
        the CUDA expandable-segments check the daemon applied unconditionally
        before this was a hook.
        """
        # The expandable-segments allocator hands out memory that cannot be
        # shared through torch's _share_cuda_ handle, so the export would fail
        # partway with an opaque CUDA error.
        for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            conf = os.environ.get(var, "")
            for field in conf.split(","):
                key, _, value = field.partition(":")
                if (
                    key.strip() == "expandable_segments"
                    and value.strip().lower() == "true"
                ):
                    raise RuntimeError(
                        f"[WeightCacheDaemon] {var} sets expandable_segments:True, "
                        f"which is incompatible with CUDA IPC weight sharing: the "
                        f"expandable-segments allocator hands out memory that cannot "
                        f"be exported via _share_cuda_, so the IPC handle export "
                        f"would fail mid-way. Unset expandable_segments for the "
                        f"weight cache daemon process (it can stay enabled for the "
                        f"engine itself)."
                    )

    def prepare_weight_cache_export(
        self, state_tensors: "Mapping[str, tuple[torch.Tensor, bool]]"
    ) -> None:
        """[Active] Run device setup the handle export needs, before exporting.

        Called by the daemon with every tensor it is about to export, after
        ``validate_weight_cache_allocator`` and before the transport backend is
        chosen. The place for an import whose side effect registers a reducer,
        or for a probe that warns when zero-copy is unavailable on this driver
        stack. Raise to abort the daemon; log to let it continue degraded.
        """
        if not self.is_npu():
            return
        # torch_npu must be imported before the first ForkingPickler dump of an
        # NPU tensor so its reducer is registered; otherwise the tensor falls
        # back to a CPU copy instead of zero-copy IPC. Probe once and warn
        # (rather than error) if zero-copy is unavailable, so weight caching
        # still works by copying the weights through the CPU.
        import torch.multiprocessing.reductions as _reductions
        import torch_npu  # noqa: F401

        first = next(iter(state_tensors.values()), None)
        if first is None:
            return
        rebuild_fn, _ = _reductions.reduce_tensor(first[0].data)
        if rebuild_fn.__name__ != "rebuild_npu_tensor":
            logger.warning(
                "NPU weight-cache zero-copy IPC is unavailable: tensor "
                "reduced to %r instead of torch_npu's rebuild_npu_tensor; "
                "falling back to a CPU copy. Zero-copy requires Ascend HDK "
                ">= 25.3.RC1 and CANN >= 8.3.RC1, and is unsupported on "
                "Ascend 950DT.",
                rebuild_fn,
            )

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        """[Active] Get current peak memory usage in bytes."""
        raise NotImplementedError

    def is_pin_memory_available(self, device=None) -> bool:
        """[Active] Whether pinned host memory is available for a target device."""
        return False

    @contextmanager
    def reindex_device_id(self, device_id: int) -> Iterator[int]:
        """[Active] Temporarily remap a physical device to logical device 0."""
        yield device_id

    # ------------------------------------------------------------------
    # Planned methods — reserved interface.  Core still uses hardcoded
    # calls (e.g. torch.cuda.*).  OOT implementations will NOT take
    # effect until the core is migrated in a future PR.
    # ------------------------------------------------------------------

    # ---- Device management ----

    def get_device(self, device_id: int = 0) -> str:
        """[Planned] Return ``torch.device`` for the given device id."""
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
        """Return the torch.distributed backend string (e.g. "nccl", "hccl").

        Default: lookup ``self.device_type`` in ``_DEVICE_TO_DISTRIBUTED_BACKEND``,
        falling back to ``"gloo"``. Subclasses override only when they need a
        non-default backend (e.g. mooncake, or a brand-new device).
        """
        return _DEVICE_TO_DISTRIBUTED_BACKEND.get(self.device_type, "gloo")

    def get_communicator_class(self) -> type | None:
        """[Planned] Return platform-specific communicator class, or None for default."""
        return None

    # ---- Misc ----

    @classmethod
    def inference_mode(cls):
        """[Planned] Return inference mode context manager."""
        return torch.inference_mode(mode=True)

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """[Planned] Set random seeds for reproducibility across all libraries."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def verify_quantization(self, quant: str) -> None:
        """[Planned] Validate that a quantization method is supported. No-op by default."""
        pass

    @cached_property
    def cpu_arch(self) -> "CpuArchEnum":
        return self.get_cpu_architecture()

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

    def get_torch_profiler_activity_str(self) -> str:
        """[Planned] Return the torch profiler activity string."""
        raise NotImplementedError

    def get_torch_profiler_activity(self) -> torch.profiler.ProfilerActivity:
        """[Planned] Return the torch profiler activity."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device_name})"
