"""Lightweight metadata for the unified ``sglang.kernels`` namespace.

This module defines small, dependency-free descriptors used to *inventory*
kernel implementations and drive a simple, heuristic dispatch. It intentionally
does not import ``torch``, ``sgl_kernel`` or ``sglang.jit_kernel`` at module
import time so that ``import sglang.kernels`` stays cheap and works on a CPU-only
box (see RFC #29630, Phase 2).

The concrete callable behind a :class:`KernelSpec` is resolved lazily through
``KernelSpec.load()``; nothing is imported until a kernel is actually called.
"""

from __future__ import annotations

import importlib
from enum import Enum
from typing import Callable, Optional, Tuple

import msgspec


class KernelBackend(str, Enum):
    """Implementation backend for a kernel.

    Values mirror the backends called out in RFC #29630: JIT CUDA, AOT
    CUDA/C++ (the ``sgl_kernel`` wheel), Triton, CuTe DSL, FlashInfer, DeepGEMM,
    and the pure-``torch`` fallback path.
    """

    TORCH = "torch"
    TRITON = "triton"
    CUDA_JIT = "cuda_jit"  # sglang.jit_kernel
    CUDA_AOT = "cuda_aot"  # sgl_kernel wheel
    CUTE_DSL = "cute_dsl"
    FLASHINFER = "flashinfer"
    DEEPGEMM = "deepgemm"


class PlatformInfo(msgspec.Struct, frozen=True):
    """A minimal snapshot of the runtime accelerator platform.

    Kept torch-free at import time; use :meth:`detect` to build one from the
    live process (which does import ``torch``).
    """

    device_type: str = "cpu"  # "cuda", "hip", "cpu", ...
    cuda_arch_major: Optional[int] = None
    cuda_arch_minor: Optional[int] = None

    @property
    def is_cuda(self) -> bool:
        return self.device_type == "cuda"

    @property
    def is_hip(self) -> bool:
        return self.device_type == "hip"

    @classmethod
    def detect(cls) -> "PlatformInfo":
        """Build a :class:`PlatformInfo` from the current process.

        Never raises: if ``torch`` is missing or no accelerator is visible the
        default CPU platform is returned.
        """
        try:
            import torch
        except Exception:
            return cls()

        try:
            if torch.version.hip is not None and torch.cuda.is_available():
                return cls(device_type="hip")
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                return cls(
                    device_type="cuda",
                    cuda_arch_major=major,
                    cuda_arch_minor=minor,
                )
        except Exception:
            pass
        return cls()


class CapabilityRequirement(msgspec.Struct, frozen=True):
    """Coarse hardware requirement used to filter out unusable backends.

    ``min_cuda_arch`` / ``max_cuda_arch`` are ``(major, minor)`` tuples, e.g.
    ``(9, 0)`` for SM90. They only apply when the kernel requires CUDA.
    """

    requires_cuda: bool = False
    requires_hip: bool = False
    min_cuda_arch: Optional[Tuple[int, int]] = None
    max_cuda_arch: Optional[Tuple[int, int]] = None

    def is_satisfied_by(self, platform: PlatformInfo) -> bool:
        if self.requires_hip and not platform.is_hip:
            return False
        if self.requires_cuda and not platform.is_cuda:
            return False
        if platform.is_cuda and platform.cuda_arch_major is not None:
            arch = (platform.cuda_arch_major, platform.cuda_arch_minor or 0)
            if self.min_cuda_arch is not None and arch < self.min_cuda_arch:
                return False
            if self.max_cuda_arch is not None and arch > self.max_cuda_arch:
                return False
        return True


class FormatSignature(msgspec.Struct, frozen=True):
    """A light description of a kernel's data contract.

    This is deliberately loose in the first version — enough to document intent
    and support future inventory tooling, not a strict schema.
    """

    supported_dtypes: Tuple[str, ...] = ()
    in_place: bool = False
    description: str = ""


class KernelSpec(msgspec.Struct, frozen=True):
    """A single callable kernel implementation and its metadata.

    Parameters
    ----------
    op:
        Fully-qualified operator id, ``"<group>.<name>"`` (e.g.
        ``"layernorm.rmsnorm"``). This is the public lookup key.
    backend:
        Which :class:`KernelBackend` provides this implementation.
    target:
        Import path of the callable in ``"module:attr"`` form, resolved lazily
        by :meth:`load` (e.g. ``"sgl_kernel:rmsnorm"``).
    capability:
        Hardware requirement used by the selector to skip unusable backends.
    format_signature:
        Optional data-contract description for inventory/documentation.
    description:
        Human-readable one-liner.
    """

    op: str
    backend: KernelBackend
    target: str
    capability: CapabilityRequirement = msgspec.field(
        default_factory=CapabilityRequirement
    )
    format_signature: FormatSignature = msgspec.field(default_factory=FormatSignature)
    description: str = ""

    @property
    def group(self) -> str:
        return self.op.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.op.split(".", 1)[1] if "." in self.op else self.op

    def is_available(self, platform: PlatformInfo) -> bool:
        """Whether this backend can run on ``platform`` (metadata-only check)."""
        return self.capability.is_satisfied_by(platform)

    def load(self) -> Callable:
        """Import and return the backing callable.

        Raises the underlying ``ImportError`` / ``AttributeError`` if the
        backend is not installed on this platform — call sites decide how to
        handle that.
        """
        module_path, sep, attr = self.target.partition(":")
        if not sep or not attr:
            raise ValueError(
                f"KernelSpec.target must be 'module:attr', got {self.target!r}"
            )
        module = importlib.import_module(module_path)
        return getattr(module, attr)
