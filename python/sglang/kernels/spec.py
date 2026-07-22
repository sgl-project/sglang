"""Lightweight metadata for the unified ``sglang.kernels`` namespace.

This module defines small, dependency-free descriptors used to *inventory*
kernel implementations and drive a simple, heuristic dispatch. It intentionally
does not import ``torch``, ``sgl_kernel`` or ``sglang.jit_kernel`` at module
import time so that ``import sglang.kernels`` stays cheap and works on a CPU-only
box (see RFC #29630, Phase 2).

The concrete callable behind a :class:`KernelSpec` is resolved lazily through
``KernelSpec.load()``; nothing is imported until a kernel is actually called.

Backend vs. device (RFC #29630 follow-up): :class:`KernelBackend` names only the
*provenance* of an implementation (how it is built / where it comes from), not
the hardware it runs on. Both JIT and AOT sources already build for CUDA *and*
ROCm, and a wheel may ship only a per-op subset, so platform support is
per-``(op, backend)`` metadata carried by :class:`CapabilityRequirement`, not
derivable from the backend name.
"""

from __future__ import annotations

import importlib
from enum import Enum
from typing import Callable, ClassVar, FrozenSet, Optional, Tuple, Union

import msgspec


class KernelBackend(str, Enum):
    """Provenance of a kernel implementation (how it is built), not its device.

    ``JIT`` (``sglang.jit_kernel``, compiles under nvcc *and* hipcc) and ``AOT``
    (the ``sgl_kernel`` wheel, built for CUDA *and* ROCm) are both cross-device;
    which devices a given op supports is expressed by its
    :class:`CapabilityRequirement` list. Platform-specific libraries (e.g.
    ``aiter`` on AMD, ``torch_npu`` on Ascend) are just additional provenance
    values, each pinned to its device by its ``CapabilityRequirement``.
    """

    TORCH = "torch"  # pure-torch reference (forward_native)
    TORCH_COMPILE = "torch_compile"  # torch.compile(forward_native)
    TRITON = "triton"
    JIT = "jit"  # sglang.jit_kernel (nvcc / hipcc)
    AOT = "aot"  # sgl_kernel wheel (CUDA / ROCm builds)
    CUTE_DSL = "cute_dsl"
    FLASHINFER = "flashinfer"  # FlashInfer auto-tuned wrapper (e.g. bmm_fp8/mm_fp4)
    DEEPGEMM = "deepgemm"
    AITER = "aiter"  # AMD aiter library (device=HIP)
    TORCH_NPU = "torch_npu"  # Ascend NPU vendor runtime (device=NPU)
    MARLIN = "marlin"  # Marlin weight-only fallback kernels
    # Named FlashInfer GEMM sub-kernels that are independently selectable
    # (unlike the generic FLASHINFER wrapper, which auto-tunes internally).
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_DEEPGEMM = "flashinfer_deepgemm"
    FLASHINFER_CUDNN = "flashinfer_cudnn"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    # TODO(RFC #29630): more provenance as needed (cpu-avx, sgl_kernel_npu, ...)


class DeviceType(str, Enum):
    """Accelerator device family a kernel can run on."""

    CUDA = "cuda"
    HIP = "hip"
    NPU = "npu"  # Ascend NPU (torch_npu / sgl_kernel_npu)
    CPU = "cpu"
    # TODO(RFC #29630): XPU / MUSA / ... as backends land.


class PlatformInfo(msgspec.Struct, frozen=True):
    """A minimal snapshot of the runtime accelerator platform.

    Kept torch-free at import time; use :meth:`detect` to build one from the
    live process (which does import ``torch``).
    """

    device_type: str = "cpu"  # "cuda", "hip", "cpu", ...
    cuda_arch_major: Optional[int] = None
    cuda_arch_minor: Optional[int] = None

    @property
    def device(self) -> DeviceType:
        try:
            return DeviceType(self.device_type)
        except (ValueError, TypeError):
            return DeviceType.CPU

    @property
    def is_cuda(self) -> bool:
        return self.device_type == "cuda"

    @property
    def is_hip(self) -> bool:
        return self.device_type == "hip"

    @classmethod
    def detect(cls) -> PlatformInfo:
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
            npu = getattr(torch, "npu", None)
            if npu is not None and npu.is_available():
                return cls(device_type="npu")
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
    """One device (plus an optional CUDA-arch window) a backend can run on.

    A :class:`KernelSpec` / :class:`~sglang.kernels.fused_op.BaseFusedOp` backend
    carries a *set* of these with **OR** semantics — any matching entry makes the
    backend eligible, and an empty set means unrestricted (runs anywhere). A set
    (not a tuple) because order and duplicates are meaningless here: ``{CUDA,
    HIP}`` and ``{HIP, CUDA}`` describe the same thing. This replaces the old
    ``requires_cuda`` / ``requires_hip`` booleans (whose AND semantics could not
    express "CUDA or HIP"); arch bounds now attach to the device they describe
    (``min_cuda_arch`` / ``max_cuda_arch`` apply only when ``device == CUDA``).

    The device-only cases are so common that they are exposed as class constants
    (``CapabilityRequirement.CUDA`` / ``.HIP`` / ``.NPU``); use :meth:`cuda` for an
    arch-bounded CUDA requirement (e.g. ``CapabilityRequirement.cuda(
    min_sm=(10, 0))`` for SM100+).
    """

    device: DeviceType
    min_cuda_arch: Optional[Tuple[int, int]] = None
    max_cuda_arch: Optional[Tuple[int, int]] = None

    # Common device-only shortcuts, assigned after the class body (they are
    # instances of the class itself). ClassVar keeps them out of msgspec fields.
    CUDA: ClassVar[CapabilityRequirement]
    HIP: ClassVar[CapabilityRequirement]
    NPU: ClassVar[CapabilityRequirement]

    @classmethod
    def cuda(
        cls,
        min_sm: Optional[Tuple[int, int]] = None,
        max_sm: Optional[Tuple[int, int]] = None,
    ) -> CapabilityRequirement:
        """A CUDA requirement bounded to an SM-arch window (inclusive)."""
        return cls(device=DeviceType.CUDA, min_cuda_arch=min_sm, max_cuda_arch=max_sm)

    def is_satisfied_by(self, platform: PlatformInfo) -> bool:
        if self.device != platform.device:
            return False
        if self.device == DeviceType.CUDA and platform.cuda_arch_major is not None:
            arch = (platform.cuda_arch_major, platform.cuda_arch_minor or 0)
            if self.min_cuda_arch is not None and arch < self.min_cuda_arch:
                return False
            if self.max_cuda_arch is not None and arch > self.max_cuda_arch:
                return False
        return True


CapabilityRequirement.CUDA = CapabilityRequirement(device=DeviceType.CUDA)
CapabilityRequirement.HIP = CapabilityRequirement(device=DeviceType.HIP)
CapabilityRequirement.NPU = CapabilityRequirement(device=DeviceType.NPU)


def capabilities_satisfied(
    capabilities: Union[
        FrozenSet[CapabilityRequirement],
        Tuple[CapabilityRequirement, ...],
        CapabilityRequirement,
    ],
    platform: PlatformInfo,
) -> bool:
    """OR over ``capabilities`` (empty = unrestricted).

    Accepts a set/tuple of requirements, or tolerates a single
    :class:`CapabilityRequirement` (the pre-decouple API used one) by wrapping it.
    """
    if isinstance(capabilities, CapabilityRequirement):
        capabilities = (capabilities,)
    return (not capabilities) or any(c.is_satisfied_by(platform) for c in capabilities)


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
        Which :class:`KernelBackend` (provenance) provides this implementation.
    target:
        Import path of the callable in ``"module:attr"`` form, resolved lazily
        by :meth:`load` (e.g. ``"sgl_kernel:rmsnorm"``). ``attr`` may be a
        dotted path into a module-level object, e.g.
        ``"sglang.kernels.ops.layernorm:_RMSNORM.forward_aot"`` for a bound
        :class:`~sglang.kernels.fused_op.BaseFusedOp` backend method.
    capabilities:
        Set of :class:`CapabilityRequirement` (OR semantics; empty = runs on
        any device) used by the selector to skip backends unusable on the
        detected platform.
    format_signature:
        Optional data-contract description for inventory/documentation.
    description:
        Human-readable one-liner.
    """

    op: str
    backend: KernelBackend
    target: str
    capabilities: FrozenSet[CapabilityRequirement] = frozenset()
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
        return capabilities_satisfied(self.capabilities, platform)

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
        obj = importlib.import_module(module_path)
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj
