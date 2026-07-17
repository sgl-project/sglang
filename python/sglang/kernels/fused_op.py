"""Multi-backend operator contract for the unified kernels namespace.

:class:`BaseFusedOp` is the per-operator implementation object behind the
``sglang.kernels.ops.*`` wrappers (RFC #29630): one logical operator,
implemented once, with multiple interchangeable backends behind a single
``forward()``.

Each subclass implements one ``forward_<backend>`` method per backend it
supports:

- ``forward_native`` — **required**; the pure-``torch`` correctness reference
  every other backend is checked against.
- ``forward_torch_compile`` — provided by the base class as
  ``torch.compile(forward_native)``.
- ``forward_triton`` / ``forward_jit`` / ``forward_aot`` /
  ``forward_cute_dsl`` / ``forward_flashinfer`` / ``forward_deepgemm`` —
  opt-in overrides.

A backend is *available* iff the subclass overrides its method (``native`` and
``torch_compile`` are always available). ``forward()`` picks the best
available backend by :attr:`BaseFusedOp.priority`, filtered per call through
:meth:`BaseFusedOp.backend_eligible` (which checks
:class:`~sglang.kernels.spec.CapabilityRequirement` against the detected
:class:`~sglang.kernels.spec.PlatformInfo`). The ``SGLANG_FORCE_FUSED_OP_BACKEND``
env var (or :func:`set_fused_op_backend`) forces every fused op onto one
backend — e.g. ``native`` to bisect numerical bugs against the reference
implementations with a single switch.

Like the rest of ``sglang.kernels``, importing this module (and instantiating
subclasses) never imports a kernel backend (``sgl_kernel`` /
``sglang.jit_kernel``) or triggers JIT compilation; backends are imported
lazily inside the ``forward_<backend>`` methods.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import (
    AbstractSet,
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

import msgspec

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
    capabilities_satisfied,
)

# backend (provenance) -> forward_<backend> method name.
BACKEND_METHODS: Dict[KernelBackend, str] = {
    KernelBackend.TORCH: "forward_native",
    KernelBackend.TORCH_COMPILE: "forward_torch_compile",
    KernelBackend.TRITON: "forward_triton",
    KernelBackend.JIT: "forward_jit",
    KernelBackend.AOT: "forward_aot",
    KernelBackend.CUTE_DSL: "forward_cute_dsl",
    KernelBackend.FLASHINFER: "forward_flashinfer",
    KernelBackend.DEEPGEMM: "forward_deepgemm",
    KernelBackend.AITER: "forward_aiter",
    KernelBackend.TORCH_NPU: "forward_npu",
}

# best -> fallback. ``torch_compile`` is deliberately absent: auto-selection
# must never trigger a surprise compilation in a serving process; force it
# explicitly when wanted. Per-op priority overrides this (see BaseFusedOp).
DEFAULT_PRIORITY: Tuple[KernelBackend, ...] = (
    KernelBackend.AOT,
    KernelBackend.JIT,
    KernelBackend.FLASHINFER,
    KernelBackend.DEEPGEMM,
    KernelBackend.CUTE_DSL,
    KernelBackend.AITER,
    KernelBackend.TORCH_NPU,
    KernelBackend.TRITON,
    KernelBackend.TORCH,
)

# Backends every op supports structurally: forward_native is abstract (so a
# concrete subclass always has it) and forward_torch_compile derives from it.
_ALWAYS_AVAILABLE = (KernelBackend.TORCH, KernelBackend.TORCH_COMPILE)


@functools.lru_cache(maxsize=1)
def _platform() -> PlatformInfo:
    return PlatformInfo.detect()


# --- global backend override ------------------------------------------------

# Sentinel distinguishing "not resolved yet" from "resolved to None (no force)".
_UNRESOLVED = object()
_forced_backend: Any = _UNRESOLVED


def get_fused_op_backend() -> Optional[KernelBackend]:
    """The process-wide forced backend, or ``None`` for auto-selection.

    Resolved once from ``SGLANG_FORCE_FUSED_OP_BACKEND`` on first use; tests
    and tools flip it afterwards via :func:`set_fused_op_backend`.
    """
    global _forced_backend
    if _forced_backend is _UNRESOLVED:
        from sglang.srt.environ import envs

        value = envs.SGLANG_FORCE_FUSED_OP_BACKEND.get()
        _forced_backend = KernelBackend(value) if value is not None else None
    return _forced_backend


def set_fused_op_backend(backend: Optional[KernelBackend]) -> None:
    """Force every :class:`BaseFusedOp` onto ``backend`` (``None`` = auto)."""
    global _forced_backend
    _forced_backend = backend


# --- optional call tracing ----------------------------------------------------


class FusedOpTraceRecord(msgspec.Struct, frozen=True):
    """One traced ``forward()`` call: which op ran on which backend, and the
    tensor shapes/dtypes it saw."""

    op: str
    backend: str
    tensor_args: Tuple[str, ...]  # e.g. "torch.bfloat16[128, 4096]"


_trace_enabled: bool = False
_trace_records: List[FusedOpTraceRecord] = []


def enable_fused_op_trace() -> None:
    """Record every fused-op call (op, backend, tensor shapes/dtypes).

    Gives an accurate inventory of which kernels a model actually exercises
    and at what shapes — the raw material for representative-shape test sets.
    """
    global _trace_enabled
    _trace_enabled = True


def disable_fused_op_trace() -> None:
    global _trace_enabled
    _trace_enabled = False


def get_fused_op_trace() -> List[FusedOpTraceRecord]:
    return list(_trace_records)


def clear_fused_op_trace() -> None:
    _trace_records.clear()


def _describe_tensors(args: tuple, kwargs: dict) -> Tuple[str, ...]:
    import torch

    described = []
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            described.append(f"{value.dtype}[{', '.join(map(str, value.shape))}]")
    return tuple(described)


# --- the per-operator contract ------------------------------------------------


class BaseFusedOp(ABC):
    """One logical operator with interchangeable backends behind ``forward()``.

    Subclasses set :attr:`op` and implement :meth:`forward_native` plus any
    optimized ``forward_<backend>`` methods. All backend methods of one op
    must share the same signature and semantics — each override adapts its
    underlying kernel's calling convention so call sites never care which
    backend ran.

    Class attributes
    ----------------
    op:
        Operator id, ``"<group>.<name>"`` (e.g. ``"layernorm.rmsnorm"``).
    priority:
        Backend preference for auto-selection, best first. Defaults to
        :data:`DEFAULT_PRIORITY`.
    capabilities:
        Per-backend set of :class:`CapabilityRequirement` (OR semantics;
        omitted / empty = runs on any device), consulted by
        :meth:`backend_eligible` (and exported into the registry specs). Use the
        ``CapabilityRequirement.CUDA`` / ``.HIP`` / ``.NPU`` shortcuts, e.g.
        ``{KernelBackend.AOT: {CapabilityRequirement.CUDA, CapabilityRequirement.HIP}}``.
    format_signature:
        Data-contract description shared by all backends of this op.
    descriptions:
        Optional per-backend one-liners for the registry inventory.
    """

    op: ClassVar[str]
    priority: ClassVar[Tuple[KernelBackend, ...]] = DEFAULT_PRIORITY
    capabilities: ClassVar[
        Mapping[KernelBackend, AbstractSet[CapabilityRequirement]]
    ] = {}
    format_signature: ClassVar[FormatSignature] = FormatSignature()
    descriptions: ClassVar[Mapping[KernelBackend, str]] = {}

    def __init__(self) -> None:
        # Cache the structural backend set and the priority-ordered subset once
        # so forward() avoids repeated introspection on the hot path.
        available = []
        for backend in KernelBackend:
            if backend in _ALWAYS_AVAILABLE or self._overrides(
                BACKEND_METHODS[backend]
            ):
                available.append(backend)
        self._available: Tuple[KernelBackend, ...] = tuple(available)
        self._ordered: Tuple[KernelBackend, ...] = tuple(
            b for b in self.priority if b in set(available)
        )
        self._compiled_native = None

    def _overrides(self, method_name: str) -> bool:
        for klass in type(self).__mro__:
            if klass is BaseFusedOp:
                return False
            if method_name in klass.__dict__:
                return True
        return False

    # --- backends: native is required; the rest are opt-in overrides ---

    @abstractmethod
    def forward_native(self, *args, **kwargs):
        """Pure-``torch`` reference implementation (correctness ground truth)."""

    def forward_torch_compile(self, *args, **kwargs):
        if self._compiled_native is None:
            import torch

            self._compiled_native = torch.compile(self.forward_native)
        return self._compiled_native(*args, **kwargs)

    def forward_triton(self, *args, **kwargs):
        raise NotImplementedError(f"{self.op}: no triton backend")

    def forward_jit(self, *args, **kwargs):
        raise NotImplementedError(f"{self.op}: no jit backend")

    def forward_aot(self, *args, **kwargs):
        raise NotImplementedError(f"{self.op}: no aot backend")

    def forward_cute_dsl(self, *args, **kwargs):
        raise NotImplementedError(f"{self.op}: no cute_dsl backend")

    def forward_flashinfer(self, *args, **kwargs):
        raise NotImplementedError(f"{self.op}: no flashinfer backend")

    def forward_deepgemm(self, *args, **kwargs):
        raise NotImplementedError(f"{self.op}: no deepgemm backend")

    def forward_aiter(self, *args, **kwargs):
        raise NotImplementedError(f"{self.op}: no aiter backend")

    def forward_npu(self, *args, **kwargs):
        raise NotImplementedError(f"{self.op}: no npu backend")

    # --- selection ---

    def available_backends(self) -> List[KernelBackend]:
        """Backends this op implements (structural check, platform-agnostic)."""
        return list(self._available)

    def backend_eligible(self, backend: KernelBackend, *args, **kwargs) -> bool:
        """Whether ``backend`` may run *this* call.

        The base implementation checks the backend's
        :class:`CapabilityRequirement` set (OR semantics) against the detected
        platform. Subclasses may extend it with per-call shape/dtype gates so
        auto-selection bounces to the next backend instead of raising.
        """
        return capabilities_satisfied(
            self.capabilities.get(backend, frozenset()), _platform()
        )

    def _resolve_backend(self, *args, **kwargs) -> KernelBackend:
        forced = get_fused_op_backend()
        if forced is not None:
            return forced
        for backend in self._ordered:
            if self.backend_eligible(backend, *args, **kwargs):
                return backend
        return KernelBackend.TORCH

    # --- dispatch ---

    def forward(self, *args, backend: Optional[KernelBackend] = None, **kwargs):
        """Run the op on ``backend``, or on the best eligible one when omitted."""
        if backend is None:
            backend = self._resolve_backend(*args, **kwargs)
        result = getattr(self, BACKEND_METHODS[backend])(*args, **kwargs)
        if _trace_enabled:
            _trace_records.append(
                FusedOpTraceRecord(
                    op=self.op,
                    backend=backend.value,
                    tensor_args=_describe_tensors(args, kwargs),
                )
            )
        return result

    __call__ = forward


def register_fused_op(instance: BaseFusedOp, module: str, attr: str) -> BaseFusedOp:
    """Register every available backend of ``instance`` in the kernel registry.

    ``module``/``attr`` locate the module-level instance so that
    ``KernelSpec.load()`` can lazily resolve e.g.
    ``"<module>:<attr>.forward_aot"`` to the bound backend method. Returns
    ``instance`` so group packages can write
    ``_RMSNORM = register_fused_op(_RMSNormOp(), __name__, "_RMSNORM")``.
    """
    for backend in instance.available_backends():
        register_kernel(
            KernelSpec(
                op=instance.op,
                backend=backend,
                target=f"{module}:{attr}.{BACKEND_METHODS[backend]}",
                capabilities=frozenset(instance.capabilities.get(backend, ())),
                format_signature=instance.format_signature,
                description=instance.descriptions.get(backend, ""),
            )
        )
    return instance
