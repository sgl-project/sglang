"""Unified multi-backend / multi-platform operator contract (RFC #29630, #26426).

:class:`BaseFusedOp` is the single operator abstraction of the unified
``sglang.kernels`` namespace: one logical operator, implemented once, with
multiple interchangeable implementations behind a single ``forward()``. It
subsumes the former ``MultiPlatformOp`` (``sglang.srt.layers.utils``), so it is
a proper :class:`torch.nn.Module` and covers **two independent dimensions**:

- **Kernel backend (provenance)** — where an implementation comes from
  (:class:`~sglang.kernels.spec.KernelBackend`): ``forward_native`` (pure
  torch, required), ``forward_torch_compile``, ``forward_triton``,
  ``forward_jit``, ``forward_aot``, ``forward_cute_dsl``,
  ``forward_flashinfer``, ``forward_deepgemm``, ``forward_aiter``,
  ``forward_torch_npu``. Which *devices* a backend supports is per-``(op,
  backend)`` metadata (:attr:`BaseFusedOp.capabilities`), never implied by the
  backend name.
- **Platform / device** — device-specific composite paths inherited from
  ``MultiPlatformOp``: ``forward_cuda``, ``forward_hip``, ``forward_npu``,
  ``forward_xpu``, ``forward_musa``, ``forward_cpu``, plus ``forward_<key>``
  for out-of-tree (OOT) platform plugins. CUDA / HIP are **not** kernel
  backends.

Dispatch priority (highest first), resolved by :meth:`BaseFusedOp.forward`:

1. **Explicit backend** — ``forward(..., backend=KernelBackend.X)``.
2. **Global forced backend** — ``SGLANG_FORCE_FUSED_OP_BACKEND`` /
   :func:`set_fused_op_backend` (e.g. ``native`` to bisect numerical bugs).
   Best-effort: an op that does not implement the forced backend falls back
   to normal dispatch with a one-time warning, so the debug switch works on
   whole models that contain device-only ops.
3. **OOT platform override** — on an out-of-tree platform, a forward
   registered via :meth:`BaseFusedOp.register_oot_forward`, then a
   ``forward_<dispatch_key>`` method, then ``forward_native``.
4. **Optimized kernel backends** — the first backend in :attr:`priority`
   whose method is overridden, that is *declared* in :attr:`capabilities`,
   and whose :class:`~sglang.kernels.spec.CapabilityRequirement` set matches
   the detected platform. Ops may extend :meth:`backend_eligible` with
   per-call shape/dtype gates; overriding it switches this step from a
   statically cached choice to per-call selection.
5. **Platform-specific forward** — ``forward_cuda`` on CUDA, ``forward_hip``
   (falling back to ``forward_cuda``) on ROCm, ``forward_musa`` on MUSA,
   ``forward_npu`` / ``forward_xpu`` on Ascend / XPU, ``forward_cpu`` on
   AMX-capable CPUs.
6. **Native fallback** — ``forward_native``.

Steps 3-6 are static per process, so their outcome is resolved once (lazily,
on first call) and cached in ``self._forward_method``; subclass ``__init__``
may pre-seed that attribute to pin an instance to a specific path (e.g. the
env-gated aiter paths in ``srt/layers``). Steps 1-2 stay per-call so tests and
tools can flip backends on live instances.

torch.compile integration: :meth:`enter_torch_compile` switches the op to its
compile-safe path (``forward_native`` by default; see
:meth:`_torch_compile_forward` for the TopK / FusedMoE special cases) so an
*outer* ``torch.compile`` never traces device-specific kernels, and
:meth:`leave_torch_compile` restores the original dispatch. Both are
idempotent because one module instance may be shared by many layers.

Like the rest of ``sglang.kernels``, importing this module (and instantiating
subclasses) never imports a kernel backend (``sgl_kernel`` /
``sglang.kernels.jit``), performs platform detection, or triggers JIT
compilation; backends are imported lazily inside the ``forward_<backend>``
methods and dispatch is resolved on first call.
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import (
    AbstractSet,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

import msgspec
import torch
from torch import nn

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
    PlatformInfo,
    capabilities_satisfied,
)

logger = logging.getLogger(__name__)

# backend (provenance) -> forward_<backend> method name. ``forward_torch_npu``
# (not ``forward_npu``) so the torch_npu *backend* method never collides with
# the NPU *platform* method.
BACKEND_METHODS: Dict[KernelBackend, str] = {
    KernelBackend.TORCH: "forward_native",
    KernelBackend.TORCH_COMPILE: "forward_torch_compile",
    KernelBackend.TRITON: "forward_triton",
    KernelBackend.JIT: "forward_jit",
    KernelBackend.AOT: "forward_aot",
    KernelBackend.CUTE_DSL: "forward_cute_dsl",
    KernelBackend.FLYDSL: "forward_flydsl",
    KernelBackend.KDA: "forward_kda",
    KernelBackend.FLASHINFER: "forward_flashinfer",
    KernelBackend.DEEPGEMM: "forward_deepgemm",
    KernelBackend.AITER: "forward_aiter",
    KernelBackend.TORCH_NPU: "forward_torch_npu",
}

_METHOD_BACKEND_LABELS: Dict[str, str] = {
    name: backend.value for backend, name in BACKEND_METHODS.items()
}

# best -> fallback. ``torch_compile`` is deliberately absent: auto-selection
# must never trigger a surprise compilation in a serving process; force it
# explicitly when wanted. Per-op priority overrides this (see BaseFusedOp).
DEFAULT_PRIORITY: Tuple[KernelBackend, ...] = (
    KernelBackend.KDA,
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

# In-tree platform key -> platform forward candidates, best first. A candidate
# counts only when the subclass actually overrides it; otherwise dispatch
# falls through to forward_native. Only HIP keeps the implicit CUDA-path
# fallback (ROCm kernels are hipified CUDA and sgl_kernel builds for both);
# MUSA deliberately does not chain into forward_cuda — srt module-level
# kernel imports are gated on is_cuda(), so a CUDA path reached implicitly on
# a MUSA box can NameError instead of degrading. MUSA ops that want the CUDA
# path opt in with an explicit forward_musa.
_PLATFORM_METHODS: Dict[str, Tuple[str, ...]] = {
    "cuda": ("forward_cuda",),
    "hip": ("forward_hip", "forward_cuda"),
    "musa": ("forward_musa",),
    "npu": ("forward_npu",),
    "xpu": ("forward_xpu",),
    "cpu": ("forward_cpu",),
}


@functools.lru_cache(maxsize=1)
def _platform() -> PlatformInfo:
    return PlatformInfo.detect()


@functools.lru_cache(maxsize=1)
def _platform_key() -> str:
    """In-tree platform dispatch key, or ``""`` for plain-native platforms.

    Checked in the same order as the former ``MultiPlatformOp``: CPU counts
    only when AMX is available (otherwise the pure-torch reference is faster
    than pretending there is an optimized CPU path).
    """
    from sglang.srt.utils import (
        cpu_has_amx_support,
        is_cpu,
        is_cuda,
        is_hip,
        is_musa,
        is_npu,
        is_xpu,
    )

    if is_cuda():
        return "cuda"
    if is_hip():
        return "hip"
    if is_cpu() and cpu_has_amx_support():
        return "cpu"
    if is_npu():
        return "npu"
    if is_xpu():
        return "xpu"
    if is_musa():
        return "musa"
    return ""


@functools.lru_cache(maxsize=1)
def _oot_dispatch_key() -> Optional[str]:
    """The active out-of-tree platform's dispatch key, or ``None`` in-tree."""
    from sglang.srt.platforms import current_platform

    if current_platform.is_out_of_tree():
        return current_platform.get_dispatch_key_name()
    return None


def clear_platform_caches() -> None:
    """Drop the cached platform detection (used by tests that mock platforms)."""
    _platform.cache_clear()
    _platform_key.cache_clear()
    _oot_dispatch_key.cache_clear()


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
    described = []
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            described.append(f"{value.dtype}[{', '.join(map(str, value.shape))}]")
    return tuple(described)


def _dispatch_label(method: Callable) -> str:
    """Trace label for a resolved dispatch target.

    Backend methods map to their :class:`KernelBackend` value (so
    ``forward_native`` traces as ``"torch"``, matching explicit-backend
    calls); platform forwards trace as their device key (``"cuda"``, ...).
    """
    name = getattr(method, "__name__", "")
    label = _METHOD_BACKEND_LABELS.get(name)
    if label is not None:
        return label
    if name.startswith("forward_"):
        return name[len("forward_") :]
    return name or "unknown"


def _record_trace(op: BaseFusedOp, label: str, args: tuple, kwargs: dict) -> None:
    _trace_records.append(
        FusedOpTraceRecord(
            op=op.op or type(op).__name__,
            backend=label,
            tensor_args=_describe_tensors(args, kwargs),
        )
    )


_warned_forced_fallbacks: set = set()


def _warn_forced_backend_unavailable(*, op: BaseFusedOp, backend: KernelBackend):
    key = (type(op), backend)
    if key not in _warned_forced_fallbacks:
        _warned_forced_fallbacks.add(key)
        logger.warning(
            "Forced fused-op backend %r is not implemented by %s; "
            "falling back to normal dispatch for this op.",
            backend.value,
            op._op_label(),
        )


# --- the per-operator contract ------------------------------------------------


class BaseFusedOp(nn.Module, ABC):
    """One logical operator with interchangeable implementations behind
    ``forward()``.

    Subclasses implement :meth:`forward_native` plus any optimized
    ``forward_<backend>`` methods and/or platform-specific
    ``forward_<device>`` methods. All implementations of one op must share the
    same signature and semantics — each override adapts its underlying
    kernel's calling convention so call sites never care which one ran.

    This is a standard :class:`torch.nn.Module`: it participates in module
    traversal / state dicts, and calls go through ``nn.Module.__call__`` so
    forward hooks keep working.

    Class attributes
    ----------------
    op:
        Operator id, ``"<group>.<name>"`` (e.g. ``"layernorm.rmsnorm"``).
        Required for :func:`register_fused_op`; layer-style subclasses that
        are not registered in the kernel registry may leave it empty.
    priority:
        Kernel-backend preference for auto-selection, best first. Defaults to
        :data:`DEFAULT_PRIORITY`. ``KernelBackend.TORCH`` entries are ignored:
        the native reference is always the final fallback, after
        platform-specific forwards.
    capabilities:
        Per-backend set of :class:`CapabilityRequirement` (OR semantics;
        an empty set value = runs on any device), consulted by
        :meth:`backend_eligible` and exported into the registry specs. A
        kernel backend joins **auto-selection** only when it is declared here
        (explicit ``backend=`` / forced selection can still target any
        overridden method) — device support is metadata, not guesswork. Use
        the ``CapabilityRequirement.CUDA`` / ``.HIP`` / ``.NPU`` shortcuts,
        e.g. ``{KernelBackend.AOT: {CapabilityRequirement.CUDA,
        CapabilityRequirement.HIP}}``.
    format_signature:
        Data-contract description shared by all backends of this op.
    descriptions:
        Optional per-backend one-liners for the registry inventory.
    """

    op: ClassVar[str] = ""
    priority: ClassVar[Tuple[KernelBackend, ...]] = DEFAULT_PRIORITY
    capabilities: ClassVar[
        Mapping[KernelBackend, AbstractSet[CapabilityRequirement]]
    ] = {}
    format_signature: ClassVar[FormatSignature] = FormatSignature()
    descriptions: ClassVar[Mapping[KernelBackend, str]] = {}

    # OOT forward registry shared by all fused ops: platform dispatch key ->
    # {op class -> forward fn}. Populated by out-of-tree platform plugins.
    _oot_forward_registry: ClassVar[Dict[str, Dict[type, Callable]]] = {}

    @classmethod
    def register_oot_forward(cls, op_cls: type, fn: Callable, platform_key: str):
        """Register an OOT forward implementation for a specific op class and
        platform. ``fn`` is bound to the instance at dispatch time (it receives
        the op as ``self``) and takes precedence over ``forward_<platform_key>``
        methods on that exact class."""
        cls._oot_forward_registry.setdefault(platform_key, {})[op_cls] = fn

    def __init__(self) -> None:
        super().__init__()
        # Statically resolved dispatch target (priority steps 3-6). ``None``
        # means "not resolved yet": resolution is deferred to the first call
        # so module-level op instances never trigger platform detection at
        # import time. Subclass __init__ may overwrite it to pin a path.
        self._forward_method: Optional[Callable] = None
        # torch.compile mode bookkeeping (see enter/leave_torch_compile).
        self._original_forward_method: Optional[Callable] = None
        self.is_torch_compile = False
        self._compiled_native = None

    def _defined_method(self, method_name: str) -> Optional[Callable]:
        """The bound method if any class below ``BaseFusedOp`` defines it."""
        for klass in type(self).__mro__:
            if klass is BaseFusedOp:
                return None
            if method_name in klass.__dict__:
                return getattr(self, method_name)
        return None

    # --- kernel backends: native is required; the rest are opt-in overrides ---

    @abstractmethod
    def forward_native(self, *args, **kwargs):
        """Pure-``torch`` reference implementation (correctness ground truth)."""

    def forward_torch_compile(self, *args, **kwargs):
        if self._compiled_native is None:
            self._compiled_native = torch.compile(self.forward_native)
        return self._compiled_native(*args, **kwargs)

    def forward_triton(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no triton backend")

    def forward_jit(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no jit backend")

    def forward_aot(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no aot backend")

    def forward_cute_dsl(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no cute_dsl backend")

    def forward_kda(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no KDA backend")

    def forward_flashinfer(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no flashinfer backend")

    def forward_deepgemm(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no deepgemm backend")

    def forward_aiter(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no aiter backend")

    def forward_torch_npu(self, *args, **kwargs):
        raise NotImplementedError(f"{self._op_label()}: no torch_npu backend")

    def _op_label(self) -> str:
        return self.op or type(self).__name__

    # --- platform forwards (forward_cuda / forward_hip / forward_npu /
    # forward_xpu / forward_musa / forward_cpu) are *not* defined here: a
    # platform path exists exactly when a subclass defines it, and dispatch
    # falls back to forward_native otherwise. ---

    # --- selection ---

    def available_backends(self) -> List[KernelBackend]:
        """Kernel backends this op implements (structural, platform-agnostic)."""
        return [
            backend
            for backend in KernelBackend
            if backend in _ALWAYS_AVAILABLE
            or self._defined_method(BACKEND_METHODS[backend]) is not None
        ]

    def backend_eligible(self, backend: KernelBackend, *args, **kwargs) -> bool:
        """Whether ``backend`` may run *this* call.

        The base implementation checks the backend's
        :class:`CapabilityRequirement` set (OR semantics) against the detected
        platform. Subclasses may extend it with per-call shape/dtype gates so
        auto-selection bounces to the next backend instead of raising;
        overriding this method switches backend auto-selection from a cached
        static choice to per-call resolution.
        """
        return capabilities_satisfied(
            self.capabilities.get(backend, frozenset()), _platform()
        )

    def _auto_backend_candidates(self) -> Tuple[KernelBackend, ...]:
        """Kernel backends participating in auto-selection, best first.

        A backend qualifies when its method is overridden *and* the op
        declares it in :attr:`capabilities` (``TORCH_COMPILE`` needs no
        declaration — it always exists and is device-agnostic — but must be
        listed in :attr:`priority` explicitly). ``TORCH`` entries are skipped:
        the native reference is the final fallback after platform forwards.
        """
        candidates = []
        for backend in self.priority:
            if backend is KernelBackend.TORCH:
                continue
            if backend is KernelBackend.TORCH_COMPILE:
                candidates.append(backend)
                continue
            if (
                backend in self.capabilities
                and self._defined_method(BACKEND_METHODS[backend]) is not None
            ):
                candidates.append(backend)
        return tuple(candidates)

    def auto_selected_backend(self) -> Optional[KernelBackend]:
        """The kernel backend auto-selection picks on this platform, or
        ``None`` when dispatch falls through to a platform forward / native
        (introspection only — per-call :meth:`backend_eligible` gates are not
        consulted)."""
        platform = _platform()
        for backend in self._auto_backend_candidates():
            if capabilities_satisfied(
                self.capabilities.get(backend, frozenset()), platform
            ):
                return backend
        return None

    def _platform_method(self, platform_key: str) -> Optional[Callable]:
        """The platform forward for ``platform_key``, or ``None``.

        In-tree keys use :data:`_PLATFORM_METHODS` (with the HIP → CUDA
        fallback chain); OOT keys look up ``forward_<key>`` directly.
        """
        if not platform_key:
            return None
        names = _PLATFORM_METHODS.get(platform_key, (f"forward_{platform_key}",))
        for name in names:
            method = self._defined_method(name)
            if method is not None:
                return method
        return None

    def _resolve_forward_method(self) -> Callable:
        """Resolve dispatch steps 3-6 (see module docstring) to one callable."""
        # 3) OOT platform override.
        oot_key = _oot_dispatch_key()
        if oot_key is not None:
            registered = self._oot_forward_registry.get(oot_key, {}).get(type(self))
            if registered is not None:
                return registered.__get__(self)
            method = self._platform_method(oot_key)
            if method is not None:
                return method
            return self.forward_native

        # 4) Optimized kernel backends by priority.
        candidates = self._auto_backend_candidates()
        if candidates:
            if self._defined_method("backend_eligible") is not None:
                # Per-call shape/dtype gates: keep selection dynamic.
                self._dynamic_backend_candidates = candidates
                return self._forward_backend_dynamic
            platform = _platform()
            for backend in candidates:
                if capabilities_satisfied(
                    self.capabilities.get(backend, frozenset()), platform
                ):
                    return getattr(self, BACKEND_METHODS[backend])

        # 5) Platform-specific forward; 6) native fallback.
        method = self._platform_method(_platform_key())
        return method if method is not None else self.forward_native

    def _forward_backend_dynamic(self, *args, **kwargs):
        """Per-call backend selection for ops with input-dependent gates."""
        for backend in self._dynamic_backend_candidates:
            if self.backend_eligible(backend, *args, **kwargs):
                return getattr(self, BACKEND_METHODS[backend])(*args, **kwargs)
        method = self._platform_method(_platform_key())
        return (method or self.forward_native)(*args, **kwargs)

    def dispatch_forward(self) -> Callable:
        """The static dispatch target for this op on the current platform."""
        return self._resolve_forward_method()

    # --- torch.compile mode ---

    def _torch_compile_forward(self, num_tokens: int) -> Optional[Callable]:
        """The forward to use while the outer model is under ``torch.compile``.

        Returning ``None`` keeps the current dispatch (used by the TopK /
        FusedMoE overrides, whose optimized paths stay active for
        ``num_tokens > 1`` where compiling the native path is a regression).
        The default routes to the compile-safe pure-torch reference — the op
        itself is *not* wrapped in a nested per-op ``torch.compile``.
        """
        return self.forward_native

    def enter_torch_compile(self, num_tokens: int) -> None:
        """Switch to the compile-safe forward. Idempotent.

        Some ops (e.g. RotaryEmbedding) are reused among layers, so this may
        be called many times; only the first call snapshots the original
        dispatch, otherwise :meth:`leave_torch_compile` could not restore it.
        """
        if self.is_torch_compile:
            return
        # Warm the lazy globals now (still eager) so a compiled trace of
        # forward() only reads plain module globals / instance attributes and
        # never graph-breaks on an import inside get_fused_op_backend() or
        # _resolve_forward_method().
        get_fused_op_backend()
        self._original_forward_method = self._forward_method
        compile_forward = self._torch_compile_forward(num_tokens=num_tokens)
        if compile_forward is not None:
            self._forward_method = compile_forward
        elif self._forward_method is None:
            self._forward_method = self._resolve_forward_method()
        self.is_torch_compile = True

    def leave_torch_compile(self) -> None:
        """Restore the pre-compile forward. Idempotent."""
        if not self.is_torch_compile:
            return
        self._forward_method = self._original_forward_method
        self._original_forward_method = None
        self.is_torch_compile = False

    # --- dispatch ---

    # Do not override forward(): implement forward_native / forward_<backend> /
    # forward_<platform> instead, so backend forcing, OOT overrides and
    # torch.compile mode switching keep working.
    @debug_kernel_api
    def forward(self, *args, backend: Optional[KernelBackend] = None, **kwargs):
        """Run the op on ``backend``, or on the best eligible path when omitted."""
        if backend is not None:
            # Explicit per-call selection is strict: an unimplemented backend
            # raises so the caller's intent never degrades silently.
            result = getattr(self, BACKEND_METHODS[backend])(*args, **kwargs)
            if _trace_enabled:
                _record_trace(self, backend.value, args, kwargs)
            return result
        forced = _forced_backend
        if forced is _UNRESOLVED:
            forced = get_fused_op_backend()
        if forced is not None:
            # The global debug switch is best-effort: ops that do not
            # implement the forced backend (e.g. forcing "torch" on an op
            # whose only paths are device-specific) fall back to normal
            # dispatch with a one-time warning instead of taking the whole
            # model down.
            try:
                result = getattr(self, BACKEND_METHODS[forced])(*args, **kwargs)
            except NotImplementedError:
                _warn_forced_backend_unavailable(op=self, backend=forced)
            else:
                if _trace_enabled:
                    _record_trace(self, forced.value, args, kwargs)
                return result
        method = self._forward_method
        if method is None:
            method = self._forward_method = self._resolve_forward_method()
        result = method(*args, **kwargs)
        if _trace_enabled:
            _record_trace(self, _dispatch_label(method), args, kwargs)
        return result


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
