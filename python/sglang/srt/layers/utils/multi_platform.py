import logging
import os
import types
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

from torch import nn

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_musa = is_musa()

# ============================================================
# Dispatch logging (similar to flag_gems' record mode)
# SGLANG_OOT_DISPATCH_LOG=/tmp/dispatch.log → write dispatch decisions to file
# ============================================================
_dispatch_log_path = os.environ.get("SGLANG_OOT_DISPATCH_LOG", "").strip() or None
_dispatched_ops: Dict[str, str] = {}  # {op_name: "kind:impl_id"} for inspection


def _log_dispatch(op_name: str, source: str, detail: str = ""):
    """Record a dispatch decision. Only writes to file once per unique op_name."""
    entry = f"{source}" + (f"({detail})" if detail else "")
    already_logged = op_name in _dispatched_ops
    _dispatched_ops[op_name] = entry
    msg = f"[OOT-DISPATCH] {op_name} → {entry}"
    logger.info(msg)
    if _dispatch_log_path and not already_logged:
        try:
            with open(_dispatch_log_path, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass


# ============================================================
# Multi-backend dispatch types
# ============================================================

class OOTBackendKind(str, Enum):
    """Backend implementation kind, matching vllm-plugin-FL's design.

    - FLAGOS: FlagOS default implementation (FlagGems Triton kernels), highest priority
    - VENDOR: Chip vendor's own implementation, medium priority
    - REFERENCE: PyTorch native fallback, lowest priority
    """
    FLAGOS = "flagos"
    VENDOR = "vendor"
    REFERENCE = "reference"


# Default priority values (higher = preferred)
_DEFAULT_PRIORITIES = {
    OOTBackendKind.FLAGOS: 150,
    OOTBackendKind.VENDOR: 100,
    OOTBackendKind.REFERENCE: 50,
}

# Default selection order: flagos > vendor > reference
_DEFAULT_ORDER = [OOTBackendKind.FLAGOS, OOTBackendKind.VENDOR, OOTBackendKind.REFERENCE]


@dataclass(frozen=True)
class OOTOpImpl:
    """Operator implementation descriptor.

    Attributes:
        op_name: Operator class name, e.g. "SiluAndMul"
        kind: Backend kind (FLAGOS / VENDOR / REFERENCE)
        fn: The implementation function. Signature: fn(self, *args, **kwargs)
        priority: Selection priority (higher = preferred). Auto-set from kind if 0.
        vendor: Vendor name (required if kind is VENDOR)
        impl_id: Unique identifier, auto-generated if not provided
    """
    op_name: str
    kind: OOTBackendKind
    fn: Callable
    priority: int = 0
    vendor: Optional[str] = None
    impl_id: str = ""

    def __post_init__(self):
        # Auto-set priority from kind if not explicitly provided
        if self.priority == 0:
            object.__setattr__(self, "priority", _DEFAULT_PRIORITIES.get(self.kind, 50))
        # Auto-generate impl_id
        if not self.impl_id:
            if self.vendor:
                object.__setattr__(self, "impl_id", f"{self.kind.value}.{self.vendor}")
            else:
                object.__setattr__(self, "impl_id", self.kind.value)


def _get_selection_order() -> List[OOTBackendKind]:
    """Get selection order from SGLANG_OOT_PREFER env var.

    Values: "flagos" (default), "vendor", "reference"
    Controls which backend is tried first. Others follow as fallback.
    """
    prefer = os.environ.get("SGLANG_OOT_PREFER", "flagos").strip().lower()
    if prefer == "vendor":
        return [OOTBackendKind.VENDOR, OOTBackendKind.FLAGOS, OOTBackendKind.REFERENCE]
    elif prefer == "reference":
        return [OOTBackendKind.REFERENCE, OOTBackendKind.FLAGOS, OOTBackendKind.VENDOR]
    else:
        return list(_DEFAULT_ORDER)


_per_op_prefer: Dict[str, OOTBackendKind] = {}

def _parse_per_op_prefer():
    """Parse SGLANG_OOT_OP_PREFER env var for per-op backend override.

    Format: "SiluAndMul:flagos,RMSNorm:reference,SiluAndMul:vendor:ascend"
    The "vendor:{name}" syntax selects a specific vendor backend.
    """
    raw = os.environ.get("SGLANG_OOT_OP_PREFER", "").strip()
    if not raw:
        return
    _kind_map = {k.value: k for k in OOTBackendKind}
    for item in raw.split(","):
        item = item.strip()
        if ":" not in item:
            continue
        parts = item.split(":")
        op_name = parts[0].strip()
        kind_str = parts[1].strip().lower()
        if kind_str == "vendor" and len(parts) >= 3:
            # vendor:{name} syntax — store as (kind, vendor_name)
            vendor_name = parts[2].strip().lower()
            _per_op_prefer[op_name] = (OOTBackendKind.VENDOR, vendor_name)
        elif kind_str in _kind_map:
            _per_op_prefer[op_name] = _kind_map[kind_str]
        else:
            logger.warning(f"Unknown backend '{kind_str}' for op '{op_name}' in SGLANG_OOT_OP_PREFER")

_parse_per_op_prefer()


# ============================================================
# Plugin loading
# ============================================================

def _load_oot_plugins():
    """Load OOT plugins from SGLANG_OOT_PLUGINS env var.

    Plugins are comma-separated module names. Each module should call
    MultiPlatformOp.register_oot() or register_oot_impl() when imported.
    """
    import importlib
    plugins = os.environ.get("SGLANG_OOT_PLUGINS", "").strip()
    if not plugins:
        return
    for mod_name in plugins.split(","):
        mod_name = mod_name.strip()
        if mod_name:
            try:
                importlib.import_module(mod_name)
                logger.info(f"Loaded OOT plugin: {mod_name}")
            except Exception as e:
                logger.warning(f"Failed to load OOT plugin {mod_name}: {e}")


# ============================================================
# MultiPlatformOp
# ============================================================

class MultiPlatformOp(nn.Module):
    # Legacy single-impl registry: {op_name: forward_fn}
    # Kept for backward compatibility with existing plugins.
    _oot_forward_registry: Dict[str, Callable] = {}

    # Multi-backend registry: {op_name: [OOTOpImpl, ...]}
    _oot_impl_registry: Dict[str, List[OOTOpImpl]] = {}

    # ---- Legacy API (backward compatible) ----

    @classmethod
    def register_oot(cls, op_name: str, forward_fn: Callable):
        """Register an OOT forward implementation for an operator.

        This is the simple API — registers as FLAGOS kind by default.
        For multi-backend support, use register_oot_impl() instead.

        Args:
            op_name: Operator class name, e.g. "SiluAndMul", "RMSNorm"
            forward_fn: Replacement forward function. Signature: fn(self, *args, **kwargs)
        """
        cls._oot_forward_registry[op_name] = forward_fn
        logger.info(f"Registered OOT forward for {op_name}")

    @classmethod
    def unregister_oot(cls, op_name: str):
        """Remove an OOT forward registration."""
        if cls._oot_forward_registry.pop(op_name, None) is not None:
            logger.info(f"Unregistered OOT forward for {op_name}")

    @classmethod
    def get_oot_registry(cls) -> Dict[str, Callable]:
        """Return a copy of the current OOT registry (legacy single-impl)."""
        return dict(cls._oot_forward_registry)

    # ---- Multi-backend API ----

    @classmethod
    def register_oot_impl(cls, impl: OOTOpImpl):
        """Register an OOT operator implementation with backend kind and priority.

        Multiple implementations can be registered for the same op_name.
        dispatch_forward() selects the best one based on policy and fallback.

        Args:
            impl: OOTOpImpl descriptor
        """
        impls = cls._oot_impl_registry.setdefault(impl.op_name, [])
        # Prevent duplicate impl_id for same op
        for existing in impls:
            if existing.impl_id == impl.impl_id:
                logger.warning(
                    f"Replacing OOT impl {impl.impl_id} for {impl.op_name}"
                )
                impls.remove(existing)
                break
        impls.append(impl)
        logger.info(
            f"Registered OOT impl: op={impl.op_name}, kind={impl.kind.value}, "
            f"priority={impl.priority}, impl_id={impl.impl_id}"
        )

    @classmethod
    def get_dispatched_ops(cls) -> Dict[str, str]:
        """Return dispatch log: {op_name: 'kind(impl_id)'} for all dispatched ops."""
        return dict(_dispatched_ops)

    @classmethod
    def unregister_oot_impl(cls, op_name: str, impl_id: str):
        """Remove a specific OOT implementation."""
        impls = cls._oot_impl_registry.get(op_name, [])
        cls._oot_impl_registry[op_name] = [
            i for i in impls if i.impl_id != impl_id
        ]

    @classmethod
    def get_oot_impl_registry(cls) -> Dict[str, List[OOTOpImpl]]:
        """Return a copy of the multi-backend OOT registry."""
        return {k: list(v) for k, v in cls._oot_impl_registry.items()}

    @classmethod
    def _is_impl_available(cls, impl: OOTOpImpl) -> bool:
        """Check if an impl is available via its _is_available attribute."""
        avail_fn = getattr(impl.fn, "_is_available", None)
        if callable(avail_fn):
            return avail_fn()
        return True  # no check → assume available

    @classmethod
    def _resolve_oot_impl(cls, op_name: str) -> Optional[Callable]:
        """Resolve the best OOT implementation for an operator.

        Selection logic:
        1. Get selection order from SGLANG_OOT_PREFER (default: flagos > vendor > reference)
        2. For each kind in order, find matching impls sorted by priority (desc)
        3. Skip impls whose _is_available() returns False
        4. Return the first available one

        Returns:
            The best forward_fn, or None if no multi-backend impl registered.
        """
        impls = cls._oot_impl_registry.get(op_name)
        if not impls:
            return None

        # Per-op override
        if op_name in _per_op_prefer:
            pref = _per_op_prefer[op_name]
            if isinstance(pref, tuple):
                # (OOTBackendKind.VENDOR, vendor_name) — match specific vendor
                preferred_kind, vendor_name = pref
                candidates = [
                    i for i in impls
                    if i.kind == preferred_kind
                    and i.vendor and i.vendor.lower() == vendor_name
                ]
            else:
                preferred_kind = pref
                candidates = [i for i in impls if i.kind == preferred_kind]
            if candidates:
                candidates.sort(key=lambda x: (x.priority, x.impl_id), reverse=True)
                for c in candidates:
                    if cls._is_impl_available(c):
                        _log_dispatch(op_name, c.kind.value, c.impl_id)
                        return c.fn
            # preferred kind not registered or not available → fall through to global order

        order = _get_selection_order()

        for kind in order:
            candidates = [i for i in impls if i.kind == kind]
            if not candidates:
                continue
            # Sort by priority descending, then impl_id for stability
            candidates.sort(key=lambda x: (x.priority, x.impl_id), reverse=True)
            for c in candidates:
                if cls._is_impl_available(c):
                    _log_dispatch(op_name, c.kind.value, c.impl_id)
                    return c.fn

        # Fallback: return highest priority available regardless of kind
        all_sorted = sorted(impls, key=lambda x: (x.priority, x.impl_id), reverse=True)
        for c in all_sorted:
            if cls._is_impl_available(c):
                _log_dispatch(op_name, "fallback", c.impl_id)
                return c.fn
        return None

    # ---- Core ----

    def __init__(self):
        super().__init__()
        self._forward_method: Callable = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    @debug_kernel_api
    def forward_native(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not have a native forward implementation."
        )

    @debug_kernel_api
    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_npu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_musa(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        op_name = self.__class__.__name__

        # Priority 1: Multi-backend registry (with kind/priority/fallback)
        resolved_fn = self._resolve_oot_impl(op_name)
        if resolved_fn is not None:
            return types.MethodType(resolved_fn, self)

        # Priority 2: Legacy single-impl registry (backward compatible)
        if op_name in self._oot_forward_registry:
            oot_fn = self._oot_forward_registry[op_name]
            _log_dispatch(op_name, "legacy_oot")
            return types.MethodType(oot_fn, self)

        # Priority 3: Built-in platform dispatch
        if _is_cuda:
            _log_dispatch(op_name, "builtin", "cuda")
            return self.forward_cuda
        elif _is_hip:
            _log_dispatch(op_name, "builtin", "hip")
            return self.forward_hip
        elif _is_cpu and _is_cpu_amx_available:
            _log_dispatch(op_name, "builtin", "cpu_amx")
            return self.forward_cpu
        elif _is_npu:
            _log_dispatch(op_name, "builtin", "npu")
            return self.forward_npu
        elif _is_xpu:
            _log_dispatch(op_name, "builtin", "xpu")
            return self.forward_xpu
        elif _is_musa:
            _log_dispatch(op_name, "builtin", "musa")
            return self.forward_musa
        else:
            _log_dispatch(op_name, "builtin", "native")
            return self.forward_native


# Auto-load OOT plugins when this module is imported (works in all processes)
_load_oot_plugins()
