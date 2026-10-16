# ruff: noqa: RUF012, SIM101
"""Base class for fused operations with multi-backend dispatch."""

from __future__ import annotations

import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tracing: dump every BaseFusedOp call to JSON when SGLANG_DUMP_FUSED_OPS is set
# ---------------------------------------------------------------------------
_DUMP_DIR: str | None = os.environ.get("SGLANG_DUMP_FUSED_OPS")
_trace_call_counter: int = 0
_triton_native_fallback_logged: set[str] = set()

# ---------------------------------------------------------------------------
# Global fused-op mode (set once at server startup via --fused-op-mode)
# ---------------------------------------------------------------------------
_global_fused_op_mode: str = "default"


def set_fused_op_mode(mode: str) -> None:
    global _global_fused_op_mode
    _global_fused_op_mode = mode


def get_fused_op_mode() -> str:
    return _global_fused_op_mode


class FusedOpBackend(Enum):
    NATIVE = "native"
    TORCH_COMPILE = "torch_compile"
    TRITON = "triton"
    CUTEDSL = "cutedsl"


# Priority order for "default" mode: best → fallback
_DEFAULT_PRIORITY = [
    FusedOpBackend.CUTEDSL,
    FusedOpBackend.TRITON,
    FusedOpBackend.NATIVE,
]

# Priority order for "torch_compile_fallback" mode: optimized kernels first,
# then torch.compile'd native instead of plain native.
_TORCH_COMPILE_FALLBACK_PRIORITY = [
    FusedOpBackend.CUTEDSL,
    FusedOpBackend.TRITON,
    FusedOpBackend.TORCH_COMPILE,
]


def _is_hip() -> bool:
    return torch.version.hip is not None


class BaseFusedOp(ABC):
    """
    Abstract base class for fused operations with multi-backend dispatch.

    Backends are Triton, CuTeDSL, and a torch-native reference (optionally
    torch.compile'd). A subclass implements ``forward_native`` plus any
    optimized ``forward_triton`` / ``forward_cutedsl`` it provides; the base
    resolves the best available backend per call.

    Code style guidelines
    1. During inference, many weights are constant. Cache computation results
       on the constant weight tensors as much as possible. Compute them in
       ``__init__`` and reuse the cached results; do not recompute them in the
       model forward code.
    """

    # Attributes set by BaseFusedOp that should be excluded from trace config
    _BASE_SKIP_ATTRS = frozenset(
        {
            "_available_backends",
            "_default_backend",
            "_torch_compile_fallback_backend",
            "_compiled_forward_native",
        }
    )

    def __init__(self):
        # Cache available backends and the default-mode resolution so
        # forward() hot-path avoids repeated introspection.
        available = set(self.available_backends())
        self._available_backends = available
        # Subclasses may override _DEFAULT_PRIORITY to change auto-selection.
        priority = getattr(type(self), "_DEFAULT_PRIORITY", _DEFAULT_PRIORITY)
        # Priority-ordered list of available backends. Default-mode resolution
        # walks this and bounces to the next backend when one is not
        # runtime-eligible for a given call (see _backend_runtime_eligible).
        self._ordered_available_backends = [b for b in priority if b in available]
        self._default_backend = (
            self._ordered_available_backends[0]
            if self._ordered_available_backends
            else FusedOpBackend.NATIVE
        )
        # Pre-compute backend for torch_compile_fallback mode.
        for backend in _TORCH_COMPILE_FALLBACK_PRIORITY:
            if backend in available:
                self._torch_compile_fallback_backend = backend
                break
        else:
            self._torch_compile_fallback_backend = FusedOpBackend.TORCH_COMPILE

    @abstractmethod
    def forward_native(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement forward_native"
        )

    def forward_torch_compile(self, *args, **kwargs):
        if not hasattr(self, "_compiled_forward_native"):
            self._compiled_forward_native = torch.compile(self.forward_native)
        return self._compiled_forward_native(*args, **kwargs)

    def forward_triton(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement forward_triton"
        )

    def forward_cutedsl(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement forward_cutedsl"
        )

    # Backends whose base implementation raises NotImplementedError
    _OPTIONAL_BACKENDS = {
        FusedOpBackend.TRITON,
        FusedOpBackend.CUTEDSL,
    }

    # Keep platform-specific backends on their platform. Triton is portable.
    _HIP_DISABLED_OPTIONAL_BACKENDS = {
        FusedOpBackend.CUTEDSL,
    }
    _CUDA_DISABLED_OPTIONAL_BACKENDS = set()

    def _should_skip_backend(self, backend: FusedOpBackend) -> bool:
        if _is_hip():
            return backend in self._HIP_DISABLED_OPTIONAL_BACKENDS
        return backend in self._CUDA_DISABLED_OPTIONAL_BACKENDS

    def available_backends(self) -> list[FusedOpBackend]:
        """Return list of backends that are implemented."""
        backends = []
        for b in FusedOpBackend:
            if b in self._OPTIONAL_BACKENDS:
                if self._should_skip_backend(b):
                    continue
                # Only available if overridden by the subclass
                if f"forward_{b.value}" in type(self).__dict__:
                    backends.append(b)
            else:
                backends.append(b)
        return backends

    def _backend_runtime_eligible(self, backend, *args, **kwargs) -> bool:
        """Per-call eligibility hook for default-mode backend selection.

        Returns True by default: a statically-available backend is always
        eligible. Subclasses override to gate a backend on runtime arguments
        (e.g. tensor shapes/dtypes); when it returns False, default-mode
        resolution bounces back to the next priority backend.
        """
        return True

    def _resolve_backend(self, *args, **kwargs) -> FusedOpBackend:
        """Pick the backend based on the global fused-op mode.

        In "default" mode, walk the priority-ordered available backends and
        return the first that is runtime-eligible for this call, bouncing back
        to the next backend otherwise.
        """
        mode = get_fused_op_mode()
        if mode == "native":
            return FusedOpBackend.NATIVE
        if mode == "torch_compile":
            return FusedOpBackend.TORCH_COMPILE
        if mode == "torch_compile_fallback":
            return self._torch_compile_fallback_backend
        # mode == "default": first runtime-eligible backend in priority order.
        for backend in self._ordered_available_backends:
            if self._backend_runtime_eligible(backend, *args, **kwargs):
                return backend
        return FusedOpBackend.NATIVE

    _DISPATCH = {
        FusedOpBackend.NATIVE: "forward_native",
        FusedOpBackend.TORCH_COMPILE: "forward_torch_compile",
        FusedOpBackend.TRITON: "forward_triton",
        FusedOpBackend.CUTEDSL: "forward_cutedsl",
    }

    def forward(self, *args, backend: FusedOpBackend | None = None, **kwargs):
        if backend is None:
            backend = self._resolve_backend(*args, **kwargs)
        result = getattr(self, self._DISPATCH[backend])(*args, **kwargs)
        if _DUMP_DIR is not None and not torch.compiler.is_compiling():
            self._trace_call(args, kwargs, result)
        return result

    def _fallback_triton_to_native(self, *args, **kwargs):
        op_name = type(self).__name__
        if (
            op_name not in _triton_native_fallback_logged
            and not torch.compiler.is_compiling()
        ):
            _triton_native_fallback_logged.add(op_name)
            logger.warning(
                "Selected Triton backend for %s but falling back to native",
                op_name,
            )
        return self.forward_native(*args, **kwargs)

    # ------------------------------------------------------------------
    # Tracing helpers
    # ------------------------------------------------------------------

    def _extract_trace_config(self) -> dict:
        """Auto-extract config by introspecting instance attributes.

        Walks ``self.__dict__``, skips BaseFusedOp internals, and serializes
        values by type.  For ``nn.Module`` values, recursively extracts
        public scalar attributes with dot-flattened keys.
        """
        config: dict = {}
        for key, value in self.__dict__.items():
            if key in self._BASE_SKIP_ATTRS:
                continue
            if isinstance(value, bool) or isinstance(value, (int, float, str)):
                config[key] = value
            elif value is None:
                config[key] = None
            elif isinstance(value, torch.dtype):
                config[key] = str(value)
            elif isinstance(value, Enum):
                config[key] = value.value
            elif isinstance(value, torch.nn.Module):
                sub = _extract_module_config(value)
                for sub_key, sub_val in sub.items():
                    config[f"{key}.{sub_key}"] = sub_val
            elif isinstance(value, (torch.Tensor, torch.nn.Parameter)):
                config[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
        return config

    def _trace_call(self, args: tuple, kwargs: dict, result) -> None:
        """Dump a single forward call to a JSON file."""
        cls_name = type(self).__name__

        # Get argument names from forward_native signature
        sig = inspect.signature(type(self).forward_native)
        param_names = [p.name for p in sig.parameters.values() if p.name != "self"]

        # Match args/kwargs to parameter names
        arg_infos: list[dict] = []
        filename_parts: list[str] = [cls_name]
        for i, value in enumerate(args):
            name = param_names[i] if i < len(param_names) else f"arg{i}"
            arg_infos.append(_describe_arg(name, value))
            if isinstance(value, torch.Tensor):
                shape_str = "x".join(str(d) for d in value.shape)
                dtype_str = str(value.dtype).replace("torch.", "")
                filename_parts.append(f"{name}_{shape_str}_{dtype_str}")
        for name, value in kwargs.items():
            arg_infos.append(_describe_arg(name, value))
            if isinstance(value, torch.Tensor):
                shape_str = "x".join(str(d) for d in value.shape)
                dtype_str = str(value.dtype).replace("torch.", "")
                filename_parts.append(f"{name}_{shape_str}_{dtype_str}")

        # Output info
        output_infos = _describe_result(result)

        # Build JSON payload
        global _trace_call_counter
        counter = _trace_call_counter
        _trace_call_counter += 1
        payload = {
            "class": cls_name,
            "call_index": counter,
            "inputs": arg_infos,
            "outputs": output_infos,
            "config": self._extract_trace_config(),
        }

        # Build filename: ClassName__counter__arg1_shape_dtype__arg2_shape_dtype.json
        filename_parts.insert(1, f"{counter:08d}")
        filename = "__".join(filename_parts) + ".json"
        # Truncate long filenames (keep class + counter prefix for uniqueness)
        if len(filename) > 200:
            prefix = f"{cls_name}__{counter:08d}__"
            filename = prefix + filename[len(prefix) : 200 - len(".json")] + ".json"

        # Write
        dump_dir = Path(_DUMP_DIR) / cls_name
        dump_dir.mkdir(parents=True, exist_ok=True)
        with open(dump_dir / filename, "w") as f:
            json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Tracing helper functions
# ---------------------------------------------------------------------------


def _extract_module_config(module: torch.nn.Module) -> dict:
    """Extract public scalar attrs and Parameter shapes from an nn.Module."""
    config: dict = {"__class__": type(module).__name__}
    for key, value in vars(module).items():
        if key.startswith("_"):
            continue
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            config[key] = value
        elif isinstance(value, torch.dtype):
            config[key] = str(value)
        elif isinstance(value, (torch.Tensor, torch.nn.Parameter)):
            config[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return config


def _describe_arg(name: str, value) -> dict:
    """Build a description dict for a single forward argument."""
    if isinstance(value, torch.Tensor):
        return {
            "name": name,
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, torch.dtype):
        return {"name": name, "type": "dtype", "value": str(value)}
    if value is None:
        return {"name": name, "type": "NoneType", "value": None}
    return {"name": name, "type": type(value).__name__, "value": value}


def _describe_result(result) -> list[dict]:
    """Build description dicts for forward return values."""
    if result is None:
        return [{"name": "output", "type": "NoneType", "value": None}]
    if isinstance(result, torch.Tensor):
        return [
            {
                "name": "output",
                "type": "tensor",
                "shape": list(result.shape),
                "dtype": str(result.dtype),
            }
        ]
    if isinstance(result, tuple):
        descs = []
        for i, v in enumerate(result):
            name = f"output_{i}"
            if isinstance(v, torch.Tensor):
                descs.append(
                    {
                        "name": name,
                        "type": "tensor",
                        "shape": list(v.shape),
                        "dtype": str(v.dtype),
                    }
                )
            elif v is None:
                descs.append({"name": name, "type": "NoneType", "value": None})
            else:
                descs.append({"name": name, "type": type(v).__name__, "value": v})
        return descs
    return [{"name": "output", "type": type(result).__name__}]
