"""
Automatic tensor shape metadata for Triton / FlashInfer / aiter kernels
in PyTorch profiler traces.

When enabled, targeted kernel entry-point functions are registered as
torch custom ops via torch.library so they appear as ``cpu_op`` events
with ``Input Dims`` and ``Input type`` in profiler traces.

Usage:
    from sglang.srt.utils.kernel_shape_profiler import enable, disable
    enable()   # before profiling starts
    disable()  # after profiling stops

Design:
    We maintain an explicit registry of kernel entry points.  For each one
    we create a ``torch.library`` custom-op wrapper and then replace **every
    module-level reference** to the original function across all of
    ``sys.modules``.  This handles the common ``from X import Y`` pattern
    where patching only the definition module would miss callers that
    already captured a local binding.

    Functions whose references were captured as *instance attributes*
    before ``enable()`` (e.g. ``self.fn = dispatch()``) cannot be
    intercepted directly.  For those cases the registry should target the
    **inner kernel** that the wrapper calls via module-global lookup at
    call time (e.g. ``gemm_a8w8_blockscale`` inside
    ``aiter_w8a8_block_fp8_linear``).
"""

import contextlib
import functools
import importlib
import inspect
import logging
import pkgutil
import sys
import threading
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.library import Library

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _preserve_global_torch_state():
    """Snapshot and restore process-global torch defaults.

    ``enable()`` imports a large number of modules (both the explicit
    registry via ``_resolve_target`` and the auto-discovery
    ``_force_import_submodules``). Some of those modules mutate
    process-global torch state at import time — most notably
    ``torch.set_default_device("cuda")`` — and that mutation is NOT undone
    by ``disable()`` (which only restores patched *function references*).

    A leaked default device corrupts downstream CPU tensor creation: e.g.
    a buffer such as ``seq_lens_cpu`` is suddenly allocated on cuda, which
    surfaces as ``Buffer seq_lens_cpu has different device than before`` in
    the input-buffer pool and, later, as out-of-bounds GPU memory access
    faults in index kernels (e.g. ``write_req_to_token_pool_triton``).

    Restoring the default device and dtype around the import-heavy regions
    keeps these side effects from escaping into the serving path. In the
    healthy case (nothing mutates the defaults) this is a no-op.
    """
    get_default_device = getattr(torch, "get_default_device", None)
    saved_device = get_default_device() if get_default_device is not None else None
    saved_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        if saved_device is not None:
            try:
                torch.set_default_device(saved_device)
            except Exception:
                pass
        try:
            torch.set_default_dtype(saved_dtype)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_enabled = False
# The torch.library.Library is created once and kept alive for the whole
# process lifetime. It is intentionally NEVER torn down: dropping it destroys
# the registered custom ops (freeing their OperatorName / schema). A wrapper
# reference can outlive a disable() — e.g. a module imported lazily *during* a
# profiling window captured the wrapper via ``from X import Y`` and is not in
# ``_patches`` to be restored. If the backing op were freed, that leaked
# wrapper would later dispatch into freed memory and segfault. Keeping the
# Library (and the monotonic op counter) alive makes such a stale dispatch
# safe; the ``if not _enabled`` guard in each wrapper then routes it straight
# to the original function.
_lib: Optional[Library] = None
# Monotonic — NEVER reset, so op names from earlier enable() cycles stay valid.
_op_counter = 0
# Each entry: (module_obj, attr_name, original_fn)
_patches: List[Tuple[Any, str, Callable]] = []
# Persistent cache of built wrappers keyed by qualified function name.
# Value: (wrapper_fn, original_fn). Reused across enable()/disable() cycles so
# each op is defined exactly once and a given function maps to a stable wrapper
# object (so references that leaked across cycles still point at a live op).
_built_wrappers: dict = {}


def _get_or_create_lib() -> Library:
    """Return the process-wide custom-op Library, creating it on first use."""
    global _lib
    if _lib is None:
        _lib = Library("sglang_profiler", "FRAGMENT")
    return _lib


# ---------------------------------------------------------------------------
# Registry of kernel entry points to wrap.
#
# Each entry is (module_path, function_name).
#
# **Guidelines for choosing what to register:**
#
# 1. Prefer functions that are called via *module-global name lookup*
#    at call time.  These are always patchable because Python resolves
#    the name in the module's ``__dict__`` on every call.
#
# 2. Avoid outer "dispatch" wrappers whose references get captured as
#    instance attributes (e.g. ``self.w8a8_block_fp8_linear =
#    dispatch_w8a8_block_fp8_linear()``).  Instead register the *inner*
#    kernel they call.
#
# 3. For functions imported via ``from X import Y`` into multiple
#    modules, the ``_patch_all_references()`` helper will find and
#    replace them everywhere in ``sys.modules``.
# ---------------------------------------------------------------------------
_KERNEL_ENTRY_POINTS = [
    # ── Triton attention ──
    ("sglang.srt.layers.attention.triton_ops.decode_attention", "decode_attention_fwd"),
    (
        "sglang.srt.layers.attention.triton_ops.decode_attention",
        "decode_attention_fwd_normal",
    ),
    (
        "sglang.srt.layers.attention.triton_ops.decode_attention",
        "decode_attention_fwd_grouped",
    ),
    ("sglang.srt.layers.attention.triton_ops.extend_attention", "extend_attention_fwd"),
    (
        "sglang.srt.layers.attention.triton_ops.prefill_attention",
        "context_attention_fwd",
    ),
    # ── Fused MoE ──
    ("sglang.srt.layers.moe.fused_moe_triton.fused_moe", "invoke_fused_moe_kernel"),
    ("sglang.srt.layers.moe.fused_moe_triton.fused_moe", "moe_align_block_size"),
    (
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels",
        "fused_append_shared_experts",
    ),
    # ── MoE TopK ──
    ("sglang.srt.layers.moe.topk", "biased_grouped_topk_gpu"),
    # ── Layer norm ──
    # The actual module-level names are rmsnorm / fused_add_rmsnorm
    # (imported from sgl_kernel on CUDA, or aiter on HIP).
    ("sglang.srt.layers.layernorm", "rmsnorm"),
    ("sglang.srt.layers.layernorm", "fused_add_rmsnorm"),
    ("sglang.srt.layers.layernorm", "gemma_rmsnorm"),
    ("sglang.srt.layers.layernorm", "gemma_fused_add_rmsnorm"),
    # ── FP8 quantization ──
    ("sglang.srt.layers.quantization.fp8_utils", "per_token_group_quant_fp8"),
    ("sglang.srt.layers.quantization.fp8_utils", "scaled_fp8_quant"),
    # Inner kernel called by triton_w8a8_block_fp8_linear via global lookup:
    ("sglang.srt.layers.quantization.fp8_utils", "w8a8_block_fp8_matmul_triton"),
    # Inner kernel called by aiter_w8a8_block_fp8_linear via global lookup
    # (the outer wrapper is captured by reference at model init, but this
    # inner kernel is looked up from the module __dict__ on every call):
    ("sglang.srt.layers.quantization.fp8_utils", "gemm_a8w8_blockscale"),
    # ── LoRA Triton (inner kernel functions, no *args/**kwargs) ──
    ("sglang.srt.lora.triton_ops.sgemm_lora_a", "sgemm_lora_a_fwd"),
    ("sglang.srt.lora.triton_ops.sgemm_lora_b", "sgemm_lora_b_fwd"),
    # ── aiter (AMD) ops — definition-site patching ──
    ("aiter.ops.triton.gemm_a8w8_blockscale", "gemm_a8w8_blockscale"),
    ("aiter.ops.triton.batched_gemm_a8w8_blockscale", "batched_gemm_a8w8_blockscale"),
    ("aiter.ops.norm", "rms_norm"),
    ("aiter.ops.norm", "fused_add_rms_norm"),
    # ── FlashInfer MoE (cutedsl) ──
    ("flashinfer.moe", "moe_gemm_fp8_nt_groupwise"),
]

# ---------------------------------------------------------------------------
# Auto-discovery prefixes.
#
# In addition to the explicit registry above, ``enable()`` scans every
# already-loaded module whose name starts with one of these prefixes and
# wraps only functions likely to launch kernels.  Discovery is filtered by
# signature/source heuristics to avoid wrapping unrelated utility code.
# ---------------------------------------------------------------------------
_AUTO_DISCOVER_PREFIXES: Tuple[str, ...] = (
    "flashinfer.",
    "sglang.srt.",
    "aiter.ops.",
)


# ---------------------------------------------------------------------------
# Schema building — works with or without type annotations
# ---------------------------------------------------------------------------

# Python type → torch schema type
_TYPE_MAP = {
    torch.Tensor: "Tensor",
    Optional[torch.Tensor]: "Tensor?",
    int: "int",
    float: "float",
    bool: "bool",
    str: "str",
    torch.dtype: "ScalarType",
}

# String annotation variants produced by ``from __future__ import annotations``
# (PEP 563) – annotations are stored as literal strings in the source code.
_STRING_TYPE_MAP = {
    "torch.Tensor": "Tensor",
    "Tensor": "Tensor",
    "Optional[torch.Tensor]": "Tensor?",
    "Optional[Tensor]": "Tensor?",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "str": "str",
    "torch.dtype": "ScalarType",
}


def _infer_schema_type(param: inspect.Parameter) -> Optional[str]:
    """Map a parameter's annotation to a torch schema type string."""
    annotation = param.annotation
    if annotation is inspect._empty:
        return None

    # ── Handle string annotations (PEP 563) ──
    if isinstance(annotation, str):
        if annotation in _STRING_TYPE_MAP:
            return _STRING_TYPE_MAP[annotation]
        # Check "Optional[X]" pattern in string form
        if annotation.startswith("Optional[") and annotation.endswith("]"):
            inner = annotation[len("Optional[") : -1]
            base = _STRING_TYPE_MAP.get(inner)
            if base is not None:
                return base if base.endswith("?") else base + "?"
        return None

    # ── Handle real type annotations ──
    # Check direct match
    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]
    # Check Optional[X] (Union[X, None])
    origin = getattr(annotation, "__origin__", None)
    if origin is type(None):
        return None
    args = getattr(annotation, "__args__", ())
    if args and type(None) in args:
        for a in args:
            if a is not type(None) and a in _TYPE_MAP:
                return _TYPE_MAP[a] + "?"
    return None


def _build_schema_from_sig(
    sig: inspect.Signature,
    skip_self: bool = False,
) -> Optional[Tuple[str, List[str], List[str]]]:
    """
    Build schema string from signature annotations.
    Returns (schema_str, tensor_param_names, non_tensor_param_names) or None
    if there are no tensor params or the signature can't be mapped.
    """
    tensor_params: List[str] = []
    non_tensor_params: List[str] = []
    schema_parts: List[str] = []

    for name, param in sig.parameters.items():
        if skip_self and name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            return None

        stype = _infer_schema_type(param)
        if stype is not None and "Tensor" in stype:
            tensor_params.append(name)
            schema_parts.append(f"{stype} {name}")
        else:
            non_tensor_params.append(name)

    if not tensor_params:
        return None

    schema_str = f"({', '.join(schema_parts)}) -> ()"
    return schema_str, tensor_params, non_tensor_params


# ---------------------------------------------------------------------------
# Thread-local side channel for non-tensor args
# ---------------------------------------------------------------------------
_tls = threading.local()


def _stash_non_tensor_args(op_name: str, values: dict):
    if not hasattr(_tls, "stash"):
        _tls.stash = {}
    _tls.stash[op_name] = values


def _pop_non_tensor_args(op_name: str) -> dict:
    if not hasattr(_tls, "stash"):
        return {}
    return _tls.stash.pop(op_name, {})


def _stash_return_value(op_name: str, value: Any):
    if not hasattr(_tls, "returns"):
        _tls.returns = {}
    _tls.returns[op_name] = value


def _pop_return_value(op_name: str) -> Any:
    if not hasattr(_tls, "returns"):
        return None
    return _tls.returns.pop(op_name, None)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _next_op_name(base: str) -> str:
    global _op_counter
    sanitized = base.replace(".", "_").replace("::", "_").replace("-", "_")
    name = f"{sanitized}_{_op_counter}"
    _op_counter += 1
    return name


def _register_op(
    op_name: str,
    schema_str: str,
    original_fn: Callable,
    tensor_param_names: List[str],
    non_tensor_param_names: List[str],
    sig: inspect.Signature,
    skip_self: bool = False,
) -> Optional[Callable]:
    """
    Register a function as a torch custom op and return a dispatch wrapper.
    Returns None if registration fails.
    """
    try:
        lib = _get_or_create_lib()
        lib.define(op_name + schema_str)

        def impl(*tensor_args):
            nt_args = _pop_non_tensor_args(op_name)
            full_kwargs = {}
            t_idx = 0
            for pname, param in sig.parameters.items():
                if skip_self and pname == "self":
                    continue
                if pname in tensor_param_names:
                    full_kwargs[pname] = tensor_args[t_idx]
                    t_idx += 1
                elif pname in non_tensor_param_names:
                    if pname in nt_args:
                        full_kwargs[pname] = nt_args[pname]
                    elif param.default is not inspect._empty:
                        full_kwargs[pname] = param.default
            result = original_fn(**full_kwargs)
            # Schema is -> () so we can't return the actual value
            # through the dispatcher.  Stash it for the caller.
            _stash_return_value(op_name, result)

        lib.impl(op_name, impl, dispatch_key="CompositeExplicitAutograd")

        torch_op = getattr(torch.ops.sglang_profiler, op_name)

        @functools.wraps(original_fn)
        def dispatch_wrapper(*args, **kwargs):
            # When profiling is not active, never route through the custom op.
            # A wrapper reference may outlive disable() (captured by a module
            # imported lazily while profiling was on, so _patch_all_references
            # could not restore it). Falling back to the original keeps such
            # leaked bindings correct and crash-free.
            if not _enabled:
                return original_fn(*args, **kwargs)
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                return original_fn(*args, **kwargs)

            tensor_args = []
            nt_vals = {}
            for pname, val in bound.arguments.items():
                if skip_self and pname == "self":
                    continue
                if pname in tensor_param_names:
                    tensor_args.append(val)
                elif pname in non_tensor_param_names:
                    nt_vals[pname] = val

            # If every tensor arg is None (all Optional[Tensor] and not
            # provided), torch dispatch will fail with "no tensor arguments".
            # Fall back to calling the original function directly.
            if not any(isinstance(t, torch.Tensor) for t in tensor_args):
                return original_fn(*args, **kwargs)

            _stash_non_tensor_args(op_name, nt_vals)
            try:
                torch_op(*tensor_args)
                return _pop_return_value(op_name)
            except Exception:
                # Dispatch failed (e.g. device/type mismatch, None for
                # non-optional Tensor, schema arity error).  Clean up
                # thread-local stash and fall back to the original call.
                _pop_non_tensor_args(op_name)
                _pop_return_value(op_name)
                return original_fn(*args, **kwargs)

        return dispatch_wrapper

    except Exception as e:
        logger.debug("Failed to register %s: %s", op_name, e)
        return None


# ---------------------------------------------------------------------------
# Module + attribute resolution
# ---------------------------------------------------------------------------


def _resolve_target(module_path: str, attr_name: str):
    """
    Resolve a target function from *module_path* and *attr_name*.
    *attr_name* can be ``"func_name"`` or ``"ClassName.method_name"``.

    Returns ``(container, attr_name, original_fn, is_method)`` or ``None``.
    """
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return None

    if "." in attr_name:
        cls_name, method_name = attr_name.split(".", 1)
        cls = getattr(mod, cls_name, None)
        if cls is None:
            return None
        fn = getattr(cls, method_name, None)
        if fn is None:
            return None
        return cls, method_name, fn, True
    else:
        fn = getattr(mod, attr_name, None)
        if fn is None:
            return None
        return mod, attr_name, fn, False


def _patch_all_references(original_fn: Callable, wrapper_fn: Callable):
    """
    Scan ``sys.modules`` and replace **every** module-level attribute that
    points to *original_fn* with *wrapper_fn*.

    This handles the common ``from X import Y`` pattern: if module A
    defines ``Y`` and module B does ``from A import Y``, both A and B
    will have their binding replaced.

    Returns a list of ``(module, attr_name, original_fn)`` for later
    restoration.
    """
    patches = []
    for _mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            mod_dict = vars(mod)
        except TypeError:
            continue
        for attr_name in list(mod_dict.keys()):
            if attr_name.startswith("__"):
                continue
            try:
                if mod_dict[attr_name] is original_fn:
                    setattr(mod, attr_name, wrapper_fn)
                    patches.append((mod, attr_name, original_fn))
            except Exception:
                pass
    return patches


# ---------------------------------------------------------------------------
# Lightweight wrappers for functions that can't use torch.library
# ---------------------------------------------------------------------------


def _make_record_function_wrapper(
    qualified_name: str,
    original_fn: Callable,
) -> Callable:
    """
    Create a wrapper that uses ``torch.profiler.record_function`` to emit
    a ``cpu_op`` event with tensor shapes embedded in the event name.

    Used for functions where we can't build a ``torch.library`` schema
    (e.g. no type annotations, ``*args``/``**kwargs``, etc.).
    """

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        # See dispatch_wrapper: a leaked reference must be a no-op when
        # profiling is inactive.
        if not _enabled:
            return original_fn(*args, **kwargs)
        shape_parts: List[str] = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                shape_parts.append(f"arg{i}:{list(arg.shape)}")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                shape_parts.append(f"{k}:{list(v.shape)}")
        if shape_parts:
            event_name = f"{qualified_name}({', '.join(shape_parts)})"
        else:
            event_name = qualified_name
        with torch.profiler.record_function(event_name):
            return original_fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Kernel-launch detection heuristics
# ---------------------------------------------------------------------------

# Substrings that strongly indicate a function launches a GPU kernel.
_KERNEL_SOURCE_INDICATORS = (
    "[grid",  # Triton launch pattern: kernel[grid](...)
    "torch.ops.",  # Custom C++/CUDA op dispatch
    "sgl_kernel.",  # sgl-kernel extension entry points
)


def _source_launches_kernel(fn: Callable) -> bool:
    """Return True if *fn* source contains known kernel-launch patterns."""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return any(marker in source for marker in _KERNEL_SOURCE_INDICATORS)


def _is_likely_kernel_launcher(fn: Callable, sig: inspect.Signature) -> bool:
    """
    Decide whether *fn* is likely to launch a GPU kernel.

    Priority:
    1) Tensor annotation exists -> include.
    2) Non-tensor annotations only -> exclude.
    3) No annotations -> fallback to source pattern matching.
    """
    has_any_annotation = False
    for param in sig.parameters.values():
        if param.annotation is inspect._empty:
            continue
        has_any_annotation = True
        stype = _infer_schema_type(param)
        if stype is not None and "Tensor" in stype:
            return True

    if has_any_annotation:
        return False

    return _source_launches_kernel(fn)


def _force_import_submodules(prefix: str) -> None:
    """
    Recursively import submodules under *prefix* so they appear in
    ``sys.modules`` before auto-discovery runs.

    *prefix* should be a top-level package name without a trailing dot
    (e.g. ``"sglang.srt"``).
    """
    try:
        pkg = importlib.import_module(prefix)
    except ImportError:
        return

    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return

    # Snapshot the global defaults once; restore them after *every* import so
    # a module that calls torch.set_default_device("cuda") at import time
    # cannot taint subsequently imported modules during discovery (nor leak
    # into the serving path). The leaf-name skip list below catches the known
    # offenders, but this restore makes the discovery robust to any other
    # module with the same import-time side effect.
    get_default_device = getattr(torch, "get_default_device", None)
    saved_device = get_default_device() if get_default_device is not None else None
    saved_dtype = torch.get_default_dtype()

    def _restore_defaults():
        if saved_device is not None:
            try:
                torch.set_default_device(saved_device)
            except Exception:
                pass
        try:
            torch.set_default_dtype(saved_dtype)
        except Exception:
            pass

    for _importer, mod_name, _is_pkg in pkgutil.walk_packages(
        pkg_path, prefix=prefix + "."
    ):
        if mod_name in sys.modules:
            continue
        # Skip test / benchmark / autotune modules.  These are not kernel
        # entry points and several of them (e.g. aiter.ops.flydsl.test_*)
        # call ``torch.set_default_device("cuda")`` at import time, which
        # leaks a CUDA default-device mode into the importing process and
        # corrupts CPU tensor creation downstream (e.g. seq_lens_cpu).
        leaf = mod_name.rsplit(".", 1)[-1]
        if (
            "test" in leaf
            or leaf.startswith("test_")
            or leaf.startswith("bench_")
            or leaf.endswith("_test")
            or leaf.endswith("_tune")
        ):
            continue
        try:
            importlib.import_module(mod_name)
        except Exception:
            # Optional modules may fail to import depending on environment.
            pass
        finally:
            _restore_defaults()


def _discover_kernel_entry_points() -> List[Tuple[str, str]]:
    """
    Scan already-loaded ``sys.modules`` for modules whose name starts
    with one of ``_AUTO_DISCOVER_PREFIXES`` and collect only functions
    likely to launch GPU kernels.

    Returns a list of ``(module_path, function_name)`` pairs.
    """
    # Force-import submodules under each discovery prefix first so deeper
    # kernels become visible in sys.modules.
    for prefix in _AUTO_DISCOVER_PREFIXES:
        _force_import_submodules(prefix.rstrip("."))

    results: List[Tuple[str, str]] = []
    seen_ids: set = set()

    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not any(mod_name.startswith(p) for p in _AUTO_DISCOVER_PREFIXES):
            continue
        try:
            mod_dict = vars(mod)
        except TypeError:
            continue
        for attr_name in list(mod_dict.keys()):
            if attr_name.startswith("__"):
                continue
            obj = mod_dict[attr_name]

            # Only wrap regular Python functions.
            # @triton.jit objects (JITFunction / Autotuner) must NOT be
            # wrapped — replacing them in module globals breaks the
            # Triton compiler's global resolution for device-side calls
            # between JIT kernels (e.g. remap_xcd, _rmsmorm_op, etc.).
            if not inspect.isfunction(obj):
                continue

            # Only include functions *defined* within a target namespace
            fn_module = getattr(obj, "__module__", "") or ""
            if not any(fn_module.startswith(p) for p in _AUTO_DISCOVER_PREFIXES):
                continue

            obj_id = id(obj)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)

            # Require at least one parameter
            try:
                sig = inspect.signature(obj)
            except (ValueError, TypeError):
                continue
            if not sig.parameters:
                continue

            # Skip signatures we cannot map to a torch schema.
            if any(
                p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                for p in sig.parameters.values()
            ):
                continue

            # Core filter: only keep likely kernel launchers.
            if not _is_likely_kernel_launcher(obj, sig):
                continue

            results.append((mod_name, attr_name))

    logger.debug(
        "Auto-discovered %d kernel candidates from %s",
        len(results),
        ", ".join(_AUTO_DISCOVER_PREFIXES),
    )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enable():
    """Patch registered kernel entry points to appear as cpu_op."""
    global _enabled
    with _lock:
        if _enabled:
            return

        # NOTE: the Library and op counter are process-persistent (see the
        # _lib / _op_counter docs). We do NOT recreate or reset them here so
        # ops registered in earlier cycles stay valid. Only the per-cycle
        # reference patches are rebuilt.
        _patches.clear()
        _wrapped_ids: set = set()  # track function ids to avoid double-wrapping

        # Backstop guard around the import-heavy region. Both the explicit
        # registry (_resolve_target -> importlib.import_module) and the
        # auto-discovery (_discover_kernel_entry_points -> _force_import_*)
        # import modules that may mutate process-global torch defaults at
        # import time; restore them on exit so the leak cannot escape into
        # the serving path. (No-op when nothing mutates the defaults.)
        with _preserve_global_torch_state():
            # Merge explicit registry with filtered auto-discovered functions.
            # Duplicates are harmless — _wrapped_ids prevents double-wrapping.
            all_entry_points = (
                list(_KERNEL_ENTRY_POINTS) + _discover_kernel_entry_points()
            )

            for module_path, attr_name in all_entry_points:
                resolved = _resolve_target(module_path, attr_name)
                if resolved is None:
                    continue

                container, name, original_fn, is_method = resolved
                is_plain_function = not is_method

                # Skip if this exact function object was already wrapped
                fn_id = id(original_fn)
                if fn_id in _wrapped_ids:
                    logger.debug("Skipping duplicate %s.%s", module_path, attr_name)
                    continue
                _wrapped_ids.add(fn_id)

                qualified_name = f"{module_path}.{name}"

                # Reuse a wrapper built in a previous cycle if the underlying
                # function object is unchanged. This keeps each op defined
                # exactly once and ensures a given function maps to a stable
                # wrapper, so a reference that leaked across cycles still
                # targets a live op.
                wrapper = None
                cached = _built_wrappers.get(qualified_name)
                if cached is not None and cached[1] is original_fn:
                    wrapper = cached[0]

                if wrapper is None:
                    # ── Regular functions ──
                    try:
                        sig = inspect.signature(original_fn)
                    except (ValueError, TypeError):
                        logger.debug(
                            "Cannot inspect signature of %s — skipping",
                            qualified_name,
                        )
                        continue

                    base = f"{module_path.split('.')[-1]}_{name}"
                    schema_info = _build_schema_from_sig(sig, skip_self=is_method)

                    if schema_info is not None:
                        # Full tensor annotations → use torch.library custom op
                        schema_str, t_names, nt_names = schema_info
                        op_name = _next_op_name(base)
                        wrapper = _register_op(
                            op_name,
                            schema_str,
                            original_fn,
                            t_names,
                            nt_names,
                            sig,
                            skip_self=is_method,
                        )
                        if wrapper is not None:
                            logger.debug(
                                "Registered %s as custom op %s", qualified_name, op_name
                            )

                    if wrapper is None:
                        # Fallback: no annotations or schema registration failed
                        # → use record_function wrapper (shapes in event name)
                        wrapper = _make_record_function_wrapper(
                            qualified_name,
                            original_fn,
                        )
                        logger.debug(
                            "Registered %s via record_function", qualified_name
                        )

                    _built_wrappers[qualified_name] = (wrapper, original_fn)

                # --- Apply patches ---
                if is_plain_function:
                    ref_patches = _patch_all_references(original_fn, wrapper)
                    _patches.extend(ref_patches)
                    if not ref_patches:
                        setattr(container, name, wrapper)
                        _patches.append((container, name, original_fn))
                else:
                    setattr(container, name, wrapper)
                    _patches.append((container, name, original_fn))

            n_discovered = len(all_entry_points) - len(_KERNEL_ENTRY_POINTS)

        _enabled = True
        logger.info(
            "kernel_shape_profiler enabled: %d references patched across "
            "%d entry points (%d explicit + %d auto-discovered)",
            len(_patches),
            len(_KERNEL_ENTRY_POINTS) + n_discovered,
            len(_KERNEL_ENTRY_POINTS),
            n_discovered,
        )


def disable():
    """Restore all patched functions to originals."""
    global _enabled
    with _lock:
        if not _enabled:
            return
        # Flip the flag first so any wrapper invoked concurrently — or one that
        # leaked past restoration — short-circuits to the original instead of
        # dispatching into a custom op.
        _enabled = False
        for container, name, original_fn in reversed(_patches):
            try:
                setattr(container, name, original_fn)
            except Exception:
                pass
        _patches.clear()
        # Intentionally keep _lib, _op_counter and _built_wrappers alive: the
        # registered ops must outlive any wrapper reference that may have
        # leaked (see module-level _lib docs).
        logger.info("kernel_shape_profiler disabled: all patches restored")


def is_enabled() -> bool:
    return _enabled
