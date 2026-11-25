"""Common helpers shared by kernel modules.

Keep helper naming consistent with other kernel helpers (e.g. `mfma_preshuffle_pipeline.py`),
but this module is intentionally small and MLIR-dialect facing.
"""

from collections.abc import Callable
from threading import Lock
from typing import Any

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import builtin
from flydsl._mlir.dialects import gpu as _gpu
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

LOG2E = 1.4426950408889634


def ceildiv(numer, denom):
    """Ceiling division preserving Python-int or DSL-scalar operand types."""
    return (numer + denom - 1) // denom


def format_kernel_name(name: str) -> str:
    """Sanitize a kernel symbol name for the amdhsa assembler.

    Config values interpolated into a kernel name may be negative (e.g. the
    grouped-contiguous sentinel ``topk=-1`` renders as ``tk-1``). A hyphen is
    not a legal symbol character, so the assembler misparses the
    ``.amdhsa_kernel`` directive and the whole module fails to link.
    """
    return name.replace("-", "_")


def kernel_signature(**params: object) -> str:
    """Render build parameters into a kernel-name suffix.

    Every build parameter that changes a kernel's body belongs here. Two builds
    of one module that differ only in an omitted parameter otherwise emit the
    same symbol, and are then indistinguishable in a profile, in a disassembly
    dump, and to anything keyed on the name.

    Booleans render as 0/1 so the suffix stays short, and the whole string goes
    through ``format_kernel_name`` because a negative config value is legal here
    and a hyphen is not legal in a symbol.
    """
    parts = [
        f"{name}{int(value) if isinstance(value, bool) else value}"
        for name, value in params.items()
    ]
    return format_kernel_name("_".join(parts))


# Exponent-all-ones with a zero mantissa; anything above it is a NaN.
F32_INF_BITS = 0x7F800000
F32_NAN_KEY = 2147483647
_F32_INT32_MIN = -2147483648


def ord_signed_f32(value):
    """Map fp32 to an int32 that compares the same way under `<`, NaN highest.

    fp32 is sign-magnitude, so flipping the magnitude bits of negatives yields a
    signed-integer total order. -0.0 and 0.0 are one score with two bit patterns
    and must not become two keys.

    NaN sorts above +inf, matching `torch.topk`. The per-row selectors are
    dispatched by shape, so a row holding a NaN must not answer differently
    depending on a choice the caller did not make -- which is why this lives
    here rather than once per selector. Testing the bits rather than `x != x`
    keeps it in the integer domain and leaves the infinities where they belong.
    """
    bits = value.bitcast(fx.Int32)
    bits = (bits == fx.Int32(_F32_INT32_MIN)).select(fx.Int32(0), bits)
    ordered = bits ^ ((bits >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF))
    is_nan = (bits & fx.Int32(0x7FFFFFFF)) > fx.Int32(F32_INF_BITS)
    return is_nan.select(fx.Int32(F32_NAN_KEY), ordered)


def uint32_to_int32(x: int) -> int:
    """Return the signed int32 value with the same low 32-bit pattern."""
    return x - (1 << 32) if x >= (1 << 31) else x


def _atomic_rmw_i32(atomic_op, memref, val, offset, syncscope):
    ptr = fx.get_iter(memref) + offset
    val = fx.Int32(val) if isinstance(val, int) else val
    return atomic_op(ptr, val, syncscope=syncscope)


def atomic_add_i32(memref, val, offset, syncscope):
    """Atomically add an int32 value and return the previous value."""
    return _atomic_rmw_i32(fx.atomic_add, memref, val, offset, syncscope)


def atomic_or_i32(memref, val, offset, syncscope):
    """Atomically OR an int32 value in and return the previous value."""
    return _atomic_rmw_i32(fx.atomic_or, memref, val, offset, syncscope)


def atomic_max_i32(memref, val, offset, syncscope):
    """Atomically take the signed max and return the previous value.

    Unlike a fetch-and-add, the result does not depend on the order the lanes
    are served, so a reduction built on this is reproducible.
    """
    return _atomic_rmw_i32(fx.atomic_max, memref, val, offset, syncscope)


def get_warp_size(arch=None):
    """Return the wavefront/warp size for the given GPU architecture.

    CDNA (gfx9xx) uses wave64, RDNA (gfx10xx/gfx11xx/gfx12xx) uses wave32.
    NOTE: we do not defer the gfx12 case to ``flydsl.runtime.device.is_rdna_arch``:
    that helper only matches ``gfx120*`` and so misclassifies gfx1250 (which is
    wave32, RDNA-family) as CDNA, yielding wave64. Building a kernel for wave64
    while gfx1250 dispatches wave32 corrupts every >1-warp-per-block kernel
    (the phantom upper lanes silently drop their work). Classify all gfx10/11/12
    as wave32 directly here so the kernel body matches the wave32 dispatch.
    """
    if arch is None:
        arch = get_rocm_arch()
    arch_l = (arch or "").lower()
    if arch_l.startswith(("gfx10", "gfx11", "gfx12")):
        return 32
    return 32 if is_rdna_arch(arch) else 64


def default_f8_type() -> ir.Type:
    """Select the E4M3 f8 type compatible with the current GPU arch.

    - gfx95* (MI350): FP8 E4M3FN (OCP)
    - gfx12*: FP8 E4M3FN (OCP)
    - gfx94* (MI300): FP8 E4M3FNUZ

    Raises ``RuntimeError`` on gfx11* (RDNA3/RDNA3.5): these chips have no
    native FP8 instructions, so FP8 compute would surface as a late LLVM
    "cannot select" error. Fail early with a clear message instead.

    Replaces the ``T.f8`` shortcut removed from ``flydsl.expr.typing`` in
    flydsl 0.3.0 (upstream moved arch-specific FP8 selection into the kernel
    layer); mirrors that helper here for the vendored kernels.
    """
    arch = ""
    try:
        arch = str(get_rocm_arch())
    except Exception:  # noqa: BLE001
        arch = ""
    if "gfx95" in arch or "gfx12" in arch:
        return fx.Float8E4M3FN.ir_type
    if arch.startswith("gfx11"):
        raise RuntimeError(
            f"default_f8_type(): no native FP8 support on {arch}; "
            "FP8 instructions are available on gfx94*, gfx95*, and gfx12*."
        )
    return fx.Float8E4M3FNUZ.ir_type


def dtype_to_elem_type(dtype_str: str):
    """Map a dtype string to its MLIR scalar type.

    Supported: ``'f32'``, ``'f16'``, ``'bf16'``.
    """
    if dtype_str == "f32":
        return T.f32
    if dtype_str == "f16":
        return T.f16
    if dtype_str == "bf16":
        return T.bf16
    raise ValueError(
        f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', or 'bf16')"
    )


# LLVM address-space numbers as fx spaces: Global(1) and Shared, which is 2 in
# fx terms but lowers to !llvm.ptr<3>. to_llvm_ptr resolves it, so the backend's
# number never appears at a call site.
FX_ADDRESS_SPACE = {1: fx.AddressSpace.Global, 3: fx.AddressSpace.Shared}


def create_llvm_ptr(value, address_space=1):
    """Raw LLVM pointer for atomics and intrinsic APIs."""
    # Accept either the LLVM number (1 global / 3 LDS) or an fx.AddressSpace,
    # so a caller cannot silently pass the wrong one.
    space = FX_ADDRESS_SPACE.get(address_space, address_space)
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=space, alignment=4)
    ptr = fx.to_llvm_ptr(fx.inttoptr(pt, value))
    return ptr._value if hasattr(ptr, "_value") else ptr


def stream_ptr_to_async_token(stream_ptr_value, loc=None, ip=None):
    stream_llvm_ptr = create_llvm_ptr(stream_ptr_value)

    async_token_type = _gpu.AsyncTokenType.get()
    cast_op = builtin.UnrealizedConversionCastOp(
        [async_token_type], [stream_llvm_ptr], loc=loc, ip=ip
    )
    return cast_op.results[0]


_compiled_cache_lock = Lock()


def run_cached(
    jit_func: Any,
    *compile_args: Any,
    constexpr_param: Any,
    compiler: Callable[..., Any],
    dispatch_args: tuple[Any, ...],
) -> Any:
    """Cache a layout-dynamic FlyDSL dispatcher by constexpr param."""
    cache_key = constexpr_param.__cache_signature__()
    compiled_cache = getattr(jit_func, "_compiled_cache", None)
    if compiled_cache is not None:
        compiled = compiled_cache.get(cache_key)
        if compiled is not None:
            compiled(*dispatch_args)
            return compiled

    dispatch_after_wait = False
    with _compiled_cache_lock:
        compiled_cache = getattr(jit_func, "_compiled_cache", None)
        if compiled_cache is None:
            compiled_cache = {}
            jit_func._compiled_cache = compiled_cache

        compiled = compiled_cache.get(cache_key)
        if compiled is None:
            compiled = compiler(jit_func, *compile_args)
            compiled_cache[cache_key] = compiled
        else:
            dispatch_after_wait = True

    if dispatch_after_wait:
        compiled(*dispatch_args)
    return compiled


_SLP_VECTORIZE_ONCE_USED = "slp-vectorize-once-used"
_slp_once_used_available = None


def set_slp_vectorize_once_used(jit_fn, enabled):
    """Set LLVM slp-vectorize-once-used on this JIT.

    ``enabled`` is the value the kernel wants. flydsl 0.4.1 defaults the flag
    on. flydsl 0.4.0 has no such option, and passing it would fail compilation,
    so the hint is attached only when the cl::opt exists.
    """
    global _slp_once_used_available
    if _slp_once_used_available is None:
        try:
            from flydsl._mlir._mlir_libs import _mlirDialectsFly as fly

            previous = fly.set_llvm_option_bool(_SLP_VECTORIZE_ONCE_USED, False)
            fly.set_llvm_option_bool(_SLP_VECTORIZE_ONCE_USED, previous)
            _slp_once_used_available = True
        except RuntimeError:
            _slp_once_used_available = False
    if _slp_once_used_available:
        options = jit_fn.compile_hints.setdefault("llvm_options", {})
        options[_SLP_VECTORIZE_ONCE_USED] = bool(enabled)
    return jit_fn
