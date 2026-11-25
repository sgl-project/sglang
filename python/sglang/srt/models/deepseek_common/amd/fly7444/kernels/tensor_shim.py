# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import threading
from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import product

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.protocol import extract_to_ir_values
from flydsl.expr import ptrtoint, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils import env

from .kernels_common import get_warp_size

# Global toggle for the amdgpu-kernarg-preload compile hint used by the flydsl
# kernels. Enabled by default; set AITER_FLYDSL_KERNARG_PRELOAD=0 to disable it
# globally for all kernels. AITER_FLYDSL_KERNARG_PRELOAD_COUNT overrides the
# number of kernel arguments to preload.
AITER_FLYDSL_KERNARG_PRELOAD = bool(
    int(os.environ.get("AITER_FLYDSL_KERNARG_PRELOAD", "1"))
)
AITER_FLYDSL_KERNARG_PRELOAD_COUNT = int(
    os.environ.get("AITER_FLYDSL_KERNARG_PRELOAD_COUNT", "32")
)

# Toggle for the amdgpu-expert-scheduling-mode compile hint on the MoE GEMM
# kernels. Disabled by default; set AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
# to enable it.
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE = bool(
    int(os.environ.get("AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE", "1"))
)

_PRELOAD_COMPILE_LOCK = threading.RLock()


def ptr_rsrc(ptr, num_records_bytes=None):
    """Convert an fx.Pointer kernel arg to a buffer resource for buffer_load/store.

    ``num_records_bytes`` may be a runtime value, for a hardware OOB check that
    zero-fills rather than reading stale bytes.
    """
    from . import buffer_ops

    return buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(ptrtoint(ptr)), num_records_bytes=num_records_bytes
    )


def buf_load_scalar(rsrc, dword_index, dwords=4):
    """Uniform load of ``dwords`` dwords from *rsrc*, landing directly in SGPRs."""
    from . import buffer_ops

    return buffer_ops.buffer_load(rsrc, dword_index, vec_width=dwords, is_scalar=True)


_BUF_COPY_ATOM = {
    16: fx.rocdl.BufferCopy128b,
    8: fx.rocdl.BufferCopy64b,
    4: fx.rocdl.BufferCopy32b,
    2: fx.rocdl.BufferCopy16b,
    1: fx.rocdl.BufferCopy8b,
}

# Nominal extent of a raw-pointer buffer view. The real bound is the V#
# num_records field, which make_buffer_tensor(max_size=True) pins to 0xFFFFFFFF
# bytes, so this only has to be large enough not to constrain any caller's
# indices.
BUF_VIEW_MAX_ELEMS = 0xFFFFFFFF


def buf_base_i64(base):
    """i64 base address of *base*: an fx pointer, a tensor/memref, or an address.

    Lets a caller narrow a descriptor to one row (``ptr + row * row_bytes``)
    without first materialising a pointer for it.
    """
    raw = extract_to_ir_values(base)[0]
    if str(raw.type).startswith(("!fly.ptr", "!llvm.ptr")):
        return fx.Int64(ptrtoint(base))
    if isinstance(raw.type, (ir.IntegerType, ir.IndexType)):
        return fx.Int64(base)
    # tensor / memref kernel arg: take its aligned base pointer.
    aligned = fly.extract_aligned_pointer_as_index(ir.Type.parse("!llvm.ptr<1>"), raw)
    return fx.Int64(llvm.PtrToIntOp(T.i64, aligned).result)


def ptr_buf_tensor(
    ptr,
    elem=fx.Int32,
    n=BUF_VIEW_MAX_ELEMS,
    unit_elems=1,
    num_records_bytes=None,
    unit_stride=None,
):
    """Buffer-resource (V#) view of *ptr*, so ``t[i]`` / ``fx.slice`` index it.

    *ptr* is an fx.Pointer kernel arg or a raw i64 address (see
    :func:`buf_base_i64`).

    Keeps the addressing `buffer_ops` used: descriptor in SGPRs, 32-bit voffset
    per access. Indexing a plain typed pointer instead builds a full 64-bit
    address in VGPRs on every access.

    ``unit_elems`` sets the access width and hence the rank:
      1  -> flat ``(n,)``; ``t[i]`` is one element. No atom, no fragment.
      >1 -> ``(n, unit_elems)``; ``fx.slice(t, (u, None))`` is one wide access
            and ``fx.copy`` over it emits a single ``buffer_load_dwordx{2,4}``.
            A vector *element type* would be the obvious spelling for this but
            MLIR rejects it (``AlignAttr`` takes integer/float only), so the
            width lives in the layout.

    ``unit_stride`` is the element distance between consecutive units and
    defaults to ``unit_elems`` (units tile the buffer, so ``u`` counts whole
    units). Pass 1 for a wide access at an arbitrary *element* offset: units
    then overlap -- meaningless to iterate but exact for the one slice a caller
    takes, and the only way to express e.g. a dwordx4 store at a row base that
    is only dword-aligned. It also sets the pointer alignment, which is all such
    an access can rely on.

    ``num_records_bytes`` may be a runtime value, for a hardware OOB check that
    zero-fills rather than reading stale bytes.
    """
    unit_stride = unit_elems if unit_stride is None else unit_stride
    layout = (
        fx.make_layout((n,), (1,))
        if unit_elems == 1
        else fx.make_layout((n, unit_elems), (unit_stride, 1))
    )
    pt = fx.PointerType.get(
        elem.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=unit_stride * (elem.width // 8),
    )
    view = fx.make_view(fx.inttoptr(pt, buf_base_i64(ptr)), layout)
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


def buf_copy_atom(unit_bytes, elem=fx.Int32, cache_modifier=0):
    """Copy atom for a ``unit_bytes``-wide buffer access (0=cached, 2=nt).

    ``cache_modifier`` is the instruction's raw aux/CPOL field, and its width is
    architecture-dependent: two bits (sc0, nt) on CDNA3, but gfx1250 puts scope
    bits above them and passes values like ``CPOL_DEV`` (0x10). It is therefore
    NOT range-checked here -- a bound that fits one generation rejects a legal
    value on another, which is what happened when this was briefly capped at 3
    and five gfx1250 mxfp8 GEMM kernels stopped building. Callers that know
    their own encoding bound it themselves; the M3 scorers do.
    """
    if unit_bytes not in _BUF_COPY_ATOM:
        raise ValueError(
            f"unit_bytes {unit_bytes} is not a buffer access width; "
            f"expected one of {sorted(_BUF_COPY_ATOM)}"
        )
    return fx.make_copy_atom(_BUF_COPY_ATOM[unit_bytes](cache_modifier), elem)


def buf_scalar_load(t, index, cache_modifier=0):
    """``t[index]`` as an ``s_buffer_load``: one dword landing in an SGPR.

    *index* must be wave-uniform; nothing here checks that. Use it where the
    result has to stay scalar -- e.g. a row that then narrows a per-row buffer
    descriptor, which otherwise goes through a readfirstlane waterfall.

    The layout API has no scalar spelling: ``t[i]`` and every ``BufferCopy*``
    atom lower to ``rocdl.raw.ptr.buffer.load`` (VGPR), and ROCDL exposes no
    ``s.buffer.load`` op to wrap. Reuse the scalar buffer path, which emits the
    intrinsic with its required v4i32 resource operand.
    """
    from . import buffer_ops

    rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(t))
    return buffer_ops.buffer_load(
        rsrc, index, vec_width=1, is_scalar=True, cache_modifier=cache_modifier
    )


# GTensor takes MLIR scalar types (``T.f32``); the buffer views take fx classes.
_FX_ELEM = {
    "i8": fx.Int8,
    "i16": fx.Int16,
    "i32": fx.Int32,
    "i64": fx.Int64,
    "f16": fx.Float16,
    "bf16": fx.BFloat16,
    "f32": fx.Float32,
}


def _fx_elem(dtype):
    """fx element class for an MLIR scalar type."""
    key = str(dtype() if callable(dtype) else dtype)
    try:
        return _FX_ELEM[key]
    except KeyError:
        raise TypeError(f"no fx element class for MLIR type {key!r}") from None


def _fx_value_elem(value):
    """(fx element class, element count) of a scalar or vector SSA value."""
    ty = extract_to_ir_values(value)[0].type
    # Duck-typed vector check, as buffer_store did: this MLIR binding has no
    # ir.VectorType.isinstance.
    if hasattr(ty, "element_type"):
        return _fx_elem(ty.element_type), ty.shape[0]
    return _fx_elem(ty), 1


def _buf_copy_slice(buffer, index, unit_elems):
    if unit_elems == 1:
        grouped = fx.logical_divide(buffer, fx.make_layout(1, 1))
        return fx.slice(grouped, (None, index))
    return fx.slice(buffer, (index, None))


def buf_copy_load(buffer, index, elem=fx.Int32, unit_elems=1, cache_modifier=0):
    """Load one vector unit, preserving an explicit buffer cache policy."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    fx.copy(
        buf_copy_atom(
            unit_elems * (elem.width // 8), elem, cache_modifier=cache_modifier
        ),
        _buf_copy_slice(buffer, index, unit_elems),
        fragment,
    )
    value = Vec(fragment.load())
    return value[0] if unit_elems == 1 else value


def buf_copy_store(buffer, index, value, elem=fx.Int32, unit_elems=1, cache_modifier=0):
    """Store one vector unit, preserving an explicit buffer cache policy."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    fragment.store(Vec.from_elements([value], elem) if unit_elems == 1 else Vec(value))
    fx.copy(
        buf_copy_atom(
            unit_elems * (elem.width // 8), elem, cache_modifier=cache_modifier
        ),
        fragment,
        _buf_copy_slice(buffer, index, unit_elems),
    )


@lru_cache(maxsize=8)
def wave_size_of(device_index: int | None = None) -> int:
    """Wave width of the GPU a kernel will dispatch on.

    Not ``get_warp_size(get_gfx())``: ``get_gfx()`` honours ``GPU_ARCHS``, so a
    build cross-targeting another arch answers for that arch while the kernel
    still dispatches here. A caller that picks a backend by one and builds it by
    the other picks for a machine it is not running on.
    """
    return get_warp_size(torch.cuda.get_device_properties(device_index).gcnArchName)


def check_e8m0(
    scale: torch.Tensor, name: str, owner: str, shape: tuple[int, ...] | None = None
) -> None:
    """Validate an e8m0 scale operand: e8m0/uint8 storage, contiguous, and
    ``shape`` when given. ``owner`` prefixes the error, e.g. "FlyDSL gfx950 bmm"."""
    from aiter.utility import dtypes

    if shape is not None and tuple(scale.shape) != shape:
        raise RuntimeError(
            f"[{owner}] {name} must have shape {shape}, got {tuple(scale.shape)}"
        )
    if scale.dtype not in (dtypes.fp8_e8m0, torch.uint8):
        raise RuntimeError(f"[{owner}] {name} must be e8m0/uint8, got {scale.dtype}")
    if not scale.is_contiguous():
        raise RuntimeError(f"[{owner}] {name} must be contiguous")


# Resolve the optional API once, but query its device/stream on every call.
# This is not a stream cache: graph capture and thread-local stream switches
# must still be observed. Older torch builds keep the public API fallback.
_get_current_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def current_stream_arg():
    if _get_current_raw_stream is None:
        return torch.cuda.current_stream()
    return fx.Stream(_get_current_raw_stream(torch.cuda.current_device()))


def ptr_arg(t: torch.Tensor, dtype=None):
    """Wrap a torch.Tensor as an fx.Pointer (PointerJitArg) for kernel launch."""
    if dtype is None:
        dtype = fx.Uint8
    type_name = type(t).__name__
    module_name = type(t).__module__
    if type_name == "FakeTensor" or "fake_tensor" in module_name:
        return flyc.from_c_void_p(dtype, 0)
    return flyc.from_c_void_p(dtype, t.data_ptr())


def _compile_and_run(exe, *args):
    """``flyc.compile(exe, *args)``: compiles **and** executes the kernel, and
    returns its ``CompiledFunction`` (None under compile-only)."""
    try:
        return flyc.compile(exe, *args)
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise


def _run_compiled(exe, *args, specialization_key=None):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.

    A specialization key gives a multi-constexpr JitFunction one compiled
    callable per configuration. Factory-style launchers should leave it unset.
    """
    if env.compile.compile_only:
        # A callable warmed during execution must not dispatch when a later
        # preload switches to compile-only. Preserve it for execution to resume.
        _compile_and_run(exe, *args)
        return
    if specialization_key is None:
        cf = getattr(exe, "_cf", None)
    else:
        cache = getattr(exe, "_cf_by_specialization", None)
        cf = cache.get(specialization_key) if cache is not None else None
    if cf is not None:
        cf(*args)
        return
    cf = _compile_and_run(exe, *args)
    if specialization_key is None:
        exe._cf = cf
    else:
        if not hasattr(exe, "_cf_by_specialization"):
            exe._cf_by_specialization = {}
        exe._cf_by_specialization[specialization_key] = cf


def _preload_compiled(exe, *args):
    """Materialize a JIT artifact without dispatching its GPU kernels.

    Newer FlyDSL versions expose ``JitFunction.preload()`` for this operation.
    Older supported versions provide the same no-dispatch behavior through the
    public ``COMPILE_ONLY`` mode.  Calling ``flyc.compile()`` without either
    guard executes the launcher and is unsafe for preload callers that pass
    placeholder pointers.
    """
    preload = getattr(exe, "preload", None)
    if callable(preload):
        return preload(*args)

    with _PRELOAD_COMPILE_LOCK:
        old_compile_only = os.environ.get("COMPILE_ONLY")
        os.environ["COMPILE_ONLY"] = "1"
        try:
            return flyc.compile(exe, *args)
        finally:
            if old_compile_only is None:
                os.environ.pop("COMPILE_ONLY", None)
            else:
                os.environ["COMPILE_ONLY"] = old_compile_only


def _to_raw(v):
    """Convert ArithValue / Numeric (Int32, Boolean, …) to raw ir.Value."""
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return _to_raw(v.ir_value())
    return ir.Value._CAPICreate(v._CAPIPtr)


def get_dtype_str(dtype):
    if dtype == torch.float:
        return "f32"
    elif dtype == torch.half:
        return "f16"
    elif dtype == torch.bfloat16:
        return "bf16"


class TensorView:
    def __init__(self, dtype, shape, stride, base_offset, load_impl, store_impl):
        self.dtype = dtype
        self.shape = shape
        if stride is None:
            self.stride = tuple(
                (
                    np.cumprod(shape[::-1])[::-1].tolist()
                    + [
                        1,
                    ]
                )[1:]
            )
        else:
            self.stride = stride
        self.base_offset = base_offset
        self.load_impl = load_impl
        self.store_impl = store_impl

    def _linear_offset(self, idxs):
        slice_shape = []
        slice_stride = []
        d_offset = self.base_offset
        for i in range_constexpr(len(idxs)):
            md_id = idxs[i]
            if md_id is None:
                slice_shape.append(self.shape[i])
                slice_stride.append(self.stride[i])
            elif isinstance(md_id, int):
                d_offset = d_offset + md_id * self.stride[i]
            else:
                d_offset = d_offset + md_id * self.stride[i]
        if len(slice_shape) > 0:
            return d_offset, tuple(slice_shape), tuple(slice_stride)
        else:
            return (d_offset,)

    def __repr__(self):
        return f"TensorView(offset={self.base_offset}, shape={self.shape}, stride={self.stride}, dtype={self.dtype})"

    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        if len(offset) == 1:
            return self.load_impl(offset[0])
        else:
            return TensorView(
                self.dtype,
                offset[1],
                offset[2],
                offset[0],
                self.load_impl,
                self.store_impl,
            )

    def __setitem__(self, idxs, value):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        self.store_impl(offset[0], value)

    def vec_load(self, idxs, vec_size):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        return self.load_impl(offset[0], vec_size=vec_size)

    def vec_store(self, idxs, value, vec_size):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        self.store_impl(offset[0], value, vec_size=vec_size)

    def linear_offset(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        return offset[0]

    def local_tile(self, tile_shape, tile_idxs):
        d_offset = self.base_offset
        stride = []
        for i in range_constexpr(len(tile_idxs)):
            d_offset = d_offset + tile_idxs[i] * tile_shape[i] * self.stride[i]
            stride.append(self.stride[i])
        return TensorView(
            self.dtype,
            tile_shape,
            tuple(stride),
            d_offset,
            self.load_impl,
            self.store_impl,
        )

    def copy_(self, src_tensor, thread_layout, value_layout, thread_idxs, vec_size):
        ndim = len(thread_layout)
        src_offset = src_tensor.base_offset
        dst_offset = self.base_offset
        for d in range_constexpr(ndim):
            src_offset = (
                src_offset + thread_idxs[d] * value_layout[d] * src_tensor.stride[d]
            )
            dst_offset = dst_offset + thread_idxs[d] * value_layout[d] * self.stride[d]
        value_layout_v = value_layout[:-1] + (value_layout[-1] // vec_size,)
        coords = tuple(product(*(range_constexpr(s) for s in value_layout_v)))
        for coord in coords:
            src_vec_offset = src_offset
            dst_vec_offset = dst_offset
            for d in range_constexpr(len(coord)):
                if d == len(coord) - 1:
                    src_vec_offset = (
                        src_vec_offset + coord[d] * src_tensor.stride[d] * vec_size
                    )
                    dst_vec_offset = (
                        dst_vec_offset + coord[d] * self.stride[d] * vec_size
                    )
                else:
                    src_vec_offset = src_vec_offset + coord[d] * src_tensor.stride[d]
                    dst_vec_offset = dst_vec_offset + coord[d] * self.stride[d]
            value = src_tensor.load_impl(src_vec_offset, vec_size=vec_size)
            self.store_impl(dst_vec_offset, value, vec_size=vec_size)


class TensorBase(TensorView, ABC):
    """A ``TensorView`` whose element access a subclass supplies.

    ``load``/``store`` are bound as the view's impls at construction, so every
    indexing method is inherited rather than forwarded.
    """

    def __init__(self, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset, self.load, self.store)

    @abstractmethod
    def load(self, offset):
        return None

    @abstractmethod
    def store(self, offset, value):
        pass


class TorchTensor(TensorBase):
    def __init__(self, torch_tensor, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.torch_tensor = torch_tensor

    def load(self, offset, vec_size=1):
        return self.torch_tensor.view(-1)[offset : offset + vec_size]

    def store(self, offset, value, vec_size=1):
        self.torch_tensor.view(-1)[offset : offset + vec_size] = value


class GTensor(TensorBase):
    def __init__(
        self,
        memref,
        dtype,
        shape,
        stride=None,
        base_offset=0,
        cache_modifier=0,
        static_bytes_offset_i64=None,
    ):
        super().__init__(dtype, shape, stride, base_offset)
        base = buf_base_i64(memref)
        if static_bytes_offset_i64 is not None:
            base = base + fx.Int64(static_bytes_offset_i64)
        self.base_i64 = base
        self.cache_modifier = cache_modifier

    def _view(self, elem, unit_elems):
        """Buffer view whose unit is one access.

        ``unit_stride=1`` because both offsets here are in *elements* while an
        access may be several elements wide (a vec4 load at offset N reads
        elements N..N+3), so units overlap and only element alignment holds.
        Built per access rather than cached: a cached view would be pinned to
        the block that first used it and not dominate later sibling blocks.
        """
        return ptr_buf_tensor(self.base_i64, elem, unit_elems=unit_elems, unit_stride=1)

    def load(self, offset, vec_size=1):
        elem = _fx_elem(self.dtype)
        t = self._view(elem, vec_size)
        if vec_size == 1:
            return t[offset]
        frag = fx.make_fragment_like(fx.slice(t, (0, None)))
        atom = buf_copy_atom(vec_size * (elem.width // 8), elem)
        fx.copy(atom, fx.slice(t, (offset, None)), frag)
        return fx.memref_load_vec(frag)

    def store(self, offset, value, vec_size=1):
        # Width comes from the value, not from self.dtype or vec_size: the
        # offset is scaled by the *stored* element, so a dword-typed store into
        # a bf16 view addresses dwords. Matches the buffer_store this replaces.
        elem, n = _fx_value_elem(value)
        t = self._view(elem, n)
        if n == 1:
            t[offset] = value
            return
        frag = fx.make_fragment_like(fx.slice(t, (0, None)))
        fx.memref_store_vec(value, frag)
        atom = buf_copy_atom(n * (elem.width // 8), elem, self.cache_modifier)
        fx.copy(atom, frag, fx.slice(t, (offset, None)))

    @property
    def rsrc(self):
        """The raw V#, for callers still issuing `buffer_ops` byte-offset ops."""
        return fx.rocdl.get_buffer_rsrc(
            fx.get_iter(self._view(_fx_elem(self.dtype), 1))
        )
