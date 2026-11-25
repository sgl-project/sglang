# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""AMD buffer load/store operations, vendored into aiter.

flydsl moved these from ``flydsl.expr.buffer_ops`` to its repo-level
``kernels/common/``, which its wheel does not ship, so aiter keeps an
equivalent copy here. Only published flydsl APIs are used.

Buffer instructions are an AMD hardware feature (buffer resource descriptor
plus ROCDL intrinsics) providing out-of-bounds protection and better memory
throughput; plain memref load/store is not a substitute.

Upstream: FlyDSL ``kernels/common/buffer_ops.py`` @ ROCm/FlyDSL#880, minus
``create_llvm_ptr`` (now ``kernels_common.create_llvm_ptr``, built on fx so the
backend resolves the address space). Everything kept behaves as upstream.

``buffer_load(is_scalar=True)`` stays here for kernels not yet moved to the
buffer-view API; migrated kernels should prefer ``tensor_shim.buf_scalar_load``.

Example:
    >>> from . import buffer_ops
    >>> from flydsl.expr import arith
    >>>
    >>> rsrc = buffer_ops.create_buffer_resource(A)
    >>> offset = row * arith.index(4096) + col
    >>> data = buffer_ops.buffer_load(rsrc, offset, vec_width=4)
    >>> buffer_ops.buffer_store(data, rsrc, offset)
"""

from __future__ import annotations

import inspect

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.extras import types as T
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.runtime.device import is_rdna_arch

# FlyDSL changed raw buffer cache policy from an i32 operand to an enum
# attribute. Keep this vendored compatibility layer usable with both forms.
_RAW_PTR_BUFFER_AUX_IS_ATTRIBUTE = (
    inspect.signature(fx.rocdl.RawPtrBufferLoadOp).parameters["aux"].kind
    is inspect.Parameter.KEYWORD_ONLY
)


def _get_buffer_flags(arch=None):
    """Get AMD buffer resource descriptor (V#) flags word (bits 127:96).

    Constructs the 32-bit flags field for rocdl.make.buffer.rsrc, following the
    same logic as LLVM's AMDGPUToROCDL makeBufferRsrc():
      https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp

    Bit layout (common to all architectures):
      bits [11:0]  - DST_SEL: ignored by raw buffer intrinsics
      bits [14:12] - DATA_FORMAT: must be nonzero, 7 = float
      bits [18:15] - NUM_FORMAT:  must be nonzero, 4 = 32-bit
      bit  [19]    - In nested heap (0)
      bit  [20]    - Behavior on unmap (0 = return 0 / ignore)
      bits [22:21] - Index stride for swizzles (0)
      bit  [23]    - Add thread ID (0)
      bit  [24]    - Reserved: must be 1 on RDNA, 0 on CDNA
      bits [26:25] - Reserved (0)
      bit  [27]    - Non-volatile (CDNA only, 0)
      bits [29:28] - OOB_SELECT (RDNA only): 0=structured, 2=none, 3=check offset
      bits [31:30] - Type (must be 0)

    CDNA (gfx9xx):    (7 << 12) | (4 << 15)                         = 0x20070
    RDNA (gfx10+):    (7 << 12) | (4 << 15) | (1 << 24) | (2 << 28) = 0x21020070
      - bit 24 set to 1 (required on RDNA)
      - OOB_SELECT=2 (no bounds checking, matching LLVM boundsCheck=false)
    """
    import os

    if arch is None:
        arch = os.environ.get("FLYDSL_GPU_ARCH")
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(arch):
        flags |= 1 << 24  # reserved bit, must be 1 on RDNA
        flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)
    return flags


__all__ = [
    "buffer_load",
    "buffer_store",
    "create_buffer_resource",
    "create_buffer_resource_from_addr",
    "get_element_ptr",
]


def _unwrap_value(value):
    """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value.

    Handles:
    - FlyDSL ArithValue (has ._value)
    - flyc DSL Numeric like fx.Int32 (has .ir_value() method)
    - flyc ArithValue (is already ir.Value subclass)
    """
    # DSL Numeric (Int32, Float32, etc.) — use ir_value() to materialize
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    max_depth = 10  # Safety limit
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


@dsl_loc_tracing
def _ptr8_to_v4i32(ptr8_val) -> ir.Value:
    """Reinterpret a buffer resource (!llvm.ptr<8>) as a <4 x i32> vector.

    Required by the scalar ``s.buffer.load`` intrinsic, whose resource operand is
    a v4i32 rather than the opaque buffer pointer used by the vector path.
    """
    i128_ty = ir.IntegerType.get_signless(128)
    v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    i128_val = llvm.ptrtoint(i128_ty, _unwrap_value(ptr8_val))
    return llvm.bitcast(v4i32_ty, i128_val)


@dsl_loc_tracing
def get_element_ptr(
    base_ptr,
    byte_offset: int | ir.Value | None = None,
    static_byte_offset: int = 0,
    elem_type: ir.Type | None = None,
    no_wrap_flags=None,
) -> ir.Value:
    """Build an LLVM GEP from a base pointer plus byte offsets."""
    _gep_dynamic_index_sentinel = -(2**31)

    base_ptr = _unwrap_value(base_ptr)
    if not isinstance(static_byte_offset, int):
        raise TypeError(
            f"static_byte_offset must be int, got {type(static_byte_offset).__name__}"
        )
    if elem_type is None:
        elem_type = T.i8()
    elif callable(elem_type):
        elem_type = elem_type()

    if byte_offset is None:
        dynamic_indices = []
        raw_constant_indices = [int(static_byte_offset)]
    elif isinstance(byte_offset, int):
        dynamic_indices = []
        raw_constant_indices = [int(byte_offset) + int(static_byte_offset)]
    else:
        offset_val = _unwrap_value(byte_offset)
        if isinstance(offset_val.type, ir.IndexType):
            offset_val = fx.Int64(offset_val).ir_value()
        elif not isinstance(offset_val.type, ir.IntegerType):
            raise TypeError(
                "byte_offset must be int, index, or integer-typed MLIR value; "
                f"got {offset_val.type}"
            )

        if static_byte_offset != 0:
            # GEP accepts arbitrary IR integer widths, including widths without
            # a corresponding fx scalar type.
            from flydsl._mlir.dialects import arith as std_arith

            static_type = offset_val.type
            static_attr = ir.IntegerAttr.get(static_type, int(static_byte_offset))
            static_const = _unwrap_value(
                std_arith.ConstantOp(static_type, static_attr).result
            )
            offset_val = _unwrap_value(
                std_arith.AddIOp(offset_val, static_const).result
            )

        dynamic_indices = [offset_val]
        raw_constant_indices = [_gep_dynamic_index_sentinel]

    return llvm.GEPOp(
        base_ptr.type,
        base_ptr,
        dynamic_indices,
        raw_constant_indices,
        elem_type,
        no_wrap_flags,
    ).result


class BufferResourceDescriptor:
    """AMD Buffer Resource Descriptor

    A buffer resource descriptor contains:
    - base_pointer: Scalar base pointer (wave-uniform, stored in SGPRs)
    - stride: Stride for structured buffers (typically 0 for contiguous)
    - num_records: Buffer size in bytes
    - flags: Data format and access flags

    The descriptor is stored in a special LLVM pointer type (!llvm.ptr<8>)
    """

    def __init__(self, rsrc: ir.Value):
        """Initialize with ROCDL resource descriptor value."""
        self.rsrc = rsrc

    @staticmethod
    @dsl_loc_tracing
    def from_memref(
        memref_val: ir.Value,
        stride: int = 0,
        max_size: bool = True,
        data_format: str = "f32",
        num_records_bytes: int | ir.Value | None = None,
        base_byte_offset: int | ir.Value | None = None,
    ) -> BufferResourceDescriptor:
        """Create buffer resource descriptor from memref.

        Args:
            memref_val: Memref value to create descriptor for
            stride: Stride in elements (0 for contiguous)
            max_size: If True, use max buffer size for flexibility
            num_records_bytes: Override buffer size (in BYTES) used by hardware OOB checking.
                              If provided, this takes precedence over `max_size`.
            base_byte_offset: Optional byte offset added to the descriptor base pointer.
            data_format: Data format ('f32', 'f16', 'i32', etc.)

        Returns:
            BufferResourceDescriptor instance

        Example:
            >>> rsrc = BufferResourceDescriptor.from_memref(A)
        """
        # Extract raw pointer from fly.memref.
        raw_val = _unwrap_value(memref_val)
        from flydsl._mlir.dialects import fly as _fly

        ptr_type = ir.Type.parse("!llvm.ptr")
        base_ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw_val)
        if base_byte_offset is not None:
            base_ptr = get_element_ptr(base_ptr, byte_offset=base_byte_offset)

        # Create buffer resource descriptor
        flags_val = _get_buffer_flags()
        flags = fx.Int32(flags_val).ir_value()
        stride_val = fx.Int16(stride).ir_value()

        def _num_records_from_memref_type() -> int | None:
            """Best-effort: derive logical buffer size (in bytes) from static memref type."""
            try:
                mt = ir.MemRefType(_unwrap_value(memref_val).type)
                shape = list(mt.shape)
                if any(int(d) < 0 for d in shape):
                    return None
                # Compute element size in bytes (scalar element type).
                elem_t = mt.element_type
                elem_bits = getattr(elem_t, "width", None)
                if elem_bits is None:
                    return None
                elem_bytes = int(elem_bits) // 8
                if elem_bytes <= 0:
                    return None
                num_elems = 1
                for d in shape:
                    num_elems *= int(d)
                return int(num_elems) * int(elem_bytes)
            # best-effort size probe: any failure just means "unknown"
            except Exception:  # noqa: BLE001
                return None

        if num_records_bytes is not None:
            # Caller-provided size in BYTES (preferred for exact hardware OOB behavior).
            if isinstance(num_records_bytes, int):
                nbytes = int(num_records_bytes)
                nbytes = max(0, nbytes)
                # Descriptor uses i32 bytes; clamp to the max representable.
                nbytes = min(nbytes, 0xFFFFFFFF)
                num_records = fx.Int64(nbytes).ir_value()
            else:
                num_records = fx.Int64(_unwrap_value(num_records_bytes)).ir_value()
        elif max_size:
            # Use max for flexibility (hardware will check actual bounds)
            # Note: FlyDSL's rocdl.make.buffer.rsrc requires i32, not i64
            num_records = fx.Int64(0xFFFFFFFF).ir_value()  # FALLBACK_MAX_SIZE
        else:
            # Use the logical memref size (in bytes) for hardware OOB checking.
            nbytes = _num_records_from_memref_type()
            if nbytes is None:
                # Fall back to max-size if we can't infer statically.
                num_records = fx.Int64(0xFFFFFFFF).ir_value()
            else:
                nbytes = min(nbytes, 0xFFFFFFFF)
                num_records = fx.Int64(int(nbytes)).ir_value()

        # Create resource descriptor (returns !llvm.ptr<8>)
        rsrc_type = ir.Type.parse("!llvm.ptr<8>")
        rsrc = fx.rocdl.make_buffer_rsrc(
            rsrc_type, base_ptr, stride_val, num_records, flags
        )

        return BufferResourceDescriptor(rsrc)


@dsl_loc_tracing
def create_buffer_resource_from_addr(
    addr_i64: ir.Value,
    *,
    num_records_bytes: int | ir.Value | None = None,
) -> ir.Value:
    """Create AMD buffer resource descriptor from a raw i64 device address.

    Useful when working with runtime pointer arrays (e.g. IPC-mapped addresses
    or device-side pointer tables) where no fly.memref is available.
    The full address is encoded as the buffer base; callers should pass
    byte offset 0 to buffer_load / buffer_store.

    Args:
        addr_i64: Raw 64-bit device address (i64 MLIR value).
        num_records_bytes: Optional buffer size in bytes for hardware OOB checking.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>).

    Example:
        >>> rsrc = create_buffer_resource_from_addr(raw_addr_i64)
        >>> data = buffer_load(rsrc, i32_zero, vec_width=4, dtype=T.i32)
    """
    addr_i64 = _unwrap_value(addr_i64)
    ptr_type = fx.PointerType.get(T.i8(), fx.AddressSpace.Generic, 1)
    base_ptr = fx.to_llvm_ptr(fx.inttoptr(ptr_type, addr_i64))
    flags = fx.Int32(_get_buffer_flags()).ir_value()
    stride = fx.Int16(0).ir_value()
    if num_records_bytes is None:
        num_records = fx.Int64(0xFFFFFFFF).ir_value()
    elif isinstance(num_records_bytes, int):
        nbytes = max(0, min(int(num_records_bytes), 0xFFFFFFFF))
        num_records = fx.Int64(nbytes).ir_value()
    else:
        num_records = fx.Int64(_unwrap_value(num_records_bytes)).ir_value()
    rsrc_type = ir.Type.parse("!llvm.ptr<8>")
    return fx.rocdl.make_buffer_rsrc(rsrc_type, base_ptr, stride, num_records, flags)


@dsl_loc_tracing
def create_buffer_resource(
    memref_val: ir.Value,
    stride: int = 0,
    max_size: bool = True,
    *,
    num_records_bytes: int | ir.Value | None = None,
    base_byte_offset: int | ir.Value | None = None,
) -> ir.Value:
    """Create AMD buffer resource descriptor from memref.

    This is a simplified wrapper around BufferResourceDescriptor.from_memref()
    that returns the raw ROCDL resource value.

    Args:
        memref_val: Memref value
        stride: Buffer stride (0 for contiguous)
        max_size: Use maximum buffer size
        num_records_bytes: Override buffer size in bytes.
        base_byte_offset: Optional byte offset added to the descriptor base pointer.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>)

    Example:
        >>> rsrc = create_buffer_resource(A)
        >>> data = buffer_load(rsrc, offset)
    """
    desc = BufferResourceDescriptor.from_memref(
        memref_val,
        stride,
        max_size,
        num_records_bytes=num_records_bytes,
        base_byte_offset=base_byte_offset,
    )
    return desc.rsrc


@dsl_loc_tracing
def buffer_load(
    rsrc: ir.Value,
    offset: ir.Value,
    vec_width: int = 4,
    dtype=None,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    soffset_bytes: int | ir.Value | None = None,
    is_scalar: bool = False,
) -> ir.Value:
    """AMD buffer load operation.

    Load data from global memory using buffer descriptor and offset.
    Uses hardware-level bounds checking and vectorization.

    Args:
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        vec_width: Vector width (1, 2, or 4)
        dtype: Element data type (None for f32, or ir.F32Type, etc.)
        mask: Optional mask for predicated load (i1 type)
        cache_modifier: Cache control flags (0 for default)
        soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
                      Use this to fold small constant deltas into the instruction instead of emitting
                      extra VGPR address arithmetic.
        is_scalar: Emit a uniform/SGPR scalar load (llvm.amdgcn.s.buffer.load) instead of the
                      vector buffer load. Use only for wave-uniform addresses to route through the
                      SMEM cache and land the result directly in SGPRs. Restricted to vec_width 1, 4, 8 or 16;
                      dtype is forced to i32 (the result is raw i32 dwords). mask and soffset_bytes
                      are not supported in this mode and raise ValueError if provided.

    Returns:
        Loaded data (scalar or vector depending on vec_width)

    Example:
        >>> # Load 4xf32
        >>> data = buffer_load(rsrc, offset, vec_width=4)
        >>>
        >>> # Load with mask
        >>> data = buffer_load(rsrc, offset, vec_width=4, mask=valid)
    """
    # Scalar (uniform) loads return raw i32 dwords; force the element type so the
    # element->byte offset math below uses 4 and the result type is i32 / v4i32.
    if is_scalar:
        if vec_width not in (1, 4, 8, 16):
            raise ValueError(
                f"buffer_load(is_scalar=True): unsupported vec_width={vec_width}"
            )
        if mask is not None or soffset_bytes is not None:
            raise ValueError(
                "buffer_load(is_scalar=True) does not support mask or soffset_bytes"
            )
        dtype = T.i32()
    # Default dtype to f32
    elif dtype is None:
        dtype = T.f32()
    # Accept DSL Numeric class (e.g. fx.Int32) as dtype: unwrap to ir.Type
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    # Buffer offsets truncate wider integers and sign-extend narrower ones.
    offset = fx.Int32(_unwrap_value(offset))

    # IMPORTANT: Buffer load offset is in BYTES, not elements!
    # For vec4xf32, each element is 4 bytes, so multiply offset by 4
    element_bytes = dtype.width // 8
    offset = offset * element_bytes

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        offset = fx.Boolean(_unwrap_value(mask)).select(offset, 0x7FFFFFFF)

    # Create vector type
    if vec_width == 1:
        result_type = dtype
    else:
        result_type = ir.VectorType.get([vec_width], dtype)

    # Scalar/uniform load path: emit s.buffer.load with a v4i32 resource and the
    # byte offset computed above. Returns i32 (vec_width 1) or v4i32 / v8i32.
    if is_scalar:
        rsrc_v4 = _ptr8_to_v4i32(rsrc)
        cache_policy = fx.Int32(cache_modifier).ir_value()
        suffix = "i32" if vec_width == 1 else f"v{vec_width}i32"
        return llvm.call_intrinsic(
            result_type,
            f"llvm.amdgcn.s.buffer.load.{suffix}",
            [rsrc_v4, offset.ir_value(), cache_policy],
            [],
            [],
        )

    # Create instruction offset and aux flags
    soffset = fx.Int32(
        0 if soffset_bytes is None else _unwrap_value(soffset_bytes)
    ).ir_value()
    aux = (
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)
        if _RAW_PTR_BUFFER_AUX_IS_ATTRIBUTE
        else fx.Int32(cache_modifier).ir_value()
    )

    # Emit buffer load
    return fx.rocdl.raw_ptr_buffer_load(
        result_type,
        rsrc,
        offset.ir_value(),
        soffset,
        aux=aux,
    )


@dsl_loc_tracing
def buffer_store(
    data: ir.Value,
    rsrc: ir.Value,
    offset: ir.Value,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    *,
    soffset_bytes: int | ir.Value | None = None,
    offset_is_bytes: bool = False,
):
    """AMD buffer store operation.

    Store data to global memory using buffer descriptor and offset.

    Args:
        data: Data to store (scalar or vector)
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        mask: Optional mask for predicated store (i1 type)
        cache_modifier: Cache control flags (0 for default)

    Example:
        >>> buffer_store(data, rsrc, offset)
        >>>
        >>> # Store with mask
        >>> buffer_store(data, rsrc, offset, mask=valid)
    """
    data = _unwrap_value(data)
    rsrc = _unwrap_value(rsrc)
    offset = fx.Int32(_unwrap_value(offset))

    # IMPORTANT: RawPtrBufferStoreOp offset is in BYTES.
    # For backward compat, `buffer_store()` accepts element offsets by default
    # and scales them to bytes. Set `offset_is_bytes=True` to skip scaling.
    if not offset_is_bytes:
        # Get element size from data type
        data_type = data.type
        if hasattr(data_type, "element_type"):  # Vector type
            element_type = data_type.element_type
        else:  # Scalar type
            element_type = data_type
        element_bytes = element_type.width // 8
        offset = offset * element_bytes

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        offset = fx.Boolean(_unwrap_value(mask)).select(offset, 0x7FFFFFFF)

    # Create instruction offset (soffset) and aux flags
    soffset = fx.Int32(
        0 if soffset_bytes is None else _unwrap_value(soffset_bytes)
    ).ir_value()
    aux = (
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)
        if _RAW_PTR_BUFFER_AUX_IS_ATTRIBUTE
        else fx.Int32(cache_modifier).ir_value()
    )

    # Emit buffer store
    fx.rocdl.raw_ptr_buffer_store(
        data,
        rsrc,
        offset.ir_value(),
        soffset,
        aux=aux,
    )
