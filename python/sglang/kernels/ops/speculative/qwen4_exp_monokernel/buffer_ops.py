# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""AMD Buffer Load/Store Operations - High-level Python API

This module provides high-level Python wrappers for AMD CDNA3/CDNA4 buffer operations.
Buffer operations use a scalar base pointer and per-thread offsets for efficient memory access.

``create_buffer_resource`` returns a byte-addressed buffer-descriptor pointer

Example:
    >>> from sglang.kernels.ops.speculative.qwen4_exp_monokernel import buffer_ops
    >>> from flydsl.expr import arith
    >>> from flydsl._mlir.extras import types as T
    >>>
    >>> # Create buffer resource from memref
    >>> rsrc = buffer_ops.create_buffer_resource(A)
    >>>
    >>> # Compute offset
    >>> offset = row * arith.index(4096) + col
    >>>
    >>> # Buffer load (4xf32)
    >>> data = buffer_ops.buffer_load(rsrc, offset, vec_width=4)
    >>>
    >>> # Buffer store
    >>> buffer_ops.buffer_store(data, rsrc, offset)
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.extras import types as T
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.runtime.device import is_rdna_arch


def _get_buffer_flags(arch=None):
    """Get AMD buffer resource descriptor (V#) flags word (bits 127:96).

     ``flash_attn_utils`` builds a couple of descriptors by hand and still calls it.

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
    "BufferResourceDescriptor",
    "buffer_load",
    "buffer_store",
    "create_buffer_resource",
    "create_buffer_resource_from_addr",
    "create_llvm_ptr",
    "extract_base_index",
    "get_element_ptr",
]

_MAX_NUM_RECORDS = 0xFFFFFFFF


def _unwrap_value(value):
    """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value.

    Handles:
    - FlyDSL ArithValue (has ._value)
    - flyc DSL Numeric like fx.Int32 (has .ir_value() method)
    - flyc ArithValue (is already ir.Value subclass)
    - a bare single-result OpView (has .result), e.g. ``arith.TruncIOp(...)``
    """
    # DSL Numeric (Int32, Float32, etc.) — use ir_value() to materialize
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    max_depth = 10  # Safety limit
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "result"):
            value = value.result
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


def _as_num_records(num_records_bytes) -> fx.Int64 | None:
    """Normalize a descriptor byte count to ``fx.Int64`` for ``make_buffer_ptr``."""
    if num_records_bytes is None:
        return None
    if isinstance(num_records_bytes, int):
        num_records_bytes = max(0, min(int(num_records_bytes), _MAX_NUM_RECORDS))
    return fx.Int64(num_records_bytes)


@dsl_loc_tracing
def _byte_buffer_ptr(global_ptr, num_records_bytes, base_byte_offset=None):
    """Wrap a global pointer in a byte-addressed buffer-resource pointer."""
    ptr = fx.recast_iter(fx.Int8, global_ptr)
    if base_byte_offset is not None:
        if isinstance(base_byte_offset, ir.Value) and not isinstance(
            base_byte_offset.type, ir.IndexType
        ):
            base_byte_offset = fx.Int32(base_byte_offset)
        ptr = ptr + base_byte_offset
    return fx.rocdl.make_buffer_ptr(
        ptr, num_records_bytes=_as_num_records(num_records_bytes)
    )


@dsl_loc_tracing
def _add_soffset_bytes(byte_offset: ir.Value, soffset_bytes) -> ir.Value:
    """Fold ``soffset_bytes`` into the byte offset."""
    if soffset_bytes is None:
        return byte_offset
    return (fx.Int32(byte_offset) + fx.Int32(soffset_bytes)).ir_value()


@dsl_loc_tracing
def create_llvm_ptr(value, address_space: int = 0) -> ir.Value:
    """Create an LLVM pointer from an integer or index value."""
    value = _unwrap_value(value)
    if isinstance(value.type, ir.IndexType):
        value = fx.Int64(value).ir_value()
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return llvm.IntToPtrOp(ptr_type, value).result


@dsl_loc_tracing
def extract_base_index(tensor, address_space: int = 1) -> ir.Value:
    """Extract the base address of a fly.memref as an index value.

    Inverse of :func:`create_llvm_ptr` (index -> ptr). Useful when ISA
    requires a raw pointer instead of a buffer resource descriptor
    (e.g. global_atomic_pk_add_bf16 on gfx942).
    """
    from flydsl._mlir.dialects import fly as _fly
    from flydsl._mlir.dialects import memref as _memref

    raw = _unwrap_value(tensor)
    try:
        ir.MemRefType(raw.type)
        return _memref.extract_aligned_pointer_as_index(raw)
    except ValueError:
        pass

    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw)
    i64_val = llvm.PtrToIntOp(ir.IntegerType.get_signless(64), ptr).result
    return fx.Index(i64_val).ir_value()


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
            dtype = fx.Numeric.from_ir_type(offset_val.type)
            offset_val = (dtype(offset_val) + dtype(static_byte_offset)).ir_value()

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


def _num_records_from_memref_type(memref_val) -> int | None:
    try:
        mt = ir.MemRefType(_unwrap_value(memref_val).type)
        shape = list(mt.shape)
        if any(int(d) < 0 for d in shape):
            return None
        # Compute element size in bytes (scalar element type).
        elem_bits = getattr(mt.element_type, "width", None)
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


class BufferResourceDescriptor:
    """AMD Buffer Resource Descriptor

    A buffer resource descriptor contains:
    - base_pointer: Scalar base pointer (wave-uniform, stored in SGPRs)
    - stride: Stride for structured buffers (typically 0 for contiguous)
    - num_records: Buffer size in bytes
    - flags: Data format and access flags
    """

    def __init__(self, rsrc):
        """Initialize with the buffer-descriptor pointer."""
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
        return BufferResourceDescriptor(
            create_buffer_resource(
                memref_val,
                stride,
                max_size,
                num_records_bytes=num_records_bytes,
                base_byte_offset=base_byte_offset,
            )
        )


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
        Byte-addressed buffer-descriptor pointer (``!fly.ptr<i8, BufferDesc>``).

    Example:
        >>> rsrc = create_buffer_resource_from_addr(raw_addr_i64)
        >>> data = buffer_load(rsrc, i32_zero, vec_width=4, dtype=T.i32)
    """
    ptr_ty = fx.PointerType.get(
        T.i8(), address_space=fx.AddressSpace.Global, alignment=16
    )
    base_ptr = fx.inttoptr(ptr_ty, fx.Int64(_unwrap_value(addr_i64)))
    return _byte_buffer_ptr(base_ptr, num_records_bytes)


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

    Args:
        memref_val: Memref value
        stride: Buffer stride (0 for contiguous); only 0 is supported
        max_size: Use maximum buffer size
        num_records_bytes: Override buffer size in bytes.
        base_byte_offset: Optional byte offset added to the descriptor base pointer.

    Returns:
        Byte-addressed buffer-descriptor pointer (``!fly.ptr<i8, BufferDesc>``), for
        :func:`buffer_load` / :func:`buffer_store`. Pass it through
        :func:`flydsl.expr.rocdl.get_buffer_rsrc` to recover the raw ``!llvm.ptr<8>``
        resource that raw ROCDL intrinsics and inline asm expect.

    Example:
        >>> rsrc = create_buffer_resource(A)
        >>> data = buffer_load(rsrc, offset)
    """
    if stride != 0:
        raise ValueError(
            f"create_buffer_resource: only stride=0 (contiguous) is supported, got {stride}"
        )

    if num_records_bytes is None and not max_size:
        num_records_bytes = _num_records_from_memref_type(memref_val)

    return _byte_buffer_ptr(
        fx.get_iter(memref_val), num_records_bytes, base_byte_offset
    )


@dsl_loc_tracing
def buffer_load(
    rsrc: ir.Value,
    offset: ir.Value,
    vec_width: int = 4,
    dtype=None,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    soffset_bytes: int | ir.Value | None = None,
) -> ir.Value:
    """AMD buffer load operation.

    Load data from global memory using buffer descriptor and offset.
    Uses hardware-level bounds checking and vectorization.

    Args:
        rsrc: Buffer descriptor pointer from :func:`create_buffer_resource`
        offset: Offset in elements (i32 type)
        vec_width: Vector width (1, 2, or 4)
        dtype: Element data type (None for f32, or ir.F32Type, etc.)
        mask: Optional mask for predicated load (i1 type)
        cache_modifier: Cache control flags (0 for default)
        soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
                      Use this to fold small constant deltas into the instruction instead of emitting
                      extra VGPR address arithmetic.
    Returns:
        Loaded data (scalar or vector depending on vec_width)

    Example:
        >>> # Load 4xf32
        >>> data = buffer_load(rsrc, offset, vec_width=4)
        >>>
        >>> # Load with mask
        >>> data = buffer_load(rsrc, offset, vec_width=4, mask=valid)
    """
    if dtype is None:
        dtype = T.f32()
    # Accept DSL Numeric class (e.g. fx.Int32) as dtype: unwrap to ir.Type
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    offset = fx.Int32(_unwrap_value(offset)).ir_value()

    # Convert the API's element offset to the buffer instruction's byte offset.
    element_bytes = dtype.width // 8
    offset = (fx.Int32(offset) * fx.Int32(element_bytes)).ir_value()

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        offset = (
            fx.Boolean(mask).select(fx.Int32(offset), fx.Int32(0x7FFFFFFF)).ir_value()
        )

    result_type = dtype if vec_width == 1 else ir.VectorType.get([vec_width], dtype)

    offset = _add_soffset_bytes(offset, soffset_bytes)
    src = fx.make_view(
        rsrc + fx.Int32(offset), fx.make_layout(vec_width * element_bytes, 1)
    )
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(vec_width * dtype.width, cache_modifier), fx.Int8
    )
    reg = fx.make_rmem_tensor(
        fx.make_layout(vec_width, 1), fx.Numeric.from_ir_type(dtype)
    )
    fx.copy(atom, src, reg)
    loaded = fx.memref_load_vec(reg)
    result = _unwrap_value(loaded[0] if vec_width == 1 else loaded)
    if result.type != result_type:
        if vec_width == 1:
            result = (
                fx.Numeric.from_ir_type(result.type)(result)
                .bitcast(fx.Numeric.from_ir_type(dtype))
                .ir_value()
            )
        else:
            result = (
                fx.Vector(result).bitcast(fx.Numeric.from_ir_type(dtype)).ir_value()
            )
    return result


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
        rsrc: Buffer descriptor pointer from :func:`create_buffer_resource`
        offset: Offset in elements (i32 type)
        mask: Optional mask for predicated store (i1 type)
        cache_modifier: Cache control flags (0 for default)

    Example:
        >>> buffer_store(data, rsrc, offset)
        >>>
        >>> # Store with mask
        >>> buffer_store(data, rsrc, offset, mask=valid)
    """
    # Keep `rsrc` as a !fly.ptr<i8, BufferDesc> for the copy atom.
    data = _unwrap_value(data)
    offset = fx.Int32(_unwrap_value(offset)).ir_value()

    data_type = data.type
    if hasattr(data_type, "element_type"):  # Vector type
        element_type = data_type.element_type
        vec_width = data_type.shape[0]
    else:  # Scalar type
        element_type = data_type
        vec_width = 1
    element_bytes = element_type.width // 8

    # Convert element offsets unless the caller already supplied bytes.
    if not offset_is_bytes:
        offset = (fx.Int32(offset) * fx.Int32(element_bytes)).ir_value()

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        offset = (
            fx.Boolean(mask).select(fx.Int32(offset), fx.Int32(0x7FFFFFFF)).ir_value()
        )

    offset = _add_soffset_bytes(offset, soffset_bytes)
    dst = fx.make_view(
        rsrc + fx.Int32(offset), fx.make_layout(vec_width * element_bytes, 1)
    )
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(vec_width * element_type.width, cache_modifier), fx.Int8
    )
    reg = fx.make_rmem_tensor(
        fx.make_layout(vec_width, 1), fx.Numeric.from_ir_type(element_type)
    )
    fx.memref_store_vec(data, reg)
    fx.copy(atom, reg, dst)
