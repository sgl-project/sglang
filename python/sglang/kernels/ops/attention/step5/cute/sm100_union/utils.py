# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import operator
import inspect
import math
from typing import Callable, Optional, Tuple, Type, Union
import torch
import cutlass
import cutlass.cute as cute

from cutlass import Float16, Float32, Int32, Int16, Int64, Uint8, Uint32
from cutlass.base_dsl.typing import Int128
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.runtime import from_dlpack as cute_from_dlpack
from torch import Tensor

_FLOAT8_ELEMENT_MAP = {}
if hasattr(torch, "float8_e4m3fn") and hasattr(cutlass, "Float8E4M3FN"):
    _FLOAT8_ELEMENT_MAP[torch.float8_e4m3fn] = cutlass.Float8E4M3FN
if hasattr(torch, "float8_e5m2") and hasattr(cutlass, "Float8E5M2"):
    _FLOAT8_ELEMENT_MAP[torch.float8_e5m2] = cutlass.Float8E5M2

SCALE_FORMAT_FLOAT32 = "float32"
SCALE_FORMAT_PACKED_UE8M0 = "packed_ue8m0"
SCALE_FORMAT_SM100_1D1D = "sm100_1d1d"
SCALE_FORMAT_FLOAT32_CODE = 0
SCALE_FORMAT_PACKED_UE8M0_CODE = 1
SCALE_FORMAT_SM100_1D1D_CODE = 2


def normalize_group_quant_scale_format(
    scale_format: Optional[str],
    *,
    use_packed_ue8m0: bool = False,
) -> str:
    if scale_format is None:
        return SCALE_FORMAT_PACKED_UE8M0 if use_packed_ue8m0 else SCALE_FORMAT_FLOAT32
    normalized = str(scale_format).lower()
    aliases = {
        "float": SCALE_FORMAT_FLOAT32,
        "fp32": SCALE_FORMAT_FLOAT32,
        "float32": SCALE_FORMAT_FLOAT32,
        "packed_ue8m0": SCALE_FORMAT_PACKED_UE8M0,
        "ue8m0_packed": SCALE_FORMAT_PACKED_UE8M0,
        "sm100_1d1d": SCALE_FORMAT_SM100_1D1D,
        "mn_major_tma_packed_ue8m0": SCALE_FORMAT_SM100_1D1D,
    }
    try:
        return aliases[normalized]
    except KeyError as err:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported scale_format={scale_format!r}; expected one of {supported}.") from err


def group_quant_scale_format_code(scale_format: str) -> int:
    if scale_format == SCALE_FORMAT_FLOAT32:
        return SCALE_FORMAT_FLOAT32_CODE
    if scale_format == SCALE_FORMAT_PACKED_UE8M0:
        return SCALE_FORMAT_PACKED_UE8M0_CODE
    if scale_format == SCALE_FORMAT_SM100_1D1D:
        return SCALE_FORMAT_SM100_1D1D_CODE
    raise ValueError(f"Unsupported scale_format={scale_format!r}.")


def _tma_aligned_size(x: int, element_size: int) -> int:
    return ((x + (16 // element_size) - 1) // (16 // element_size)) * (16 // element_size)


def make_group_quant_packed_ue8m0_scale_tensor(
    rows: int,
    groups_per_row: int,
    *,
    device: torch.device,
    scale_format: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    packed_groups = (groups_per_row + 3) // 4
    if scale_format == SCALE_FORMAT_PACKED_UE8M0:
        padded_groups = packed_groups * 4
        storage = torch.empty((rows, padded_groups), device=device, dtype=torch.uint8)
        if padded_groups != groups_per_row:
            storage.zero_()
        return storage.view(torch.int32), storage, 1

    if scale_format == SCALE_FORMAT_SM100_1D1D:
        aligned_rows = _tma_aligned_size(rows, 4)
        storage = torch.zeros(
            (packed_groups, aligned_rows, 4),
            device=device,
            dtype=torch.uint8,
        )
        scales = storage.view(torch.int32).reshape(packed_groups, aligned_rows).mT[:rows, :]
        return scales, storage, 2

    raise ValueError(f"Unsupported packed UE8M0 scale_format={scale_format!r}.")


def _normalize_dlpack_tensor(x: Optional[Tensor]) -> Optional[Tensor]:
    if x is None:
        return None
    # Note(wangbojun/codex): CuTe only consumes the tensor storage plus layout
    # metadata here. Only grad-tracked tensors need detaching to satisfy
    # PyTorch's DLPack autograd guard without changing the underlying device
    # buffer.
    if x.requires_grad:
        return x.detach()
    return x


def _to_cute_tensor_from_raw(
    raw_x: Tensor,
    *,
    leading_dim: int,
    alignment: int = 16,
    enable_tvm_ffi: bool = True,
) -> cute.Tensor:
    element_override = _FLOAT8_ELEMENT_MAP.get(getattr(raw_x, "dtype", None))
    dlpack_source = raw_x.view(torch.uint8) if element_override is not None else raw_x
    kwargs = {"assumed_align": alignment}
    if enable_tvm_ffi:
        kwargs["enable_tvm_ffi"] = True
    tensor = cute_from_dlpack(dlpack_source, **kwargs).mark_layout_dynamic(
        leading_dim=leading_dim
    )
    if element_override is not None:
        tensor.element_type = element_override
    return tensor


def _from_dlpack_layout(
    x: Tensor,
    *,
    leading_dim: int,
    alignment: int = 16,
    enable_tvm_ffi: bool = True,
) -> tuple[Tensor, cute.Tensor]:
    raw_x = _normalize_dlpack_tensor(x)
    assert raw_x is not None
    return raw_x, _to_cute_tensor_from_raw(
        raw_x,
        leading_dim=leading_dim,
        alignment=alignment,
        enable_tvm_ffi=enable_tvm_ffi,
    )


def _mark_compact_dynamic(
    tensor: cute.Tensor,
    raw_x: Tensor,
    *,
    leading_dim: int,
    divisibility: int = 1,
) -> cute.Tensor:
    use_inferred_stride_order = (
        raw_x.ndim == 2 and ((raw_x.stride(0) == 1) ^ (raw_x.stride(1) == 1))
    )
    try:
        if use_inferred_stride_order:
            return tensor.mark_compact_shape_dynamic(
                mode=leading_dim,
                divisibility=divisibility,
            )
        return tensor.mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=raw_x.dim_order(),
            divisibility=divisibility,
        )
    except RuntimeError as err:
        msg = str(err)
        if use_inferred_stride_order and "could not be deduced" in msg:
            return tensor.mark_compact_shape_dynamic(
                mode=leading_dim,
                stride_order=raw_x.dim_order(),
                divisibility=divisibility,
            )
        if "stride_order" not in msg:
            raise
        # Some views (e.g., column slices) have a valid stride layout but violate
        # the compact stride-order check. Retry without forcing a stride ordering
        # so we can preserve the original strides for the kernel.
        return tensor.mark_compact_shape_dynamic(
            mode=leading_dim,
            divisibility=divisibility,
        )


class _RuntimeLayoutTensorArg:
    def __init__(
        self,
        tensor: Tensor,
        *,
        leading_dim: int,
        alignment: int = 16,
        enable_tvm_ffi: bool = True,
    ) -> None:
        self._tensor = _normalize_dlpack_tensor(tensor)
        self._leading_dim = leading_dim
        self._alignment = alignment
        self._enable_tvm_ffi = enable_tvm_ffi
        self._cute_tensor = None

    def _wrapped(self):
        if self._cute_tensor is None:
            self._cute_tensor = _to_cute_tensor_from_raw(
                self._tensor,
                leading_dim=self._leading_dim,
                alignment=self._alignment,
                enable_tvm_ffi=self._enable_tvm_ffi,
            )
        return self._cute_tensor

    def __c_pointers__(self):
        return self._wrapped().__c_pointers__()

    def __get_mlir_types__(self):
        return self._wrapped().__get_mlir_types__()

    def __tvm_ffi_object__(self):
        return self._wrapped().__tvm_ffi_object__()


def runtime_layout_tensor_arg(
    x: Tensor,
    *,
    leading_dim: int,
    alignment: int = 16,
    enable_tvm_ffi: bool = True,
) -> _RuntimeLayoutTensorArg:
    return _RuntimeLayoutTensorArg(
        x,
        leading_dim=leading_dim,
        alignment=alignment,
        enable_tvm_ffi=enable_tvm_ffi,
    )


def tvm_ffi_tensor_spec(
    x: Tensor,
    *,
    leading_dim: int,
    alignment: int = 16,
) -> cute.Tensor:
    # Note(wangbojun/codex): Keep the compile-time TVM-FFI tensor ABI in one
    # place so the CuTeDSL kernels stop duplicating the same wrapper boilerplate.
    _, tensor = _from_dlpack_layout(
        x,
        leading_dim=leading_dim,
        alignment=alignment,
        enable_tvm_ffi=True,
    )
    return tensor


def convert_from_dlpack(
    x,
    leading_dim,
    alignment=16,
    divisibility=1,
    *,
    compact=True,
) -> cute.Tensor:
    raw_x, tensor = _from_dlpack_layout(
        x,
        leading_dim=leading_dim,
        alignment=alignment,
        enable_tvm_ffi=True,
    )
    if not compact:
        return tensor
    return _mark_compact_dynamic(
        tensor,
        raw_x,
        leading_dim=leading_dim,
        divisibility=divisibility,
    )


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    val = cute.make_fragment(1, type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(val_i32[i], offset, mask_and_clamp=mask_and_clamp)
    return val[0]


@cute.jit
def block_reduce(
    val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, init_val: cute.Numeric = 0.0
) -> cute.Numeric:
    """reduction_buffer has shape (num_warps / warp_per_row, warps_per_row)"""
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def load_int64(
    tensor: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> Int64:
    ptr = elem_pointer(tensor, coord, loc=loc, ip=ip).align(8)
    int_ptr = cute.make_ptr(
        Int64,
        ptr.toint(),
        tensor.memspace,
        assumed_align=min(ptr.max_alignment, 8),
    )
    src = cute.make_tensor(int_ptr, cute.make_layout((1,), stride=(1,)))
    return src[0]


@cute.jit
def load_int64x2(
    tensor: cute.Tensor,
    coord: cute.Coord,
) -> tuple[Int64, Int64]:
    ptr = elem_pointer(tensor, coord).align(16)
    int_ptr = cute.make_ptr(
        Int64,
        ptr.toint(),
        tensor.memspace,
        assumed_align=min(ptr.max_alignment, 16),
    )
    src = cute.make_tensor(int_ptr, cute.make_layout((2,), stride=(1,)))
    vals = src.load()
    return Int64(vals[0]), Int64(vals[1])


@dsl_user_op
def store_int128(
    tensor: cute.Tensor,
    coord: cute.Coord,
    packed: Int128,
    *,
    loc=None,
    ip=None,
) -> None:
    ptr = elem_pointer(tensor, coord, loc=loc, ip=ip).align(16)
    int_ptr = cute.make_ptr(
        Int128,
        ptr.toint(),
        tensor.memspace,
        assumed_align=min(ptr.max_alignment, 16),
    )
    dst = cute.make_tensor(int_ptr, cute.make_layout((1,), stride=(1,)))
    dst[0] = packed


@dsl_user_op
def pack_int64x2_to_int128(
    v0: Int64,
    v1: Int64,
    *,
    loc=None,
    ip=None,
) -> Int128:
    vec_i64 = vector.from_elements(
        T.vector(2, T.i64()),
        (
            Int64(v0).ir_value(loc=loc, ip=ip),
            Int64(v1).ir_value(loc=loc, ip=ip),
        ),
        loc=loc,
        ip=ip,
    )
    vec_i128 = vector.bitcast(T.vector(1, T.i(128)), vec_i64)
    packed = vector.extract(vec_i128, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    return Int128(packed)


@dsl_user_op
def pack_uint8x2_to_int16(
    val_lo: Uint8, val_hi: Uint8, *, loc=None, ip=None
) -> Int16:
    vec_u8 = vector.from_elements(
        T.vector(2, T.i8()),
        (
            Uint8(val_lo).ir_value(loc=loc, ip=ip),
            Uint8(val_hi).ir_value(loc=loc, ip=ip),
        ),
        loc=loc,
        ip=ip,
    )
    vec_i16 = vector.bitcast(T.vector(1, T.i16()), vec_u8)
    packed = vector.extract(vec_i16, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    return Int16(packed)


@dsl_user_op
def bitcast_f32_to_u32(v: Float32, *, loc=None, ip=None) -> Uint32:
    vec_f32 = vector.from_elements(
        T.vector(1, T.f32()),
        (Float32(v).ir_value(loc=loc, ip=ip),),
        loc=loc,
        ip=ip,
    )
    vec_i32 = vector.bitcast(T.vector(1, T.i32()), vec_f32)
    bits = vector.extract(vec_i32, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    return Uint32(bits)


@dsl_user_op
def bitcast_u32_to_f32(v: Uint32, *, loc=None, ip=None) -> Float32:
    vec_i32 = vector.from_elements(
        T.vector(1, T.i32()),
        (Uint32(v).ir_value(loc=loc, ip=ip),),
        loc=loc,
        ip=ip,
    )
    vec_f32 = vector.bitcast(T.vector(1, T.f32()), vec_i32)
    out = vector.extract(vec_f32, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    return Float32(out)


@dsl_user_op
def pack_int16x4_to_int64(
    v0: Int16, v1: Int16, v2: Int16, v3: Int16, *, loc=None, ip=None
) -> Int64:
    vec_i16 = vector.from_elements(
        T.vector(4, T.i16()),
        (
            Int16(v0).ir_value(loc=loc, ip=ip),
            Int16(v1).ir_value(loc=loc, ip=ip),
            Int16(v2).ir_value(loc=loc, ip=ip),
            Int16(v3).ir_value(loc=loc, ip=ip),
        ),
        loc=loc,
        ip=ip,
    )
    vec_i64 = vector.bitcast(T.vector(1, T.i64()), vec_i16)
    packed = vector.extract(vec_i64, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    return Int64(packed)


@dsl_user_op
def store_fp8x8(
    tensor: cute.Tensor, coord: cute.Coord, packed: Int64, *, loc=None, ip=None
) -> None:
    ptr = elem_pointer(tensor, coord, loc=loc, ip=ip).align(8)
    int_ptr = cute.make_ptr(
        Int64,
        ptr.toint(),
        tensor.memspace,
        assumed_align=min(ptr.max_alignment, 8),
    )
    dst = cute.make_tensor(int_ptr, cute.make_layout((1,), stride=(1,)))
    dst[0] = packed


@cute.jit
def pack_fp32x8_to_fp8x8(
    lane_vals: cute.Tensor,
    inv_scale: Float32,
    fp8_min: Float32,
    fp8_max: Float32,
) -> Int64:
    # Note(wangbojun/codex): Keep the repeated 8-lane FP8 pack path in one
    # place without hiding the surrounding load/scale flow that still differs
    # across kernels.
    packed_pairs = cute.make_rmem_tensor((4,), Int16)
    for pair in cutlass.range_constexpr(4):
        idx0 = pair * 2
        idx1 = idx0 + 1
        val0 = Float32(lane_vals[idx0]) * inv_scale
        val0 = cute.arch.fmax(-fp8_max, -val0)
        val0 = -val0
        val0 = cute.arch.fmax(fp8_min, val0)
        val1 = Float32(lane_vals[idx1]) * inv_scale
        val1 = cute.arch.fmax(-fp8_max, -val1)
        val1 = -val1
        val1 = cute.arch.fmax(fp8_min, val1)
        packed_pairs[pair] = cvt_fp32x2_to_e4m3x2(val1, val0)
    return pack_int16x4_to_int64(
        packed_pairs[0],
        packed_pairs[1],
        packed_pairs[2],
        packed_pairs[3],
    )


# Layout contract shared with the 128-wide group-quant kernels: one quant group
# is covered by a half warp (16 lanes x 8 values), element index within the
# group is `lane_in_half * 8 + elem` (bit0-2 in-register, bit3-6 across lanes).
_FWHT_VALS_PER_THREAD = 8
_FWHT_HALF_WARP = 16
_FWHT_MAX_LOG2_GROUP_SIZE = 7  # group <= 128 stays inside one half warp
_FWHT_HALF_WARP_MASK_AND_CLAMP = ((cute.arch.WARP_SIZE - _FWHT_HALF_WARP) << 8) | (
    _FWHT_HALF_WARP - 1
)

_FWHT_FRAGMENT_CACHE: dict = {}


def make_fwht_halfwarp_fragment(log2_group_size: int) -> Callable:
    """Build an in-place unnormalized FWHT over an 8-value fp32 fragment.

    Butterfly network for a Hadamard group of size 2**log2_group_size laid out
    as `idx = lane_in_half * 8 + elem`: the first min(3, k) stages are
    in-register add/sub pairs, the remaining k-3 stages exchange values with
    `lane ^ (1 << j)` via shuffle_sync_bfly. The caller must fold the g**-0.5
    normalization into its own scale computation. The high-side sign flip is
    done by XOR-ing the fp32 sign bit so every lane executes the same
    instruction stream (no divergence around the shuffles).
    """
    assert 1 <= log2_group_size <= _FWHT_MAX_LOG2_GROUP_SIZE, (
        f"log2_group_size={log2_group_size} outside supported range "
        f"[1, {_FWHT_MAX_LOG2_GROUP_SIZE}] (hadamard group must fit a half warp)."
    )
    cached = _FWHT_FRAGMENT_CACHE.get(log2_group_size)
    if cached is not None:
        return cached

    num_inreg_stages = min(3, log2_group_size)
    num_xlane_stages = max(0, log2_group_size - 3)

    @cute.jit
    def _fwht_halfwarp_fragment(lane_vals: cute.Tensor, lane_in_half: Int32) -> None:
        for stage in cutlass.range_constexpr(num_inreg_stages):
            stride = 1 << stage
            for elem in cutlass.range_constexpr(_FWHT_VALS_PER_THREAD):
                if (elem & stride) == 0:
                    low = Float32(lane_vals[elem])
                    high = Float32(lane_vals[elem + stride])
                    lane_vals[elem] = low + high
                    lane_vals[elem + stride] = low - high

        if num_xlane_stages > 0:
            mask_and_clamp = Int32(_FWHT_HALF_WARP_MASK_AND_CLAMP)
            full_mask = Int32(-1)
            for stage in cutlass.range_constexpr(num_xlane_stages):
                offset = 1 << stage
                # +0.0 on the low side, -0.0 on the high side: XOR-ing this
                # into the fp32 bits maps val -> +/-val without a branch.
                lane_bit = (Uint32(lane_in_half) >> Uint32(stage)) & Uint32(1)
                sign_mask = lane_bit << Uint32(31)
                for elem in cutlass.range_constexpr(_FWHT_VALS_PER_THREAD):
                    val = Float32(lane_vals[elem])
                    other = cute.arch.shuffle_sync_bfly(
                        val,
                        offset=Int32(offset),
                        mask=full_mask,
                        mask_and_clamp=mask_and_clamp,
                    )
                    flipped = bitcast_u32_to_f32(bitcast_f32_to_u32(val) ^ sign_mask)
                    lane_vals[elem] = flipped + other

    _FWHT_FRAGMENT_CACHE[log2_group_size] = _fwht_halfwarp_fragment
    return _fwht_halfwarp_fragment


def make_group_quant_scale_tensor(
    rows: int,
    groups_per_row: int,
    *,
    device: torch.device,
    column_major_scales: bool,
) -> tuple[torch.Tensor, int]:
    if column_major_scales:
        return (
            torch.empty_strided(
                (rows, groups_per_row),
                (1, rows),
                device=device,
                dtype=torch.float32,
            ),
            0,
        )
    return (
        torch.empty((rows, groups_per_row), device=device, dtype=torch.float32),
        1,
    )


@dsl_user_op
def atomic_add_fp32(a: Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
            Float32(a).ir_value(loc=loc, ip=ip),
        ],
        "red.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def atomic_max_nonnegative_f32(a: Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    # Non-negative finite f32 values have the same ordering as their u32 bit patterns.
    llvm.inline_asm(
        None,
        [gmem_ptr.toint(loc=loc, ip=ip).ir_value(), bitcast_f32_to_u32(a).ir_value()],
        "red.global.max.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: cute.Int32, *, loc=None, ip=None
) -> cutlass.Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: float | Float32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    if cutlass.const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, cutlass.Int64)), "val must be Float32 or Int64"
    suffix = "f32" if cutlass.const_expr(isinstance(val, Float32)) else "s64"
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{'f' if cutlass.const_expr(isinstance(val, Float32)) else 'l'},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: Optional[cutlass.Int32] = None,
) -> cute.Numeric:
    """reduction_buffer has shape (num_warps / warps_per_row, (warps_per_row, cluster_n))"""
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                num_warps * cluster_n * reduction_buffer.element_type.width // 8,
            )
    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])
    return warp_reduce(block_reduce_val, op)


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[cutlass.Int32] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Perform either block or cluster reduction based on whether mbar_ptr is provided."""
    if cutlass.const_expr(mbar_ptr is None):
        return block_reduce(val, op, reduction_buffer, init_val=init_val)
    else:
        return cluster_reduce(val, op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val)


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[cutlass.Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
) -> cute.Numeric:
    """reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n))"""
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax if cutlass.const_expr(x.dtype == Float32) else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = warp_reduce(
        val,
        warp_op,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if cutlass.const_expr(hook_fn is not None):
        hook_fn()
    if cutlass.const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        assert (
            cluster_n == 1 or mbar_ptr is not None
        ), "mbar_ptr must be provided for cluster reduction"
        if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
            val = block_or_cluster_reduce(
                val, warp_op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val
            )
    return val


@cute.jit
def online_softmax_reduce(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    hook_fn: Optional[Callable] = None,
    phase: Optional[cutlass.Int32] = None,
    return_exp_x: bool = False,
) -> [Float32, Float32, Optional[cute.TensorSSA]]:
    assert x.dtype == Float32, "x must be of type Float32"
    """reduction_buffer must have shape (num_warps / warps_per_row, (warps_per_row, cluster_n), 2)"""
    max_x = warp_reduce(
        x.reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
        cute.arch.fmax,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    log2_e = math.log2(math.e)
    exp_x = exp2f(x * log2_e - (max_x * log2_e))
    # exp_x = exp2f((x - max_x) * log2_e)
    sum_exp_x = warp_reduce(
        exp_x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
        operator.add,
        width=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if cutlass.const_expr(hook_fn is not None):
        hook_fn()
    if cutlass.const_expr(reduction_buffer is not None):
        rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
        assert (
            cluster_n == 1 or mbar_ptr is not None
        ), "mbar_ptr must be provided for cluster reduction"
        if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
            assert (
                reduction_buffer.element_type == cutlass.Int64
            ), "reduction_buffer must be of type cute.Int64"
            lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
            row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
            if cutlass.const_expr(mbar_ptr is None):
                if lane_idx == 0:
                    reduction_buffer[row_idx, col_idx] = f32x2_to_i64(max_x, sum_exp_x)
                cute.arch.barrier()
                max_x_single_warp = -Float32.inf
                sum_exp_x = 0.0
                if lane_idx < warps_per_row:
                    max_x_single_warp, sum_exp_x = i64_to_f32x2(reduction_buffer[row_idx, lane_idx])
                max_x_final = warp_reduce(max_x_single_warp, cute.arch.fmax)
                sum_exp_x *= exp2f((max_x_single_warp - max_x_final) * log2_e)
                sum_exp_x = warp_reduce(sum_exp_x, operator.add)
                if cutlass.const_expr(return_exp_x):
                    exp_x *= exp2f((max_x - max_x_final) * log2_e)
                max_x = max_x_final
            else:
                cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        num_warps = rows_per_block * warps_per_row
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr,
                            num_warps * cluster_n * reduction_buffer.element_type.width // 8,
                        )
                if lane_idx < cluster_n:
                    store_shared_remote(
                        f32x2_to_i64(max_x, sum_exp_x),
                        elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
                        mbar_ptr,
                        peer_cta_rank_in_cluster=lane_idx,
                    )
                cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
                num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
                max_x_single_warp = cute.make_fragment(num_iter, Float32)
                max_x_single_warp.fill(-Float32.inf)
                sum_exp_x_single_warp = cute.make_fragment(num_iter, Float32)
                sum_exp_x_single_warp.fill(0.0)
                for i in cutlass.range_constexpr(num_iter):
                    idx = lane_idx + i * cute.arch.WARP_SIZE
                    if idx < cute.size(reduction_buffer, mode=[1]):
                        max_x_single_warp[i], sum_exp_x_single_warp[i] = i64_to_f32x2(
                            reduction_buffer[row_idx, idx]
                        )
                max_x_final = max_x_single_warp.load().reduce(
                    cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0
                )
                max_x_final = warp_reduce(max_x_final, cute.arch.fmax)
                sum_exp_x = 0.0
                for i in cutlass.range_constexpr(num_iter):
                    sum_exp_x += sum_exp_x_single_warp[i] * exp2f(
                        (max_x_single_warp[i] - max_x_final) * log2_e
                    )
                sum_exp_x = warp_reduce(sum_exp_x, operator.add)
                if cutlass.const_expr(return_exp_x):
                    exp_x *= exp2f((max_x - max_x_final) * log2_e)
                max_x = max_x_final
    return max_x, sum_exp_x, (exp_x if cutlass.const_expr(return_exp_x) else None)


@dsl_user_op
def fmin(a: Union[float, Float32], b: Union[float, Float32], *, loc=None, ip=None) -> Float32:
    a_ir = Float32(a).ir_value(loc=loc, ip=ip)
    b_ir = Float32(b).ir_value(loc=loc, ip=ip)
    positional_params = tuple(
        p
        for p in inspect.signature(nvvm.fmin).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    if len(positional_params) >= 3:
        return Float32(nvvm.fmin(T.f32(), a_ir, b_ir, loc=loc, ip=ip))
    return Float32(
        nvvm.fmin(
            a_ir,
            b_ir,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def exp2f(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    """exp2f calculation for both vector and scalar.
    :param x: input value
    :type x: cute.TensorSSA or Float32
    :return: exp2 value
    :rtype: cute.TensorSSA or Float32
    """
    if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        for i in cutlass.range(cute.size(x.shape), unroll_full=True):
            res[i] = cute.arch.exp2(res[i])
        return res.load()
    else:
        return cute.arch.exp2(x)


@dsl_user_op
def log2f(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def sqrt(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "sqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def rsqrt(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ceil(a: float | Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rpi.ftz.s32.f32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def silu(a: float | Float32, *, loc=None, ip=None) -> Float32:
    """
    silu(a) = a * sigmoid(a) = a * (1 + tanh(a / 2)) / 2 = (0.5 * a) * tanh(0.5 * a) + (0.5 * a)
    This compiles down to 3 SASS instructions: FMUL to get 0.5 * a, MUFU.TANH, and FFMA.
    """
    a_half = 0.5 * a
    return a_half * tanh(a_half) + a_half


@dsl_user_op
def prmt(a: int | Int32, b: int | Int32, c: int | Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                Int32(c).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def permute_gated_Cregs_b16(t: cute.Tensor) -> None:
    assert t.element_type.width == 16
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b16 permutation"
    t_u32 = cute.recast_tensor(t, Int32)

    quad_idx = cute.arch.lane_idx() % 4
    lane_03 = quad_idx == 0 or quad_idx == 3
    selector_upper = Int32(0x5410) if lane_03 else Int32(0x1054)
    selector_lower = Int32(0x7632) if lane_03 else Int32(0x3276)
    # upper_map = [0, 3, 1, 2]
    # lower_map = [1, 2, 0, 3]
    # upper_idx = upper_map[quad_idx]
    # indexing isn't supported so we have to do arithmetic
    upper_idx = quad_idx // 2 if quad_idx % 2 == 0 else 3 - quad_idx // 2
    lower_idx = upper_idx ^ 1

    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range(cute.size(t_u32.shape) // 2, unroll_full=True):
        upper, lower = t_u32[i * 2 + 0], t_u32[i * 2 + 1]
        upper0 = upper if lane_03 else lower
        lower0 = lower if lane_03 else upper
        upper0 = cute.arch.shuffle_sync(upper0, offset=upper_idx, mask_and_clamp=mask_and_clamp)
        lower0 = cute.arch.shuffle_sync(lower0, offset=lower_idx, mask_and_clamp=mask_and_clamp)
        t_u32[i * 2 + 0] = prmt(upper0, lower0, selector_upper)
        t_u32[i * 2 + 1] = prmt(upper0, lower0, selector_lower)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_fragment(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    """Fill out-of-bounds values in shared memory tensor.

    Args:
        tXsX: Shared memory tensor to fill
        tXpX: Predicate tensor indicating valid elements
        fill_value: Value to fill OOB locations with
    """
    tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), 0, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if cutlass.const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


@dsl_user_op
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int64:
    vec_f32x2 = vector.from_elements(
        T.vector(2, T.f32()), (a.ir_value(), b.ir_value()), loc=loc, ip=ip
    )
    vec_i64x1 = vector.bitcast(T.vector(1, T.i64()), vec_f32x2)
    res = cutlass.Int64(
        vector.extract(vec_i64x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    return res


@dsl_user_op
def i64_to_f32x2(c: cutlass.Int64, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    vec_i64x1 = vector.from_elements(T.vector(1, T.i64()), (c.ir_value(),), loc=loc, ip=ip)
    vec_f32x2 = vector.bitcast(T.vector(2, T.f32()), vec_i64x1)
    res0 = Float32(
        vector.extract(vec_f32x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    res1 = Float32(
        vector.extract(vec_f32x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    )
    return res0, res1


@dsl_user_op
def cvt_fp32x2_to_e4m3x2(
    val_hi: Float32, val_lo: Float32, *, loc=None, ip=None
) -> cutlass.Int16:
    packed = llvm.inline_asm(
        T.i16(),
        [Float32(val_hi).ir_value(loc=loc, ip=ip), Float32(val_lo).ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int16(packed)


@dsl_user_op
def cvt_e4m3x2_to_f32x2(
    packed: Int16, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    packed_f16x2 = llvm.inline_asm(
        Int32.mlir_type,
        [Int16(packed).ir_value(loc=loc, ip=ip)],
        """{\n\t
            .reg .b16 b;\n\t
            mov.b16 b, $1;\n\t
            cvt.rn.f16x2.e4m3x2 $0, b;\n\t
        }""",
        "=r,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    vec_f16x2 = llvm.bitcast(
        ir.VectorType.get([2], Float16.mlir_type, loc=loc),
        packed_f16x2,
        loc=loc,
        ip=ip,
    )
    val_lo_f16 = Float16(
        vector.extract(vec_f16x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    val_hi_f16 = Float16(
        vector.extract(vec_f16x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    )
    return Float32(val_lo_f16), Float32(val_hi_f16)


@dsl_user_op
def store_fp8x2(
    tensor: cute.Tensor, coord: cute.Coord, packed: cutlass.Int16, *, loc=None, ip=None
) -> None:
    ptr = elem_pointer(tensor, coord, loc=loc, ip=ip)
    int_ptr = cute.make_ptr(
        cutlass.Int16,
        ptr.toint(),
        tensor.memspace,
        assumed_align=min(ptr.max_alignment, 2),
    )
    dst = cute.make_tensor(int_ptr, cute.make_layout((1,), stride=(1,)))
    dst[0] = packed


@dsl_user_op
def domain_offset_i64(coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(
        flat_stride
    ), "Coordinate and stride must have the same length"
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def coord_offset_i64(
    idx: cute.typing.Int, tensor: cute.Tensor, dim: int, *, loc=None, ip=None
) -> cute.Tensor:
    offset = cutlass.Int64(idx) * cute.size(tensor.stride[dim])
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@cute.jit
def warp_prefix_sum(val: cutlass.Int32, lane: Optional[cutlass.Int32] = None) -> cutlass.Int32:
    if cutlass.const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val


def convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
            (
                acc_layout_col_major.shape[0][0],
                *acc_layout_col_major.shape[0][2:],
                acc_layout_col_major.shape[2],
            ),  # MMA_N
            *acc_layout_col_major.shape[3:],
        ),
        stride=(
            (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
            (
                acc_layout_col_major.stride[0][0],
                *acc_layout_col_major.stride[0][2:],
                acc_layout_col_major.stride[2],
            ),  # MMA_N
            *acc_layout_col_major.stride[3:],
        ),
    )
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))


@dsl_user_op
def sm90_get_smem_load_op(
    layout_c: cutlass.utils.LayoutEnum,
    elem_ty_c: Type[cutlass.Numeric],
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    """
    Selects the largest vectorized smem load atom available subject to constraint of gmem layout.

    Parameters:
    -----------
    layout_c : LayoutEnum
        The layout enum of the output tensor D.

    elem_ty_c : Type[Numeric]
        The element type for output tensor D.

    Returns:
    --------
    Either SmemLoadMatrix or SimtSyncCopy, based on the input parameters.
    """

    if not isinstance(elem_ty_c, cutlass.cutlass_dsl.NumericMeta):
        raise TypeError(f"elem_ty_c must be a Numeric, but got {elem_ty_c}")
    is_m_major = layout_c.is_m_major_c()
    if elem_ty_c.width == 16:
        return cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(is_m_major, 4), elem_ty_c, loc=loc, ip=ip
        )
    else:
        return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), elem_ty_c, loc=loc, ip=ip)


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
}
if hasattr(torch, "float8_e4m3fn") and hasattr(cutlass, "Float8E4M3FN"):
    torch2cute_dtype_map[torch.float8_e4m3fn] = cutlass.Float8E4M3FN
if hasattr(torch, "float8_e5m2") and hasattr(cutlass, "Float8E5M2"):
    torch2cute_dtype_map[torch.float8_e5m2] = cutlass.Float8E5M2
