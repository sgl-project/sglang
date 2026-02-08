# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

"""
This module provides MLIR Arith Dialect helper functions
"""

import array
import numpy as np

from ..common import *
from ..._mlir import ir  # type: ignore
from ..._mlir.extras import types as T  # type: ignore
from ..._mlir.dialects import arith, nvgpu, math, builtin  # type: ignore

from .lru_cache_ir import lru_cache_ir

# =============================================================================
# Arith Dialect Helper functions
# =============================================================================


def recast_type(src_type, res_elem_type) -> ir.Type:
    if isinstance(src_type, T.VectorType):
        if src_type.scalable:
            res_type = T.vector(
                *src_type.shape,
                res_elem_type,
                scalable=src_type.scalable,
                scalable_dims=src_type.scalable_dims,
            )
        else:
            res_type = T.vector(*src_type.shape, res_elem_type)
    elif isinstance(src_type, T.RankedTensorType):
        res_type = T.RankedTensorType.get(
            element_type=res_elem_type, shape=src_type.shape, strides=src_type.strides
        )
    elif isinstance(src_type, T.UnrankedTensorType):
        res_type = T.UnrankedTensorType.get(element_type=res_elem_type)
    elif isinstance(src_type, T.MemRefType):
        res_type = T.MemRefType.get(
            element_type=res_elem_type, shape=src_type.shape, strides=src_type.strides
        )
    else:
        res_type = res_elem_type
    return res_type


def is_scalar(ty) -> bool:
    return not isinstance(
        ty, (T.VectorType, T.RankedTensorType, T.UnrankedTensorType, T.MemRefType)
    )


def element_type(ty) -> ir.Type:
    if not is_scalar(ty):
        return ty.element_type
    else:
        return ty


def is_narrow_precision(ty) -> bool:
    narrow_types = {
        T.f8E8M0FNU(),
        T.f8E4M3FN(),
        T.f8E4M3(),
        T.f8E5M2(),
        T.f8E4M3B11FNUZ(),
        T.f4E2M1FN(),
        T.f6E3M2FN(),
        T.f6E2M3FN(),
    }
    return ty in narrow_types


def is_float_type(ty) -> bool:
    return (
        arith._is_float_type(ty)
        # TODO-upstream: prediction is not correct. Patch here and fix in upstream later
        or is_narrow_precision(ty)
        or ty in (T.bf16(), T.tf32())
    )


def truncf_to_narrow(res_ty, src, loc, ip):
    res_elem_ty = element_type(res_ty)
    if res_elem_ty == T.f8E8M0FNU():
        rnd = nvgpu.RoundingMode.RP
    else:
        rnd = nvgpu.RoundingMode.RN
    return nvgpu.cvt_fptrunc(res_ty, src, rnd=rnd, loc=loc, ip=ip)


def extf_from_narrow(res_ty, src, loc, ip):
    src_elem_ty = element_type(src.type)

    # When source type is E8M0, temporary element type has to be bf16
    tmp_elem_ty = T.bf16() if src_elem_ty == T.f8E8M0FNU() else T.f16()
    tmp_ty = recast_type(src.type, tmp_elem_ty)

    # narrow -> bf16/f16 -> target type
    tmp = nvgpu.cvt_fpext(tmp_ty, src, loc=loc, ip=ip)
    return arith.extf(res_ty, tmp, loc=loc, ip=ip)


def bitcast(src, res_elem_type, *, loc=None, ip=None):
    res_type = recast_type(src.type, res_elem_type)
    return arith.bitcast(res_type, src, loc=loc, ip=ip)


def cvtf(src, res_elem_type, *, loc=None, ip=None):
    src_elem_type = element_type(src.type)

    if res_elem_type == src_elem_type:
        return src

    res_type = recast_type(src.type, res_elem_type)

    # Treat TF32 as F32 and use i32 as intermediate data
    # TODO-upstream: update arith to support tf32 <-> f32 conversion
    if src_elem_type == T.tf32():
        # tf32 -> i32
        tmp_type = recast_type(src.type, T.i32())
        src = builtin.unrealized_conversion_cast([tmp_type], [src], loc=loc, ip=ip)
        # i32 -> f32
        src = bitcast(src, T.f32(), loc=loc, ip=ip)
        # f32 -> X with `cvtf` recursively
        return cvtf(src, res_elem_type, loc=loc, ip=ip)

    if res_elem_type == T.tf32():
        # X -> f32 with `cvtf`` recursively
        tmp = cvtf(src, T.f32(), loc=loc, ip=ip)
        # f32 -> i32
        tmp = bitcast(tmp, T.i32(), loc=loc, ip=ip)
        # i32 -> tf32
        return builtin.unrealized_conversion_cast([res_type], [tmp], loc=loc, ip=ip)

    if res_elem_type.width > src_elem_type.width:
        if is_narrow_precision(src_elem_type):
            return extf_from_narrow(res_type, src, loc, ip)
        else:
            return arith.extf(res_type, src, loc=loc, ip=ip)
    else:
        tmp_mlir_type = recast_type(src.type, T.f32())

        # f16 -- extf -> f32 -- truncf -> bf16
        # TODO-upstream: update arith to support bf16 <-> f16 conversion?
        if (src_elem_type == T.f16() and res_elem_type == T.bf16()) or (
            src_elem_type == T.bf16() and res_elem_type == T.f16()
        ):
            tmp = arith.extf(tmp_mlir_type, src, loc=loc, ip=ip)
            return arith.truncf(res_type, tmp, loc=loc, ip=ip)

        # {f8, f6, f4} -> f16, f32, ...
        elif is_narrow_precision(res_elem_type):
            return truncf_to_narrow(res_type, src, loc, ip)
        else:
            return arith.truncf(res_type, src, loc=loc, ip=ip)


def fptoi(src, signed: Union[bool, None], res_elem_type, *, loc=None, ip=None):
    res_type = recast_type(src.type, res_elem_type)
    # TODO-upstream: update arith to support this kind of conversion
    if element_type(src.type) in (T.tf32(), T.bf16()):
        src = cvtf(src, T.f32(), loc=loc, ip=ip)

    if signed:
        return arith.fptosi(res_type, src, loc=loc, ip=ip)
    else:
        return arith.fptoui(res_type, src, loc=loc, ip=ip)


def itofp(src, signed: Union[bool, None], res_elem_type, *, loc=None, ip=None):
    res_type = recast_type(src.type, res_elem_type)

    orig_res_type = res_type
    # TODO-upstream: update arith to support this kind of conversion
    if res_elem_type in (T.tf32(), T.bf16()):
        res_type = recast_type(src.type, T.f32())

    if signed and element_type(src.type).width > 1:
        res = arith.sitofp(res_type, src, loc=loc, ip=ip)
    else:
        res = arith.uitofp(res_type, src, loc=loc, ip=ip)

    if orig_res_type == res_type:
        return res

    return cvtf(res, element_type(orig_res_type), loc=loc, ip=ip)


def int_to_int(a, dst_elem_type, *, loc=None, ip=None):
    src_signed = a.signed
    dst_signed = dst_elem_type.signed
    src_width = element_type(a.type).width
    dst_width = dst_elem_type.width

    dst_mlir_type = recast_type(a.type, dst_elem_type.mlir_type)

    if dst_width == src_width:
        return a
    elif src_signed != False and not dst_signed:
        # Signed -> Unsigned
        if dst_width > src_width:
            return arith.extui(dst_mlir_type, a, loc=loc, ip=ip)
        else:
            return arith.trunci(dst_mlir_type, a, loc=loc, ip=ip)
    elif src_signed == dst_signed:
        # Same signedness
        if dst_width > src_width:
            if src_signed != False and src_width > 1:
                return arith.extsi(dst_mlir_type, a, loc=loc, ip=ip)
            else:
                return arith.extui(dst_mlir_type, a, loc=loc, ip=ip)
        else:
            return arith.trunci(dst_mlir_type, a, loc=loc, ip=ip)
    else:
        # Unsigned -> Signed
        if dst_width > src_width:
            return arith.extui(dst_mlir_type, a, loc=loc, ip=ip)
        else:
            # For truncation from unsigned to signed, we need to handle overflow
            # First truncate to the target width
            trunc = arith.trunci(dst_mlir_type, a, loc=loc, ip=ip)
            # Then reinterpret as signed
            if dst_signed:
                return arith.bitcast(dst_mlir_type, trunc, loc=loc, ip=ip)
            return trunc


# =============================================================================
# Arith Ops Emitter Helpers
#   - assuming type of lhs and rhs match each other
#   - op name matches python module operator
# =============================================================================


def _cast(res_elem_ty, src, is_signed=None, *, loc=None, ip=None):
    """
    This function provides simplified interface to upstream op builder
        arith.truncf(T.vector(shape, new_type), src)

    is simplified as because it's element-wise op which can't change shape
        arith.truncf(new_type, src)
    """
    if isinstance(src, ir.Value):
        src_ty = src.type
    else:
        src_ty = type(src).mlir_type
        src = src.ir_value()

    src_elem_ty = element_type(src_ty)

    if src_elem_ty == res_elem_ty:
        return src
    elif is_float_type(src_elem_ty) and is_float_type(res_elem_ty):
        # float-to-float
        return cvtf(src, res_elem_ty, loc=loc, ip=ip)
    elif arith._is_integer_like_type(src_elem_ty) and arith._is_integer_like_type(
        res_elem_ty
    ):
        if src_elem_ty.width >= res_elem_ty.width:
            cast_op = arith.trunci
        else:
            if is_signed:
                cast_op = arith.extsi
            else:
                cast_op = arith.extui

        res_ty = recast_type(src_ty, res_elem_ty)
        return cast_op(res_ty, src, loc=loc, ip=ip)
    elif is_float_type(src_elem_ty) and arith._is_integer_like_type(res_elem_ty):
        return fptoi(src, is_signed, res_elem_ty, loc=loc, ip=ip)
    elif arith._is_integer_like_type(src_elem_ty) and is_float_type(res_elem_ty):
        return itofp(src, is_signed, res_elem_ty, loc=loc, ip=ip)
    else:
        raise DSLRuntimeError(
            f"cast from {src_elem_ty} to {res_elem_ty} is not supported"
        )


@lru_cache_ir()
def const(value, ty=None, *, loc=None, ip=None):
    """
    Generates dynamic expression for constant values.
    """
    from ..typing import Numeric, NumericMeta
    from ..dsl import is_dynamic_expression, _numpy_type_to_mlir_type

    if isinstance(value, Numeric):
        value = value.value

    # Early return
    if is_dynamic_expression(value) and (
        value.type.isinstance(value.type) or T.bool().isinstance(value.type)
    ):
        return value

    # Assume type
    if ty is None:
        if isinstance(value, float):
            ty = T.f32()
        elif isinstance(value, bool):
            ty = T.bool()
        elif isinstance(value, int):
            ty = T.i32()
        elif isinstance(value, np.ndarray):
            ty = T.vector(*value.shape, _numpy_type_to_mlir_type(value.dtype))
            value = array.array(value.dtype.kind, value.flatten().tolist())
        else:
            raise DSLNotImplemented(f"{type(value)} is not supported")
    elif isinstance(ty, NumericMeta):
        ty = ty.mlir_type
    elif isinstance(ty, ir.Type):
        if ir.RankedTensorType.isinstance(ty) or ir.VectorType.isinstance(ty):
            elem_ty = ty.element_type
            if isinstance(elem_ty, ir.IntegerType):
                attr = ir.IntegerAttr.get(elem_ty, value)
            else:
                attr = ir.FloatAttr.get(elem_ty, value)
            value = ir.DenseElementsAttr.get_splat(ty, attr)
        elif arith._is_float_type(ty) and isinstance(value, (bool, int)):
            value = float(value)
        elif arith._is_integer_like_type(ty) and isinstance(value, float):
            value = int(value)
    else:
        raise DSLNotImplemented(f"type {ty} is not supported")

    return arith.constant(ty, value, loc=loc, ip=ip)


def _dispatch_to_rhs_r_op(op):
    """Decorator that dispatches to the right-hand-side's reverse operation.

    If the other operand is not an ArithValue or is a subclass (more specific)
    of ArithValue, this allows proper method resolution for binary operations.
    """

    def wrapper(self, other, **kwargs):
        if not isinstance(other, ArithValue):
            if not isinstance(other, (int, float, bool)):
                # allows to call other.__rmul__
                return NotImplemented

        return op(self, other, **kwargs)

    return wrapper


def _binary_op(op):
    """
    Decorator to check if the 'other' argument is an ArithValue.
    If not, returns NotImplemented.
    """

    def wrapper(self, other, **kwargs):
        # When reach this point, `self` must be cast to base `ArithValue` type
        if isinstance(other, (int, float, bool)):
            other = const(other, self.type).with_signedness(self.signed)

        # Call the original function
        # If sub-class doesn't implement overloaded arithmetic, cast to base class
        return op(self, other, **kwargs)

    return wrapper


# Operator overloading
@ir.register_value_caster(ir.Float4E2M1FNType.static_typeid)
@ir.register_value_caster(ir.Float6E2M3FNType.static_typeid)
@ir.register_value_caster(ir.Float6E3M2FNType.static_typeid)
@ir.register_value_caster(ir.Float8E4M3FNType.static_typeid)
@ir.register_value_caster(ir.Float8E4M3B11FNUZType.static_typeid)
@ir.register_value_caster(ir.Float8E5M2Type.static_typeid)
@ir.register_value_caster(ir.Float8E4M3Type.static_typeid)
@ir.register_value_caster(ir.Float8E8M0FNUType.static_typeid)
@ir.register_value_caster(ir.BF16Type.static_typeid)
@ir.register_value_caster(ir.F16Type.static_typeid)
@ir.register_value_caster(ir.FloatTF32Type.static_typeid)
@ir.register_value_caster(ir.F32Type.static_typeid)
@ir.register_value_caster(ir.F64Type.static_typeid)
@ir.register_value_caster(ir.IntegerType.static_typeid)
@ir.register_value_caster(ir.VectorType.static_typeid)
@ir.register_value_caster(ir.RankedTensorType.static_typeid)
class ArithValue(ir.Value):
    """Overloads operators for MLIR's Arith dialects binary operations."""

    def __init__(self, v, signed: Union[bool, None] = None):
        if isinstance(v, int):
            v = arith.constant(self.type, v)
        super().__init__(v)

        elem_ty = element_type(self.type)
        self.is_float = arith._is_float_type(elem_ty)
        # arith dialect consider `1` in `i1` as `-1`, treat it as unsigned for DSL
        self.signed = signed and elem_ty.width > 1

    def with_signedness(self, signed: Union[bool, None]):
        return type(self)(self, signed)

    def __neg__(self, *, loc=None, ip=None):
        if self.type == T.bool():
            raise TypeError(
                "Negation, the operator `-` is not supported for boolean type"
            )

        if self.is_float:
            return arith.negf(self, loc=loc, ip=ip)
        else:
            c0 = arith.constant(self.type, 0, loc=loc, ip=ip)
            return arith.subi(c0, self, loc=loc, ip=ip)

    @_binary_op
    def __pow__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float and other.is_float:
            return math.powf(self, other, loc=loc, ip=ip)
        elif self.is_float and not other.is_float:
            return math.fpowi(self, other, loc=loc, ip=ip)
        elif not self.is_float and other.is_float:
            lhs = itofp(self, self.signed, T.f32(), loc=loc, ip=ip)
            rhs = cvtf(other, T.f32(), loc=loc, ip=ip)
            return math.powf(lhs, rhs, loc=loc, ip=ip)
        elif not self.is_float and not other.is_float:
            return math.ipowi(self, other, loc=loc, ip=ip)
        else:
            raise DSLNotImplemented(f"Unsupported '{self} ** {other}'")

    @_binary_op
    def __rpow__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return other.__pow__(self, loc=loc, ip=ip)

    # arith operators

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __add__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            return arith.addf(self, other, loc=loc, ip=ip)
        else:
            return arith.addi(self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __sub__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            return arith.subf(self, other, loc=loc, ip=ip)
        else:
            return arith.subi(self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __mul__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            return arith.mulf(self, other, loc=loc, ip=ip)
        else:
            return arith.muli(self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __truediv__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            return arith.divf(self, other, loc=loc, ip=ip)
        else:
            lhs = itofp(self, self.signed, T.f32(), loc=loc, ip=ip)
            rhs = itofp(other, other.signed, T.f32(), loc=loc, ip=ip)
            return arith.divf(lhs, rhs, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __floordiv__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            q = arith.divf(self, other, loc=loc, ip=ip)
            return math.floor(q, loc=loc, ip=ip)
        elif self.signed != False:
            return arith.floordivsi(self, other, loc=loc, ip=ip)
        else:
            return arith.divui(self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __mod__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            return arith.remf(self, other, loc=loc, ip=ip)
        elif self.signed != False:
            return arith.remsi(self, other, loc=loc, ip=ip)
        else:
            return arith.remui(self, other, loc=loc, ip=ip)

    @_binary_op
    def __radd__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return other.__add__(self, loc=loc, ip=ip)

    @_binary_op
    def __rsub__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return other.__sub__(self, loc=loc, ip=ip)

    @_binary_op
    def __rmul__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return other.__mul__(self, loc=loc, ip=ip)

    @_binary_op
    def __rtruediv__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return other.__truediv__(self, loc=loc, ip=ip)

    @_binary_op
    def __rfloordiv__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return other.__floordiv__(self, loc=loc, ip=ip)

    @_binary_op
    def __rmod__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return other.__mod__(self, loc=loc, ip=ip)

    # Comparison operators (comparison doesn't have right-hand-side variants)
    @_dispatch_to_rhs_r_op
    @_binary_op
    def __lt__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            return arith.cmpf(arith.CmpFPredicate.OLT, self, other, loc=loc, ip=ip)
        elif self.signed != False:
            return arith.cmpi(arith.CmpIPredicate.slt, self, other, loc=loc, ip=ip)
        else:
            return arith.cmpi(arith.CmpIPredicate.ult, self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __le__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            return arith.cmpf(arith.CmpFPredicate.OLE, self, other, loc=loc, ip=ip)
        elif self.signed != False:
            return arith.cmpi(arith.CmpIPredicate.sle, self, other, loc=loc, ip=ip)
        else:
            return arith.cmpi(arith.CmpIPredicate.ule, self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __eq__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            return arith.cmpf(arith.CmpFPredicate.OEQ, self, other, loc=loc, ip=ip)
        else:
            return arith.cmpi(arith.CmpIPredicate.eq, self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __ne__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            # In Python, bool(float("nan")) is True, so use unordered comparison here
            return arith.cmpf(arith.CmpFPredicate.UNE, self, other, loc=loc, ip=ip)
        else:
            return arith.cmpi(arith.CmpIPredicate.ne, self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __gt__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            return arith.cmpf(arith.CmpFPredicate.OGT, self, other, loc=loc, ip=ip)
        elif self.signed != False:
            return arith.cmpi(arith.CmpIPredicate.sgt, self, other, loc=loc, ip=ip)
        else:
            return arith.cmpi(arith.CmpIPredicate.ugt, self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __ge__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.is_float:
            return arith.cmpf(arith.CmpFPredicate.OGE, self, other, loc=loc, ip=ip)
        elif self.signed != False:
            return arith.cmpi(arith.CmpIPredicate.sge, self, other, loc=loc, ip=ip)
        else:
            return arith.cmpi(arith.CmpIPredicate.uge, self, other, loc=loc, ip=ip)

    # Unary operators
    def __invert__(self, *, loc=None, ip=None) -> "ArithValue":
        return arith.xori(self, arith.constant(self.type, -1))

    # Bitwise operations
    @_dispatch_to_rhs_r_op
    @_binary_op
    def __and__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return arith.andi(self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __or__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return arith.ori(self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __xor__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return arith.xori(self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __rshift__(self, other, *, loc=None, ip=None) -> "ArithValue":
        if self.signed != False:
            return arith.shrsi(self, other, loc=loc, ip=ip)
        else:
            return arith.shrui(self, other, loc=loc, ip=ip)

    @_dispatch_to_rhs_r_op
    @_binary_op
    def __lshift__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return arith.shli(self, other, loc=loc, ip=ip)

    @_binary_op
    def __rand__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return arith.andi(other, self, loc=loc, ip=ip)

    @_binary_op
    def __ror__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return arith.ori(other, self, loc=loc, ip=ip)

    @_binary_op
    def __rxor__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return arith.xori(other, self, loc=loc, ip=ip)

    @_binary_op
    def __rrshift__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return other.__rshift__(self, loc=loc, ip=ip)

    @_binary_op
    def __rlshift__(self, other, *, loc=None, ip=None) -> "ArithValue":
        return other.__lshift__(self, loc=loc, ip=ip)

    def __hash__(self):
        return super().__hash__()

    def __str__(self):
        return "?"

    def __repr__(self):
        return self.__str__()


def _min(lhs, rhs, *, loc=None, ip=None):
    """
    This function provides a unified interface for building arith min

    Assuming the operands have the same type
    """
    from ..dsl import is_dynamic_expression

    if not is_dynamic_expression(lhs):
        if not is_dynamic_expression(rhs):
            return min(lhs, rhs)
        else:
            lhs = arith.constant(rhs.type, lhs, loc=loc, ip=ip)
    else:
        if not is_dynamic_expression(rhs):
            rhs = arith.constant(lhs.type, rhs, loc=loc, ip=ip)

    if arith._is_integer_like_type(lhs.type):
        if lhs.signed != False:
            return arith.minsi(lhs, rhs, loc=loc, ip=ip)
        else:
            return arith.minui(lhs, rhs, loc=loc, ip=ip)
    else:
        return arith.minimumf(lhs, rhs, loc=loc, ip=ip)


def _max(lhs, rhs, *, loc=None, ip=None):
    """
    This function provides a unified interface for building arith max

    Assuming the operands have the same type
    """
    from ..dsl import is_dynamic_expression

    if not is_dynamic_expression(lhs):
        if not is_dynamic_expression(rhs):
            return max(lhs, rhs)
        else:
            lhs = arith.constant(rhs.type, lhs, loc=loc, ip=ip)
    else:
        if not is_dynamic_expression(rhs):
            rhs = arith.constant(lhs.type, rhs, loc=loc, ip=ip)

    if arith._is_integer_like_type(lhs.type):
        if lhs.signed != False:
            return arith.maxsi(lhs, rhs, loc=loc, ip=ip)
        else:
            return arith.maxui(lhs, rhs, loc=loc, ip=ip)
    else:
        return arith.maximumf(lhs, rhs, loc=loc, ip=ip)
