# Copyright (c) 2025, Tri Dao.

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@cute.jit
def clz(x: Int32) -> Int32:
    # for i in cutlass.range_constexpr(32):
    #     if (1 << (31 - i)) & x:
    #         return Int32(i)
    # return Int32(32)
    # Early exit is not supported yet
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res


def find_log2(x: Int32) -> Int32:
    a: Int32 = Int32(31 - clz(x))
    return a + ((x & (x - 1)) != 0)  # Round up, add 1 if not a power of 2.


@dsl_user_op
def umulhi(a: Int32, b: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class FastDivmod:
    def __init__(
        self, divisor: Int32, multipler: Uint32, shift_right: Uint32, *, loc=None, ip=None
    ):
        self.divisor = divisor
        self.multiplier = multipler
        self.shift_right = shift_right
        self._loc = loc

    # called by host
    @staticmethod
    def create(divisor: Int32, *, loc=None, ip=None) -> "FastDivmod":
        """Construct the FastDivmod object, in host code.
        This precomputes some values based on the divisor and is computationally expensive.
        """
        p = Uint32(31 + find_log2(divisor))
        divisor_u32 = Uint32(divisor)
        multiplier = Uint32(((cutlass.Uint64(1) << p) + divisor_u32 - 1) // divisor_u32)
        shift_right = Uint32(p - 32)
        return FastDivmod(divisor, multiplier, shift_right, loc=loc, ip=ip)

    @cute.jit
    def div(self, dividend: Int32) -> Int32:
        return (
            Int32(umulhi(dividend, self.multiplier) >> self.shift_right)
            if self.divisor != 1
            else dividend
        )

    def divmod(self, dividend: Int32) -> Tuple[Int32, Int32]:
        quotient = self.div(dividend)
        remainder = dividend - quotient * self.divisor
        return quotient, remainder

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.divisor, self.multiplier, self.shift_right]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.divisor, self.multiplier, self.shift_right], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FastDivmod(*(tuple(obj_list)), loc=self._loc)
