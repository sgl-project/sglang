"""Raw-pointer helpers for the FlyDSL Muse Spark attention kernels.

aiter's ``buffer_ops`` used to provide ``extract_base_index`` / ``create_llvm_ptr``; aiter main dropped them in its
pointer API cleanup. The kernels need them where an instruction takes a 64-bit address instead of a buffer resource
(the address of an operand above 2**31 bytes of a buffer, an i64 base for the far K/V pages).
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as _memref


def _unwrap_value(value):
    """The MLIR value under a FlyDSL wrapper (ArithValue, or a Numeric such as fx.Int32)."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    for _ in range(10):
        if isinstance(value, ir.Value):
            break
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
    return value


def create_llvm_ptr(value, address_space: int = 0) -> ir.Value:
    """An LLVM pointer from an integer or index value."""
    value = _unwrap_value(value)
    if isinstance(value.type, ir.IndexType):
        value = _unwrap_value(
            std_arith.IndexCastOp(ir.IntegerType.get_signless(64), value).result
        )
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return llvm.IntToPtrOp(ptr_type, value).result


def extract_base_index(tensor, address_space: int = 1) -> ir.Value:
    """The base address of a fly.memref (or a builtin memref) as an index value."""
    raw = _unwrap_value(tensor)
    try:
        ir.MemRefType(raw.type)
        return _memref.extract_aligned_pointer_as_index(raw)
    except ValueError:
        pass
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw)
    i64_val = llvm.PtrToIntOp(ir.IntegerType.get_signless(64), ptr).result
    return _unwrap_value(std_arith.IndexCastOp(ir.IndexType.get(), i64_val).result)
