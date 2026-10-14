import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def ld_acquire(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(state)


@dsl_user_op
def red_relaxed(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def red_release(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_release(lock_ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "st.global.release.gpu.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def atom_add_i32(lock_ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> cutlass.Int32:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    old = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "atom.global.add.s32 $0, [$1], $2;",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(old)


@dsl_user_op
def atom_cas_i32(
    lock_ptr: cute.Pointer, compare: Int32, val: Int32, *, loc=None, ip=None
) -> cutlass.Int32:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    old = llvm.inline_asm(
        T.i32(),
        [
            lock_ptr_i64,
            Int32(compare).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "atom.global.cas.b32 $0, [$1], $2, $3;",
        "=r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(old)


@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    val = llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int64(val)


@dsl_user_op
def st_i64(ptr: cute.Pointer, val: Int64, *, loc=None, ip=None) -> None:
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [ptr_i64, Int64(val).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def launch_dependents(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "griddepcontrol.launch_dependents;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def wait_eq(lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: Int32) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        read_val = Int32(0)
        while read_val != val:
            read_val = ld_acquire(flag_ptr)


@cute.jit
def arrive_inc(
    lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: cutlass.Constexpr[Int32]
) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        red_release(flag_ptr, val)
        # red_relaxed(flag_ptr, val)
