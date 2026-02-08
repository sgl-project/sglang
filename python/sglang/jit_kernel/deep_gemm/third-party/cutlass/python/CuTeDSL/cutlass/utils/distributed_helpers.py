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

from functools import partial
from typing import Tuple

import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op, while_generate

from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, scf
from cutlass._mlir.dialects.nvvm import (
    MemOrderKind,
    MemScopeKind,
    AtomicOpKind,
)
from cutlass.cute.typing import Pointer, Int32, Boolean


@dsl_user_op
def atomicAdd(dst_ptr: Pointer, val: Int32, loc=None, ip=None) -> Int32:
    return nvvm.atomicrmw(
        T.i32(),
        AtomicOpKind.ADD,
        dst_ptr.llvm_ptr,
        val.ir_value(loc=loc, ip=ip),
        mem_order=MemOrderKind.RELAXED,
        syncscope=MemScopeKind.SYS,
        loc=loc,
        ip=ip,
    )


@cute.jit
def ld_bypass(input_tensor: cute.Tensor):
    fragment = cute.make_fragment(input_tensor.layout, input_tensor.element_type)
    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        input_tensor.element_type,
        memory_order=cute.nvgpu.common.MemoryOrder.VOLATILE,
        memory_scope=cute.nvgpu.common.MemoryScope.SYS,
    )
    cute.copy(copy_atom_load, input_tensor, fragment)
    vals = fragment.load()
    return vals

@cute.jit
def spin_lock_wait(lock_ptr: Pointer, expect_count: Int32, mem_order : str = "relaxed", mem_scope : str = "gpu", loc=None, ip=None) -> None:
    """
    wait on a spin lock until the expected count is reached.
    """
    res = 0
    while res != expect_count:
        res = nvvm.atomicrmw(
            T.i32(),    
            AtomicOpKind.CAS, 
            lock_ptr.llvm_ptr, 
            Int32(0).ir_value(loc=loc, ip=ip),
            b=Int32(expect_count).ir_value(loc=loc, ip=ip),
            mem_order=MemOrderKind.ACQUIRE if mem_order == "acquire" else MemOrderKind.RELAXED,
            syncscope=MemScopeKind.GPU if mem_scope == "gpu" else MemScopeKind.SYS
        )


@dsl_user_op
def multimem_red_add_sys_release(mc_ptr: Pointer, loc=None, ip=None) -> None:
    """
    add 1 to the multimem address
    """
    llvm.inline_asm(
        None,
        [mc_ptr.toint().ir_value()],
        "multimem.red.release.sys.global.add.u32 [$0], 1;",
        "l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )

@dsl_user_op
def multimem_red_add_gpu_relaxed(mc_ptr: Pointer, loc=None, ip=None) -> None:
    """
    add 1 to the multimem address
    """
    llvm.inline_asm(
        None,
        [mc_ptr.toint().ir_value()],
        "multimem.red.relaxed.gpu.global.add.u32 [$0], 1;",
        "l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


def spin_lock_multimem_arrive(lock_ptr: Pointer, loc=None, ip=None) -> None:
    """
    arrive a spin lock when the lock_ptr is a multimem address.
    """
    multimem_red_add_gpu_relaxed(lock_ptr, loc=loc, ip=ip)


def sm_wise_inter_gpu_multimem_barrier(barrier : Pointer, barrier_mc : Pointer, num_ranks, loc=None, ip=None) -> None :
    """
    barrier for inter-gpu sm-wise
    """
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.grid_dim()
    pid = bidx + bidy * bdimx + bidz * bdimx * bdimy
    multimem_red_add_sys_release(barrier_mc + pid, loc=loc, ip=ip)
    cute.arch.fence_proxy(cute.arch.ProxyKind.alias)
    spin_lock_wait(barrier + pid, num_ranks, mem_order="acquire", mem_scope="sys", loc=loc, ip=ip)


@dsl_user_op
def multimem_ld_reduce_base(
    mc_ptr: Pointer,
    *,
    ptx_string: str = "",
    loc=None,
    ip=None,
)  -> Tuple[Int32, Int32, Int32, Int32]:
    # ld reduce 8xf16 elts
    mc_ptr_int = mc_ptr.toint(loc=loc, ip=ip).ir_value()
    return_struct = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [mc_ptr_int],
        ptx_string,
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return_regs = [llvm.extractvalue(T.i32(), return_struct, [i]) for i in range(4)]
    return return_regs[0], return_regs[1], return_regs[2], return_regs[3]


multimem_ld_reduce_8xf16 = partial(multimem_ld_reduce_base, ptx_string="multimem.ld_reduce.sys.relaxed.global.add.acc::f32.v4.f16x2 {$0, $1, $2, $3}, [$4];")
multimem_ld_reduce_4xf32 = partial(multimem_ld_reduce_base, ptx_string="multimem.ld_reduce.sys.relaxed.global.add.v4.f32 {$0, $1, $2, $3}, [$4];")
multimem_ld_reduce_8xbf16 = partial(multimem_ld_reduce_base, ptx_string="multimem.ld_reduce.sys.relaxed.global.add.acc::f32.v4.bf16x2 {$0, $1, $2, $3}, [$4];")
multimem_ld_reduce_16xe4m3 = partial(multimem_ld_reduce_base, ptx_string="multimem.ld_reduce.sys.relaxed.global.add.acc::f16.v4.e4m3x4 {$0, $1, $2, $3}, [$4];")
multimem_ld_reduce_16xe5m2 = partial(multimem_ld_reduce_base, ptx_string="multimem.ld_reduce.sys.relaxed.global.add.acc::f16.v4.e5m2x4 {$0, $1, $2, $3}, [$4];")


@dsl_user_op
def multimem_st_4xb32(
    mc_ptr: Pointer,
    x: Int32,
    y: Int32,
    z: Int32,
    w: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    # st 4x32 bits of data
    mc_ptr_int = mc_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        T.i32(),
        [mc_ptr_int, x, y, z, w],
        "multimem.st.sys.relaxed.global.v4.f32 [$1], {$2, $3, $4, $5};",
        "=r,l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )

