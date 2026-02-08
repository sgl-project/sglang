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
from typing import Optional, Tuple, Union, Callable
from typing_extensions import deprecated

from cutlass.cutlass_dsl import T, dsl_user_op

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector

# Forward nvvm enums
from cutlass._mlir.dialects.nvvm import (
    ProxyKind,
    SharedSpace,
    Tcgen05WaitKind,
    SetMaxRegisterAction,
    RoundingModeKind,
)

from ..typing import (
    Int,
    Boolean,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Float32,
    BFloat16,
    Numeric,
    as_numeric,
)

WARP_SIZE = 32
FULL_MASK = 0xFFFFFFFF


@dsl_user_op
def lane_idx(*, loc=None, ip=None) -> Int32:
    """
    Returns the lane index of the current thread within the warp.
    """
    return Int32(nvvm.read_ptx_sreg_laneid(T.i32(), loc=loc, ip=ip))


@dsl_user_op
def warp_idx(*, loc=None, ip=None) -> Int32:
    """
    Returns the warp index within a CTA.
    """
    warp_size = 32
    tid_x = Int32(nvvm.read_ptx_sreg_tid_x(T.i32(), loc=loc, ip=ip))
    tid_y = Int32(nvvm.read_ptx_sreg_tid_y(T.i32(), loc=loc, ip=ip))
    tid_z = Int32(nvvm.read_ptx_sreg_tid_z(T.i32(), loc=loc, ip=ip))
    ntid_x = Int32(nvvm.read_ptx_sreg_ntid_x(T.i32(), loc=loc, ip=ip))
    ntid_y = Int32(nvvm.read_ptx_sreg_ntid_y(T.i32(), loc=loc, ip=ip))
    tid = tid_x + tid_y * ntid_x + tid_z * ntid_x * ntid_y
    return tid // warp_size


@dsl_user_op
def thread_idx(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """
    Returns the thread index within a CTA.
    """
    return (
        Int32(nvvm.read_ptx_sreg_tid_x(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_tid_y(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_tid_z(T.i32(), loc=loc, ip=ip)),
    )


@dsl_user_op
def block_dim(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """
    Returns the number of threads in each dimension of the CTA.
    """
    return (
        Int32(nvvm.read_ptx_sreg_ntid_x(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_ntid_y(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_ntid_z(T.i32(), loc=loc, ip=ip)),
    )


@dsl_user_op
def block_idx(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """
    Returns the CTA identifier within a grid.
    """
    return (
        Int32(nvvm.read_ptx_sreg_ctaid_x(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_ctaid_y(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_ctaid_z(T.i32(), loc=loc, ip=ip)),
    )


@dsl_user_op
def grid_dim(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """
    Returns the number of CTAs in each dimension of the grid.
    """
    return (
        Int32(nvvm.read_ptx_sreg_nctaid_x(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_nctaid_y(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_nctaid_z(T.i32(), loc=loc, ip=ip)),
    )


@dsl_user_op
def cluster_idx(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """
    Returns the cluster identifier within a grid.
    """
    return (
        Int32(nvvm.read_ptx_sreg_clusterid_x(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_clusterid_y(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_clusterid_z(T.i32(), loc=loc, ip=ip)),
    )


@dsl_user_op
def cluster_dim(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """
    Returns the number of clusters in each dimension of the grid.
    """
    return (
        Int32(nvvm.read_ptx_sreg_nclusterid_x(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_nclusterid_y(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_nclusterid_z(T.i32(), loc=loc, ip=ip)),
    )


@dsl_user_op
def block_in_cluster_idx(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """
    Returns the CTA index within a cluster across all dimensions.
    """
    return (
        Int32(nvvm.read_ptx_sreg_cluster_ctaid_x(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_cluster_ctaid_y(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_cluster_ctaid_z(T.i32(), loc=loc, ip=ip)),
    )


@dsl_user_op
def block_in_cluster_dim(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """
    Returns the dimensions of the cluster.
    """
    return (
        Int32(nvvm.read_ptx_sreg_cluster_nctaid_x(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_cluster_nctaid_y(T.i32(), loc=loc, ip=ip)),
        Int32(nvvm.read_ptx_sreg_cluster_nctaid_z(T.i32(), loc=loc, ip=ip)),
    )


@dsl_user_op
def block_idx_in_cluster(*, loc=None, ip=None) -> Int32:
    """
    Returns the linearized identifier of the CTA within the cluster.
    """
    return Int32(nvvm.read_ptx_sreg_cluster_ctarank(T.i32(), loc=loc, ip=ip))


@dsl_user_op
def shuffle_sync_op(
    value: Numeric,
    offset: Int,
    mask: Int = FULL_MASK,
    mask_and_clamp: Int = WARP_SIZE - 1,
    kind: nvvm.ShflKind = nvvm.ShflKind.idx,
    *,
    loc=None,
    ip=None,
) -> Numeric:
    """
    Shuffles a value within the threads of a warp.

    :param value:          The value to shuffle
    :type value:           Numeric
    :param mask:           A mask describing the threads participating in this operation
    :type mask:            Int
    :param offset:         A source lane or a source lane offset depending on kind
    :type offset:          Int
    :param mask_and_clamp: An integer containing two packed values specifying a mask for logically
                           splitting warps into sub-segments and an upper bound for clamping the
                           source lane index.
    :type mask_and_clamp:  Int
    :param kind:           The kind of shuffle, can be idx, up, down, or bfly
    :type kind:            ShflKind
    :return:               The shuffled value
    :rtype:                Numeric
    """
    if not isinstance(value, Numeric):
        value = as_numeric(value)
    if value.width > 64:
        raise ValueError("shuffle_sync only supports values up to 64 bits")

    orig_type = type(value)
    if value.width < 32:
        if value.dtype.is_float:
            value = value.to(Float32)
        else:
            if value.signed:
                value = value.to(Int32)
            else:
                value = value.to(Uint32)
        return orig_type(
            nvvm.shfl_sync(
                type(value).mlir_type,
                Int32(mask).ir_value(loc=loc, ip=ip),
                value.ir_value(loc=loc, ip=ip),
                Int32(offset).ir_value(loc=loc, ip=ip),
                Int32(mask_and_clamp).ir_value(loc=loc, ip=ip),
                kind,
                loc=loc,
                ip=ip,
            )
        )
    elif value.width == 32:
        return orig_type(
            nvvm.shfl_sync(
                type(value).mlir_type,
                Int32(mask).ir_value(loc=loc, ip=ip),
                value.ir_value(loc=loc, ip=ip),
                Int32(offset).ir_value(loc=loc, ip=ip),
                Int32(mask_and_clamp).ir_value(loc=loc, ip=ip),
                kind,
                loc=loc,
                ip=ip,
            )
        )
    else:
        if value.width != 64:
            raise ValueError(
                "shuffle_sync only supports 64 bits values when the bit width is larger than 32"
            )
        value = llvm.bitcast(
            T.i64(), value.to(ir.Value, loc=loc, ip=ip), loc=loc, ip=ip
        )
        # extract low 32 bits
        low_32_bits = llvm.trunc(
            T.i32(), value, llvm.IntegerOverflowFlags.none, loc=loc, ip=ip
        )
        # extract high 32 bits
        high_32_bits = llvm.lshr(
            value, Int64(32).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
        high_32_bits = llvm.trunc(
            T.i32(), high_32_bits, llvm.IntegerOverflowFlags.none, loc=loc, ip=ip
        )

        low_32_bits_shfl = nvvm.shfl_sync(
            T.i32(),
            Int32(mask).ir_value(loc=loc, ip=ip),
            low_32_bits,
            Int32(offset).ir_value(loc=loc, ip=ip),
            Int32(mask_and_clamp).ir_value(loc=loc, ip=ip),
            kind,
            loc=loc,
            ip=ip,
        )
        high_32_bits_shfl = nvvm.shfl_sync(
            T.i32(),
            Int32(mask).ir_value(loc=loc, ip=ip),
            high_32_bits,
            Int32(offset).ir_value(loc=loc, ip=ip),
            Int32(mask_and_clamp).ir_value(loc=loc, ip=ip),
            kind,
            loc=loc,
            ip=ip,
        )

        # combine low and high 32 bits
        low_64_bit = llvm.zext(T.i64(), low_32_bits_shfl, loc=loc, ip=ip)
        high_64_bit = llvm.zext(T.i64(), high_32_bits_shfl, loc=loc, ip=ip)
        shlf_res = llvm.shl(
            high_64_bit,
            Int64(32).ir_value(loc=loc, ip=ip),
            llvm.IntegerOverflowFlags.none,
            loc=loc,
            ip=ip,
        )
        shlf_res = llvm.or_(shlf_res, low_64_bit, loc=loc, ip=ip)
        shlf_res = llvm.bitcast(orig_type.mlir_type, shlf_res, loc=loc, ip=ip)
        return orig_type(shlf_res)

shuffle_sync = partial(shuffle_sync_op, kind=nvvm.ShflKind.idx)
shuffle_sync_up = partial(shuffle_sync_op, kind=nvvm.ShflKind.up)
shuffle_sync_down = partial(shuffle_sync_op, kind=nvvm.ShflKind.down)
shuffle_sync_bfly = partial(shuffle_sync_op, kind=nvvm.ShflKind.bfly)


@dsl_user_op
def barrier(*, barrier_id=None, number_of_threads=None, loc=None, ip=None) -> None:
    """
    Creates a barrier, optionally named.
    """
    if barrier_id is not None:
        barrier_id = Int32(barrier_id).ir_value(loc=loc, ip=ip)

    if number_of_threads is not None:
        number_of_threads = Int32(number_of_threads).ir_value(loc=loc, ip=ip)

    nvvm.barrier(
        barrier_id=barrier_id, number_of_threads=number_of_threads, loc=loc, ip=ip
    )


@dsl_user_op
def barrier_arrive(
    *, barrier_id=None, number_of_threads=None, loc=None, ip=None
) -> None:
    if barrier_id is not None:
        barrier_id = Int32(barrier_id).ir_value(loc=loc, ip=ip)

    if number_of_threads is None:
        raise ValueError(
            "barrier_arrive needs pass number_of_threads to arrive the barrier",
        )
    number_of_threads = Int32(number_of_threads).ir_value(loc=loc, ip=ip)

    nvvm.barrier_arrive(
        barrier_id=barrier_id, number_of_threads=number_of_threads, loc=loc, ip=ip
    )


@dsl_user_op
def sync_threads(*, loc=None, ip=None) -> None:
    """
    Synchronizes all threads within a CTA.
    """
    nvvm.barrier(loc=loc, ip=ip)


@dsl_user_op
def sync_warp(mask: Int = FULL_MASK, *, loc=None, ip=None) -> None:
    """
    Performs a warp-wide sync with an optional mask.
    """
    nvvm.bar_warp_sync(Int32(mask).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def fence_acq_rel_cta(*, loc=None, ip=None) -> None:
    """
    Fence operation with acquire-release semantics.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>`__.
    """
    nvvm.fence_acq_rel_cta(loc=loc, ip=ip)


@dsl_user_op
def fence_acq_rel_cluster(*, loc=None, ip=None) -> None:
    """
    Fence operation with acquire-release semantics.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>`__.
    """
    nvvm.fence_acq_rel_cluster(loc=loc, ip=ip)


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None) -> None:
    """
    Fence operation with acquire-release semantics.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>`__.
    """
    nvvm.fence_acq_rel_gpu(loc=loc, ip=ip)


@dsl_user_op
def fence_acq_rel_sys(*, loc=None, ip=None) -> None:
    """
    Fence operation with acquire-release semantics.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>`__.
    """
    nvvm.fence_acq_rel_sys(loc=loc, ip=ip)


@dsl_user_op
def cp_async_commit_group(*, loc=None, ip=None) -> None:
    """
    Commits all prior initiated but uncommitted cp.async instructions.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group>`__.
    """
    nvvm.cp_async_commit_group(loc=loc, ip=ip)


@dsl_user_op
def cp_async_wait_group(n, *, loc=None, ip=None) -> None:
    """
    Waits till only a specified numbers of cp.async groups are pending.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all>`__.
    """
    nvvm.cp_async_wait_group(n, loc=loc, ip=ip)


@dsl_user_op
def cp_async_bulk_commit_group(*, loc=None, ip=None) -> None:
    """
    Commits all prior initiated but uncommitted cp.async.bulk instructions.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group>`__.
    """
    nvvm.cp_async_bulk_commit_group(loc=loc, ip=ip)


@dsl_user_op
def cp_async_bulk_wait_group(group, *, read=None, loc=None, ip=None) -> None:
    """
    Waits till only a specified numbers of cp.async.bulk groups are pending.

    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group>`__.
    """
    nvvm.cp_async_bulk_wait_group(group, read=read, loc=loc, ip=ip)


@dsl_user_op
def cluster_wait(*, loc=None, ip=None) -> None:
    """
    A cluster-wide wait operation.
    """
    nvvm.cluster_wait(loc=loc, ip=ip)


@dsl_user_op
def cluster_arrive(*, aligned=None, loc=None, ip=None) -> None:
    """
    A cluster-wide arrive operation.
    """
    nvvm.cluster_arrive(aligned=aligned, loc=loc, ip=ip)


@dsl_user_op
def cluster_arrive_relaxed(*, aligned=None, loc=None, ip=None) -> None:
    """
    A cluster-wide arrive operation with relaxed semantics.
    """
    nvvm.cluster_arrive_relaxed(aligned=aligned, loc=loc, ip=ip)


@dsl_user_op
def fence_proxy(
    kind: ProxyKind,
    *,
    space: Optional[SharedSpace] = None,
    use_intrinsic=None,
    loc=None,
    ip=None,
) -> None:
    nvvm.fence_proxy(
        kind=kind, space=space, use_intrinsic=use_intrinsic, loc=loc, ip=ip
    )


@dsl_user_op
def vote_ballot_sync(
    pred: Boolean, mask: Int = FULL_MASK, *, loc=None, ip=None
) -> Int32:
    """
    Performs a ballot operation across the warp.
    """
    return Int32(
        nvvm.vote_ballot_sync(
            T.i32(),
            Int32(mask).ir_value(loc=loc, ip=ip),
            Boolean(pred).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def popc(value: Numeric, *, loc=None, ip=None) -> Numeric:
    """
    Performs a population count operation.
    """
    if not isinstance(value, Numeric):
        value = as_numeric(value)
    return type(value)(llvm.intr_ctpop(value.ir_value(), loc=loc, ip=ip))


@dsl_user_op
def fence_view_async_tmem_op(
    kind: Tcgen05WaitKind,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    Perform a fence operation on the async TMEM load or store.

    .. note::
        This function is only available on sm_100a and above.
        The fence is required to synchronize the TMEM load/store
        and let the pipeline release or commit the buffer.

        Take a mma2acc pipeline as an example of LOAD fence, the ACC tensor is from TMEM.
        ```
        # Start to copy ACC from TMEM to register
        cute.copy(tmem_load, tACC, rACC)
        fence_view_async_tmem_load()
        # After fence, we can ensure the TMEM buffer is consumed totally.
        # Release the buffer to let the MMA know it can overwrite the buffer.
        mma2accum_pipeline.consumer_release(curr_consumer_state)
        ```
        Take a TS GEMM kernel as an example of STORE fence, the A tensor is from TMEM.
        ```
        # Start to copy A from register to TMEM
        cute.copy(tmem_store, rA, tA)
        fence_view_async_tmem_store()
        # After fence, we can ensure the TMEM buffer is ready.
        # Commit the buffer to let the MMA know it can start to load A.
        tmem_mma_pipeline.producer_commit(curr_producer_state)
        ```


    :param kind: The kind of fence operation to perform including LOAD and STORE.
    :type kind: Tcgen05WaitKind
    """
    nvvm.tcgen05_wait(kind, loc=loc, ip=ip)


fence_view_async_tmem_load = partial(
    fence_view_async_tmem_op, kind=Tcgen05WaitKind.LOAD
)
fence_view_async_tmem_store = partial(
    fence_view_async_tmem_op, kind=Tcgen05WaitKind.STORE
)


@dsl_user_op
def warpgroup_reg_realloc_op(
    reg_count: int,
    kind: SetMaxRegisterAction,
    *,
    loc=None,
    ip=None,
) -> None:
    nvvm.setmaxregister(reg_count, kind, loc=loc, ip=ip)


warpgroup_reg_alloc = partial(
    warpgroup_reg_realloc_op, kind=SetMaxRegisterAction.increase
)
warpgroup_reg_dealloc = partial(
    warpgroup_reg_realloc_op, kind=SetMaxRegisterAction.decrease
)


@dsl_user_op
def calc_packed_f32x2_op(
    src_a: Tuple[Float32, Float32],
    src_b: Tuple[Float32, Float32],
    src_c: Tuple[Float32, Float32] | None,
    calc_func: Callable,
    *,
    rnd=RoundingModeKind.RZ,
    ftz=True,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    vec_type = ir.VectorType.get([2], Float32.mlir_type, loc=loc)
    vec_src_a = vector.from_elements(
        vec_type, tuple(as_numeric(a).ir_value() for a in src_a), loc=loc, ip=ip
    )
    vec_src_b = vector.from_elements(
        vec_type, tuple(as_numeric(b).ir_value() for b in src_b), loc=loc, ip=ip
    )
    if src_c is not None:
        vec_src_c = vector.from_elements(
            vec_type, tuple(as_numeric(c).ir_value() for c in src_c), loc=loc, ip=ip
        )
        vec_res = calc_func(
            vec_type, vec_src_a, vec_src_b, vec_src_c, rnd=rnd, ftz=ftz, loc=loc, ip=ip
        )
    else:
        vec_res = calc_func(
            vec_type, vec_src_a, vec_src_b, rnd=rnd, ftz=ftz, loc=loc, ip=ip
        )

    res0 = Float32(
        vector.extract(
            vec_res, dynamic_position=[], static_position=[0], loc=loc, ip=ip
        )
    )
    res1 = Float32(
        vector.extract(
            vec_res, dynamic_position=[], static_position=[1], loc=loc, ip=ip
        )
    )
    return res0, res1


fma_packed_f32x2 = partial(calc_packed_f32x2_op, calc_func=nvvm.fma_packed_f32x2)
mul_packed_f32x2 = partial(
    calc_packed_f32x2_op, src_c=None, calc_func=nvvm.mul_packed_f32x2
)
add_packed_f32x2 = partial(
    calc_packed_f32x2_op, src_c=None, calc_func=nvvm.add_packed_f32x2
)


@dsl_user_op
def fmax(
    a: Union[float, Float32], b: Union[float, Float32], *, loc=None, ip=None
) -> Float32:
    return Float32(
        nvvm.fmax(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def rcp_approx(a: Union[float, Float32], *, loc=None, ip=None):
    return Float32(
        nvvm.rcp_approx_ftz_f(
            T.f32(), Float32(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


@dsl_user_op
@deprecated(
    "cute.arch.exp2 is deprecated, use cute.math.exp2 with `fastmath=True` instead"
)
def exp2(a: Union[float, Float32], *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
@deprecated(
    "cute.arch.exp is deprecated, use cute.math.exp with `fastmath=True` instead"
)
def exp(a: Union[float, Float32], *, loc=None, ip=None) -> Float32:
    LOG2_E = 1.4426950408889634
    return exp2(a * LOG2_E, loc=loc, ip=ip)


@dsl_user_op
@deprecated(
    "cute.arch.exp_packed_f32x2 is deprecated, use cute.arch.mul_packed_f32x2 and cute.math.exp2 with `fastmath=True` instead"
)
def exp_packed_f32x2(
    a: Tuple[Float32, Float32], *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    LOG2_E = Float32(1.4426950408889634)
    b = mul_packed_f32x2(a, (LOG2_E, LOG2_E), loc=loc, ip=ip)
    return exp2(b[0], loc=loc, ip=ip), exp2(b[1], loc=loc, ip=ip)
