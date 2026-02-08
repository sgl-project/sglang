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
from typing import Optional

from cutlass.cutlass_dsl import CuTeDSL, T, if_generate, dsl_user_op

from cutlass._mlir.dialects import nvvm
from cutlass._mlir import ir

from ..typing import Pointer, Int, Boolean, Int32
from ...impl_utils import check_value_in


####################################################################################################
#
# Mbarrier management utilities
#
####################################################################################################


@dsl_user_op
def mbarrier_init(mbar_ptr: Pointer, cnt: Int, *, loc=None, ip=None) -> None:
    """
    Initializes a mbarrier with the specified thread arrival count.

    :param mbar_ptr: A pointer to the mbarrier in SMEM
    :type mbar_ptr:  Pointer
    :param cnt:      The arrival count of the mbarrier
    :type cnt:       Int
    """
    nvvm.mbarrier_init_shared(
        mbar_ptr.llvm_ptr, Int32(cnt).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )


@dsl_user_op
def mbarrier_init_fence(*, loc=None, ip=None) -> None:
    """
    A fence operation that applies to the mbarrier initializations.
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(
        arch,
        [
            "sm_90",
            "sm_90a",
            "sm_100a",
            "sm_100f",
        ],
        "arch",
    )
    nvvm.fence_mbarrier_init(loc=loc, ip=ip)


@dsl_user_op
def mbarrier_arrive_and_expect_tx(
    mbar_ptr: Pointer, bytes: Int, peer_cta_rank_in_cluster=None, *, loc=None, ip=None
) -> None:
    """
    Arrives on a mbarrier and expects a specified number of transaction bytes.

    :param mbar_ptr:                 A pointer to the mbarrier in SMEM
    :type mbar_ptr:                  Pointer
    :param bytes:                    The number of transaction bytes
    :type bytes:                     Int
    :param peer_cta_rank_in_cluster: An optional CTA rank in cluster. If provided, the pointer to
                                     the mbarrier is converted to a remote address in the peer CTA's
                                     SMEM.
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(
        arch,
        [
            "sm_90",
            "sm_90a",
            "sm_100a",
            "sm_100f",
        ],
        "arch",
    )

    mbar_llvm_ptr = mbar_ptr.llvm_ptr
    if peer_cta_rank_in_cluster is not None:
        mbar_llvm_ptr = nvvm.mapa_shared_cluster(
            mbar_llvm_ptr.type,
            mbar_llvm_ptr,
            Int32(peer_cta_rank_in_cluster).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        space = nvvm.MBarrierSpaceKind.CLUSTER
    else:
        space = nvvm.MBarrierSpaceKind.CTA

    nvvm.mbarrier_txn(
        mbar_llvm_ptr,
        Int32(bytes).ir_value(loc=loc, ip=ip),
        kind=nvvm.MBarrierTxnKind.ARRIVE_EXPECT_TX,
        space=space,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_expect_tx(
    mbar_ptr: Pointer, bytes: Int, peer_cta_rank_in_cluster=None, *, loc=None, ip=None
) -> None:
    """
    Expects a specified number of transaction bytes without an arrive.

    :param mbar_ptr:                 A pointer to the mbarrier in SMEM
    :type mbar_ptr:                  Pointer
    :param bytes:                    The number of transaction bytes
    :type bytes:                     Int
    :param peer_cta_rank_in_cluster: An optional CTA rank in cluster. If provided, the pointer to
                                     the mbarrier is converted to a remote address in the peer CTA's
                                     SMEM.
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(
        arch,
        [
            "sm_90",
            "sm_90a",
            "sm_100a",
            "sm_100f",
        ],
        "arch",
    )

    mbar_llvm_ptr = mbar_ptr.llvm_ptr
    if peer_cta_rank_in_cluster is not None:
        mbar_llvm_ptr = nvvm.mapa(
            mbar_llvm_ptr.type,
            mbar_llvm_ptr,
            Int32(peer_cta_rank_in_cluster).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        space = nvvm.MBarrierSpaceKind.CLUSTER
    else:
        space = nvvm.MBarrierSpaceKind.CTA

    nvvm.mbarrier_txn(
        mbar_llvm_ptr,
        Int32(bytes).ir_value(loc=loc, ip=ip),
        kind=nvvm.MBarrierTxnKind.EXPECT_TX,
        space=space,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_wait(mbar_ptr: Pointer, phase: Int, *, loc=None, ip=None) -> None:
    """
    Waits on a mbarrier with a specified phase.

    :param mbar_ptr: A pointer to the mbarrier in SMEM
    :type mbar_ptr:  Pointer
    :param phase:    The phase to wait for (either 0 or 1)
    :type phase:     Int
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(
        arch,
        [
            "sm_90",
            "sm_90a",
            "sm_100a",
            "sm_100f",
        ],
        "arch",
    )

    timeout_ns = 10000000
    # This NVVM Op is a spin-loop wrapping the mbarrier.try_wait.parity.shared.b64 PTX
    # The timeout in ns only applies to the latter and this call is truly blocking
    nvvm.mbarrier_try_wait_parity_shared(
        mbar_ptr.llvm_ptr,
        Int32(phase).ir_value(loc=loc, ip=ip),
        Int32(timeout_ns).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_try_wait(mbar_ptr: Pointer, phase: Int, *, loc=None, ip=None) -> Boolean:
    """
    Attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

    :param mbar_ptr: A pointer to the mbarrier in SMEM
    :type mbar_ptr:  Pointer
    :param phase:    The phase to wait for (either 0 or 1)
    :type phase:     Int
    :return:         A boolean value indicating whether the wait operation was successful
    :rtype:          Boolean
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(
        arch,
        [
            "sm_90",
            "sm_90a",
            "sm_100a",
            "sm_100f",
        ],
        "arch",
    )

    return Boolean(
        nvvm.mbarrier_wait_parity(
            T.bool(),
            mbar_ptr.llvm_ptr,
            Int32(phase).ir_value(loc=loc, ip=ip),
            nvvm.MBarrierWaitKind.TRY,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mbarrier_conditional_try_wait(
    cond, mbar_ptr: Pointer, phase: Int, *, loc=None, ip=None
) -> Boolean:
    """
    Conditionally attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

    :param cond:     A boolean predicate
    :param mbar_ptr: A pointer to the mbarrier in SMEM
    :type mbar_ptr:  Pointer
    :param phase:    The phase to wait for (either 0 or 1)
    :type phase:     Int
    :return:         A boolean value indicating whether the wait operation was successful
    :rtype:          Boolean
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(
        arch,
        [
            "sm_90",
            "sm_90a",
            "sm_100a",
            "sm_100f",
        ],
        "arch",
    )
    return if_generate(
        cond,
        lambda: mbarrier_try_wait(mbar_ptr, phase, loc=loc, ip=ip),
        lambda: Boolean(True).ir_value(loc=loc, ip=ip),
        None,
        [Boolean],
    )


@dsl_user_op
def mbarrier_arrive(
    mbar_ptr: Pointer,
    peer_cta_rank_in_cluster: Optional[Int] = None,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    Arrives on an mbarrier.

    :param mbar_ptr:                 A pointer to the mbarrier in SMEM
    :type mbar_ptr:                  Pointer
    :param peer_cta_rank_in_cluster: An optional CTA rank in cluster. If provided, the pointer to
                                     the mbarrier is converted to a remote address in the peer CTA's
                                     SMEM.
    """
    mbar_llvm_ptr = mbar_ptr.llvm_ptr
    if peer_cta_rank_in_cluster is not None:
        arch = CuTeDSL._get_dsl().envar.arch
        check_value_in(
            arch,
            [
                "sm_90",
                "sm_90a",
                "sm_100a",
                "sm_100f",
            ],
            "arch",
        )

        mbar_llvm_ptr = nvvm.mapa_shared_cluster(
            mbar_llvm_ptr.type,
            mbar_llvm_ptr,
            Int32(peer_cta_rank_in_cluster).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        space = nvvm.MBarrierSpaceKind.CLUSTER
    else:
        space = nvvm.MBarrierSpaceKind.CTA

    nvvm.mbarrier_txn(
        mbar_llvm_ptr,
        Int32(1).ir_value(loc=loc, ip=ip),
        kind=nvvm.MBarrierTxnKind.ARRIVE,
        space=space,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_mbarrier_arrive_noinc(mbar_ptr: Pointer, *, loc=None, ip=None) -> None:
    """
    Arrives on an mbarrier for async load **without incrementing** the arrival count
    (`cp.async.mbarrier.arrive.shared ..., noinc=1`).
    Used in the warp-specialized kernel when the non-TMA load warp(producer) is not the same
    as the math/epilogue warp(consumer).

    :param mbar_ptr: A pointer to the mbarrier in SMEM
    :type mbar_ptr:  Pointer
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(
        arch,
        [
            "sm_90",
            "sm_90a",
            "sm_100a",
            "sm_100f",
        ],
        "arch",
    )

    mbar_llvm_ptr = mbar_ptr.llvm_ptr
    nvvm.cp_async_mbarrier_arrive_shared(
        mbar_llvm_ptr,
        noinc=True,
        loc=loc,
        ip=ip,
    )
