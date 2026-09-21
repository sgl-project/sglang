# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Vendored from flashinfer-ai/flashinfer PR #4266 at 629147317d4149a12e53bcef27808bac380c283f.
"""Blackwell low-M BF16/FP16 GEMM with an in-kernel cluster split-K reduction.

Each cluster rank accumulates an exact K slice in FP32. Peers publish partials
to rank 0 through DSMEM; rank 0 reduces, casts, and stores once. The public
``A[M, K] @ B[K, N]`` problem is swapped internally, so tile dimensions below
use kernel coordinates: kernel-M carries public N and kernel-N carries public M.
"""

from __future__ import annotations

import dataclasses

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

#: Per-CTA SMEM capacity reported by CuTeDSL on SM100/SM103.
_SMEM_CAPACITY_BYTES = 227 * 1024

#: K extent of one CTA tile.
_CTA_K = 128

#: Kernel-M tiles; 64 increases CTA count for low-M decode shapes.
_SUPPORTED_MMA_M = (64, 128)

#: Kernel-N carries public M, which is limited to 32.
_SUPPORTED_MMA_N = (8, 16, 32)

#: Physical cluster-K sizes; split 1 compiles out the DSMEM path.
_SUPPORTED_SPLIT_K = (1, 2, 3, 4, 8, 16)

#: Largest public M this low-M policy serves.
_MAX_M = 32

#: Bytes per FP32 partial exchanged through DSMEM.
_FP32_BYTES = 4

#: DSMEM mailbox base alignment, in bytes.
_MAILBOX_ALIGN_BYTES = 128

#: Size and alignment of one mbarrier, in bytes.
_MBARRIER_BYTES = 8

#: Size and alignment of the TMEM base pointer slot.
_TMEM_POINTER_BYTES = 4

#: Bytes per BF16/FP16 element.
_AB_ELEMENT_BYTES = 2

#: Alignment of the A/B shared-memory buffers.
_AB_BUFFER_ALIGN_BYTES = 1024

#: A/B pipeline stage bounds.
_MIN_AB_STAGES = 2
_MAX_AB_STAGES = 12


@dataclasses.dataclass(frozen=True, slots=True)
class SplitKTactic:
    """One specialization; mma_m carries public N and mma_n carries public M."""

    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _smem_bytes(
    tactic: SplitKTactic,
    ab_stages: int,
) -> int:
    """Mirror the device allocator's shared-memory layout."""
    cursor = (
        _align_up(
            tactic.mma_m * _CTA_K * _AB_ELEMENT_BYTES * ab_stages,
            _AB_BUFFER_ALIGN_BYTES,
        )
        + tactic.mma_n * _CTA_K * _AB_ELEMENT_BYTES * ab_stages
    )

    cursor = _align_up(cursor, _MBARRIER_BYTES)
    cursor += 2 * ab_stages * _MBARRIER_BYTES
    cursor += 3 * _MBARRIER_BYTES
    cursor = _align_up(cursor, _TMEM_POINTER_BYTES)
    cursor += _TMEM_POINTER_BYTES

    if tactic.split_k == 1:
        return cursor

    return (
        _align_up(
            _align_up(cursor, _MAILBOX_ALIGN_BYTES)
            + (tactic.split_k - 1) * tactic.mma_m * tactic.mma_n * _FP32_BYTES,
            _MBARRIER_BYTES,
        )
        + _MBARRIER_BYTES
    )


def _max_ab_stages_for(
    tactic: SplitKTactic,
    smem_capacity: int,
) -> int:
    return next(
        (
            stages
            for stages in range(_MAX_AB_STAGES, -1, -1)
            if _smem_bytes(tactic, stages) <= smem_capacity
        ),
        0,
    )


def validate_tactic(
    tactic: SplitKTactic,
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int = _SMEM_CAPACITY_BYTES,
) -> None:
    """Reject a tactic that cannot serve ``(m, n, k)``."""
    if tactic.mma_m not in _SUPPORTED_MMA_M:
        raise ValueError(f"unsupported mma_m={tactic.mma_m}")
    if tactic.mma_n not in _SUPPORTED_MMA_N:
        raise ValueError(f"unsupported mma_n={tactic.mma_n}")
    if tactic.split_k not in _SUPPORTED_SPLIT_K:
        raise ValueError(f"unsupported split_k={tactic.split_k}")
    if not _MIN_AB_STAGES <= tactic.ab_stages <= _MAX_AB_STAGES:
        raise ValueError(
            f"ab_stages must be in [{_MIN_AB_STAGES}, {_MAX_AB_STAGES}], "
            f"got {tactic.ab_stages}"
        )
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"this low-M policy requires 1 <= M <= {_MAX_M}, got {m}")
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    if k <= 0 or k % _CTA_K or (k // _CTA_K) % tactic.split_k:
        raise ValueError(
            f"K={k} with CTA_K={_CTA_K} does not divide evenly across "
            f"split_k={tactic.split_k}"
        )
    smem_bytes = _smem_bytes(tactic, tactic.ab_stages)
    if smem_bytes > smem_capacity:
        raise ValueError(
            f"tactic {tactic} needs {smem_bytes} B of shared memory but only "
            f"{smem_capacity} B are available; max ab_stages is "
            f"{_max_ab_stages_for(tactic, smem_capacity)}"
        )


def autotune_tactics(
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int = _SMEM_CAPACITY_BYTES,
) -> list[SplitKTactic]:
    """Return valid tactics in the shape-specific stage window."""
    tactics: list[SplitKTactic] = []
    for mma_m in _SUPPORTED_MMA_M:
        for mma_n in _SUPPORTED_MMA_N:
            for split_k in _SUPPORTED_SPLIT_K:
                base = SplitKTactic(mma_m, mma_n, split_k, _MIN_AB_STAGES)
                try:
                    validate_tactic(base, m, n, k, smem_capacity=smem_capacity)
                except ValueError:
                    continue
                max_stages = _max_ab_stages_for(base, smem_capacity)
                # Short K favors shallow pipelines; long K stays near the cap.
                tactics.extend(
                    dataclasses.replace(base, ab_stages=ab_stages)
                    for ab_stages in (
                        range(_MIN_AB_STAGES, min(max_stages, 6) + 1)
                        if k <= 4 * _CTA_K
                        else range(
                            min(max(5, max_stages - 2), max_stages),
                            max_stages + 1,
                        )
                    )
                )
    return tactics


def default_tactic(m: int, n: int, k: int) -> SplitKTactic:
    """Choose the default occupancy-oriented tactic."""
    if n <= 512:
        mma_m = 64
        mma_n = 16 if n == 512 and m > 24 else 8
        requested_split = 4
    else:
        mma_n = 8 if m <= 8 else 16 if m <= 16 else 32
        if n <= 3072:
            mma_m = 128 if m <= 16 else 64
            requested_split = 4 if m <= 16 else 2
        elif n < 8192:
            mma_m = 64
            requested_split = 2
        else:
            mma_m = 128 if k <= 1024 and m <= 24 else 64
            requested_split = 1

    if k <= 4 * _CTA_K:
        requested_split = 1
    split_k = next(
        split_k
        for split_k in reversed(_SUPPORTED_SPLIT_K)
        if split_k <= requested_split and (k // _CTA_K) % split_k == 0
    )
    tactic = SplitKTactic(mma_m, mma_n, split_k, _MIN_AB_STAGES)
    max_stages = _max_ab_stages_for(tactic, _SMEM_CAPACITY_BYTES)
    tactic = dataclasses.replace(
        tactic,
        ab_stages=(
            _MIN_AB_STAGES
            if k <= 2 * _CTA_K and m > 8
            else min(max_stages, 6)
            if k <= 4 * _CTA_K
            else max_stages
        ),
    )
    validate_tactic(tactic, m, n, k)
    return tactic


__all__ = [
    "SplitKTactic",
    "autotune_tactics",
    "default_tactic",
    "run_splitk_dense",
    "run_splitk_dense_silu",
    "run_splitk_dense_silu_aux",
    "run_splitk_dense_gate",
]


@dsl_user_op
def _map_shared_rank(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Map an SMEM pointer into a peer CTA's address space."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                smem_ptr.toint(loc=loc, ip=ip).ir_value(),
                peer_cta_rank_in_cluster.ir_value(),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_shared_remote_v4(
    value0,
    value1,
    value2,
    value3,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Publish four FP32 partials into a peer's SMEM, crediting 16 bytes."""
    llvm.inline_asm(
        None,
        [
            _map_shared_rank(
                smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
            ).ir_value(),
            value0.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value1.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value2.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value3.bitcast(Int32).ir_value(loc=loc, ip=ip),
            _map_shared_rank(
                mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
            ).ir_value(),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


#: Rank that gathers partials and stores the output.
OWNER_RANK = 0


def _sigmoid_f32(v):
    return cute_math.rcp(cute_math.exp(v * -1.0) + 1.0)


#: Epilogue modes; "none" preserves the vendored store path byte-for-byte.
_EPILOGUE_MODES = (
    "none",
    "silu",
    "gate",
    "silu_aux_direct",
    "silu_aux_alpha_only",
)

#: Public selector names for packed-aux epilogues.
_SILU_AUX_IMPLS = {
    "direct": "silu_aux_direct",
    "alpha_only": "silu_aux_alpha_only",
}


def resolve_splitk_dense_silu_aux_impl(
    aux_impl: str,
    aux_start: int,
    aux_width: int,
    n: int,
    tactic: SplitKTactic,
) -> str:
    """Resolve the requested packed-aux epilogue to its executable policy."""
    if aux_impl not in _SILU_AUX_IMPLS:
        raise ValueError(
            f"aux_impl must be one of {tuple(_SILU_AUX_IMPLS)}, got {aux_impl!r}"
        )
    if aux_impl == "alpha_only" and (
        aux_start + aux_width != n or aux_start % tactic.mma_m != 0
    ):
        # A mixed or non-trailing tile may contain useful ordinary output.
        return "direct"
    return aux_impl


#: Named barrier for the gate epilogue's SMEM staging round-trip.
_GATE_BARRIER_ID = 7

#: Alignment of the gate epilogue SMEM tile.
_GATE_TILE_ALIGN_BYTES = 16


class SplitKDenseGemmKernel:
    """Standalone BF16/FP16 GEMM with a cluster-local split-K reduction."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        has_bias: bool,
        epilogue_mode: str = "none",
        epilogue_scale: float = 1.0,
        epilogue_group: int = 1,
        epilogue_aux_start: int = 0,
        epilogue_aux_width: int = 0,
        raw_rms: bool = False,
        raw_sum: bool = False,
        reset_sum: bool = False,
        rms_eps: float = 0.0,
        rms_inv_h: float = 0.0,
        branch_packed: bool = False,
    ) -> None:
        self.acc_dtype = cutlass.Float32
        self.cta_m = tactic.mma_m
        self.cta_n = tactic.mma_n
        self.cta_k = _CTA_K
        self.num_ab_stage = tactic.ab_stages
        self.split_k = tactic.split_k
        self.use_pdl = use_pdl
        self.has_bias = has_bias
        self.epilogue_mode = epilogue_mode
        self.epilogue_scale = epilogue_scale
        self.epilogue_group = epilogue_group
        self.epilogue_aux_start = epilogue_aux_start
        self.epilogue_aux_width = epilogue_aux_width
        self.raw_rms = raw_rms
        self.raw_sum = raw_sum
        self.reset_sum = reset_sum
        self.branch_packed = branch_packed
        if branch_packed and (not raw_rms or epilogue_mode == "gate"):
            raise ValueError("branch-packed weights require raw HC Down")
        # Retained small-T sum Down: preserve the existing gate/aux staging
        # reservations on every other path. No shape-specific H/T dispatch.
        self.launch_smem_bytes = (
            _smem_bytes(tactic, tactic.ab_stages)
            if raw_sum and epilogue_mode == "silu_aux_alpha_only"
            else utils.get_smem_capacity_in_bytes("sm_100")
        )
        self.rms_eps = rms_eps
        self.rms_inv_h = rms_inv_h
        if raw_sum and not raw_rms:
            raise ValueError("sum-state requires raw RMS")
        if reset_sum and (not raw_rms or epilogue_mode != "gate"):
            raise ValueError("next-sum reset requires a raw gate epilogue")
        if raw_rms and epilogue_mode != "gate":
            if epilogue_mode not in _SILU_AUX_IMPLS.values():
                raise ValueError("raw RMS requires a gate or packed-aux epilogue")
            if epilogue_group != 4 or tactic.split_k % epilogue_group:
                raise ValueError("raw Down requires C=4 and branch-aligned split-K")

        if epilogue_mode not in _EPILOGUE_MODES:
            raise ValueError(f"unsupported epilogue_mode={epilogue_mode}")
        if epilogue_mode == "gate":
            if tactic.split_k != 1:
                raise ValueError("gate epilogue requires split_k=1")
            if has_bias:
                raise ValueError("gate epilogue does not support bias")
            if epilogue_group < 2 or tactic.mma_m % epilogue_group:
                raise ValueError(
                    f"gate epilogue_group={epilogue_group} must divide "
                    f"mma_m={tactic.mma_m}"
                )
            gate_out_elems = (tactic.mma_m // epilogue_group) * tactic.mma_n
            if gate_out_elems % 128:
                raise ValueError(
                    f"gate tile ({tactic.mma_m}, {tactic.mma_n}) gives "
                    f"{gate_out_elems} outputs; must be a multiple of 128"
                )
            gate_smem = (
                _align_up(_smem_bytes(tactic, tactic.ab_stages), _GATE_TILE_ALIGN_BYTES)
                + tactic.mma_m * tactic.mma_n * _FP32_BYTES
            )
            if gate_smem > _SMEM_CAPACITY_BYTES:
                raise ValueError(
                    f"gate epilogue needs {gate_smem} B of shared memory; "
                    f"only {_SMEM_CAPACITY_BYTES} B available"
                )
            if raw_sum:
                # Include the FP32 gate tile and its allocator alignment.
                self.launch_smem_bytes = gate_smem
        if epilogue_mode in (
            "silu_aux_direct",
            "silu_aux_alpha_only",
        ):
            if has_bias:
                raise ValueError("silu_aux epilogues do not support bias")
            if epilogue_aux_start < 0 or epilogue_aux_width <= 0:
                raise ValueError(
                    "silu_aux epilogues require non-negative aux_start and "
                    "positive aux_width"
                )
        if epilogue_mode == "silu_aux_alpha_only" and epilogue_aux_start % tactic.mma_m:
            raise ValueError("silu_aux_alpha_only requires aux_start aligned to mma_m")

        self.threads_per_cta = 256
        self.epilog_threads = 128
        self.mma_tiler_mn = (tactic.mma_m, tactic.mma_n)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
        self.cluster_shape = (1, tactic.split_k, 1)

        values_per_thread = (tactic.mma_m * tactic.mma_n) // self.epilog_threads
        if values_per_thread % 4:
            raise ValueError(
                f"CTA tile ({tactic.mma_m}, {tactic.mma_n}) gives "
                f"{values_per_thread} "
                "values per epilogue thread; remote stores require a multiple of 4"
            )
        self.mailbox_elements = (
            (tactic.split_k - 1) * self.epilog_threads * values_per_thread
        )
        self.expected_transaction_bytes = self.mailbox_elements * _FP32_BYTES

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        x: cute.Tensor,
        out: cute.Tensor,
        rms_stats: cute.Tensor,
        next_sum_sq: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, bias, x, out, rms_stats, next_sum_sq).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.launch_smem_bytes),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        mBias: cute.Tensor,  # Broadcast bias; dead when has_bias=False
        mX: cute.Tensor,  # Gate activation; dead unless epilogue_mode="gate"
        mOut: cute.Tensor,  # Gate output; dead unless epilogue_mode="gate"
        mRmsStats: cute.Tensor,  # Read-only current sum if raw_sum, else inv.
        mNextSum: cute.Tensor,  # Write-only next-state reset; reset_sum only.
    ):
        """Allocate storage and dispatch the specialized warps."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_mma_epilog

        if cutlass.const_expr(self.epilogue_mode == "gate"):
            gate_tile = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout((self.cta_m, self.cta_n)),
                alignment=_GATE_TILE_ALIGN_BYTES,
            )
        else:
            gate_tile = mailbox

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish peer barriers before cross-CTA stores.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Host validation guarantees an equal, tail-free K partition.
        if cutlass.const_expr(self.branch_packed):
            # mA is [N,H,C], backed by the large-T [C,Npad,H] storage.
            # Each rank stays inside one branch; mB still spans the full C*H.
            ranks_per_branch = self.split_k // self.epilogue_group
            weight_batch = split_rank // ranks_per_branch
            k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // ranks_per_branch
            weight_k_start = (split_rank % ranks_per_branch) * k_tile_count
        else:
            weight_batch = l_idx
            k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
            weight_k_start = split_rank * k_tile_count
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(
                    mA, (self.cta_m, self.cta_k), (bidx, None, weight_batch)
                ),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                weight_k_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.local_tile(mBias, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                gate_tile,
                mX,
                mOut,
                mRmsStats,
                mNextSum,
                bidx,
                n_idx,
            )

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        bar_tma_epilog,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        stages = self.num_ab_stage
        if cutlass.const_expr(not is_a and self.use_pdl):
            cute.arch.griddepcontrol_wait()

        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s_tile.element_type,
                        cute.slice_(s_tile.layout, (None, None, None, 0)),
                    ),
                )
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        if cutlass.const_expr(is_a and self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(not is_a and self.has_bias):
            cute.arch.mbarrier_arrive(bar_tma_epilog)
        self._drain_producer(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def _drain_producer(
        self,
        bar_empty,
        empty_phase: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):
        stages = self.num_ab_stage
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def mma_warp(
        self,
        bar_full,
        bar_empty,
        bar_mma_epilog,
        bar_tmem_alloc,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
            if stage == self.num_ab_stage - 1:
                full_phase = full_phase ^ 1

        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        gBias_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        gate_tile,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        mRmsStats: cute.Tensor,
        mNextSum: cute.Tensor,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
    ):
        if cutlass.const_expr((self.raw_sum or self.reset_sum) and self.use_pdl):
            # The A/weight producer can release the next grid early. Scalar
            # state accesses must also wait, independently of the B DMA warp.
            cute.arch.griddepcontrol_wait()
        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        if cutlass.const_expr(self.has_bias):
            bias_dtype = gBias_tile.element_type
            rBias = cute_ext.allocate(
                bias_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rBiasAcc = cute_ext.allocate(
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            if split_rank == OWNER_RANK:
                cute.arch.mbarrier_wait(bar_tma_epilog, 0)
                cute_ext.partition_and_copy(
                    cute.make_tiled_copy_D(
                        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                        tiled_copy_t2r,
                    ).get_slice(epi_tid),
                    cute.flat_divide(gBias_tile, epi_tile)[None, None, 0, 0],
                    rBias,
                )
                rBiasAcc.store(rBias.load().to(self.acc_dtype))

        if cutlass.const_expr(self.raw_sum and self.epilogue_mode != "gate"):
            # Same early scalar preparation as the retained isolated Down.
            # Every split rank consumes current sums independently; no inv
            # publication or dependency on another CTA's global store.
            rEarlyInv = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            early_coords = cute.flat_divide(
                cute.make_identity_tensor(epi_tile), epi_tile
            )
            tEarlyCoord = thr_t2r.partition_D(early_coords)[(None, None, None, 0, 0)]
            early_branch = split_rank // (self.split_k // self.epilogue_group)
            for early_idx in cutlass.range_constexpr(cute.size(rmem_layout)):
                early_row = n_idx * self.cta_n + tEarlyCoord[early_idx][1]
                early_inv = cutlass.Float32(0.0)
                if early_row < cute.size(mRmsStats, mode=[0]):
                    early_inv = cute.math.rsqrt(
                        mRmsStats[early_row, early_branch] * self.rms_inv_h
                        + self.rms_eps,
                        fastmath=True,
                    )
                rEarlyInv[early_idx] = early_inv

        if cutlass.const_expr(self.raw_sum and self.epilogue_mode == "gate"):
            # These per-output inverses do not depend on MMA results. Prepare
            # them after PDL/TMEM setup, keeping current sums read-only.
            early_group = self.epilogue_group
            early_j_per_tile = self.cta_m // early_group
            early_iterations = early_j_per_tile * self.cta_n // self.epilog_threads
            rEarlyGateInv = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.rmem,
                cute.make_layout((early_group, early_iterations)),
                alignment=32,
            )
            for early_it in cutlass.range_constexpr(early_iterations):
                early_row = (
                    n_idx * self.cta_n
                    + (early_it * self.epilog_threads + epi_tid) // early_j_per_tile
                )
                for early_g in cutlass.range_constexpr(early_group):
                    early_inv = cutlass.Float32(0.0)
                    if early_row < cute.size(mOut, mode=[0]):
                        early_inv = cute.math.rsqrt(
                            mRmsStats[early_row, early_g] * self.rms_inv_h
                            + self.rms_eps,
                            fastmath=True,
                        )
                    rEarlyGateInv[early_g, early_it] = early_inv

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.raw_rms and self.epilogue_mode != "gate"):
            # A rank's contiguous K partition stays inside exactly one HC
            # branch. Scale its FP32 partial BEFORE inter-branch reduction.
            coords = cute.flat_divide(cute.make_identity_tensor(epi_tile), epi_tile)
            tRawCoord = thr_t2r.partition_D(coords)[(None, None, None, 0, 0)]
            branch = split_rank // (self.split_k // self.epilogue_group)
            for value_idx in cutlass.range_constexpr(cute.size(rmem_layout)):
                row = n_idx * self.cta_n + tRawCoord[value_idx][1]
                if row < cute.size(mRmsStats, mode=[0]):
                    if cutlass.const_expr(self.raw_sum):
                        inv = rEarlyInv[value_idx]
                    else:
                        inv = mRmsStats[row, branch]
                    rAcc[value_idx] = rAcc[value_idx] * inv

        # Peers publish FP32 partials; only rank 0 reduces and stores.
        if cutlass.const_expr(self.split_k > 1):
            assert cute.size(rmem_layout) == self.mailbox_elements // (
                (self.split_k - 1) * self.epilog_threads
            )
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(
                self.epilog_threads * values_per_thread
            )
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer
                                + epi_tid * values_per_thread
                                + value_idx
                            ]
                        )

        if split_rank == OWNER_RANK:
            if cutlass.const_expr(self.has_bias):
                rAcc.store(rAcc.load() + rBiasAcc.load())

            if cutlass.const_expr(
                self.epilogue_mode in ("silu_aux_direct", "silu_aux_alpha_only")
            ):
                # Mirror the TMEM->RMEM destination partition with an identity
                # tensor.  The coordinate at each ordinal therefore belongs to
                # the same epilogue thread and the same accumulator register as
                # rAcc[value_idx], without a SMEM transpose or CTA barrier.
                cAux_epi = cute.flat_divide(
                    cute.make_identity_tensor(epi_tile), epi_tile
                )
                tCoord = thr_t2r.partition_D(cAux_epi)[(None, None, None, 0, 0)]
                tile_start = bidx * self.cta_m
                if (
                    tile_start < self.epilogue_aux_start + self.epilogue_aux_width
                    and tile_start + self.cta_m > self.epilogue_aux_start
                ):
                    rows = cute.size(mOut, mode=[0])
                    for value_idx in cutlass.range_constexpr(cute.size(rmem_layout)):
                        coord = tCoord[value_idx]
                        # Kernel axes are swapped relative to the public GEMM:
                        # kernel-M is the output column, kernel-N is the row.
                        j_global = bidx * self.cta_m + coord[0]
                        m_global = n_idx * self.cta_n + coord[1]
                        if (
                            m_global < rows
                            and j_global >= self.epilogue_aux_start
                            and j_global
                            < self.epilogue_aux_start + self.epilogue_aux_width
                        ):
                            raw = rAcc[value_idx]
                            mOut[
                                m_global,
                                j_global - self.epilogue_aux_start,
                            ] = cutlass.Float32(2.0) * _sigmoid_f32(
                                raw * self.epilogue_scale
                            )

            if cutlass.const_expr(self.epilogue_mode == "silu_aux_alpha_only"):
                # The optimized HC contract places a trailing, CTA-aligned
                # Inject slice after the useful Down columns.  Its tile only
                # emits FP32 alpha: skip the otherwise dead SiLU, dtype cast,
                # and packed-output store.  Earlier tiles keep the ordinary
                # Down epilogue unchanged.
                tile_start = bidx * self.cta_m
                if tile_start < self.epilogue_aux_start:
                    scaled = rAcc.load() * self.epilogue_scale
                    rAcc.store(scaled * _sigmoid_f32(scaled))
                    rD.store(rAcc.load().to(c_dtype))
                    # Preserve TMEM coordinates; the copy predicates tails.
                    cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])
            elif cutlass.const_expr(self.epilogue_mode in ("silu", "silu_aux_direct")):
                scaled = rAcc.load() * self.epilogue_scale
                rAcc.store(scaled * _sigmoid_f32(scaled))
                rD.store(rAcc.load().to(c_dtype))
                # Preserve TMEM coordinates; the copy predicates output tails.
                cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])
            elif cutlass.const_expr(self.epilogue_mode == "gate"):
                group = self.epilogue_group
                rSig = cute_ext.allocate(
                    self.acc_dtype,
                    cute.AddressSpace.rmem,
                    rmem_layout,
                    alignment=32,
                )
                rSig.store(_sigmoid_f32(rAcc.load()))
                sGate_epi = cute.flat_divide(gate_tile, epi_tile)
                cute_ext.partition_and_copy(thr_t2r, rSig, sGate_epi[None, None, 0, 0])
                cute.arch.barrier(
                    barrier_id=_GATE_BARRIER_ID,
                    number_of_threads=self.epilog_threads,
                )

                rows = cute.size(mOut, mode=[0])
                hs = cute.size(mOut, mode=[1])
                if cutlass.const_expr(self.reset_sum):
                    # Exactly one CTA per token tile clears the NEXT state.
                    # Up's dependency wait precedes these stores; Apply uses a
                    # normal stream launch and cannot race this reset.
                    if bidx == 0:
                        for reset_it in cutlass.range_constexpr(
                            cute.ceil_div(self.cta_n * group, self.epilog_threads)
                        ):
                            reset_idx = reset_it * self.epilog_threads + epi_tid
                            reset_row = n_idx * self.cta_n + reset_idx // group
                            if reset_idx < self.cta_n * group and reset_row < rows:
                                mNextSum[reset_row, reset_idx % group] = (
                                    cutlass.Float32(0.0)
                                )
                j_per_tile = self.cta_m // group
                total_out = j_per_tile * self.cta_n
                for it in cutlass.range_constexpr(total_out // self.epilog_threads):
                    elem = it * self.epilog_threads + epi_tid
                    j_local = elem % j_per_tile
                    m_local = elem // j_per_tile
                    m_global = n_idx * self.cta_n + m_local
                    j_global = bidx * j_per_tile + j_local
                    if m_global < rows:
                        gated = cutlass.Float32(0.0)
                        for g in cutlass.range_constexpr(group):
                            sig = gate_tile[j_local * group + g, m_local]
                            xv = mX[m_global, g * hs + j_global].to(cutlass.Float32)
                            if cutlass.const_expr(self.raw_rms):
                                weight = gBias_tile[j_local * group + g, m_local].to(
                                    cutlass.Float32
                                )
                                if cutlass.const_expr(self.raw_sum):
                                    inv = rEarlyGateInv[g, it]
                                else:
                                    inv = mRmsStats[m_global, g]
                                # Match materialized RMSNorm's activation dtype
                                # before the gate multiply; gamma is 1+Wnorm.
                                xv = (
                                    (xv * inv * (cutlass.Float32(1.0) + weight))
                                    .to(c_dtype)
                                    .to(cutlass.Float32)
                                )
                            gated = gated + sig * xv
                        mOut[m_global, j_global] = (gated * self.epilogue_scale).to(
                            c_dtype
                        )
            else:
                rD.store(rAcc.load().to(c_dtype))
                # Preserve TMEM coordinates; the copy predicates output tails.
                cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        # The reduction mbarrier covers remote stores; no cluster barrier needed.


import torch as _torch

_SUPPORTED_TORCH_DTYPES = (_torch.bfloat16, _torch.float16)


@cute.experimental.jit
def _bmm_no_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        c,
        c,
        c,
        c,
        stream,
    )


@cute.experimental.jit
def _bmm_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    bias: cute.Tensor,
    stream: _cuda.CUstream,
):
    c_swapped = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c_swapped,
        cute.make_tensor(bias.iterator, cute.select(bias.layout, mode=[1, 2, 0])),
        c_swapped,
        c_swapped,
        c_swapped,
        c_swapped,
        stream,
    )


@cute.experimental.jit
def _bmm_gate(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    x: cute.Tensor,
    out: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        x,
        out,
        c,
        c,
        stream,
    )


@cute.experimental.jit
def _bmm_silu_aux(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    aux: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        c,
        aux,
        c,
        c,
        stream,
    )


@cute.experimental.jit
def _bmm_silu_aux_raw(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    aux: cute.Tensor,
    rms_stats: cute.Tensor,
    unused_next_sum: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        c,
        c,
        aux,
        rms_stats,
        unused_next_sum,
        stream,
    )


@cute.experimental.jit
def _bmm_gate_raw(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    x: cute.Tensor,
    out: cute.Tensor,
    rms_stats: cute.Tensor,
    norm_weight: cute.Tensor,
    next_sum_sq: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        norm_weight,
        x,
        out,
        rms_stats,
        next_sum_sq,
        stream,
    )


def _from_dlpack_dynamic(tensor, leading_dim: int, assumed_align: int = 32):
    return from_dlpack(tensor, assumed_align=assumed_align).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def _detect_leading_dim(tensor: _torch.Tensor) -> int:
    # Ignore synthetic batch stride, including 1x1.
    for dim, stride in enumerate(tensor.stride()[1:], start=1):
        if stride == 1:
            return dim
    raise ValueError("tensor has no stride-1 dimension")


def _make_layout_tensor(
    shape: tuple[int, ...], dtype: _torch.dtype, leading_dim: int
) -> _torch.Tensor:
    permutation = [dim for dim in range(len(shape)) if dim != leading_dim] + [
        leading_dim
    ]
    return _torch.empty(
        tuple(shape[dim] for dim in permutation), dtype=dtype, device="cuda"
    ).permute([permutation.index(dim) for dim in range(len(shape))])


def _make_compile_repr_tensors(
    dtype: _torch.dtype,
    has_bias: bool,
    a_leading: int,
    b_leading: int,
    c_leading: int,
    epilogue_mode: str = "none",
    epilogue_aux_width: int = 0,
    raw_rms: bool = False,
    epilogue_group: int = 1,
    branch_packed: bool = False,
):
    m, n, k, batch = 64, 8, _CTA_K, 1
    tensors = tuple(
        _from_dlpack_dynamic(
            _make_layout_tensor(shape, dtype, leading_dim), leading_dim
        )
        for shape, leading_dim in zip(
            (
                (epilogue_group if branch_packed else batch, n, k),
                (batch, k, m),
                (batch, n, m),
            ),
            (a_leading, b_leading, c_leading),
            strict=True,
        )
    )
    if epilogue_mode == "gate":
        extra = ()
        if raw_rms:
            extra = (
                _from_dlpack_dynamic(
                    _torch.empty(
                        (n, epilogue_group), dtype=_torch.float32, device="cuda"
                    ),
                    1,
                ),
                _from_dlpack_dynamic(
                    _torch.empty((m,), dtype=dtype, device="cuda").as_strided(
                        (m, n, 1), (1, 0, 0)
                    ),
                    0,
                ),
                _from_dlpack_dynamic(
                    _torch.empty(
                        (n, epilogue_group), dtype=_torch.float32, device="cuda"
                    ),
                    1,
                ),
            )
        return (
            *tensors,
            _from_dlpack_dynamic(_torch.empty((n, m), dtype=dtype, device="cuda"), 1),
            _from_dlpack_dynamic(_torch.empty((n, m), dtype=dtype, device="cuda"), 1),
            *extra,
        )
    if epilogue_mode in (
        "silu_aux_direct",
        "silu_aux_alpha_only",
    ):
        extra = ()
        if raw_rms:
            extra = (
                _from_dlpack_dynamic(
                    _torch.empty(
                        (n, epilogue_group), dtype=_torch.float32, device="cuda"
                    ),
                    1,
                ),
                _from_dlpack_dynamic(
                    _torch.empty(
                        (n, epilogue_group), dtype=_torch.float32, device="cuda"
                    ),
                    1,
                ),
            )
        return (
            *tensors,
            _from_dlpack_dynamic(
                _torch.empty(
                    (n, epilogue_aux_width),
                    dtype=_torch.float32,
                    device="cuda",
                ),
                1,
            ),
            *extra,
        )
    if not has_bias:
        return (*tensors, None)
    return (
        *tensors,
        _from_dlpack_dynamic(
            _torch.empty((n,), dtype=dtype, device="cuda").as_strided(
                size=(batch, n, m), stride=(0, 1, 0)
            ),
            1,
            2,
        ),
    )


def _to_cute_swap(a, b, out, bias, *, branch_packed=False):
    a_swap = (
        b[:, :, : out.shape[1]].transpose(-2, -1)
        if branch_packed
        else b.unsqueeze(0).transpose(-2, -1)
    )
    b_swap = a.unsqueeze(0).transpose(-2, -1)
    c_swap = out.unsqueeze(0).transpose(-2, -1)
    leading_dims = tuple(
        _detect_leading_dim(tensor) for tensor in (a_swap, b_swap, c_swap)
    )
    cute_tensors = tuple(
        _from_dlpack_dynamic(tensor, leading_dim)
        for tensor, leading_dim in zip(
            (a_swap, b_swap, c_swap), leading_dims, strict=True
        )
    )
    if bias is None:
        return (*cute_tensors, None, leading_dims)
    return (
        *cute_tensors,
        _from_dlpack_dynamic(
            bias.as_strided(
                size=(1, c_swap.shape[1], c_swap.shape[2]), stride=(0, 1, 0)
            ),
            1,
            2,
        ),
        leading_dims,
    )


# Tactic hashes all compile-time tile, split, and stage fields.
_SPLITK_COMPILE_CACHE: dict = {}


def _get_compiled_splitk_kernel(
    dtype,
    tactic: SplitKTactic,
    use_pdl: bool,
    has_bias: bool,
    leading_dims: tuple[int, int, int],
    epilogue_mode: str = "none",
    epilogue_scale: float = 1.0,
    epilogue_group: int = 1,
    epilogue_aux_start: int = 0,
    epilogue_aux_width: int = 0,
    raw_rms: bool = False,
    raw_sum: bool = False,
    reset_sum: bool = False,
    rms_eps: float = 0.0,
    rms_inv_h: float = 0.0,
    branch_packed: bool = False,
):
    key = (
        dtype,
        tactic,
        use_pdl,
        has_bias,
        epilogue_mode,
        epilogue_scale,
        epilogue_group,
        epilogue_aux_start,
        epilogue_aux_width,
        raw_rms,
        raw_sum,
        reset_sum,
        rms_eps,
        rms_inv_h,
        branch_packed,
        *leading_dims,
    )
    cached = _SPLITK_COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    if dtype not in _SUPPORTED_TORCH_DTYPES:
        raise ValueError(
            f"split-K dense GEMM supports {_SUPPORTED_TORCH_DTYPES}; got {dtype}"
        )

    kernel = SplitKDenseGemmKernel(
        tactic=tactic,
        use_pdl=use_pdl,
        has_bias=has_bias,
        epilogue_mode=epilogue_mode,
        epilogue_scale=epilogue_scale,
        epilogue_group=epilogue_group,
        epilogue_aux_start=epilogue_aux_start,
        epilogue_aux_width=epilogue_aux_width,
        raw_rms=raw_rms,
        raw_sum=raw_sum,
        reset_sum=reset_sum,
        rms_eps=rms_eps,
        rms_inv_h=rms_inv_h,
        branch_packed=branch_packed,
    )
    compile_tensors = _make_compile_repr_tensors(
        dtype,
        has_bias,
        *leading_dims,
        epilogue_mode=epilogue_mode,
        epilogue_aux_width=epilogue_aux_width,
        raw_rms=raw_rms,
        epilogue_group=epilogue_group,
        branch_packed=branch_packed,
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    if epilogue_mode == "gate":
        wrapper = _bmm_gate_raw if raw_rms else _bmm_gate
        compiled = cute_ext.compile(wrapper, kernel, *compile_tensors, stream)
    elif epilogue_mode in (
        "silu_aux_direct",
        "silu_aux_alpha_only",
    ):
        wrapper = _bmm_silu_aux_raw if raw_rms else _bmm_silu_aux
        compiled = cute_ext.compile(wrapper, kernel, *compile_tensors, stream)
    elif has_bias:
        compiled = cute_ext.compile(_bmm_bias, kernel, *compile_tensors, stream)
    else:
        compiled = cute_ext.compile(_bmm_no_bias, kernel, *compile_tensors[:3], stream)
    _SPLITK_COMPILE_CACHE[key] = compiled
    return compiled


def _validate_runtime_tensors(a, b, bias, out) -> tuple[int, int, int]:
    tensors = (a, b, out) + ((bias,) if bias is not None else ())
    if any(not isinstance(tensor, _torch.Tensor) for tensor in tensors):
        raise ValueError("a, b, out, and bias must be torch tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("split-K dense GEMM accepts only 2D tensors")
    if a.device.type != "cuda" or any(tensor.device != a.device for tensor in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if a.dtype not in _SUPPORTED_TORCH_DTYPES or any(
        tensor.dtype != a.dtype for tensor in tensors
    ):
        raise ValueError("a, b, out, and bias must share BF16 or FP16 dtype")

    def _is_dense_2d(tensor: _torch.Tensor) -> bool:
        rows, cols = tensor.shape
        return (tensor.stride(1) == 1 and tensor.stride(0) >= cols) or (
            tensor.stride(0) == 1 and tensor.stride(1) >= rows
        )

    if any(not _is_dense_2d(tensor) for tensor in (a, b, out)):
        raise ValueError(
            "a, b, and out must be row-major or column-major matrices "
            "(padded leading strides allowed)"
        )
    if any(tensor.data_ptr() % 32 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 32-byte aligned")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(
            f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    if bias is not None and (
        bias.ndim != 1 or bias.shape[0] != n or not bias.is_contiguous()
    ):
        raise ValueError(
            f"bias must be contiguous with shape {(n,)}, "
            f"got shape {tuple(bias.shape)} and stride {bias.stride()}"
        )

    return m, n, k


def _validate_branch_packed_tensors(a, b, out):
    """B is [C,H,Npad], a transpose view of contiguous [C,Npad,H]."""
    if any(not isinstance(t, _torch.Tensor) for t in (a, b, out)):
        raise ValueError("a, b, and out must be torch tensors")
    if a.ndim != 2 or b.ndim != 3 or out.ndim != 2:
        raise ValueError("branch-packed Down requires A[T,C*H], B[C,H,Npad], out[T,N]")
    c, h, npad = b.shape
    if (
        c != 4
        or h <= 0
        or a.shape[1] != c * h
        or not 0 < out.shape[1] <= npad
        or b.stride() != (npad * h, 1, h)
    ):
        raise ValueError("expected C=4, full C*H input and branch-major weight storage")
    # Reuse dtype, device, alignment and output-stride validation on one branch.
    m, n, _ = _validate_runtime_tensors(a[:, :h], b[0, :, : out.shape[1]], None, out)
    return m, n, c * h


def _validate_inv_rms(inv_rms, a, rows: int, group: int):
    if (
        inv_rms.shape != (rows, group)
        or inv_rms.dtype != _torch.float32
        or inv_rms.device != a.device
        or not inv_rms.is_contiguous()
        or inv_rms.data_ptr() % 32
    ):
        raise ValueError(
            "RMS statistics must be aligned contiguous FP32 [M,C], matching a device"
        )


def _raw_rms_input(inv_rms, sum_sq, a, rows, group, rms_eps):
    """Exactly one read-only statistics representation; never an inv output."""
    if inv_rms is not None and sum_sq is not None:
        raise ValueError("provide inv_rms or sum_sq, not both")
    if sum_sq is not None:
        import math

        if not math.isfinite(rms_eps) or rms_eps <= 0:
            raise ValueError("sum-state requires positive finite epsilon")
    stats = sum_sq if sum_sq is not None else inv_rms
    if stats is not None:
        _validate_inv_rms(stats, a, rows, group)
    return stats


def run_splitk_dense(
    a,
    b,
    bias,
    out,
    pdl: bool,
    tactic: SplitKTactic,
):
    """Run ``A[M,K] @ B[K,N]`` with the ``mm_bf16`` layouts."""
    validate_tactic(tactic, *_validate_runtime_tensors(a, b, bias, out))
    has_bias = bias is not None
    cute_tensors = _to_cute_swap(a, b, out, bias)
    compiled = _get_compiled_splitk_kernel(
        dtype=a.dtype,
        tactic=tactic,
        use_pdl=pdl,
        has_bias=has_bias,
        leading_dims=cute_tensors[4],
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    if has_bias:
        compiled(*cute_tensors[:4], stream)
    else:
        compiled(*cute_tensors[:3], stream)
    return out


def run_splitk_dense_silu(
    a,
    b,
    out,
    pdl: bool,
    tactic: SplitKTactic,
    scale: float,
):
    """Run ``bf16(silu(scale * (A[M,K] @ B[K,N])))``."""
    validate_tactic(tactic, *_validate_runtime_tensors(a, b, None, out))
    cute_tensors = _to_cute_swap(a, b, out, None)
    compiled = _get_compiled_splitk_kernel(
        dtype=a.dtype,
        tactic=tactic,
        use_pdl=pdl,
        has_bias=False,
        leading_dims=cute_tensors[4],
        epilogue_mode="silu",
        epilogue_scale=scale,
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(*cute_tensors[:3], stream)
    return out


def run_splitk_dense_silu_aux(
    a,
    b,
    out,
    aux,
    pdl: bool,
    tactic: SplitKTactic,
    scale: float,
    aux_start: int,
    aux_impl: str = "direct",
    *,
    inv_rms=None,
    sum_sq=None,
    rms_eps=1e-6,
    branch_packed=False,
):
    """Run a packed SiLU GEMM and emit final FP32 values for an N slice.

    ``direct`` stores the full packed projection to ``out`` after SiLU as a
    fallback for non-aligned auxiliary slices. Eligible ``alpha_only`` calls
    leave the trailing auxiliary slice of ``out`` untouched. In both modes, ``aux`` receives
    ``2 * sigmoid(scale * raw)`` directly from the FP32 accumulators before
    any output cast.
    Raw callers provide either read-only ``inv_rms`` or read-only ``sum_sq``.
    The latter computes its inverse RMS locally; no inv buffer is written.
    ``branch_packed`` accepts the large-T weight view ``B[C,H,Npad]`` without
    repacking it. ``out.shape[1]`` supplies logical N; physical padding is not
    computed or stored. This mode requires raw RMS statistics and C=4.
    """
    m, n, k = (
        _validate_branch_packed_tensors(a, b, out)
        if branch_packed
        else _validate_runtime_tensors(a, b, None, out)
    )
    validate_tactic(tactic, m, n, k)
    if (
        aux.ndim != 2
        or aux.shape[0] != m
        or aux.dtype != _torch.float32
        or aux.device != a.device
        or not aux.is_contiguous()
        or aux.data_ptr() % 32
    ):
        raise ValueError(
            "aux must be a 32-byte-aligned contiguous FP32 [M,width] tensor"
        )
    aux_width = aux.shape[1]
    stats = _raw_rms_input(inv_rms, sum_sq, a, m, aux_width, rms_eps)
    if branch_packed and stats is None:
        raise ValueError("branch-packed Down requires raw RMS statistics")
    if stats is not None:
        if aux_width != 4 or tactic.split_k % aux_width or k % tactic.split_k:
            raise ValueError("raw Down requires C=4 and branch-aligned split-K")
    if aux_start < 0 or aux_start + aux_width > n:
        raise ValueError(
            f"aux slice [{aux_start}:{aux_start + aux_width}] exceeds N={n}"
        )
    effective_aux_impl = resolve_splitk_dense_silu_aux_impl(
        aux_impl, aux_start, aux_width, n, tactic
    )
    epilogue_mode = _SILU_AUX_IMPLS[effective_aux_impl]
    cute_tensors = _to_cute_swap(a, b, out, None, branch_packed=branch_packed)
    compiled = _get_compiled_splitk_kernel(
        dtype=a.dtype,
        tactic=tactic,
        use_pdl=pdl,
        has_bias=False,
        leading_dims=cute_tensors[4],
        epilogue_mode=epilogue_mode,
        epilogue_scale=scale,
        epilogue_aux_start=aux_start,
        epilogue_aux_width=aux_width,
        epilogue_group=aux_width if stats is not None else 1,
        raw_rms=stats is not None,
        raw_sum=sum_sq is not None,
        rms_eps=rms_eps if sum_sq is not None else 0.0,
        rms_inv_h=aux_width / k if sum_sq is not None else 0.0,
        branch_packed=branch_packed,
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(
        *cute_tensors[:3],
        _from_dlpack_dynamic(aux, 1),
        *(
            ()
            if stats is None
            else (
                _from_dlpack_dynamic(stats, 1),
                # Shared internal ABI; reset_sum=False makes this argument
                # dead in Down. Reuse the input, without allocating a buffer.
                _from_dlpack_dynamic(stats, 1),
            )
        ),
        stream,
    )
    return out, aux


def run_splitk_dense_gate(
    a,
    b,
    x,
    out,
    pdl: bool,
    tactic: SplitKTactic,
    scale: float,
    group: int,
    *,
    inv_rms=None,
    norm_weight_permuted=None,
    sum_sq=None,
    next_sum_sq=None,
    rms_eps=1e-6,
):
    """Fused ``logits = A @ B`` (B pre-permuted so ``n = j*group + g``) with
    ``out[m, j] = bf16(scale * sum_g sigmoid(logits[m, j*group+g]) *
    fp32(x[m, g*hs+j]))``; the logits matrix is never materialized and ``x``
    doubles as its shape anchor. Raw callers supply read-only inv_rms OR
    current sum_sq. next_sum_sq is a separate reset output, never that input.
    """
    m, n, k = _validate_runtime_tensors(a, b, None, x)
    validate_tactic(tactic, m, n, k)
    if n % group:
        raise ValueError(f"N={n} must be divisible by group={group}")
    if n % tactic.mma_m:
        raise ValueError(
            f"gate epilogue requires N={n} divisible by mma_m={tactic.mma_m}"
        )
    hs = n // group
    stats = _raw_rms_input(inv_rms, sum_sq, a, m, group, rms_eps)
    if next_sum_sq is not None:
        if stats is None:
            raise ValueError("sum reset requires raw gate")
        _validate_inv_rms(next_sum_sq, a, m, group)
        # Both statistics tensors are contiguous; reject partial overlap too.
        stat_bytes = stats.numel() * stats.element_size()
        if (
            next_sum_sq.data_ptr() < stats.data_ptr() + stat_bytes
            and stats.data_ptr() < next_sum_sq.data_ptr() + stat_bytes
        ):
            raise ValueError("next sum and current RMS statistics must not overlap")
    if (stats is None) != (norm_weight_permuted is None):
        raise ValueError("raw gate requires RMS statistics and permuted norm weight")
    if stats is not None:
        w = norm_weight_permuted
        if (
            w.shape != (n,)
            or w.dtype != a.dtype
            or w.device != a.device
            or not w.is_contiguous()
            or w.data_ptr() % 32
        ):
            raise ValueError("norm weight must be aligned contiguous [C*H], matching a")
    if (
        out.ndim != 2
        or out.shape != (m, hs)
        or out.dtype != a.dtype
        or out.device != a.device
        or out.stride(1) != 1
        or out.stride(0) < hs
        or out.data_ptr() % 32
        or x.stride(1) != 1
    ):
        raise ValueError(
            f"gate out must be a 32-byte-aligned row-major {(m, hs)} tensor "
            f"matching a, and x must be row-major"
        )
    cute_tensors = _to_cute_swap(a, b, x, None)
    compiled = _get_compiled_splitk_kernel(
        dtype=a.dtype,
        tactic=tactic,
        use_pdl=pdl,
        has_bias=False,
        leading_dims=cute_tensors[4],
        epilogue_mode="gate",
        epilogue_scale=scale,
        epilogue_group=group,
        raw_rms=stats is not None,
        raw_sum=sum_sq is not None,
        reset_sum=next_sum_sq is not None,
        rms_eps=rms_eps if sum_sq is not None else 0.0,
        rms_inv_h=1.0 / hs if sum_sq is not None else 0.0,
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    compiled(
        *cute_tensors[:3],
        _from_dlpack_dynamic(x, 1),
        _from_dlpack_dynamic(out, 1),
        *(
            ()
            if stats is None
            else (
                _from_dlpack_dynamic(stats, 1),
                _from_dlpack_dynamic(
                    norm_weight_permuted.as_strided((n, m, 1), (1, 0, 0)), 0
                ),
                _from_dlpack_dynamic(stats if next_sum_sq is None else next_sum_sq, 1),
            )
        ),
        stream,
    )
    return out
