import ctypes
import os
from typing import Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import dsl_user_op


class _NoInitMbarrierArray(pipeline.MbarrierArray):
    @dsl_user_op
    def mbarrier_init(self, *, loc=None, ip=None) -> None:
        pass

    def __new_from_mlir_values__(self, values: list[object]) -> "_NoInitMbarrierArray":
        new_mbarrier_array = object.__new__(_NoInitMbarrierArray)
        barrier_ptr = values[0]
        new_mbarrier_array.barrier_storage = barrier_ptr
        new_mbarrier_array.op_type = self.op_type
        new_mbarrier_array.cg = self.cg
        new_mbarrier_array.num_stages = self.num_stages
        new_mbarrier_array.tx_count = self.tx_count
        new_mbarrier_array.arrive_count = self.arrive_count
        new_mbarrier_array.mbarrier_base = barrier_ptr
        new_mbarrier_array.mbarrier_layout = self.mbarrier_layout
        new_mbarrier_array.name = self.name
        return new_mbarrier_array


class Sm100Nvfp4GemmFast:
    # 64-bit L2 cache-policy descriptors for .L2::cache_hint (values of CUTLASS TMA::CacheHintSm90).
    EVICT_FIRST = 0x12F0000000000000
    EVICT_LAST = 0x14F0000000000000

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int] = (128, 64),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        num_ab_stage: int = 7,
        num_c_stage: int = 1,
        use_tma_store: bool = True,
        subtile_width: int = 64,
        early_tmem_free: bool = False,
        pdl_mode: int = 3,
        peel_first_tile: bool = True,
        pdl_safe: bool = True,
        store_wait: bool = True,
        prefetch_b: int = 0,
        prefetch_a: bool = False,
        l2_hint: bool = False,
        c_alias_a: bool = False,
        b_first: int = 0,
        coop_b_warp3: bool = False,
        overlap_tmem_free: bool = False,
        defer_sync: bool = False,
        pre_alloc_tmem: bool = False,
        num_tmem_alloc_cols: int = 512,
        copy_before_pf: bool = False,
        pf_init: int = 0,
        rolling_pf_b: bool = True,
        hoist_epilog_wait: bool = False,
        pf_desc_c_epi: bool = False,
        fast_single_subtile: bool = False,
        coop_b_mode: int = 0,
        coop_deep_k: int = 0,
        fast_single_pass_ring: bool = False,
        peel_producer_init: bool = False,
        elide_tail_release: bool = False,
        clean_c_prologue: bool = False,
        warp_scoped_ab_frag: bool = False,
        coop_a_next_wave: bool = False,
        coop_pf_init: int = 0,
        split_pf_b_warp1: bool = False,
        direct_consumer_wait: bool = False,
        unroll_pf0: bool = False,
        deep_b_warps23: int = 0,
        s2_wave3_b_pf: bool = False,
        elide_acc_empty_wait: bool = False,
        omit_1cta_dealloc_mbar: bool = False,
        hoist_sf0_set: bool = False,
        unroll_peel_init: bool = False,
        unroll_single_pass_tma: bool = False,
        split_tma_steady_pf: bool = False,
        unroll_tma_tail_pf: bool = False,
        early_cta_sync: bool = False,
        warp0_only_tma_c_part: bool = False,
        peel_mma_tail_last: bool = False,
        pdl_early_k: int = 0,
        single_pf_a: bool = False,
        warp0_pf_a: bool = False,
        multiwarp_pf_b: bool = False,
        hoist_tma_c_part: bool = False,
        tma_pf_split: int = 4,
        parallel_mbar_init: bool = False,
        elide_dead_mbar_init: bool = False,
        fast_tmem_bar: bool = False,
        unroll_coop_pf: bool = False,
        hoist_mma_s2t_part: bool = False,
        hoist_mma_full: bool = False,
        alg_mbar_init: bool = False,
        unroll_mma_tail: bool = False,
        early_pf_desc_c: bool = False,
        static_epilog_tmem_ptr: bool = False,
        direct_single_pass_mma: bool = False,
        alg_mbar_6warp: bool = False,
        elide_1cta_mbar_fence: bool = False,
        dom_tma_desc: int = 0,
        c_dtype: type[cutlass.Numeric] = cutlass.Float16,
        has_alpha: bool = False,
    ):
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.c_dtype = c_dtype
        self.has_alpha = has_alpha
        self.sf_vec_size = 16
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (mma_tiler_mn[0], mma_tiler_mn[1], 256)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * 6

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * len(self.epilog_warp_id)
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=(
                64
                if (fast_tmem_bar and not self.use_2cta_instrs)
                else 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            ),
        )
        self.num_tmem_alloc_cols = num_tmem_alloc_cols
        self.num_ab_stage = num_ab_stage
        self.num_acc_stage = 1
        self.num_c_stage = num_c_stage
        self.use_tma_store = use_tma_store
        self.subtile_width = subtile_width
        self.early_tmem_free = early_tmem_free
        self.pdl_mode = pdl_mode
        self.use_pdl = pdl_mode > 0
        self.peel_first_tile = peel_first_tile
        # C1: TMA warp waits on the upstream grid before issuing any SMEM input load.
        self.pdl_safe = pdl_safe
        # C2: epilogue waits until the TMA store has finished reading sC before the CTA exits.
        self.store_wait = store_wait
        # P1: L2 prefetch of B/SFB before the PDL wait. 0 = off, <0 = all k-tiles, D>0 = D tiles ahead.
        self.prefetch_b = prefetch_b
        # P1-A: distributed L2 prefetch of A/SFA k-tiles (one CTA per k-tile) before the PDL wait.
        self.prefetch_a = prefetch_a
        # P2: L2 eviction hints on TMA loads (B/SFB evict_first, A/SFA evict_last).
        self.l2_hint = l2_hint
        self.c_alias_a = c_alias_a
        self.b_first = b_first
        self.coop_b_warp3 = coop_b_warp3
        self.overlap_tmem_free = overlap_tmem_free
        self.defer_sync = defer_sync
        self.pre_alloc_tmem = pre_alloc_tmem
        self.copy_before_pf = copy_before_pf
        self.pf_init = pf_init
        self.rolling_pf_b = rolling_pf_b
        self.hoist_epilog_wait = hoist_epilog_wait
        self.pf_desc_c_epi = pf_desc_c_epi
        self.fast_single_subtile = fast_single_subtile
        self.coop_b_mode = coop_b_mode
        self.coop_deep_k = coop_deep_k
        self.fast_single_pass_ring = fast_single_pass_ring
        self.peel_producer_init = peel_producer_init
        self.elide_tail_release = elide_tail_release
        self.clean_c_prologue = clean_c_prologue
        self.warp_scoped_ab_frag = warp_scoped_ab_frag
        self.coop_a_next_wave = coop_a_next_wave
        self.coop_pf_init = coop_pf_init
        self.split_pf_b_warp1 = split_pf_b_warp1
        self.direct_consumer_wait = direct_consumer_wait
        self.unroll_pf0 = unroll_pf0
        self.deep_b_warps23 = deep_b_warps23
        self.s2_wave3_b_pf = s2_wave3_b_pf
        self.elide_acc_empty_wait = elide_acc_empty_wait
        self.omit_1cta_dealloc_mbar = omit_1cta_dealloc_mbar
        self.hoist_sf0_set = hoist_sf0_set
        self.unroll_peel_init = unroll_peel_init
        self.unroll_single_pass_tma = unroll_single_pass_tma
        self.split_tma_steady_pf = split_tma_steady_pf
        self.unroll_tma_tail_pf = unroll_tma_tail_pf
        self.early_cta_sync = early_cta_sync
        self.warp0_only_tma_c_part = warp0_only_tma_c_part
        self.peel_mma_tail_last = peel_mma_tail_last
        self.pdl_early_k = pdl_early_k
        self.single_pf_a = single_pf_a
        self.warp0_pf_a = warp0_pf_a
        self.multiwarp_pf_b = multiwarp_pf_b
        self.hoist_tma_c_part = hoist_tma_c_part
        self.tma_pf_split = tma_pf_split
        self.parallel_mbar_init = parallel_mbar_init
        self.elide_dead_mbar_init = elide_dead_mbar_init
        self.fast_tmem_bar = fast_tmem_bar
        self.unroll_coop_pf = unroll_coop_pf
        self.hoist_mma_s2t_part = hoist_mma_s2t_part
        self.hoist_mma_full = hoist_mma_full
        self.alg_mbar_init = alg_mbar_init
        self.unroll_mma_tail = unroll_mma_tail
        self.early_pf_desc_c = early_pf_desc_c
        self.static_epilog_tmem_ptr = static_epilog_tmem_ptr
        self.direct_single_pass_mma = direct_single_pass_mma
        self.alg_mbar_6warp = alg_mbar_6warp
        self.elide_1cta_mbar_fence = elide_1cta_mbar_fence
        self.dom_tma_desc = dom_tma_desc

    def _setup_attributes(self):
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        mma_inst_shape_k = 64
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        self.epi_tile = (128, min(self.subtile_width, self.cta_tile_shape_mnk[1]))

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )
        return tiled_mma, tiled_mma_sfb

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: cutlass.Constexpr[tuple],
    ):
        m, n, k, l = problem_size
        tiled_mma, tiled_mma_sfb = self._setup_attributes()
        # Initial prefetch window for prefetch_b = D > 0 (problem_size is Constexpr, so this is static).
        k_tiles_total = (k + self.mma_tiler[2] - 1) // self.mma_tiler[2]
        pf_init_val = self.pf_init if self.pf_init > 0 else max(self.prefetch_b, 0)
        self.pf0 = min(pf_init_val, k_tiles_total)
        coop_pf_val = self.coop_pf_init if self.coop_pf_init > 0 else self.pf0
        self.coop_pf0 = min(coop_pf_val, k_tiles_total)
        self.tma_pf0 = (
            min(self.tma_pf_split, self.pf0) if self.split_pf_b_warp1 else self.pf0
        )

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout((cute.assume(m, 32), n, l), stride=(n, 1, m * n)),
        )
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.ab_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c_tensor, epi_smem_layout, self.epi_tile
        )

        grid = (
            cute.ceil_div(c_tensor.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(c_tensor.shape[1], self.cta_tile_shape_mnk[1]),
            c_tensor.shape[2],
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorageWithC:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorageNoC:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorageNoCSlim:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        if cutlass.const_expr(self.c_alias_a and self.omit_1cta_dealloc_mbar):
            self.shared_storage = SharedStorageNoCSlim
        elif cutlass.const_expr(self.c_alias_a):
            self.shared_storage = SharedStorageNoC
        else:
            self.shared_storage = SharedStorageWithC

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            c_tensor,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            None,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )
        return

    @cute.jit
    def call_with_stream_alpha(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        stream: cuda.CUstream,
        problem_size: cutlass.Constexpr[tuple],
    ):
        m, n, k, l = problem_size
        tiled_mma, tiled_mma_sfb = self._setup_attributes()
        k_tiles_total = (k + self.mma_tiler[2] - 1) // self.mma_tiler[2]
        pf_init_val = self.pf_init if self.pf_init > 0 else max(self.prefetch_b, 0)
        self.pf0 = min(pf_init_val, k_tiles_total)
        coop_pf_val = self.coop_pf_init if self.coop_pf_init > 0 else self.pf0
        self.coop_pf0 = min(coop_pf_val, k_tiles_total)
        self.tma_pf0 = (
            min(self.tma_pf_split, self.pf0) if self.split_pf_b_warp1 else self.pf0
        )

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout((cute.assume(m, 32), n, l), stride=(n, 1, m * n)),
        )
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)
        alpha_tensor = cute.make_tensor(alpha_ptr, cute.make_layout((1,), stride=(1,)))

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.ab_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c_tensor, epi_smem_layout, self.epi_tile
        )

        grid = (
            cute.ceil_div(c_tensor.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(c_tensor.shape[1], self.cta_tile_shape_mnk[1]),
            c_tensor.shape[2],
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorageWithC:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorageNoC:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorageNoCSlim:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        if cutlass.const_expr(self.c_alias_a and self.omit_1cta_dealloc_mbar):
            self.shared_storage = SharedStorageNoCSlim
        elif cutlass.const_expr(self.c_alias_a):
            self.shared_storage = SharedStorageNoC
        else:
            self.shared_storage = SharedStorageWithC

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            c_tensor,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            alpha_tensor if self.has_alpha else None,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_tma_mnl: cute.Tensor,
        mC_simt_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        mAlpha: Optional[cute.Tensor],
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if cutlass.const_expr(
            self.dom_tma_desc < 2 or (self.use_tma_store and not self.pf_desc_c_epi)
        ):
            if warp_idx == self.tma_warp_id:
                if cutlass.const_expr(self.dom_tma_desc == 0):
                    cpasync.prefetch_descriptor(tma_atom_a)
                    cpasync.prefetch_descriptor(tma_atom_sfa)
                if cutlass.const_expr(self.dom_tma_desc < 2):
                    cpasync.prefetch_descriptor(tma_atom_b)
                    cpasync.prefetch_descriptor(tma_atom_sfb)
                if cutlass.const_expr(self.use_tma_store and not self.pf_desc_c_epi):
                    cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        mma_tile_coord_mnl = (
            bidx // cute.size(tiled_mma.thr_id.shape),
            bidy,
            bidz,
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        if cutlass.const_expr(
            self.parallel_mbar_init and cute.size(self.cluster_shape_mn) == 1
        ):
            producer_a = (pipeline.PipelineOp.TmaLoad, ab_pipeline_producer_group)
            consumer_a = (pipeline.PipelineOp.TCGen05Mma, ab_pipeline_consumer_group)
            ab_sync_full = _NoInitMbarrierArray(
                barrier_storage=storage.ab_full_mbar_ptr.data_ptr().align(min_align=8),
                num_stages=self.num_ab_stage,
                agent=producer_a,
                tx_count=self.num_tma_load_bytes,
                name="ab_pipeline.full",
            )
            ab_sync_empty = _NoInitMbarrierArray(
                barrier_storage=storage.ab_full_mbar_ptr.data_ptr().align(min_align=8)
                + self.num_ab_stage,
                num_stages=self.num_ab_stage,
                agent=consumer_a,
                name="ab_pipeline.empty",
            )
            ab_pipeline = pipeline.PipelineTmaUmma(
                ab_sync_full,
                ab_sync_empty,
                self.num_ab_stage,
                None,
                None,
                True,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
            )
        else:
            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=self.defer_sync,
            )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        if cutlass.const_expr(
            self.parallel_mbar_init and cute.size(self.cluster_shape_mn) == 1
        ):
            producer_acc = (pipeline.PipelineOp.TCGen05Mma, acc_pipeline_producer_group)
            consumer_acc = (
                pipeline.PipelineOp.AsyncThread,
                acc_pipeline_consumer_group,
            )
            acc_sync_full = _NoInitMbarrierArray(
                barrier_storage=storage.acc_full_mbar_ptr.data_ptr().align(min_align=8),
                num_stages=self.num_acc_stage,
                agent=producer_acc,
                name="acc_pipeline.full",
            )
            acc_sync_empty = _NoInitMbarrierArray(
                barrier_storage=storage.acc_full_mbar_ptr.data_ptr().align(min_align=8)
                + self.num_acc_stage,
                num_stages=self.num_acc_stage,
                agent=consumer_acc,
                name="acc_pipeline.empty",
            )
            acc_pipeline = pipeline.PipelineUmmaAsync(
                acc_sync_full,
                acc_sync_empty,
                self.num_acc_stage,
                None,
                None,
                cute.nvgpu.tcgen05.CtaGroup.ONE,
            )
        else:
            acc_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
                num_stages=self.num_acc_stage,
                producer_group=acc_pipeline_producer_group,
                consumer_group=acc_pipeline_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=self.defer_sync,
            )

        dealloc_mbar_ptr = (
            None
            if cutlass.const_expr(self.c_alias_a and self.omit_1cta_dealloc_mbar)
            else storage.tmem_dealloc_mbar_ptr
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=dealloc_mbar_ptr,
        )
        if cutlass.const_expr(
            self.defer_sync
            and not (self.parallel_mbar_init and cute.size(self.cluster_shape_mn) == 1)
        ):
            cute.arch.mbarrier_init_fence()

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()
        elif cutlass.const_expr(self.early_cta_sync and not self.pre_alloc_tmem):
            self.cta_sync_barrier.arrive_and_wait()

        if cutlass.const_expr(self.c_alias_a):
            sC_ptr = cute.recast_ptr(
                storage.sA.data_ptr(), c_smem_layout_staged.inner, dtype=self.c_dtype
            )
            sC = cute.make_tensor(sC_ptr, c_smem_layout_staged.outer)
        else:
            sC = storage.sC.get_tensor(
                c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
            )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        if cutlass.const_expr(not self.use_tma_store):
            gC_simt_mnl = cute.local_tile(
                mC_simt_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
        k_tile_cnt_static = cute.size(gA_mkl, mode=[3])
        k_tile_cnt = cutlass.Int32(k_tile_cnt_static)
        single_pass_ring = (
            self.fast_single_pass_ring
            and (k_tile_cnt_static <= self.num_ab_stage)
            and (cute.size(self.cluster_shape_mn) == 1)
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        if cutlass.const_expr(not self.use_tma_store):
            tCgC_simt = thr_mma.partition_C(gC_simt_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        if cutlass.const_expr(self.pre_alloc_tmem and not use_2cta_instrs):
            tmem.allocate(self.num_tmem_alloc_cols)
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        elif cutlass.const_expr(not (self.early_cta_sync and not self.pre_alloc_tmem)):
            if cutlass.const_expr(
                self.parallel_mbar_init and cute.size(self.cluster_shape_mn) == 1
            ):
                if cutlass.const_expr(self.alg_mbar_init):
                    mbar_base = storage.ab_full_mbar_ptr.data_ptr().align(min_align=8)
                    if cutlass.const_expr(
                        self.alg_mbar_6warp
                        and not (self.elide_dead_mbar_init and single_pass_ring)
                        and (2 * self.num_ab_stage + 2 * self.num_acc_stage == 18)
                    ):
                        with cute.arch.elect_one():
                            w_base = mbar_base + warp_idx * 3
                            for i in cutlass.range_constexpr(3):
                                cute.arch.mbarrier_init(w_base + i, 1)
                    else:
                        if warp_idx < 4:
                            with cute.arch.elect_one():
                                if cutlass.const_expr(
                                    self.elide_dead_mbar_init and single_pass_ring
                                ):
                                    w_base = mbar_base + warp_idx * 2
                                    for i in cutlass.range_constexpr(2):
                                        cute.arch.mbarrier_init(w_base + i, 1)
                                else:
                                    w_base = mbar_base + warp_idx * 4
                                    for i in cutlass.range_constexpr(4):
                                        cute.arch.mbarrier_init(w_base + i, 1)
                        elif warp_idx == 4:
                            with cute.arch.elect_one():
                                for i in cutlass.range_constexpr(self.num_acc_stage):
                                    cute.arch.mbarrier_init(
                                        acc_pipeline.sync_object_full.get_barrier(i), 1
                                    )
                            if cutlass.const_expr(
                                not (
                                    self.elide_dead_mbar_init
                                    and self.elide_acc_empty_wait
                                )
                            ):
                                with cute.arch.elect_one():
                                    for i in cutlass.range_constexpr(
                                        self.num_acc_stage
                                    ):
                                        cute.arch.mbarrier_init(
                                            acc_pipeline.sync_object_empty.get_barrier(
                                                i
                                            ),
                                            1,
                                        )
                else:
                    if warp_idx == 0:
                        with cute.arch.elect_one():
                            for i in cutlass.range_constexpr(self.num_ab_stage):
                                cute.arch.mbarrier_init(
                                    ab_pipeline.sync_object_full.get_barrier(i), 1
                                )
                    elif warp_idx == 1:
                        if cutlass.const_expr(
                            not (self.elide_dead_mbar_init and single_pass_ring)
                        ):
                            with cute.arch.elect_one():
                                for i in cutlass.range_constexpr(self.num_ab_stage):
                                    cute.arch.mbarrier_init(
                                        ab_pipeline.sync_object_empty.get_barrier(i), 1
                                    )
                    elif warp_idx == 2:
                        with cute.arch.elect_one():
                            for i in cutlass.range_constexpr(self.num_acc_stage):
                                cute.arch.mbarrier_init(
                                    acc_pipeline.sync_object_full.get_barrier(i), 1
                                )
                        if cutlass.const_expr(
                            not (
                                self.elide_dead_mbar_init and self.elide_acc_empty_wait
                            )
                        ):
                            with cute.arch.elect_one():
                                for i in cutlass.range_constexpr(self.num_acc_stage):
                                    cute.arch.mbarrier_init(
                                        acc_pipeline.sync_object_empty.get_barrier(i), 1
                                    )
                if cutlass.const_expr(not self.elide_1cta_mbar_fence):
                    cute.arch.mbarrier_init_fence()
            self.cta_sync_barrier.arrive_and_wait()

        if warp_idx == self.tma_warp_id:
            if cutlass.const_expr(self.dom_tma_desc >= 1):
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_sfa)
            if cutlass.const_expr(self.dom_tma_desc >= 2):
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(self.pdl_mode == 3):
                cute.arch.griddepcontrol_launch_dependents()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tAgA_slice = tAgA[
                (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
            ]
            tBgB_slice = tBgB[
                (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
            ]
            tAgSFA_slice = tAgSFA[
                (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
            ]
            slice_n = mma_tile_coord_mnl[1]
            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                slice_n = mma_tile_coord_mnl[1] // 2
            tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

            # P1: warm L2 with this CTA's operands before the PDL wait. A prefetch only moves
            # data DRAM -> L2 (the coherence point for all SMs); nothing reaches SMEM until the
            # TMA loads below, which are issued after griddepcontrol_wait. So an upstream kernel
            # that is still writing these buffers cannot leave a stale copy in this CTA.
            if cutlass.const_expr(self.prefetch_b < 0):
                cute.prefetch(tma_atom_b, tBgB_slice)
                cute.prefetch(tma_atom_sfb, tBgSFB_slice)
            if cutlass.const_expr(self.prefetch_b > 0):
                if cutlass.const_expr(self.multiwarp_pf_b):
                    pass
                elif cutlass.const_expr(self.unroll_pf0):
                    for pf_k_u in cutlass.range_constexpr(self.tma_pf0):
                        cute.prefetch(tma_atom_b, tBgB_slice[(None, pf_k_u)])
                        cute.prefetch(tma_atom_sfb, tBgSFB_slice[(None, pf_k_u)])
                else:
                    for pf_k in cutlass.range(0, self.tma_pf0, 1, unroll=1):
                        cute.prefetch(tma_atom_b, tBgB_slice[(None, pf_k)])
                        cute.prefetch(tma_atom_sfb, tBgSFB_slice[(None, pf_k)])
            if cutlass.const_expr(self.prefetch_a and not self.warp0_pf_a):
                if cutlass.const_expr(self.single_pf_a):
                    if bidy < k_tile_cnt:
                        cute.prefetch(tma_atom_a, tAgA_slice[(None, bidy)])
                        cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, bidy)])
                else:
                    # A/SFA rows are shared by every CTA of this M tile: spread the k-tiles over them.
                    grid_n = cute.arch.grid_dim()[1]
                    for pf_k in cutlass.range(bidy, k_tile_cnt, grid_n, unroll=1):
                        cute.prefetch(tma_atom_a, tAgA_slice[(None, pf_k)])
                        cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, pf_k)])

            # C1: no SMEM input load may be issued before the upstream grid has completed.
            if cutlass.const_expr(self.use_pdl and self.pdl_safe):
                cute.arch.griddepcontrol_wait()

            ab_producer_state.reset_count()
            if cutlass.const_expr(single_pass_ring):
                if cutlass.const_expr(self.unroll_single_pass_tma):
                    for k_tile_s in cutlass.range_constexpr(k_tile_cnt_static):
                        ab_pipeline.producer_acquire(
                            ab_producer_state, cutlass.Boolean(1)
                        )
                        bar_s = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cnt_s = ab_producer_state.count
                        stg_s = ab_producer_state.index

                        if cutlass.const_expr(self.b_first == 2):
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt_s)],
                                tBsB[(None, stg_s)],
                                tma_bar_ptr=bar_s,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt_s)],
                                tAsA[(None, stg_s)],
                                tma_bar_ptr=bar_s,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt_s)],
                                tBsSFB[(None, stg_s)],
                                tma_bar_ptr=bar_s,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt_s)],
                                tAsSFA[(None, stg_s)],
                                tma_bar_ptr=bar_s,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                        else:
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt_s)],
                                tAsA[(None, stg_s)],
                                tma_bar_ptr=bar_s,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt_s)],
                                tBsB[(None, stg_s)],
                                tma_bar_ptr=bar_s,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt_s)],
                                tAsSFA[(None, stg_s)],
                                tma_bar_ptr=bar_s,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt_s)],
                                tBsSFB[(None, stg_s)],
                                tma_bar_ptr=bar_s,
                                mcast_mask=sfb_full_mcast_mask,
                            )

                        if cutlass.const_expr(
                            self.prefetch_b > 0
                            and self.rolling_pf_b
                            and (k_tile_s + self.prefetch_b < k_tile_cnt_static)
                        ):
                            cute.prefetch(
                                tma_atom_b, tBgB_slice[(None, cnt_s + self.prefetch_b)]
                            )
                            cute.prefetch(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt_s + self.prefetch_b)],
                            )

                        if cutlass.const_expr(
                            self.pdl_mode == 2
                            and self.pdl_early_k > 0
                            and (k_tile_s == k_tile_cnt_static - 1 - self.pdl_early_k)
                        ):
                            cute.arch.griddepcontrol_launch_dependents()

                        ab_producer_state.advance()
                else:
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        ab_pipeline.producer_acquire(
                            ab_producer_state, cutlass.Boolean(1)
                        )
                        bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cnt = ab_producer_state.count
                        stg = ab_producer_state.index

                        if cutlass.const_expr(self.b_first == 2):
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt)],
                                tBsB[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt)],
                                tBsSFB[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                        else:
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt)],
                                tBsB[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt)],
                                tBsSFB[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                            )

                        if cutlass.const_expr(
                            self.prefetch_b > 0 and self.rolling_pf_b
                        ):
                            if cnt + self.prefetch_b < k_tile_cnt:
                                cute.prefetch(
                                    tma_atom_b,
                                    tBgB_slice[(None, cnt + self.prefetch_b)],
                                )
                                cute.prefetch(
                                    tma_atom_sfb,
                                    tBgSFB_slice[(None, cnt + self.prefetch_b)],
                                )

                        ab_producer_state.advance()
            elif cutlass.const_expr(
                self.peel_producer_init and k_tile_cnt_static > self.num_ab_stage
            ):
                if cutlass.const_expr(self.unroll_peel_init):
                    for k_tile_p in cutlass.range_constexpr(self.num_ab_stage):
                        ab_pipeline.producer_acquire(
                            ab_producer_state, cutlass.Boolean(1)
                        )
                        bar_p = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cnt_p = ab_producer_state.count
                        stg_p = ab_producer_state.index

                        if cutlass.const_expr(self.b_first == 2):
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt_p)],
                                tBsB[(None, stg_p)],
                                tma_bar_ptr=bar_p,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt_p)],
                                tAsA[(None, stg_p)],
                                tma_bar_ptr=bar_p,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt_p)],
                                tBsSFB[(None, stg_p)],
                                tma_bar_ptr=bar_p,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt_p)],
                                tAsSFA[(None, stg_p)],
                                tma_bar_ptr=bar_p,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                        else:
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt_p)],
                                tAsA[(None, stg_p)],
                                tma_bar_ptr=bar_p,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt_p)],
                                tBsB[(None, stg_p)],
                                tma_bar_ptr=bar_p,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt_p)],
                                tAsSFA[(None, stg_p)],
                                tma_bar_ptr=bar_p,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt_p)],
                                tBsSFB[(None, stg_p)],
                                tma_bar_ptr=bar_p,
                                mcast_mask=sfb_full_mcast_mask,
                            )

                        if cutlass.const_expr(
                            self.prefetch_b > 0 and self.rolling_pf_b
                        ):
                            if cutlass.const_expr(
                                self.num_ab_stage + self.prefetch_b <= k_tile_cnt_static
                            ):
                                cute.prefetch(
                                    tma_atom_b,
                                    tBgB_slice[(None, cnt_p + self.prefetch_b)],
                                )
                                cute.prefetch(
                                    tma_atom_sfb,
                                    tBgSFB_slice[(None, cnt_p + self.prefetch_b)],
                                )
                            else:
                                if cnt_p + self.prefetch_b < k_tile_cnt:
                                    cute.prefetch(
                                        tma_atom_b,
                                        tBgB_slice[(None, cnt_p + self.prefetch_b)],
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb,
                                        tBgSFB_slice[(None, cnt_p + self.prefetch_b)],
                                    )

                        ab_producer_state.advance()
                else:
                    for k_tile_d in cutlass.range(0, self.num_ab_stage, 1, unroll=1):
                        ab_pipeline.producer_acquire(
                            ab_producer_state, cutlass.Boolean(1)
                        )
                        bar_d = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cnt_d = ab_producer_state.count
                        stg_d = ab_producer_state.index

                        if cutlass.const_expr(self.b_first == 2):
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt_d)],
                                tBsB[(None, stg_d)],
                                tma_bar_ptr=bar_d,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt_d)],
                                tAsA[(None, stg_d)],
                                tma_bar_ptr=bar_d,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt_d)],
                                tBsSFB[(None, stg_d)],
                                tma_bar_ptr=bar_d,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt_d)],
                                tAsSFA[(None, stg_d)],
                                tma_bar_ptr=bar_d,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                        else:
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt_d)],
                                tAsA[(None, stg_d)],
                                tma_bar_ptr=bar_d,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt_d)],
                                tBsB[(None, stg_d)],
                                tma_bar_ptr=bar_d,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt_d)],
                                tAsSFA[(None, stg_d)],
                                tma_bar_ptr=bar_d,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt_d)],
                                tBsSFB[(None, stg_d)],
                                tma_bar_ptr=bar_d,
                                mcast_mask=sfb_full_mcast_mask,
                            )

                        if cutlass.const_expr(
                            self.prefetch_b > 0 and self.rolling_pf_b
                        ):
                            if cutlass.const_expr(
                                self.num_ab_stage + self.prefetch_b <= k_tile_cnt_static
                            ):
                                cute.prefetch(
                                    tma_atom_b,
                                    tBgB_slice[(None, cnt_d + self.prefetch_b)],
                                )
                                cute.prefetch(
                                    tma_atom_sfb,
                                    tBgSFB_slice[(None, cnt_d + self.prefetch_b)],
                                )
                            else:
                                if cnt_d + self.prefetch_b < k_tile_cnt:
                                    cute.prefetch(
                                        tma_atom_b,
                                        tBgB_slice[(None, cnt_d + self.prefetch_b)],
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb,
                                        tBgSFB_slice[(None, cnt_d + self.prefetch_b)],
                                    )

                        ab_producer_state.advance()

                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                    ab_producer_state
                )
                if cutlass.const_expr(
                    self.split_tma_steady_pf
                    and self.prefetch_b > 0
                    and self.rolling_pf_b
                    and (self.num_ab_stage + self.prefetch_b < k_tile_cnt_static)
                ):
                    for k_tile_m1 in cutlass.range(
                        self.num_ab_stage, k_tile_cnt - self.prefetch_b, 1, unroll=1
                    ):
                        ab_pipeline.producer_acquire(
                            ab_producer_state, peek_ab_empty_status
                        )
                        bar_m1 = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cnt_m1 = ab_producer_state.count
                        stg_m1 = ab_producer_state.index

                        if cutlass.const_expr(self.b_first == 2):
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt_m1)],
                                tBsB[(None, stg_m1)],
                                tma_bar_ptr=bar_m1,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt_m1)],
                                tAsA[(None, stg_m1)],
                                tma_bar_ptr=bar_m1,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt_m1)],
                                tBsSFB[(None, stg_m1)],
                                tma_bar_ptr=bar_m1,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt_m1)],
                                tAsSFA[(None, stg_m1)],
                                tma_bar_ptr=bar_m1,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                        else:
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt_m1)],
                                tAsA[(None, stg_m1)],
                                tma_bar_ptr=bar_m1,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt_m1)],
                                tBsB[(None, stg_m1)],
                                tma_bar_ptr=bar_m1,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt_m1)],
                                tAsSFA[(None, stg_m1)],
                                tma_bar_ptr=bar_m1,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt_m1)],
                                tBsSFB[(None, stg_m1)],
                                tma_bar_ptr=bar_m1,
                                mcast_mask=sfb_full_mcast_mask,
                            )

                        cute.prefetch(
                            tma_atom_b, tBgB_slice[(None, cnt_m1 + self.prefetch_b)]
                        )
                        cute.prefetch(
                            tma_atom_sfb, tBgSFB_slice[(None, cnt_m1 + self.prefetch_b)]
                        )

                        ab_producer_state.advance()
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                    if cutlass.const_expr(self.unroll_tma_tail_pf):
                        for k_tile_t in cutlass.range_constexpr(self.prefetch_b):
                            ab_pipeline.producer_acquire(
                                ab_producer_state, peek_ab_empty_status
                            )
                            bar_t = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cnt_t = ab_producer_state.count
                            stg_t = ab_producer_state.index

                            if cutlass.const_expr(self.b_first == 2):
                                cute.copy(
                                    tma_atom_b,
                                    tBgB_slice[(None, cnt_t)],
                                    tBsB[(None, stg_t)],
                                    tma_bar_ptr=bar_t,
                                    mcast_mask=b_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, cnt_t)],
                                    tAsA[(None, stg_t)],
                                    tma_bar_ptr=bar_t,
                                    mcast_mask=a_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_sfb,
                                    tBgSFB_slice[(None, cnt_t)],
                                    tBsSFB[(None, stg_t)],
                                    tma_bar_ptr=bar_t,
                                    mcast_mask=sfb_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA_slice[(None, cnt_t)],
                                    tAsSFA[(None, stg_t)],
                                    tma_bar_ptr=bar_t,
                                    mcast_mask=sfa_full_mcast_mask,
                                )
                            else:
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, cnt_t)],
                                    tAsA[(None, stg_t)],
                                    tma_bar_ptr=bar_t,
                                    mcast_mask=a_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_b,
                                    tBgB_slice[(None, cnt_t)],
                                    tBsB[(None, stg_t)],
                                    tma_bar_ptr=bar_t,
                                    mcast_mask=b_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA_slice[(None, cnt_t)],
                                    tAsSFA[(None, stg_t)],
                                    tma_bar_ptr=bar_t,
                                    mcast_mask=sfa_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_sfb,
                                    tBgSFB_slice[(None, cnt_t)],
                                    tBsSFB[(None, stg_t)],
                                    tma_bar_ptr=bar_t,
                                    mcast_mask=sfb_full_mcast_mask,
                                )

                            ab_producer_state.advance()
                            if cutlass.const_expr(k_tile_t + 1 < self.prefetch_b):
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )
                    else:
                        for k_tile_m2 in cutlass.range(
                            k_tile_cnt - self.prefetch_b, k_tile_cnt, 1, unroll=1
                        ):
                            ab_pipeline.producer_acquire(
                                ab_producer_state, peek_ab_empty_status
                            )
                            bar_m2 = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cnt_m2 = ab_producer_state.count
                            stg_m2 = ab_producer_state.index

                            if cutlass.const_expr(self.b_first == 2):
                                cute.copy(
                                    tma_atom_b,
                                    tBgB_slice[(None, cnt_m2)],
                                    tBsB[(None, stg_m2)],
                                    tma_bar_ptr=bar_m2,
                                    mcast_mask=b_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, cnt_m2)],
                                    tAsA[(None, stg_m2)],
                                    tma_bar_ptr=bar_m2,
                                    mcast_mask=a_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_sfb,
                                    tBgSFB_slice[(None, cnt_m2)],
                                    tBsSFB[(None, stg_m2)],
                                    tma_bar_ptr=bar_m2,
                                    mcast_mask=sfb_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA_slice[(None, cnt_m2)],
                                    tAsSFA[(None, stg_m2)],
                                    tma_bar_ptr=bar_m2,
                                    mcast_mask=sfa_full_mcast_mask,
                                )
                            else:
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, cnt_m2)],
                                    tAsA[(None, stg_m2)],
                                    tma_bar_ptr=bar_m2,
                                    mcast_mask=a_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_b,
                                    tBgB_slice[(None, cnt_m2)],
                                    tBsB[(None, stg_m2)],
                                    tma_bar_ptr=bar_m2,
                                    mcast_mask=b_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA_slice[(None, cnt_m2)],
                                    tAsSFA[(None, stg_m2)],
                                    tma_bar_ptr=bar_m2,
                                    mcast_mask=sfa_full_mcast_mask,
                                )
                                cute.copy(
                                    tma_atom_sfb,
                                    tBgSFB_slice[(None, cnt_m2)],
                                    tBsSFB[(None, stg_m2)],
                                    tma_bar_ptr=bar_m2,
                                    mcast_mask=sfb_full_mcast_mask,
                                )

                            ab_producer_state.advance()
                            peek_ab_empty_status = cutlass.Boolean(1)
                            if ab_producer_state.count < k_tile_cnt:
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )
                else:
                    for k_tile in cutlass.range(
                        self.num_ab_stage, k_tile_cnt, 1, unroll=1
                    ):
                        ab_pipeline.producer_acquire(
                            ab_producer_state, peek_ab_empty_status
                        )
                        bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cnt = ab_producer_state.count
                        stg = ab_producer_state.index

                        if cutlass.const_expr(self.b_first == 2):
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt)],
                                tBsB[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt)],
                                tBsSFB[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                        else:
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, cnt)],
                                tBsB[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt)],
                                tBsSFB[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                            )

                        if cutlass.const_expr(
                            self.prefetch_b > 0 and self.rolling_pf_b
                        ):
                            if cnt + self.prefetch_b < k_tile_cnt:
                                cute.prefetch(
                                    tma_atom_b,
                                    tBgB_slice[(None, cnt + self.prefetch_b)],
                                )
                                cute.prefetch(
                                    tma_atom_sfb,
                                    tBgSFB_slice[(None, cnt + self.prefetch_b)],
                                )

                        if cutlass.const_expr(
                            self.pdl_mode == 2 and self.pdl_early_k > 0
                        ):
                            if cnt == k_tile_cnt - 1 - self.pdl_early_k:
                                cute.arch.griddepcontrol_launch_dependents()

                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < k_tile_cnt:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state
                            )
            else:
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cnt = ab_producer_state.count
                    stg = ab_producer_state.index

                    if cutlass.const_expr(
                        self.prefetch_b > 0
                        and self.rolling_pf_b
                        and not self.copy_before_pf
                    ):
                        if cnt + self.prefetch_b < k_tile_cnt:
                            cute.prefetch(
                                tma_atom_b, tBgB_slice[(None, cnt + self.prefetch_b)]
                            )
                            cute.prefetch(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt + self.prefetch_b)],
                            )

                    if cutlass.const_expr(self.l2_hint):
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, cnt)],
                            tAsA[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=a_full_mcast_mask,
                            cache_policy=cutlass.Int64(self.EVICT_LAST),
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, cnt)],
                            tBsB[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=b_full_mcast_mask,
                            cache_policy=cutlass.Int64(self.EVICT_FIRST),
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, cnt)],
                            tAsSFA[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfa_full_mcast_mask,
                            cache_policy=cutlass.Int64(self.EVICT_LAST),
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, cnt)],
                            tBsSFB[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                            cache_policy=cutlass.Int64(self.EVICT_FIRST),
                        )
                    elif cutlass.const_expr(self.b_first == 2):
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, cnt)],
                            tBsB[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, cnt)],
                            tAsA[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, cnt)],
                            tBsSFB[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, cnt)],
                            tAsSFA[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                    elif cutlass.const_expr(self.b_first == 1):
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, cnt)],
                            tBsB[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, cnt)],
                            tBsSFB[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, cnt)],
                            tAsA[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, cnt)],
                            tAsSFA[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                    else:
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, cnt)],
                            tAsA[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, cnt)],
                            tBsB[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, cnt)],
                            tAsSFA[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, cnt)],
                            tBsSFB[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                        )

                    if cutlass.const_expr(
                        self.prefetch_b > 0
                        and self.rolling_pf_b
                        and self.copy_before_pf
                    ):
                        if cnt + self.prefetch_b < k_tile_cnt:
                            cute.prefetch(
                                tma_atom_b, tBgB_slice[(None, cnt + self.prefetch_b)]
                            )
                            cute.prefetch(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, cnt + self.prefetch_b)],
                            )

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

            if cutlass.const_expr(cute.size(self.cluster_shape_mn) > 1):
                ab_pipeline.producer_tail(ab_producer_state)
            if cutlass.const_expr(self.pdl_mode == 2 and self.pdl_early_k == 0):
                cute.arch.griddepcontrol_launch_dependents()

        if warp_idx == self.mma_warp_id:
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            if cutlass.const_expr(
                self.hoist_mma_s2t_part and self.c_alias_a and not use_2cta_instrs
            ):
                acc_tmem_ptr = cute.make_ptr(
                    self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16
                )
                tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
                acc_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtAcc)

                sfa_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + acc_col_offset, dtype=self.sf_dtype
                )
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

                sfa_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                sfb_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + acc_col_offset + sfa_col_offset, dtype=self.sf_dtype
                )
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

                tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
                )
                tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
                )

                if cutlass.const_expr(self.hoist_mma_full):
                    ab_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_ab_stage
                    )
                    acc_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.num_acc_stage
                    )
                    ab_consumer_state.reset_count()

                    tCtSFB_mma = tCtSFB
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                        shifted_ptr = cute.recast_ptr(
                            acc_tmem_ptr + acc_col_offset + sfa_col_offset + offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                    if cutlass.const_expr(self.peel_first_tile and self.hoist_sf0_set):
                        if is_leader_cta:
                            tiled_mma.set(
                                tcgen05.Field.SFA, tCtSFA[(None, None, 0)].iterator
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB, tCtSFB_mma[(None, None, 0)].iterator
                            )

                tmem.wait_for_alloc()
            else:
                if cutlass.const_expr(
                    not (self.pre_alloc_tmem and not use_2cta_instrs)
                ):
                    tmem.wait_for_alloc()
                acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
                acc_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtAcc)

                sfa_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + acc_col_offset, dtype=self.sf_dtype
                )
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

                sfa_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                sfb_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + acc_col_offset + sfa_col_offset, dtype=self.sf_dtype
                )
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

                tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
                )
                tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
                )

            if cutlass.const_expr(
                not (
                    self.hoist_mma_full
                    and self.hoist_mma_s2t_part
                    and self.c_alias_a
                    and not use_2cta_instrs
                )
            ):
                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                acc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_acc_stage
                )
                ab_consumer_state.reset_count()

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + acc_col_offset + sfa_col_offset + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

            peek_ab_full_status = cutlass.Boolean(1)
            if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

            if is_leader_cta:
                if cutlass.const_expr(self.elide_acc_empty_wait):
                    acc_pipeline.producer_acquire(
                        acc_producer_state, cutlass.Boolean(1)
                    )
                else:
                    acc_pipeline.producer_acquire(acc_producer_state)

            if cutlass.const_expr(self.peel_first_tile):
                if is_leader_cta:
                    if cutlass.const_expr(
                        self.hoist_sf0_set
                        and not (
                            self.hoist_mma_full
                            and self.hoist_mma_s2t_part
                            and self.c_alias_a
                            and not use_2cta_instrs
                        )
                    ):
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA[(None, None, 0)].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[(None, None, 0)].iterator
                        )
                    ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                    s2t_stage_coord0 = (None, None, None, None, ab_consumer_state.index)
                    cute.copy(
                        tiled_copy_s2t_sfa,
                        tCsSFA_compact_s2t[s2t_stage_coord0],
                        tCtSFA_compact_s2t,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfb,
                        tCsSFB_compact_s2t[s2t_stage_coord0],
                        tCtSFB_compact_s2t,
                    )
                    for kblock_idx in cutlass.range_constexpr(4):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, kblock_idx > 0)
                        kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                        if cutlass.const_expr(not self.hoist_sf0_set) or kblock_idx > 0:
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                            )
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[kblock_coord],
                            tCrB[kblock_coord],
                            tCtAcc,
                        )
                    if cutlass.const_expr(not single_pass_ring):
                        ab_pipeline.consumer_release(ab_consumer_state)

                if cutlass.const_expr(
                    not (self.direct_single_pass_mma and single_pass_ring)
                ):
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                if cutlass.const_expr(
                    self.elide_tail_release
                    and not single_pass_ring
                    and (k_tile_cnt_static > self.num_ab_stage + 1)
                    and (cute.size(self.cluster_shape_mn) == 1)
                ):
                    for k_tile in cutlass.range(
                        1, k_tile_cnt - self.num_ab_stage, 1, unroll=1
                    ):
                        if is_leader_cta:
                            if cutlass.const_expr(self.hoist_sf0_set):
                                tiled_mma.set(
                                    tcgen05.Field.SFA, tCtSFA[(None, None, 0)].iterator
                                )
                                tiled_mma.set(
                                    tcgen05.Field.SFB,
                                    tCtSFB_mma[(None, None, 0)].iterator,
                                )
                            ab_pipeline.consumer_wait(
                                ab_consumer_state, peek_ab_full_status
                            )
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                ab_consumer_state.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                            for kblock_idx in cutlass.range_constexpr(4):
                                kblock_coord = (
                                    None,
                                    None,
                                    kblock_idx,
                                    ab_consumer_state.index,
                                )
                                if (
                                    cutlass.const_expr(not self.hoist_sf0_set)
                                    or kblock_idx > 0
                                ):
                                    sf_kblock_coord = (None, None, kblock_idx)
                                    tiled_mma.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA[sf_kblock_coord].iterator,
                                    )
                                    tiled_mma.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB_mma[sf_kblock_coord].iterator,
                                    )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[kblock_coord],
                                    tCrB[kblock_coord],
                                    tCtAcc,
                                )
                            ab_pipeline.consumer_release(ab_consumer_state)

                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                    if cutlass.const_expr(self.peel_mma_tail_last):
                        for k_tile_tl in cutlass.range(
                            k_tile_cnt - self.num_ab_stage, k_tile_cnt - 1, 1, unroll=1
                        ):
                            if is_leader_cta:
                                if cutlass.const_expr(self.hoist_sf0_set):
                                    tiled_mma.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA[(None, None, 0)].iterator,
                                    )
                                    tiled_mma.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB_mma[(None, None, 0)].iterator,
                                    )
                                ab_pipeline.consumer_wait(
                                    ab_consumer_state, peek_ab_full_status
                                )
                                s2t_stage_coord_tl = (
                                    None,
                                    None,
                                    None,
                                    None,
                                    ab_consumer_state.index,
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfa,
                                    tCsSFA_compact_s2t[s2t_stage_coord_tl],
                                    tCtSFA_compact_s2t,
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfb,
                                    tCsSFB_compact_s2t[s2t_stage_coord_tl],
                                    tCtSFB_compact_s2t,
                                )
                                for kblock_idx_tl in cutlass.range_constexpr(4):
                                    kblock_coord_tl = (
                                        None,
                                        None,
                                        kblock_idx_tl,
                                        ab_consumer_state.index,
                                    )
                                    if (
                                        cutlass.const_expr(not self.hoist_sf0_set)
                                        or kblock_idx_tl > 0
                                    ):
                                        sf_kblock_coord_tl = (None, None, kblock_idx_tl)
                                        tiled_mma.set(
                                            tcgen05.Field.SFA,
                                            tCtSFA[sf_kblock_coord_tl].iterator,
                                        )
                                        tiled_mma.set(
                                            tcgen05.Field.SFB,
                                            tCtSFB_mma[sf_kblock_coord_tl].iterator,
                                        )
                                    cute.gemm(
                                        tiled_mma,
                                        tCtAcc,
                                        tCrA[kblock_coord_tl],
                                        tCrB[kblock_coord_tl],
                                        tCtAcc,
                                    )

                            ab_consumer_state.advance()
                            if is_leader_cta:
                                peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                    ab_consumer_state
                                )

                        if is_leader_cta:
                            if cutlass.const_expr(self.hoist_sf0_set):
                                tiled_mma.set(
                                    tcgen05.Field.SFA, tCtSFA[(None, None, 0)].iterator
                                )
                                tiled_mma.set(
                                    tcgen05.Field.SFB,
                                    tCtSFB_mma[(None, None, 0)].iterator,
                                )
                            ab_pipeline.consumer_wait(
                                ab_consumer_state, peek_ab_full_status
                            )
                            s2t_stage_coord_last = (
                                None,
                                None,
                                None,
                                None,
                                ab_consumer_state.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord_last],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord_last],
                                tCtSFB_compact_s2t,
                            )
                            for kblock_idx_last in cutlass.range_constexpr(4):
                                kblock_coord_last = (
                                    None,
                                    None,
                                    kblock_idx_last,
                                    ab_consumer_state.index,
                                )
                                if (
                                    cutlass.const_expr(not self.hoist_sf0_set)
                                    or kblock_idx_last > 0
                                ):
                                    sf_kblock_coord_last = (None, None, kblock_idx_last)
                                    tiled_mma.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA[sf_kblock_coord_last].iterator,
                                    )
                                    tiled_mma.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB_mma[sf_kblock_coord_last].iterator,
                                    )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[kblock_coord_last],
                                    tCrB[kblock_coord_last],
                                    tCtAcc,
                                )
                    elif cutlass.const_expr(self.unroll_mma_tail):
                        for k_tile_u in cutlass.range_constexpr(self.num_ab_stage):
                            if is_leader_cta:
                                if cutlass.const_expr(self.hoist_sf0_set):
                                    tiled_mma.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA[(None, None, 0)].iterator,
                                    )
                                    tiled_mma.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB_mma[(None, None, 0)].iterator,
                                    )
                                ab_pipeline.consumer_wait(
                                    ab_consumer_state, peek_ab_full_status
                                )
                                s2t_stage_coord_u = (
                                    None,
                                    None,
                                    None,
                                    None,
                                    ab_consumer_state.index,
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfa,
                                    tCsSFA_compact_s2t[s2t_stage_coord_u],
                                    tCtSFA_compact_s2t,
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfb,
                                    tCsSFB_compact_s2t[s2t_stage_coord_u],
                                    tCtSFB_compact_s2t,
                                )
                                for kblock_idx_u in cutlass.range_constexpr(4):
                                    kblock_coord_u = (
                                        None,
                                        None,
                                        kblock_idx_u,
                                        ab_consumer_state.index,
                                    )
                                    if (
                                        cutlass.const_expr(not self.hoist_sf0_set)
                                        or kblock_idx_u > 0
                                    ):
                                        sf_kblock_coord_u = (None, None, kblock_idx_u)
                                        tiled_mma.set(
                                            tcgen05.Field.SFA,
                                            tCtSFA[sf_kblock_coord_u].iterator,
                                        )
                                        tiled_mma.set(
                                            tcgen05.Field.SFB,
                                            tCtSFB_mma[sf_kblock_coord_u].iterator,
                                        )
                                    cute.gemm(
                                        tiled_mma,
                                        tCtAcc,
                                        tCrA[kblock_coord_u],
                                        tCrB[kblock_coord_u],
                                        tCtAcc,
                                    )

                            if cutlass.const_expr(k_tile_u + 1 < self.num_ab_stage):
                                ab_consumer_state.advance()
                                if is_leader_cta:
                                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                        ab_consumer_state
                                    )
                    else:
                        for k_tile in cutlass.range(
                            k_tile_cnt - self.num_ab_stage, k_tile_cnt, 1, unroll=1
                        ):
                            if is_leader_cta:
                                if cutlass.const_expr(self.hoist_sf0_set):
                                    tiled_mma.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA[(None, None, 0)].iterator,
                                    )
                                    tiled_mma.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB_mma[(None, None, 0)].iterator,
                                    )
                                ab_pipeline.consumer_wait(
                                    ab_consumer_state, peek_ab_full_status
                                )
                                s2t_stage_coord = (
                                    None,
                                    None,
                                    None,
                                    None,
                                    ab_consumer_state.index,
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfa,
                                    tCsSFA_compact_s2t[s2t_stage_coord],
                                    tCtSFA_compact_s2t,
                                )
                                cute.copy(
                                    tiled_copy_s2t_sfb,
                                    tCsSFB_compact_s2t[s2t_stage_coord],
                                    tCtSFB_compact_s2t,
                                )
                                for kblock_idx in cutlass.range_constexpr(4):
                                    kblock_coord = (
                                        None,
                                        None,
                                        kblock_idx,
                                        ab_consumer_state.index,
                                    )
                                    if (
                                        cutlass.const_expr(not self.hoist_sf0_set)
                                        or kblock_idx > 0
                                    ):
                                        sf_kblock_coord = (None, None, kblock_idx)
                                        tiled_mma.set(
                                            tcgen05.Field.SFA,
                                            tCtSFA[sf_kblock_coord].iterator,
                                        )
                                        tiled_mma.set(
                                            tcgen05.Field.SFB,
                                            tCtSFB_mma[sf_kblock_coord].iterator,
                                        )
                                    cute.gemm(
                                        tiled_mma,
                                        tCtAcc,
                                        tCrA[kblock_coord],
                                        tCrB[kblock_coord],
                                        tCtAcc,
                                    )

                            ab_consumer_state.advance()
                            peek_ab_full_status = cutlass.Boolean(1)
                            if ab_consumer_state.count < k_tile_cnt:
                                if is_leader_cta:
                                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                        ab_consumer_state
                                    )
                elif cutlass.const_expr(
                    self.direct_single_pass_mma and single_pass_ring
                ):
                    for k_tile in cutlass.range(1, k_tile_cnt, 1, unroll=1):
                        if is_leader_cta:
                            if cutlass.const_expr(self.hoist_sf0_set):
                                tiled_mma.set(
                                    tcgen05.Field.SFA, tCtSFA[(None, None, 0)].iterator
                                )
                                tiled_mma.set(
                                    tcgen05.Field.SFB,
                                    tCtSFB_mma[(None, None, 0)].iterator,
                                )
                            ab_pipeline.sync_object_full.wait(k_tile, cutlass.Int32(0))
                            s2t_stage_coord_d = (None, None, None, None, k_tile)
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord_d],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord_d],
                                tCtSFB_compact_s2t,
                            )
                            for kblock_idx_d in cutlass.range_constexpr(4):
                                kblock_coord_d = (None, None, kblock_idx_d, k_tile)
                                if (
                                    cutlass.const_expr(not self.hoist_sf0_set)
                                    or kblock_idx_d > 0
                                ):
                                    sf_kblock_coord_d = (None, None, kblock_idx_d)
                                    tiled_mma.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA[sf_kblock_coord_d].iterator,
                                    )
                                    tiled_mma.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB_mma[sf_kblock_coord_d].iterator,
                                    )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[kblock_coord_d],
                                    tCrB[kblock_coord_d],
                                    tCtAcc,
                                )
                else:
                    for k_tile in cutlass.range(1, k_tile_cnt, 1, unroll=1):
                        if is_leader_cta:
                            if cutlass.const_expr(self.hoist_sf0_set):
                                tiled_mma.set(
                                    tcgen05.Field.SFA, tCtSFA[(None, None, 0)].iterator
                                )
                                tiled_mma.set(
                                    tcgen05.Field.SFB,
                                    tCtSFB_mma[(None, None, 0)].iterator,
                                )
                            ab_pipeline.consumer_wait(
                                ab_consumer_state, peek_ab_full_status
                            )
                            s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                ab_consumer_state.index,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                            for kblock_idx in cutlass.range_constexpr(4):
                                kblock_coord = (
                                    None,
                                    None,
                                    kblock_idx,
                                    ab_consumer_state.index,
                                )
                                if (
                                    cutlass.const_expr(not self.hoist_sf0_set)
                                    or kblock_idx > 0
                                ):
                                    sf_kblock_coord = (None, None, kblock_idx)
                                    tiled_mma.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA[sf_kblock_coord].iterator,
                                    )
                                    tiled_mma.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB_mma[sf_kblock_coord].iterator,
                                    )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[kblock_coord],
                                    tCrB[kblock_coord],
                                    tCtAcc,
                                )
                            if cutlass.const_expr(not single_pass_ring):
                                ab_pipeline.consumer_release(ab_consumer_state)

                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                    ab_consumer_state
                                )
            else:
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_stage_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[s2t_stage_coord],
                            tCtSFB_compact_s2t,
                        )
                        for kblock_idx in cutlass.range(4, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()
            if cutlass.const_expr(self.pdl_mode == 1):
                cute.arch.griddepcontrol_launch_dependents()

        if warp_idx < self.mma_warp_id:
            if cutlass.const_expr(not (self.pre_alloc_tmem and not use_2cta_instrs)):
                tmem.allocate(self.num_tmem_alloc_cols)
                if cutlass.const_expr(self.fast_tmem_bar and not use_2cta_instrs):
                    if warp_idx == 0:
                        tmem.wait_for_alloc()
                else:
                    tmem.wait_for_alloc()

            if cutlass.const_expr(
                self.early_pf_desc_c and self.pf_desc_c_epi and self.use_tma_store
            ):
                if warp_idx == self.epilog_warp_id[0]:
                    cpasync.prefetch_descriptor(tma_atom_c)

            if cutlass.const_expr(self.warp0_pf_a and self.prefetch_a):
                if warp_idx == 0:
                    tAgA_slice0 = tAgA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    tAgSFA_slice0 = tAgSFA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    if cutlass.const_expr(self.single_pf_a):
                        if bidy < k_tile_cnt:
                            cute.prefetch(tma_atom_a, tAgA_slice0[(None, bidy)])
                            cute.prefetch(tma_atom_sfa, tAgSFA_slice0[(None, bidy)])
                    else:
                        grid_n_a = cute.arch.grid_dim()[1]
                        for pf_k in cutlass.range(bidy, k_tile_cnt, grid_n_a, unroll=1):
                            cute.prefetch(tma_atom_a, tAgA_slice0[(None, pf_k)])
                            cute.prefetch(tma_atom_sfa, tAgSFA_slice0[(None, pf_k)])

            if cutlass.const_expr(self.multiwarp_pf_b and (self.prefetch_b > 0)):
                if warp_idx < 3:
                    tBgB_mwb = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    sn_mwb = (
                        mma_tile_coord_mnl[1] // 2
                        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64)
                        else mma_tile_coord_mnl[1]
                    )
                    tBgSFB_mwb = tBgSFB[(None, sn_mwb, None, mma_tile_coord_mnl[2])]
                    if warp_idx == 0:
                        for pf_k_u in cutlass.range_constexpr(0, 3):
                            if cutlass.const_expr(pf_k_u < self.prefetch_b):
                                cute.prefetch(tma_atom_b, tBgB_mwb[(None, pf_k_u)])
                                cute.prefetch(tma_atom_sfb, tBgSFB_mwb[(None, pf_k_u)])
                    elif warp_idx == 1:
                        for pf_k_u in cutlass.range_constexpr(3, 6):
                            if cutlass.const_expr(pf_k_u < self.prefetch_b):
                                cute.prefetch(tma_atom_b, tBgB_mwb[(None, pf_k_u)])
                                cute.prefetch(tma_atom_sfb, tBgSFB_mwb[(None, pf_k_u)])
                    elif warp_idx == 2:
                        for pf_k_u in cutlass.range_constexpr(6, 8):
                            if cutlass.const_expr(pf_k_u < self.prefetch_b):
                                cute.prefetch(tma_atom_b, tBgB_mwb[(None, pf_k_u)])
                                cute.prefetch(tma_atom_sfb, tBgSFB_mwb[(None, pf_k_u)])

            if cutlass.const_expr(self.coop_b_mode >= 2):
                w = 4 - warp_idx
                if warp_idx >= (4 - self.coop_b_mode):
                    grid_n = cute.arch.grid_dim()[1]
                    ny = bidy + w * 36
                    if ny < grid_n:
                        tBgB_w = tBgB[(None, ny, None, mma_tile_coord_mnl[2])]
                        sn_w = (
                            ny // 2
                            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64)
                            else ny
                        )
                        tBgSFB_w = tBgSFB[(None, sn_w, None, mma_tile_coord_mnl[2])]
                        if cutlass.const_expr(self.unroll_coop_pf):
                            for pf_k_c in cutlass.range_constexpr(self.coop_pf0):
                                cute.prefetch(tma_atom_b, tBgB_w[(None, pf_k_c)])
                                cute.prefetch(tma_atom_sfb, tBgSFB_w[(None, pf_k_c)])
                        else:
                            for pf_k in cutlass.range(0, self.coop_pf0, 1, unroll=1):
                                cute.prefetch(tma_atom_b, tBgB_w[(None, pf_k)])
                                cute.prefetch(tma_atom_sfb, tBgSFB_w[(None, pf_k)])
            elif cutlass.const_expr(self.coop_b_warp3):
                if warp_idx == 3:
                    grid_n = cute.arch.grid_dim()[1]
                    for w in cutlass.range(1, 3, 1, unroll=1):
                        ny = bidy + w * 36
                        if ny < grid_n:
                            tBgB_w = tBgB[(None, ny, None, mma_tile_coord_mnl[2])]
                            sn_w = (
                                ny // 2
                                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64)
                                else ny
                            )
                            tBgSFB_w = tBgSFB[(None, sn_w, None, mma_tile_coord_mnl[2])]
                            if cutlass.const_expr(self.unroll_coop_pf):
                                for pf_k_c in cutlass.range_constexpr(self.pf0):
                                    cute.prefetch(tma_atom_b, tBgB_w[(None, pf_k_c)])
                                    cute.prefetch(
                                        tma_atom_sfb, tBgSFB_w[(None, pf_k_c)]
                                    )
                            else:
                                for pf_k in cutlass.range(0, self.pf0, 1, unroll=1):
                                    cute.prefetch(tma_atom_b, tBgB_w[(None, pf_k)])
                                    cute.prefetch(tma_atom_sfb, tBgSFB_w[(None, pf_k)])
            if cutlass.const_expr(self.split_pf_b_warp1 and (self.tma_pf0 < self.pf0)):
                if warp_idx == 1:
                    tBgB_cur1 = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    sn_cur1 = (
                        mma_tile_coord_mnl[1] // 2
                        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64)
                        else mma_tile_coord_mnl[1]
                    )
                    tBgSFB_cur1 = tBgSFB[(None, sn_cur1, None, mma_tile_coord_mnl[2])]
                    if cutlass.const_expr(self.unroll_coop_pf):
                        for pf_i in cutlass.range_constexpr(self.pf0 - self.tma_pf0):
                            cute.prefetch(
                                tma_atom_b, tBgB_cur1[(None, self.tma_pf0 + pf_i)]
                            )
                            cute.prefetch(
                                tma_atom_sfb, tBgSFB_cur1[(None, self.tma_pf0 + pf_i)]
                            )
                    elif cutlass.const_expr(self.unroll_pf0):
                        for pf_i in cutlass.range_constexpr(self.pf0 - self.tma_pf0):
                            cute.prefetch(
                                tma_atom_b, tBgB_cur1[(None, self.tma_pf0 + pf_i)]
                            )
                            cute.prefetch(
                                tma_atom_sfb, tBgSFB_cur1[(None, self.tma_pf0 + pf_i)]
                            )
                    else:
                        for pf_k in cutlass.range(self.tma_pf0, self.pf0, 1, unroll=1):
                            cute.prefetch(tma_atom_b, tBgB_cur1[(None, pf_k)])
                            cute.prefetch(tma_atom_sfb, tBgSFB_cur1[(None, pf_k)])
            if cutlass.const_expr(self.coop_a_next_wave):
                if warp_idx == 3:
                    pf_ka = bidy + 36
                    if pf_ka < k_tile_cnt:
                        tAgA_w3 = tAgA[
                            (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                        ]
                        tAgSFA_w3 = tAgSFA[
                            (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                        ]
                        cute.prefetch(tma_atom_a, tAgA_w3[(None, pf_ka)])
                        cute.prefetch(tma_atom_sfa, tAgSFA_w3[(None, pf_ka)])
            if cutlass.const_expr(self.coop_deep_k > 0):
                if warp_idx == 3:
                    tBgB_cur = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    sn_cur = (
                        mma_tile_coord_mnl[1] // 2
                        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64)
                        else mma_tile_coord_mnl[1]
                    )
                    tBgSFB_cur = tBgSFB[(None, sn_cur, None, mma_tile_coord_mnl[2])]
                    for pf_k in cutlass.range(
                        self.pf0, self.pf0 + self.coop_deep_k, 1, unroll=1
                    ):
                        if pf_k < k_tile_cnt:
                            cute.prefetch(tma_atom_b, tBgB_cur[(None, pf_k)])
                            cute.prefetch(tma_atom_sfb, tBgSFB_cur[(None, pf_k)])
            if cutlass.const_expr(self.deep_b_warps23 > 0):
                if warp_idx == 2 or warp_idx == 3:
                    tBgB_dp = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    sn_dp = (
                        mma_tile_coord_mnl[1] // 2
                        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64)
                        else mma_tile_coord_mnl[1]
                    )
                    tBgSFB_dp = tBgSFB[(None, sn_dp, None, mma_tile_coord_mnl[2])]
                    dp_start = self.pf0 + (warp_idx - 2) * self.deep_b_warps23
                    if cutlass.const_expr(
                        self.pf0 + 2 * self.deep_b_warps23 <= k_tile_cnt_static
                    ):
                        for pf_k in cutlass.range(
                            dp_start, dp_start + self.deep_b_warps23, 1, unroll=1
                        ):
                            cute.prefetch(tma_atom_b, tBgB_dp[(None, pf_k)])
                            cute.prefetch(tma_atom_sfb, tBgSFB_dp[(None, pf_k)])
                    else:
                        for pf_k in cutlass.range(
                            dp_start, dp_start + self.deep_b_warps23, 1, unroll=1
                        ):
                            if pf_k < k_tile_cnt:
                                cute.prefetch(tma_atom_b, tBgB_dp[(None, pf_k)])
                                cute.prefetch(tma_atom_sfb, tBgSFB_dp[(None, pf_k)])
            if cutlass.const_expr(self.s2_wave3_b_pf):
                if warp_idx == 1:
                    grid_n_w3 = cute.arch.grid_dim()[1]
                    ny_w3 = bidy + 108
                    if ny_w3 < grid_n_w3:
                        tBgB_w3 = tBgB[(None, ny_w3, None, mma_tile_coord_mnl[2])]
                        sn_w3 = (
                            ny_w3 // 2
                            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64)
                            else ny_w3
                        )
                        tBgSFB_w3 = tBgSFB[(None, sn_w3, None, mma_tile_coord_mnl[2])]
                        for pf_k in cutlass.range(0, self.coop_pf0, 1, unroll=1):
                            cute.prefetch(tma_atom_b, tBgB_w3[(None, pf_k)])
                            cute.prefetch(tma_atom_sfb, tBgSFB_w3[(None, pf_k)])

            if cutlass.const_expr(
                self.static_epilog_tmem_ptr and self.c_alias_a and not use_2cta_instrs
            ):
                acc_tmem_ptr = cute.make_ptr(
                    self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16
                )
            else:
                if cutlass.const_expr(self.fast_tmem_bar and not use_2cta_instrs):
                    self.epilog_sync_barrier.arrive_and_wait()
                acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            if cutlass.const_expr(not self.use_tma_store):
                op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x64, tcgen05.Pack.NONE)
                copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
                tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
                tTR_gC = thr_copy_t2r.partition_D(tCgC_simt)
                tTR_rAcc = cute.make_rmem_tensor(
                    tTR_gC[None, None, None, None, 0, 0, 0].shape, self.acc_dtype
                )
                tTR_rC = cute.make_rmem_tensor(
                    tTR_gC[None, None, None, None, 0, 0, 0].shape, self.c_dtype
                )
                simt_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.c_dtype
                )
                tTR_gC = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]

                alpha_val = cutlass.Float32(1.0)
                if cutlass.const_expr(self.has_alpha):
                    alpha_val = cutlass.Float32(mAlpha[0])
                acc_pipeline.consumer_wait(acc_consumer_state)
                cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
                acc_vec = tTR_rAcc.load()
                if cutlass.const_expr(self.has_alpha):
                    acc_vec = acc_vec * alpha_val
                tTR_rC.store(acc_vec.to(self.c_dtype))
                cute.copy(simt_atom, tTR_rC, tTR_gC)
                tmem.relinquish_alloc_permit()
                self.epilog_sync_barrier.arrive_and_wait()
                tmem.free(acc_tmem_ptr)
            else:
                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    self.cta_tile_shape_mnk,
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    epi_tile,
                    use_2cta_instrs,
                )
                tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
                tiled_copy_t2r = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
                )
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

                gC_tma_epi = cute.local_tile(
                    mC_tma_mnl,
                    cute.slice_(self.mma_tiler, (None, None, 0)),
                    (None, None, None),
                )
                tCgC_tma_epi = thr_mma.partition_C(gC_tma_epi)
                gC_mnl_epi = cute.flat_divide(
                    tCgC_tma_epi[((None, None), 0, 0, None, None, None)], epi_tile
                )
                tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
                tTR_rAcc = cute.make_rmem_tensor(
                    tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
                )
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

                copy_atom_r2s = sm100_utils.get_smem_store_op(
                    self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
                )
                tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sC = thr_copy_r2s.partition_D(sC)
                tRS_rC = tiled_copy_r2s.retile(tTR_rC)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                if cutlass.const_expr(self.hoist_tma_c_part):
                    sC_for_tma_hoist = cute.group_modes(sC, 0, 2)
                    gC_for_tma_hoist = cute.group_modes(gC_mnl_epi, 0, 2)
                    bSG_sC_hoist, bSG_gC_part_hoist = cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        sC_for_tma_hoist,
                        gC_for_tma_hoist,
                    )
                    bSG_gC_slice_hoist = bSG_gC_part_hoist[
                        (None, None, None, *mma_tile_coord_mnl)
                    ]
                    bSG_gC_hoist = cute.group_modes(
                        bSG_gC_slice_hoist, 1, cute.rank(bSG_gC_slice_hoist)
                    )

                if cutlass.const_expr(self.pf_desc_c_epi and not self.early_pf_desc_c):
                    if warp_idx == self.epilog_warp_id[0]:
                        cpasync.prefetch_descriptor(tma_atom_c)
                if cutlass.const_expr(self.hoist_epilog_wait):
                    tmem.relinquish_alloc_permit()
                    if warp_idx == self.epilog_warp_id[0]:
                        if cutlass.const_expr(self.use_pdl):
                            cute.arch.griddepcontrol_wait()

                alpha_val = cutlass.Float32(1.0)
                if cutlass.const_expr(self.has_alpha):
                    alpha_val = cutlass.Float32(mAlpha[0])

                acc_pipeline.consumer_wait(acc_consumer_state)

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                if cutlass.const_expr(self.early_tmem_free and subtile_cnt == 1):
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, 0)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    if cutlass.const_expr(not self.hoist_epilog_wait):
                        tmem.relinquish_alloc_permit()
                    self.epilog_sync_barrier.arrive_and_wait()
                    tmem.free(acc_tmem_ptr)
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    if cutlass.const_expr(self.has_alpha):
                        acc_vec = acc_vec * alpha_val
                    tRS_rC.store(acc_vec.to(self.c_dtype))
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, 0)])
                    cute.arch.fence_view_async_shared()
                    self.epilog_sync_barrier.arrive_and_wait()
                else:
                    if cutlass.const_expr(
                        self.fast_single_subtile and subtile_cnt == 1
                    ):
                        cute.copy(
                            tiled_copy_t2r, tTR_tAcc[(None, None, None, 0)], tTR_rAcc
                        )
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        if cutlass.const_expr(self.has_alpha):
                            acc_vec = acc_vec * alpha_val
                        tRS_rC.store(acc_vec.to(self.c_dtype))
                        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, 0)])
                    else:
                        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                            if cutlass.const_expr(self.has_alpha):
                                acc_vec = acc_vec * alpha_val
                            tRS_rC.store(acc_vec.to(self.c_dtype))
                            cute.copy(
                                tiled_copy_r2s,
                                tRS_rC,
                                tRS_sC[(None, None, None, subtile_idx)],
                            )

                    cute.arch.fence_view_async_shared()
                    if cutlass.const_expr(not self.hoist_epilog_wait):
                        tmem.relinquish_alloc_permit()
                    self.epilog_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(self.overlap_tmem_free):
                        if warp_idx == self.epilog_warp_id[0]:
                            if cutlass.const_expr(
                                self.use_pdl and not self.hoist_epilog_wait
                            ):
                                cute.arch.griddepcontrol_wait()
                            if cutlass.const_expr(self.hoist_tma_c_part):
                                if cutlass.const_expr(
                                    self.fast_single_subtile and subtile_cnt == 1
                                ):
                                    cute.copy(
                                        tma_atom_c,
                                        bSG_sC_hoist[(None, 0)],
                                        bSG_gC_hoist[(None, 0)],
                                    )
                                else:
                                    for subtile_idx_w0 in cutlass.range_constexpr(
                                        subtile_cnt
                                    ):
                                        cute.copy(
                                            tma_atom_c,
                                            bSG_sC_hoist[(None, subtile_idx_w0)],
                                            bSG_gC_hoist[(None, subtile_idx_w0)],
                                        )
                            else:
                                sC_for_tma_w0 = cute.group_modes(sC, 0, 2)
                                gC_for_tma_w0 = cute.group_modes(gC_mnl_epi, 0, 2)
                                bSG_sC_w0, bSG_gC_part_w0 = cpasync.tma_partition(
                                    tma_atom_c,
                                    0,
                                    cute.make_layout(1),
                                    sC_for_tma_w0,
                                    gC_for_tma_w0,
                                )
                                bSG_gC_slice_w0 = bSG_gC_part_w0[
                                    (None, None, None, *mma_tile_coord_mnl)
                                ]
                                bSG_gC_w0 = cute.group_modes(
                                    bSG_gC_slice_w0, 1, cute.rank(bSG_gC_slice_w0)
                                )
                                for subtile_idx_w0 in cutlass.range_constexpr(
                                    subtile_cnt
                                ):
                                    cute.copy(
                                        tma_atom_c,
                                        bSG_sC_w0[(None, subtile_idx_w0)],
                                        bSG_gC_w0[(None, subtile_idx_w0)],
                                    )
                            cute.arch.cp_async_bulk_commit_group()
                        tmem.free(acc_tmem_ptr)
                        if warp_idx == self.epilog_warp_id[0]:
                            if cutlass.const_expr(self.store_wait):
                                cute.arch.cp_async_bulk_wait_group(0, read=True)
                    else:
                        tmem.free(acc_tmem_ptr)

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


_OP_K_MAJOR = (
    cute.nvgpu.OperandMajorMode.K
    if hasattr(cute.nvgpu, "OperandMajorMode")
    else tcgen05.OperandMajorMode.K
)


class Sm100Nvfp4GemmPersistent2CTA:
    """Persistent Warp-Specialized SM100 2-CTA NVFP4 GEMM with Overlapped/Double-Buffered TMEM & PDL-Safe Bookends."""

    def __init__(
        self,
        tile_n: int = 256,
        cluster_shape_mn: Tuple[int, int] = (2, 1),
        prefetch_dist: int = 0,
        pdl_safe: bool = True,
        c_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        has_alpha: bool = True,
    ):
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.c_dtype = c_dtype
        self.has_alpha = has_alpha
        self.sf_vec_size = 16
        self.tile_n = tile_n
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (256, tile_n, 256)
        self.mma_tiler_sfb = (128, ((tile_n + 127) // 128) * 128, 256)
        self.cta_tile_shape_mnk = (128, tile_n, 256)
        self.cta_group = tcgen05.CtaGroup.TWO
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192
        self.epi_tile = (128, 64)
        self.epi_tile_n = 64
        self.num_acc_stage = 1 if tile_n == 256 else 2
        self.overlapping_accum = self.num_acc_stage == 1
        self.num_sfa_tmem_cols = 16
        self.num_sfb_tmem_cols = (self.mma_tiler_sfb[1] // 32) * 4
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            tile_n * self.num_acc_stage
            if not self.overlapping_accum
            else tile_n * 2 - self.num_sf_tmem_cols
        )
        self.iter_acc_early_release = self.num_sf_tmem_cols // self.epi_tile_n
        self.num_tmem_alloc_cols = 512
        self.pdl_safe = pdl_safe
        self.prefetch_dist_param = prefetch_dist
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=160)

    def _compute_stages_and_layouts(self):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            _OP_K_MAJOR,
            _OP_K_MAJOR,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            (256, self.tile_n),
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            _OP_K_MAJOR,
            _OP_K_MAJOR,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            (128, self.mma_tiler_sfb[1]),
        )
        a1 = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.ab_dtype, 1)
        b1 = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.ab_dtype, 1)
        sfa1 = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        sfb1 = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        c1 = sm100_utils.make_smem_layout_epi(
            self.c_dtype, utils.LayoutEnum.ROW_MAJOR, self.epi_tile, 1
        )

        ab_bytes = (
            cute.size_in_bytes(self.ab_dtype, a1)
            + cute.size_in_bytes(self.ab_dtype, b1)
            + cute.size_in_bytes(self.sf_dtype, sfa1)
            + cute.size_in_bytes(self.sf_dtype, sfb1)
        )
        c_bytes_per = cute.size_in_bytes(self.c_dtype, c1)
        reserved = 1024
        num_c_stage = 2
        num_ab_stage = (
            self.smem_capacity - (reserved + num_c_stage * c_bytes_per)
        ) // ab_bytes
        num_c_stage += (
            self.smem_capacity
            - ab_bytes * num_ab_stage
            - (reserved + num_c_stage * c_bytes_per)
        ) // c_bytes_per
        self.num_ab_stage = num_ab_stage
        self.num_c_stage = num_c_stage
        self.prefetch_dist = (
            self.prefetch_dist_param if self.prefetch_dist_param >= 0 else num_ab_stage
        )
        return tiled_mma, tiled_mma_sfb

    def _prepare_host_args(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: Optional[cute.Pointer],
        problem_size: tuple,
    ):
        m, n, k, l = problem_size
        tiled_mma, tiled_mma_sfb = self._compute_stages_and_layouts()

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma_sfb.thr_id.shape,)
        )
        self.num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])

        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage
        )
        sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, utils.LayoutEnum.ROW_MAJOR, self.epi_tile, self.num_c_stage
        )

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
        epi_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout((cute.assume(m, 32), n, l), stride=(n, 1, m * n)),
        )
        sfa_tensor = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, 16)
        )
        sfb_tensor = cute.make_tensor(
            sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, 16)
        )
        alpha_tensor = (
            cute.make_tensor(alpha_ptr, cute.make_layout((1,), stride=(1,)))
            if self.has_alpha and alpha_ptr is not None
            else None
        )

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )

        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        if cutlass.const_expr(self.tile_n == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)
            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, cute.make_layout(new_shape, stride=new_stride)
            )

        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c_tensor, epi_smem_layout, self.epi_tile
        )

        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.ab_dtype, a_smem_layout)
            + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        ) * 2

        c_shape = cute.slice_(self.cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c_tensor, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*self.cluster_shape_mn, 1)
        max_active_clusters = 74 if self.cluster_shape_mn[1] == 1 else 34
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(c_smem_layout_staged.outer)
                ],
                1024,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(a_smem_layout_staged.outer)
                ],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(b_smem_layout_staged.outer)
                ],
                1024,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(sfa_smem_layout_staged)
                ],
                1024,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(sfb_smem_layout_staged)
                ],
                1024,
            ]

        self.shared_storage = SharedStorage
        kern_call = self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            alpha_tensor,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            c_smem_layout_staged,
            tile_sched_params,
        )
        return kern_call, grid, cluster_shape_mnl

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: cutlass.Constexpr[tuple],
    ):
        kern_call, grid, cluster_shape_mnl = self._prepare_host_args(
            a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, None, problem_size
        )
        kern_call.launch(
            grid=grid,
            block=[192, 1, 1],
            cluster=cluster_shape_mnl,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.jit
    def call_with_stream_alpha(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        stream: cuda.CUstream,
        problem_size: cutlass.Constexpr[tuple],
    ):
        kern_call, grid, cluster_shape_mnl = self._prepare_host_args(
            a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, alpha_ptr, problem_size
        )
        kern_call.launch(
            grid=grid,
            block=[192, 1, 1],
            cluster=cluster_shape_mnl,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mAlpha: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        mma_tile_coord_v = bidx % 2
        is_leader_cta = mma_tile_coord_v == 0

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_tma_producer
            ),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )
        sfa_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        sfb_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        k_block_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )

        # 1. TMA Load Warp
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            if cutlass.const_expr(self.pdl_safe):
                cute.arch.griddepcontrol_wait()
            else:
                cute.arch.griddepcontrol_launch_dependents()

            while work_tile.is_valid_tile:
                cur_coord = work_tile.tile_idx
                mma_coord_mnl = (cur_coord[0] // 2, cur_coord[1], cur_coord[2])
                tAgA_s = tAgA[(None, mma_coord_mnl[0], None, mma_coord_mnl[2])]
                tBgB_s = tBgB[(None, mma_coord_mnl[1], None, mma_coord_mnl[2])]
                tAgSFA_s = tAgSFA[(None, mma_coord_mnl[0], None, mma_coord_mnl[2])]
                slice_n = (
                    mma_coord_mnl[1] // 2
                    if cutlass.const_expr(self.tile_n == 64)
                    else mma_coord_mnl[1]
                )
                tBgSFB_s = tBgSFB[(None, slice_n, None, mma_coord_mnl[2])]

                if cutlass.const_expr(self.prefetch_dist > 0):
                    for pf_k in cutlass.range(
                        0, min(self.prefetch_dist, k_block_cnt), unroll=1
                    ):
                        cute.prefetch(tma_atom_a, tAgA_s[(None, pf_k)])
                        cute.prefetch(tma_atom_b, tBgB_s[(None, pf_k)])
                        cute.prefetch(tma_atom_sfa, tAgSFA_s[(None, pf_k)])
                        cute.prefetch(tma_atom_sfb, tBgSFB_s[(None, pf_k)])

                ab_prod.reset_count()
                peek_empty = cutlass.Boolean(1)
                if ab_prod.count < k_block_cnt:
                    peek_empty = ab_pipeline.producer_try_acquire(ab_prod)

                for kt in cutlass.range(0, k_block_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_prod, peek_empty)
                    bar = ab_pipeline.producer_get_barrier(ab_prod)
                    cnt = ab_prod.count
                    stg = ab_prod.index
                    cute.copy(
                        tma_atom_a,
                        tAgA_s[(None, cnt)],
                        tAsA[(None, stg)],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_s[(None, cnt)],
                        tBsB[(None, stg)],
                        tma_bar_ptr=bar,
                        mcast_mask=b_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_s[(None, cnt)],
                        tAsSFA[(None, stg)],
                        tma_bar_ptr=bar,
                        mcast_mask=sfa_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_s[(None, cnt)],
                        tBsSFB[(None, stg)],
                        tma_bar_ptr=bar,
                        mcast_mask=sfb_mcast_mask,
                    )

                    if cutlass.const_expr(self.prefetch_dist > 0):
                        if cnt + self.prefetch_dist < k_block_cnt:
                            cute.prefetch(
                                tma_atom_a, tAgA_s[(None, cnt + self.prefetch_dist)]
                            )
                            cute.prefetch(
                                tma_atom_b, tBgB_s[(None, cnt + self.prefetch_dist)]
                            )
                            cute.prefetch(
                                tma_atom_sfa, tAgSFA_s[(None, cnt + self.prefetch_dist)]
                            )
                            cute.prefetch(
                                tma_atom_sfb, tBgSFB_s[(None, cnt + self.prefetch_dist)]
                            )

                    ab_prod.advance()
                    peek_empty = cutlass.Boolean(1)
                    if ab_prod.count < k_block_cnt:
                        peek_empty = ab_pipeline.producer_try_acquire(ab_prod)

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_pipeline.producer_tail(ab_prod)
            if cutlass.const_expr(self.pdl_safe):
                cute.arch.griddepcontrol_launch_dependents()

        # 2. MMA Warp
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            copy_atom_s2t = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO), self.sf_dtype
            )
            tCsSFA_c = cute.filter_zeros(sSFA)
            tCtSFA_c = cute.filter_zeros(tCtSFA)
            tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_c)
            thr_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
            tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_sfa, thr_s2t_sfa.partition_S(tCsSFA_c)
            )
            tCtSFA_s2t = thr_s2t_sfa.partition_D(tCtSFA_c)

            tCsSFB_c = cute.filter_zeros(sSFB)
            tCtSFB_c = cute.filter_zeros(tCtSFB)
            tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_c)
            thr_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
            tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_sfb, thr_s2t_sfb.partition_S(tCsSFB_c)
            )
            tCtSFB_s2t = thr_s2t_sfb.partition_D(tCtSFB_c)

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            tile_sched_m = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile_m = tile_sched_m.initial_work_tile_info()
            ab_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile_m.is_valid_tile:
                cur_coord_m = work_tile_m.tile_idx
                mma_coord_mnl_m = (cur_coord_m[0] // 2, cur_coord_m[1], cur_coord_m[2])
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_idx = acc_prod.phase ^ 1
                else:
                    acc_stage_idx = acc_prod.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_idx)]

                ab_cons.reset_count()
                peek_full = cutlass.Boolean(1)
                if ab_cons.count < k_block_cnt and is_leader_cta:
                    peek_full = ab_pipeline.consumer_try_wait(ab_cons)

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_prod)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.tile_n == 192 or self.tile_n == 64):
                    offset = cutlass.Int32((mma_coord_mnl_m[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kt in cutlass.range(0, k_block_cnt, 1, unroll=1):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_cons, peek_full)
                        st = ab_cons.index
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_s2t[(None, None, None, None, st)],
                            tCtSFA_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_s2t[(None, None, None, None, st)],
                            tCtSFB_s2t,
                        )
                        for kb in cutlass.range(4, unroll_full=True):
                            tiled_mma.set(
                                tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB, tCtSFB_mma[(None, None, kb)].iterator
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[(None, None, kb, st)],
                                tCrB[(None, None, kb, st)],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipeline.consumer_release(ab_cons)
                    ab_cons.advance()
                    peek_full = cutlass.Boolean(1)
                    if ab_cons.count < k_block_cnt and is_leader_cta:
                        peek_full = ab_pipeline.consumer_try_wait(ab_cons)

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_prod)
                acc_prod.advance()
                tile_sched_m.advance_to_next_work()
                work_tile_m = tile_sched_m.get_current_work()

            acc_pipeline.producer_tail(acc_prod)

        # 3. Epilogue Warps (0..3)
        if warp_idx < 4:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                utils.LayoutEnum.ROW_MAJOR,
                self.c_dtype,
                self.acc_dtype,
                self.epi_tile,
                True,
            )
            tAcc_epi = cute.flat_divide(
                tCtAcc_base[((None, None), 0, 0, None)], self.epi_tile
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)

            gC_mnl = cute.local_tile(
                mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
            )
            tCgC = thr_mma.partition_C(gC_mnl)
            gC_epi = cute.flat_divide(
                tCgC[((None, None), 0, 0, None, None, None)], self.epi_tile
            )
            tTR_gC = thr_copy_t2r.partition_D(gC_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

            copy_atom_r2s = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR, self.c_dtype, self.acc_dtype, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)

            bSG_sC, bSG_gC_part = cpasync.tma_partition(
                tma_atom_c,
                0,
                cute.make_layout(1),
                cute.group_modes(sC, 0, 2),
                cute.group_modes(gC_epi, 0, 2),
            )

            alpha_val = cutlass.Float32(1.0)
            if cutlass.const_expr(self.has_alpha):
                alpha_val = cutlass.Float32(mAlpha[0])

            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            )
            tile_sched_e = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile_e = tile_sched_e.initial_work_tile_info()
            acc_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            if cutlass.const_expr(not self.pdl_safe):
                if warp_idx == 0:
                    cute.arch.griddepcontrol_wait()

            while work_tile_e.is_valid_tile:
                cur_coord_e = work_tile_e.tile_idx
                mma_coord_mnl_e = (cur_coord_e[0] // 2, cur_coord_e[1], cur_coord_e[2])
                bSG_gC = bSG_gC_part[(None, None, None, *mma_coord_mnl_e)]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_idx_e = acc_cons.phase
                    reverse_subtile = (
                        cutlass.Boolean(True)
                        if acc_stage_idx_e == 0
                        else cutlass.Boolean(False)
                    )
                else:
                    acc_stage_idx_e = acc_cons.index

                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_idx_e)
                ]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                acc_pipeline.consumer_wait(acc_cons)

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched_e.num_tiles_executed * subtile_cnt

                for subtile_idx in cutlass.range(subtile_cnt):
                    real_sub_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_sub_idx = subtile_cnt - 1 - subtile_idx
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, real_sub_idx)],
                        tTR_rAcc,
                    )
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_cons)
                            acc_cons.advance()

                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    if cutlass.const_expr(self.has_alpha):
                        acc_vec = acc_vec * alpha_val
                    tRS_rC.store(acc_vec.to(self.c_dtype))

                    c_buf = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buf)])
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id, number_of_threads=128
                    )
                    if warp_idx == 0:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buf)],
                            bSG_gC[(None, real_sub_idx)],
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id, number_of_threads=128
                    )

                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_cons)
                    acc_cons.advance()

                tile_sched_e.advance_to_next_work()
                work_tile_e = tile_sched_e.get_current_work()

            tmem.relinquish_alloc_permit()
            c_pipeline.producer_tail()
            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=128)
            tmem.free(acc_tmem_ptr)


def _select_persistent_2cta_tile_n(m: int, n: int, l: int) -> int:
    m_clusters = _ceil_div(m, 256)
    valid_tns = [64]
    if n % 128 == 0:
        valid_tns.append(128)
    if n >= 512 and n % 64 == 0:
        valid_tns.append(192)
    if n % 256 == 0:
        valid_tns.append(256)
    weights = {64: 0.58, 128: 0.68, 192: 0.85, 256: 1.00}
    return min(
        valid_tns,
        key=lambda tn: _ceil_div(m_clusters * _ceil_div(n, tn) * l, 74) * weights[tn],
    )


_FAST_CACHE = {}
_SGLANG_GEMM_CACHE = {}
_SM100_DEV_CACHE: dict[int, bool] = {}
_KEEPALIVE = []
_ADAPTIVE_2CTA = True
# Variant flags per path (see Sm100Nvfp4GemmFast.__init__).
_VARIANT_2CTA = dict(
    pdl_safe=True,
    store_wait=True,
    prefetch_b=6,
    prefetch_a=True,
    l2_hint=False,
    c_alias_a=False,
)
_VARIANT_1CTA = dict(
    pdl_safe=True,
    store_wait=True,
    prefetch_b=6,
    prefetch_a=True,
    l2_hint=False,
    c_alias_a=True,
    b_first=2,
)
# P4: cluster along N for the 1-CTA path multicasts A/SFA across the cluster (needs (N/64) % cluster_n == 0).
_CLUSTER_1CTA = (1, 1)
# griddepcontrol.launch_dependents point for the 1-CTA path: 3 = TMA warp start, 2 = after the last load.
_PDL_MODE_1CTA = 2


def _get_fast_runner(shape: Tuple[int, int, int, int]):
    runner = _FAST_CACHE.get(shape)
    if runner is not None:
        return runner

    m, n, k, l = shape
    if _ADAPTIVE_2CTA and m >= 256 and (m % 128 == 0) and (n % 64 == 0):
        tile_n = _select_persistent_2cta_tile_n(m, n, l)
        kern = Sm100Nvfp4GemmPersistent2CTA(
            tile_n=tile_n,
            cluster_shape_mn=(2, 1),
            prefetch_dist=0,
            pdl_safe=True,
            c_dtype=cutlass.Float16,
            has_alpha=False,
        )
    else:
        cl = _CLUSTER_1CTA if (n // 64) % _CLUSTER_1CTA[1] == 0 else (1, 1)
        early_k = 0
        pdl_m = _PDL_MODE_1CTA
        if k <= 2048:
            early_k = 5
        kern = Sm100Nvfp4GemmFast(
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=cl,
            num_ab_stage=8,
            num_c_stage=1,
            use_tma_store=True,
            subtile_width=64,
            early_tmem_free=False,
            pdl_mode=pdl_m,
            pdl_early_k=early_k,
            peel_first_tile=True,
            prefetch_b=(8 if 2048 < k < 16384 else 6),
            prefetch_a=(k > 256),
            coop_b_warp3=(k <= 2048),
            overlap_tmem_free=True,
            defer_sync=True,
            pre_alloc_tmem=False,
            num_tmem_alloc_cols=128,
            copy_before_pf=True,
            pf_init=0,
            rolling_pf_b=True,
            hoist_epilog_wait=True,
            pf_desc_c_epi=True,
            fast_single_subtile=True,
            hoist_tma_c_part=True,
            coop_b_mode=(2 if (k <= 2048 or k >= 16384) else 0),
            coop_deep_k=0,
            fast_single_pass_ring=True,
            peel_producer_init=True,
            elide_tail_release=True,
            clean_c_prologue=True,
            warp_scoped_ab_frag=True,
            coop_a_next_wave=(k >= 16384),
            coop_pf_init=(2 if k >= 16384 else 0),
            split_pf_b_warp1=True,
            tma_pf_split=4,
            direct_consumer_wait=False,
            unroll_pf0=True,
            deep_b_warps23=0,
            s2_wave3_b_pf=False,
            elide_acc_empty_wait=True,
            omit_1cta_dealloc_mbar=True,
            hoist_sf0_set=True,
            unroll_peel_init=True,
            unroll_single_pass_tma=True,
            split_tma_steady_pf=(2048 < k < 16384),
            unroll_tma_tail_pf=False,
            early_cta_sync=False,
            warp0_only_tma_c_part=True,
            peel_mma_tail_last=False,
            c_alias_a=True,
            b_first=2,
            pdl_safe=True,
            store_wait=True,
            l2_hint=False,
            single_pf_a=True,
            warp0_pf_a=True,
            parallel_mbar_init=True,
            elide_dead_mbar_init=True,
            fast_tmem_bar=(k >= 16384),
            unroll_coop_pf=(k >= 16384),
            hoist_mma_s2t_part=True,
            hoist_mma_full=(k >= 16384),
            alg_mbar_init=True,
            unroll_mma_tail=False,
            direct_single_pass_mma=True,
            early_pf_desc_c=(2048 < k < 16384),
            static_epilog_tmem_ptr=(k > 2048),
            alg_mbar_6warp=True,
            elide_1cta_mbar_fence=True,
            dom_tma_desc=0,
        )

    a_p = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_p = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_p = make_ptr(cutlass.Float16, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_p = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_p = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32)

    fn = cute.compile(kern, a_p, b_p, sfa_p, sfb_p, c_p, shape)

    def _init_and_run(a, b, sfa_perm, sfb_perm, c):
        fn(
            make_ptr(
                cutlass.Float4E2M1FN,
                a.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float4E2M1FN,
                b.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                sfa_perm.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                sfb_perm.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
        )
        ex = fn._default_executor
        packed_args = ex._tls.packed_args
        ptr_buf = (ctypes.c_uint64 * 5)()
        base_addr = ctypes.addressof(ptr_buf)
        for i in range(5):
            packed_args[i] = base_addr + i * 8
        capi_func = ex.capi_func
        _KEEPALIVE.append((kern, fn, ex, packed_args, ptr_buf))

        def _fast_call(a, b, sfa_perm, sfb_perm, c):
            ptr_buf[0] = a.data_ptr()
            ptr_buf[1] = b.data_ptr()
            ptr_buf[2] = sfa_perm.data_ptr()
            ptr_buf[3] = sfb_perm.data_ptr()
            ptr_buf[4] = c.data_ptr()
            capi_func(packed_args)

        _FAST_CACHE[shape] = _fast_call

    _FAST_CACHE[shape] = _init_and_run
    return _init_and_run


def _get_sglang_gemm_runner(
    shape: Tuple[int, int, int, int],
    out_dtype: torch.dtype,
    has_alpha: bool = True,
):
    key = (shape, out_dtype, has_alpha)
    runner = _SGLANG_GEMM_CACHE.get(key)
    if runner is not None:
        return runner

    c_dtype_cutlass = (
        cutlass.BFloat16 if out_dtype == torch.bfloat16 else cutlass.Float16
    )
    m, n, k, l = shape
    if m >= 256 and (m % 128 == 0) and (n % 64 == 0):
        tile_n = _select_persistent_2cta_tile_n(m, n, l)
        kern = Sm100Nvfp4GemmPersistent2CTA(
            tile_n=tile_n,
            cluster_shape_mn=(2, 1),
            prefetch_dist=0,
            pdl_safe=True,
            c_dtype=c_dtype_cutlass,
            has_alpha=has_alpha,
        )
    elif n >= 4096 and k >= 7168 and (n // 64) % 2 == 0:
        cl_n = 4 if (n // 64) % 4 == 0 else 2
        kern = Sm100Nvfp4GemmFast(
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, cl_n),
            num_ab_stage=8,
            num_c_stage=1,
            use_tma_store=True,
            subtile_width=64,
            pdl_mode=_PDL_MODE_1CTA,
            peel_first_tile=True,
            prefetch_b=6,
            prefetch_a=True,
            overlap_tmem_free=True,
            defer_sync=True,
            num_tmem_alloc_cols=128,
            copy_before_pf=True,
            rolling_pf_b=True,
            hoist_epilog_wait=True,
            pf_desc_c_epi=True,
            fast_single_subtile=True,
            hoist_tma_c_part=True,
            peel_producer_init=True,
            clean_c_prologue=True,
            warp_scoped_ab_frag=True,
            split_pf_b_warp1=True,
            tma_pf_split=4,
            unroll_pf0=True,
            elide_acc_empty_wait=True,
            hoist_sf0_set=True,
            unroll_peel_init=True,
            c_alias_a=True,
            b_first=2,
            pdl_safe=True,
            store_wait=True,
            single_pf_a=True,
            warp0_pf_a=True,
            c_dtype=c_dtype_cutlass,
            has_alpha=has_alpha,
        )
    else:
        early_k = 5 if k <= 2048 else 0
        kern = Sm100Nvfp4GemmFast(
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
            num_ab_stage=8,
            num_c_stage=1,
            use_tma_store=True,
            subtile_width=64,
            early_tmem_free=False,
            pdl_mode=_PDL_MODE_1CTA,
            pdl_early_k=early_k,
            peel_first_tile=True,
            prefetch_b=(8 if 2048 < k < 16384 else 6),
            prefetch_a=(k > 256),
            coop_b_warp3=(k <= 2048),
            overlap_tmem_free=True,
            defer_sync=True,
            pre_alloc_tmem=False,
            num_tmem_alloc_cols=128,
            copy_before_pf=True,
            pf_init=0,
            rolling_pf_b=True,
            hoist_epilog_wait=True,
            pf_desc_c_epi=True,
            fast_single_subtile=True,
            hoist_tma_c_part=True,
            coop_b_mode=(2 if (k <= 2048 or k >= 16384) else 0),
            coop_deep_k=0,
            fast_single_pass_ring=True,
            peel_producer_init=True,
            elide_tail_release=True,
            clean_c_prologue=True,
            warp_scoped_ab_frag=True,
            coop_a_next_wave=(k >= 16384),
            coop_pf_init=(2 if k >= 16384 else 0),
            split_pf_b_warp1=True,
            tma_pf_split=4,
            direct_consumer_wait=False,
            unroll_pf0=True,
            deep_b_warps23=0,
            s2_wave3_b_pf=False,
            elide_acc_empty_wait=True,
            omit_1cta_dealloc_mbar=True,
            hoist_sf0_set=True,
            unroll_peel_init=True,
            unroll_single_pass_tma=True,
            split_tma_steady_pf=(2048 < k < 16384),
            unroll_tma_tail_pf=False,
            early_cta_sync=False,
            warp0_only_tma_c_part=True,
            peel_mma_tail_last=False,
            c_alias_a=True,
            b_first=2,
            pdl_safe=True,
            store_wait=True,
            l2_hint=False,
            single_pf_a=True,
            warp0_pf_a=True,
            parallel_mbar_init=True,
            elide_dead_mbar_init=True,
            fast_tmem_bar=(k >= 16384),
            unroll_coop_pf=(k >= 16384),
            hoist_mma_s2t_part=True,
            hoist_mma_full=(k >= 16384),
            alg_mbar_init=True,
            unroll_mma_tail=False,
            direct_single_pass_mma=True,
            early_pf_desc_c=(2048 < k < 16384),
            static_epilog_tmem_ptr=(k > 2048),
            alg_mbar_6warp=True,
            elide_1cta_mbar_fence=True,
            dom_tma_desc=0,
            c_dtype=c_dtype_cutlass,
            has_alpha=has_alpha,
        )

    a_p = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_p = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_p = make_ptr(c_dtype_cutlass, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_p = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_p = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    alpha_p = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4)
    s_p = cuda.CUstream(0)

    fn = cute.compile(
        kern.call_with_stream_alpha, a_p, b_p, sfa_p, sfb_p, c_p, alpha_p, s_p, shape
    )

    def _init_and_run_sgl(
        a_ptr_i, b_ptr_i, sfa_ptr_i, sfb_ptr_i, c_ptr_i, alpha_ptr_i, stream_i
    ):
        fn(
            make_ptr(
                cutlass.Float4E2M1FN, a_ptr_i, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float4E2M1FN, b_ptr_i, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                sfa_ptr_i,
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                sfb_ptr_i,
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                c_dtype_cutlass, c_ptr_i, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float32, alpha_ptr_i, cute.AddressSpace.gmem, assumed_align=4
            ),
            cuda.CUstream(stream_i),
        )
        ex = fn._default_executor
        packed_args = ex._tls.packed_args
        ptr_buf = (ctypes.c_uint64 * 7)()
        base_addr = ctypes.addressof(ptr_buf)
        for i in range(7):
            packed_args[i] = base_addr + i * 8
        capi_func = ex.capi_func
        _KEEPALIVE.append((kern, fn, ex, packed_args, ptr_buf))

        def _fast_call_sgl(
            a_ptr_i, b_ptr_i, sfa_ptr_i, sfb_ptr_i, c_ptr_i, alpha_ptr_i, stream_i
        ):
            ptr_buf[0] = a_ptr_i
            ptr_buf[1] = b_ptr_i
            ptr_buf[2] = sfa_ptr_i
            ptr_buf[3] = sfb_ptr_i
            ptr_buf[4] = c_ptr_i
            ptr_buf[5] = alpha_ptr_i
            ptr_buf[6] = stream_i
            capi_func(packed_args)

        _SGLANG_GEMM_CACHE[key] = _fast_call_sgl

    _SGLANG_GEMM_CACHE[key] = _init_and_run_sgl
    return _init_and_run_sgl


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def permute_nvfp4_scale(scale: torch.Tensor) -> torch.Tensor:
    squeeze_l = scale.ndim == 2
    if squeeze_l:
        scale = scale.unsqueeze(-1)
    if scale.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D scale tensor, got shape {tuple(scale.shape)}"
        )
    mn, sf_k, l = scale.shape
    rest_m = _ceil_div(mn, 128)
    rest_k = _ceil_div(sf_k, 4)
    padded = torch.zeros(
        (rest_m * 128, rest_k * 4, l), dtype=scale.dtype, device=scale.device
    )
    padded[:mn, :sf_k, :] = scale
    reshaped = padded.view(rest_m, 4, 32, rest_k, 4, l)
    phys = reshaped.permute(5, 0, 3, 2, 1, 4).contiguous()
    return phys.permute(3, 4, 1, 5, 2, 0)


def nvfp4_gemm_pdl_safe(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    alpha: Optional[torch.Tensor] = None,
    swizzled_scales: bool = False,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("nvfp4_gemm_pdl_safe requires CUDA tensors on the same device")
    is_2d = a.ndim == 2
    if is_2d:
        m, k_half = a.shape
        l = 1
    elif a.ndim == 3:
        m, k_half, l = a.shape
    else:
        raise ValueError(
            f"Expected 2D or 3D input tensor a, got shape {tuple(a.shape)}"
        )
    if b.ndim != a.ndim or b.shape[1] != k_half or (not is_2d and b.shape[2] != l):
        raise ValueError(
            f"Shape mismatch between a {tuple(a.shape)} and b {tuple(b.shape)}"
        )
    if a.stride(1) != 1 or b.stride(1) != 1:
        raise ValueError("Input tensors a and b must be K-major (stride(1) == 1)")
    n = b.shape[0]
    k = k_half * 2
    if (
        m <= 0
        or n <= 0
        or k <= 0
        or l <= 0
        or (m % 128 != 0)
        or (n % 64 != 0)
        or (k % 256 != 0)
    ):
        raise ValueError(
            f"Unsupported NVFP4 GEMM shape (M={m}, N={n}, K={k}, L={l}): "
            "M must be a multiple of 128, N a multiple of 64, and K a multiple of 256."
        )
    if out is not None:
        expected_out_shape = (m, n) if is_2d else (m, n, l)
        if out.shape != expected_out_shape or out.device != a.device:
            raise ValueError(
                f"Expected out shape {expected_out_shape} on {a.device}, got {tuple(out.shape)} on {out.device}"
            )
        if out.stride(1) != 1 or out.stride(0) < n:
            raise ValueError("Output tensor out must be row-major (stride(1) == 1)")
        if out_dtype is not None and out.dtype != out_dtype:
            raise ValueError(
                f"out.dtype ({out.dtype}) does not match out_dtype ({out_dtype})"
            )
        out_dtype = out.dtype
    elif out_dtype is None:
        out_dtype = torch.bfloat16
    if out_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"Unsupported output dtype {out_dtype}; expected float16 or bfloat16"
        )
    if not swizzled_scales:
        if sfa.ndim != 6:
            sfa = permute_nvfp4_scale(sfa)
        if sfb.ndim != 6:
            sfb = permute_nvfp4_scale(sfb)
    if out is None:
        if is_2d:
            out = torch.empty((m, n), dtype=out_dtype, device=a.device)
        else:
            out = torch.empty((l, m, n), dtype=out_dtype, device=a.device).permute(
                1, 2, 0
            )
    has_alpha = alpha is not None
    alpha_ptr = alpha.data_ptr() if has_alpha else 0
    stream_i = torch.cuda.current_stream(a.device).cuda_stream
    runner = _get_sglang_gemm_runner((m, n, k, l), out_dtype, has_alpha=has_alpha)
    runner(
        a.data_ptr(),
        b.data_ptr(),
        sfa.data_ptr(),
        sfb.data_ptr(),
        out.data_ptr(),
        alpha_ptr,
        stream_i,
    )
    return out


def try_cutedsl_nvfp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: Optional[torch.Tensor],
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> Optional[torch.Tensor]:
    """Run SM100 CuTe DSL PDL-safe NVFP4 GEMM when enabled, or return None for fallback."""
    if (
        os.environ.get("SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM", "1") != "1"
        and os.environ.get("SGLANG_USE_CUTEDSL_NVFP4_GEMM", "0") != "1"
    ):
        return None
    if (
        input.device.type != "cuda"
        or input.ndim != 2
        or input_sf is None
        or out_dtype not in (torch.float16, torch.bfloat16)
    ):
        return None
    dev_idx = input.device.index or 0
    is_sm100 = _SM100_DEV_CACHE.get(dev_idx)
    if is_sm100 is None:
        is_sm100 = torch.cuda.get_device_capability(input.device)[0] == 10
        _SM100_DEV_CACHE[dev_idx] = is_sm100
    if not is_sm100:
        return None

    m, k_half = input.shape
    k = k_half * 2
    n = int(out_features)
    min_m = int(os.environ.get("SGLANG_CUTEDSL_MIN_M", "128"))
    if m < min_m or n <= 0 or k <= 0 or (n % 64 != 0) or (k % 256 != 0):
        return None
    if input.stride(1) != 1:
        return None

    if weight.ndim != 2:
        return None
    if weight.shape == (k_half, n) and weight.stride(0) == 1:
        b_kmajor = weight.T
    elif weight.shape == (n, k_half) and weight.stride(1) == 1:
        b_kmajor = weight
    else:
        return None

    if weight_sf.ndim != 2:
        return None
    if weight_sf.stride(0) == 1 and weight_sf.stride(1) != 1:
        sfb_kmajor = weight_sf.T
    elif weight_sf.stride(1) == 1:
        sfb_kmajor = weight_sf
    else:
        return None

    if input_sf.ndim != 2 or input_sf.stride(1) != 1:
        return None
    pad_m = _ceil_div(m, 128) * 128
    if input_sf.shape[0] != pad_m:
        return None

    if pad_m != m:
        a_padded = torch.nn.functional.pad(input, (0, 0, 0, pad_m - m))
        out_padded = nvfp4_gemm_pdl_safe(
            a_padded,
            b_kmajor,
            input_sf,
            sfb_kmajor,
            alpha=alpha,
            swizzled_scales=True,
            out_dtype=out_dtype,
        )
        return out_padded[:m]

    return nvfp4_gemm_pdl_safe(
        input,
        b_kmajor,
        input_sf,
        sfb_kmajor,
        alpha=alpha,
        swizzled_scales=True,
        out_dtype=out_dtype,
    )


__all__ = [
    "Sm100Nvfp4GemmFast",
    "Sm100Nvfp4GemmPersistent2CTA",
    "nvfp4_gemm_pdl_safe",
    "permute_nvfp4_scale",
    "try_cutedsl_nvfp4_gemm",
    "_ceil_div",
]
