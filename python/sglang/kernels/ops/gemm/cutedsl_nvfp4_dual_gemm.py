import ctypes
from typing import Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def _rcp_seed_neg(nd, *, loc=None, ip=None):
    """Seed for 1/(-nd) with nd < 0: bits = 0xFEF311C3 - bits(nd) (mod 2^32).

    Equals the classic 0x7EF311C3 - bits(-nd) reciprocal seed (max rel err 10.1%).
    One integer subtract; no MUFU.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(nd).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b32 sd;\nmov.b32 sd, $1;\nsub.u32 sd, 0xFEF311C3, sd;\nmov.b32 $0, sd;\n}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _swz_sc_addr(sc_ptr, tidx, k_off, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                sc_ptr.llvm_ptr,
                cutlass.Int32(tidx).ir_value(loc=loc, ip=ip),
                cutlass.Int32(k_off).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .b32 b, o, s;\nmov.b32 b, $1;\nmad.lo.u32 o, $2, 128, b;\nadd.u32 o, o, $3;\nshr.u32 s, o, 3;\nand.b32 s, s, 112;\nxor.b32 $0, o, s;\n}",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _swz_sc_addr_fast(sc_ptr, tidx, k_off, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                sc_ptr.llvm_ptr,
                cutlass.Int32(tidx).ir_value(loc=loc, ip=ip),
                cutlass.Int32(k_off).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .b32 b, o, s;\nmov.b32 b, $1;\nmad.lo.u32 o, $2, 128, b;\nshl.b32 s, $2, 4;\nand.b32 s, s, 112;\nor.b32 o, o, s;\nxor.b32 $0, o, $3;\n}",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_st_f16x8(
    addr, o0, o1, o2, o3, o4, o5, o6, o7, sub_idx: int = 0, *, loc=None, ip=None
):
    disp = "+16384" if sub_idx == 1 else ""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
            cutlass.Float32(o0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(o1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(o2).ir_value(loc=loc, ip=ip),
            cutlass.Float32(o3).ir_value(loc=loc, ip=ip),
            cutlass.Float32(o4).ir_value(loc=loc, ip=ip),
            cutlass.Float32(o5).ir_value(loc=loc, ip=ip),
            cutlass.Float32(o6).ir_value(loc=loc, ip=ip),
            cutlass.Float32(o7).ir_value(loc=loc, ip=ip),
        ],
        f"{{\n.reg .b32 h0, h1, h2, h3;\ncvt.rn.f16x2.f32 h0, $2, $1;\ncvt.rn.f16x2.f32 h1, $4, $3;\ncvt.rn.f16x2.f32 h2, $6, $5;\ncvt.rn.f16x2.f32 h3, $8, $7;\nst.shared.v4.b32 [$0{disp}], {{h0, h1, h2, h3}};\n}}",
        "r,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _arrive_peer_dealloc_warp(mbar_ptr, peer_rank, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            mbar_ptr.llvm_ptr,
            cutlass.Int32(peer_rank).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .pred p;\n.reg .b32 r_mbar, r_peer;\nelect.sync _|p, -1;\n@p mov.b32 r_mbar, $0;\n@p mapa.shared::cluster.u32 r_peer, r_mbar, $1;\n@p mbarrier.arrive.shared::cluster.b64 _, [r_peer], 1;\n}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _arrive_peer_dealloc_warp_relaxed(mbar_ptr, peer_rank, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            mbar_ptr.llvm_ptr,
            cutlass.Int32(peer_rank).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .pred p;\n.reg .b32 r_mbar, r_peer;\nelect.sync _|p, -1;\n@p mov.b32 r_mbar, $0;\n@p mapa.shared::cluster.u32 r_peer, r_mbar, $1;\n@p mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [r_peer], 1;\n}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _arrive_peer_dealloc_warp_wait_ld(mbar_ptr, peer_rank, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            mbar_ptr.llvm_ptr,
            cutlass.Int32(peer_rank).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .pred p;\n.reg .b32 r_mbar, r_peer;\ntcgen05.wait::ld.sync.aligned;\nelect.sync _|p, -1;\n@p mov.b32 r_mbar, $0;\n@p mapa.shared::cluster.u32 r_peer, r_mbar, $1;\n@p mbarrier.arrive.shared::cluster.b64 _, [r_peer], 1;\n}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class Sm100Nvfp4DualGemmFast:
    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int] = (256, 128),
        cluster_shape_mn: Tuple[int, int] = (2, 1),
        num_ab_stage: int = 5,
        num_c_stage: int = 2,
        c_dtype: type[cutlass.Numeric] = cutlass.Float16,
        use_pdl: bool = True,
        fuse_b: bool = False,
        k_tile: int = 256,
        hint_b: bool = False,
        hint_b_last_row: int = -1,
        hint_c: bool = False,
        pf_dist: int = 0,
        pf_prolog: bool = False,
        pf_a: bool = False,
        pf_head: int = 0,
        pf_row0_only: bool = False,
        pf_head_warp2: bool = False,
        cta_desc_pf: bool = False,
        early_pdl_launch: bool = False,
        tma_order: int = 0,
        mma_leader_only: bool = False,
        peel_acc_init: bool = False,
        db_tmem_sf: bool = False,
        early_relinquish: bool = False,
        pre_wait_pdl: bool = False,
        pipe_subtile_store: bool = False,
        early_tmem_free: bool = False,
        defer_pipeline_sync: bool = False,
        first_inst_pdl_launch: bool = False,
        early_tmem_alloc: bool = False,
        pf_head_mode: int = 0,
        pf_a_bidy0: bool = False,
        async_epilog_bar: bool = False,
        preload_subtile1_t2r: bool = False,
        swiglu_ilp: bool = False,
        no_fmin_clamp: bool = False,
        warp_spec_parts: bool = False,
        split_tmem_dealloc: bool = False,
        hybrid_rcp_mod: int = 0,
        direct_acc_mbar: bool = False,
        const_mma_fields: bool = False,
        swiglu_group4: bool = False,
        pre_bar_pdl_wait: bool = False,
        pf_a_bidy0_ml: bool = False,
        peel_tma_init_try: bool = False,
        leader_tmem_bar: bool = False,
        pf_a_warp0: bool = False,
        s2t_interleave_mode: int = 0,
        t2r_rep_x32: bool = False,
        inline_swz_r2s: bool = False,
        pf_a_all_m: bool = False,
        tma_cta_split_m256: bool = False,
        single_mbar_init: bool = False,
        const_tmem_bar: bool = False,
        pre_fence_dealloc_arv: bool = False,
        prune_dead_smem: bool = False,
        pdl_wait_mode: int = 0,
        unified_mma_loop: bool = False,
        peel_mma_init_try: bool = False,
        tma_cta_split_all_m: bool = False,
        interleave_m_row_pf: bool = False,
        split_bidy_pf_a_mode: int = 0,
        single_swz_base: bool = False,
        coalesce_warp0_tmem: bool = False,
        split_mrow_ml_pf_mode: int = 0,
        hoist_cta_prolog_b_pf: bool = False,
        pf_a_prolog_mode: int = 0,
        hybrid_rcp_offset: int = 0,
        early_peer_dealloc_mode: int = 0,
        sub1_bar_mode: int = 0,
        hint_a: bool = False,
        rcp_pair_mode: int = 0,
        t2r_interleave_mode: int = 0,
        const_prolog_pf: bool = False,
        defer_mma_tmem_bar: bool = False,
        split_c_store_warps: bool = False,
        alg_mbar_mode: int = 0,
        const_cluster_sync: bool = False,
        tma_loop_split_mode: int = 0,
        warp_scoped_ab_frag: bool = False,
        mma_tmem_bar_after_k0_wait: bool = False,
        elide_mma_tail_release: int = 0,
        clean_c_prologue: bool = False,
        warp1_only_tma_c_part: bool = False,
        relaxed_peer_dealloc: int = 0,
        epilog_fence_mode: int = 0,
        fused_mma_tail_mode: int = 0,
        static_cmask: bool = False,
        uniform_coord: int = 0,
        sub1_ldtm_mode: int = 0,
        nr_steps: int = 2,
        store_drain: bool = True,
        pdl_safe: bool = False,
        has_alpha: bool = False,
    ):
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.c_dtype = c_dtype
        self.sf_vec_size = 16
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        assert k_tile in (128, 256)
        self.k_blocks = k_tile // 64
        self.mma_tiler = (mma_tiler_mn[0], mma_tiler_mn[1], k_tile)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * 6

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=64
            if const_tmem_bar
            else (32 * len((self.mma_warp_id, *self.epilog_warp_id))),
        )
        self.epilog_sub1_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.num_tmem_alloc_cols = 512
        self.num_ab_stage = num_ab_stage
        self.num_acc_stage = 1
        self.num_c_stage = num_c_stage
        self.use_pdl = use_pdl
        self.fuse_b = bool(fuse_b) and tuple(mma_tiler_mn) == (256, 128)
        self.hint_b = hint_b
        self.hint_b_last_row = hint_b_last_row
        self.hint_c = hint_c
        self.pf2 = pf_dist
        self.pf_prolog = pf_prolog
        self.pf_a = pf_a
        self.pf_head = pf_head
        self.pf_row0_only = pf_row0_only
        self.pf_head_warp2 = pf_head_warp2
        self.cta_desc_pf = cta_desc_pf
        self.early_pdl_launch = early_pdl_launch
        self.tma_order = tma_order
        self.mma_leader_only = mma_leader_only
        self.peel_acc_init = peel_acc_init
        self.db_tmem_sf = db_tmem_sf
        self.early_relinquish = early_relinquish
        self.pre_wait_pdl = pre_wait_pdl
        self.pipe_subtile_store = pipe_subtile_store
        self.early_tmem_free = early_tmem_free
        self.defer_pipeline_sync = defer_pipeline_sync
        self.first_inst_pdl_launch = first_inst_pdl_launch
        self.early_tmem_alloc = early_tmem_alloc
        self.pf_head_mode = pf_head_mode
        self.pf_a_bidy0 = pf_a_bidy0
        self.async_epilog_bar = async_epilog_bar
        self.preload_subtile1_t2r = preload_subtile1_t2r
        self.swiglu_ilp = swiglu_ilp
        self.no_fmin_clamp = no_fmin_clamp
        self.warp_spec_parts = warp_spec_parts
        self.split_tmem_dealloc = split_tmem_dealloc
        self.hybrid_rcp_mod = hybrid_rcp_mod
        self.direct_acc_mbar = direct_acc_mbar
        self.const_mma_fields = const_mma_fields
        self.swiglu_group4 = swiglu_group4
        self.pre_bar_pdl_wait = pre_bar_pdl_wait
        self.pf_a_bidy0_ml = pf_a_bidy0_ml
        self.peel_tma_init_try = peel_tma_init_try
        self.leader_tmem_bar = leader_tmem_bar
        self.pf_a_warp0 = pf_a_warp0
        self.s2t_interleave_mode = s2t_interleave_mode
        self.t2r_rep_x32 = t2r_rep_x32
        self.inline_swz_r2s = bool(inline_swz_r2s) and (c_dtype == cutlass.Float16)
        self.pf_a_all_m = pf_a_all_m
        self.tma_cta_split_m256 = tma_cta_split_m256
        self.single_mbar_init = single_mbar_init
        self.const_tmem_bar = const_tmem_bar
        self.pre_fence_dealloc_arv = pre_fence_dealloc_arv
        self.prune_dead_smem = prune_dead_smem
        self.pdl_wait_mode = pdl_wait_mode
        self.unified_mma_loop = unified_mma_loop
        self.peel_mma_init_try = peel_mma_init_try
        self.tma_cta_split_all_m = tma_cta_split_all_m
        self.interleave_m_row_pf = interleave_m_row_pf
        self.split_bidy_pf_a_mode = split_bidy_pf_a_mode
        self.single_swz_base = single_swz_base
        self.coalesce_warp0_tmem = coalesce_warp0_tmem
        self.split_mrow_ml_pf_mode = split_mrow_ml_pf_mode
        self.hoist_cta_prolog_b_pf = hoist_cta_prolog_b_pf
        self.pf_a_prolog_mode = pf_a_prolog_mode
        self.hybrid_rcp_offset = hybrid_rcp_offset
        self.early_peer_dealloc_mode = early_peer_dealloc_mode
        self.sub1_bar_mode = sub1_bar_mode
        self.hint_a = hint_a
        self.rcp_pair_mode = rcp_pair_mode
        self.t2r_interleave_mode = t2r_interleave_mode
        self.const_prolog_pf = const_prolog_pf
        self.defer_mma_tmem_bar = defer_mma_tmem_bar
        self.split_c_store_warps = split_c_store_warps
        self.alg_mbar_mode = alg_mbar_mode
        self.const_cluster_sync = const_cluster_sync
        self.tma_loop_split_mode = tma_loop_split_mode
        self.warp_scoped_ab_frag = warp_scoped_ab_frag
        self.mma_tmem_bar_after_k0_wait = mma_tmem_bar_after_k0_wait
        self.elide_mma_tail_release = elide_mma_tail_release
        self.clean_c_prologue = clean_c_prologue
        self.warp1_only_tma_c_part = warp1_only_tma_c_part
        self.relaxed_peer_dealloc = relaxed_peer_dealloc
        self.epilog_fence_mode = epilog_fence_mode
        self.fused_mma_tail_mode = fused_mma_tail_mode
        self.static_cmask = static_cmask
        self.uniform_coord = uniform_coord
        self.sub1_ldtm_mode = sub1_ldtm_mode
        self.nr_steps = nr_steps
        self.store_drain = store_drain
        self.pdl_safe = pdl_safe
        self.has_alpha = has_alpha

    def _make_tiled_mmas(self):
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
        if self.fuse_b:
            tiled_mma_f = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.ab_dtype,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                self.sf_dtype,
                self.sf_vec_size,
                tcgen05.CtaGroup.TWO,
                (self.mma_inst_shape_mn[0], 2 * self.mma_inst_shape_mn[1]),
            )
        else:
            tiled_mma_f = tiled_mma
        return tiled_mma, tiled_mma_sfb, tiled_mma_f

    def _setup_attributes(self):
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )
        tiled_mma, tiled_mma_sfb, tiled_mma_f = self._make_tiled_mmas()
        mma_inst_shape_k = 64
        mma_inst_tile_k = self.k_blocks
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
        self.epi_tile = (128, min(64, self.cta_tile_shape_mnk[1]))

        if self.fuse_b:
            self.mma_tiler_f = (
                self.mma_inst_shape_mn[0],
                2 * self.mma_inst_shape_mn[1],
                mma_inst_shape_k * mma_inst_tile_k,
            )
        else:
            self.mma_tiler_f = self.mma_tiler

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_f,
            self.mma_tiler_f,
            self.ab_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_f,
            self.mma_tiler_f,
            self.ab_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_f,
            self.mma_tiler_f,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_f,
            self.mma_tiler_f,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_half_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )
        return tiled_mma, tiled_mma_sfb, tiled_mma_f

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b1_ptr: cute.Pointer,
        b2_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb1_ptr: cute.Pointer,
        sfb2_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: cutlass.Constexpr[tuple],
    ):
        m, n, k, l = problem_size
        tiled_mma, tiled_mma_sfb, tiled_mma_f = self._setup_attributes()

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b1_tensor = cute.make_tensor(
            b1_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        b2_tensor = cute.make_tensor(
            b2_ptr,
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
            b1_tensor.shape, self.sf_vec_size
        )
        sfb1_tensor = cute.make_tensor(sfb1_ptr, sfb_layout)
        sfb2_tensor = cute.make_tensor(sfb2_ptr, sfb_layout)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler_f,
            tiled_mma_f,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b1_tensor,
            b_smem_layout,
            self.mma_tiler_f,
            tiled_mma_f,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b2_tensor,
            b_smem_layout,
            self.mma_tiler_f,
            tiled_mma_f,
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
            self.mma_tiler_f,
            tiled_mma_f,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_half_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb1_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb2_tensor,
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
        if cutlass.const_expr(self.fuse_b):
            self.num_tma_load_bytes = (
                a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size * 2
            ) * atom_thr_size
        else:
            self.num_tma_load_bytes = (
                a_copy_size + b_copy_size * 2 + sfa_copy_size + sfb_copy_size * 2
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
        b1_cosize = cute.cosize(self.b_smem_layout_staged.outer)
        b2_cosize = (0 if self.prune_dead_smem else 256) if self.fuse_b else b1_cosize
        sfb1_cosize = cute.cosize(self.sfb_smem_layout_staged)
        sfb2_cosize = (
            (0 if self.prune_dead_smem else 128) if self.fuse_b else sfb1_cosize
        )

        @cute.struct
        class SharedStorage:
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
            sB1: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b1_cosize],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b2_cosize],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb1_cosize],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb2_cosize],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorageCompact:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
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
            sB1: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b1_cosize],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b2_cosize],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb1_cosize],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb2_cosize],
                self.buffer_align_bytes,
            ]

        self.shared_storage = (
            SharedStorageCompact
            if (self.alg_mbar_mode == 3 and self.direct_acc_mbar)
            else SharedStorage
        )

        k_mma = None if self.const_mma_fields else tiled_mma
        k_mma_sfb = None if self.const_mma_fields else tiled_mma_sfb
        k_mma_f = None if self.const_mma_fields else tiled_mma_f

        self.kernel(
            k_mma,
            k_mma_sfb,
            k_mma_f,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b1,
            tma_tensor_b1,
            tma_atom_b2,
            tma_tensor_b2,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb1,
            tma_tensor_sfb1,
            tma_atom_sfb2,
            tma_tensor_sfb2,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.sfb_half_smem_layout_staged,
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
    def call_with_stream(
        self,
        a_ptr: cute.Pointer,
        b1_ptr: cute.Pointer,
        b2_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb1_ptr: cute.Pointer,
        sfb2_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        stream: cuda.CUstream,
        problem_size: cutlass.Constexpr[tuple],
    ):
        m, n, k, l = problem_size
        tiled_mma, tiled_mma_sfb, tiled_mma_f = self._setup_attributes()

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b1_tensor = cute.make_tensor(
            b1_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        b2_tensor = cute.make_tensor(
            b2_ptr,
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
            b1_tensor.shape, self.sf_vec_size
        )
        sfb1_tensor = cute.make_tensor(sfb1_ptr, sfb_layout)
        sfb2_tensor = cute.make_tensor(sfb2_ptr, sfb_layout)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler_f,
            tiled_mma_f,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b1_tensor,
            b_smem_layout,
            self.mma_tiler_f,
            tiled_mma_f,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b2_tensor,
            b_smem_layout,
            self.mma_tiler_f,
            tiled_mma_f,
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
            self.mma_tiler_f,
            tiled_mma_f,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_half_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb1_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb2_tensor,
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
        if cutlass.const_expr(self.fuse_b):
            self.num_tma_load_bytes = (
                a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size * 2
            ) * atom_thr_size
        else:
            self.num_tma_load_bytes = (
                a_copy_size + b_copy_size * 2 + sfa_copy_size + sfb_copy_size * 2
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
        b1_cosize = cute.cosize(self.b_smem_layout_staged.outer)
        b2_cosize = (0 if self.prune_dead_smem else 256) if self.fuse_b else b1_cosize
        sfb1_cosize = cute.cosize(self.sfb_smem_layout_staged)
        sfb2_cosize = (
            (0 if self.prune_dead_smem else 128) if self.fuse_b else sfb1_cosize
        )

        @cute.struct
        class SharedStorage:
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
            sB1: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b1_cosize],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b2_cosize],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb1_cosize],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb2_cosize],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorageCompact:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
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
            sB1: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b1_cosize],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b2_cosize],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb1_cosize],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb2_cosize],
                self.buffer_align_bytes,
            ]

        self.shared_storage = (
            SharedStorageCompact
            if (self.alg_mbar_mode == 3 and self.direct_acc_mbar)
            else SharedStorage
        )

        k_mma = None if self.const_mma_fields else tiled_mma
        k_mma_sfb = None if self.const_mma_fields else tiled_mma_sfb
        k_mma_f = None if self.const_mma_fields else tiled_mma_f

        self.kernel(
            k_mma,
            k_mma_sfb,
            k_mma_f,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b1,
            tma_tensor_b1,
            tma_atom_b2,
            tma_tensor_b2,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb1,
            tma_tensor_sfb1,
            tma_atom_sfb2,
            tma_tensor_sfb2,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.sfb_half_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            None,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )
        return

    @cute.jit
    def call_with_stream_alpha(
        self,
        a_ptr: cute.Pointer,
        b1_ptr: cute.Pointer,
        b2_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb1_ptr: cute.Pointer,
        sfb2_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        stream: cuda.CUstream,
        problem_size: cutlass.Constexpr[tuple],
    ):
        m, n, k, l = problem_size
        tiled_mma, tiled_mma_sfb, tiled_mma_f = self._setup_attributes()

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b1_tensor = cute.make_tensor(
            b1_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        b2_tensor = cute.make_tensor(
            b2_ptr,
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
            b1_tensor.shape, self.sf_vec_size
        )
        sfb1_tensor = cute.make_tensor(sfb1_ptr, sfb_layout)
        sfb2_tensor = cute.make_tensor(sfb2_ptr, sfb_layout)
        alpha_tensor = cute.make_tensor(
            alpha_ptr,
            cute.make_layout((1,), stride=(1,)),
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler_f,
            tiled_mma_f,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b1_tensor,
            b_smem_layout,
            self.mma_tiler_f,
            tiled_mma_f,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b2_tensor,
            b_smem_layout,
            self.mma_tiler_f,
            tiled_mma_f,
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
            self.mma_tiler_f,
            tiled_mma_f,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_half_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb1_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb2_tensor,
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
        if cutlass.const_expr(self.fuse_b):
            self.num_tma_load_bytes = (
                a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size * 2
            ) * atom_thr_size
        else:
            self.num_tma_load_bytes = (
                a_copy_size + b_copy_size * 2 + sfa_copy_size + sfb_copy_size * 2
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
        b1_cosize = cute.cosize(self.b_smem_layout_staged.outer)
        b2_cosize = (0 if self.prune_dead_smem else 256) if self.fuse_b else b1_cosize
        sfb1_cosize = cute.cosize(self.sfb_smem_layout_staged)
        sfb2_cosize = (
            (0 if self.prune_dead_smem else 128) if self.fuse_b else sfb1_cosize
        )

        @cute.struct
        class SharedStorage:
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
            sB1: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b1_cosize],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b2_cosize],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb1_cosize],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb2_cosize],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorageCompact:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
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
            sB1: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b1_cosize],
                self.buffer_align_bytes,
            ]
            sB2: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, b2_cosize],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb1_cosize],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb2_cosize],
                self.buffer_align_bytes,
            ]

        self.shared_storage = (
            SharedStorageCompact
            if (self.alg_mbar_mode == 3 and self.direct_acc_mbar)
            else SharedStorage
        )

        k_mma = None if self.const_mma_fields else tiled_mma
        k_mma_sfb = None if self.const_mma_fields else tiled_mma_sfb
        k_mma_f = None if self.const_mma_fields else tiled_mma_f

        self.kernel(
            k_mma,
            k_mma_sfb,
            k_mma_f,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b1,
            tma_tensor_b1,
            tma_atom_b2,
            tma_tensor_b2,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb1,
            tma_tensor_sfb1,
            tma_atom_sfb2,
            tma_tensor_sfb2,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.sfb_half_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            alpha_tensor,
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
        tiled_mma: Optional[cute.TiledMma],
        tiled_mma_sfb: Optional[cute.TiledMma],
        tiled_mma_f: Optional[cute.TiledMma],
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b1: cute.CopyAtom,
        mB1_nkl: cute.Tensor,
        tma_atom_b2: cute.CopyAtom,
        mB2_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb1: cute.CopyAtom,
        mSFB1_nkl: cute.Tensor,
        tma_atom_sfb2: cute.CopyAtom,
        mSFB2_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        sfb_half_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        mAlpha: Optional[cute.Tensor],
    ):
        if cutlass.const_expr(self.const_mma_fields):
            tiled_mma, tiled_mma_sfb, tiled_mma_f = self._make_tiled_mmas()

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        if cutlass.const_expr(
            self.uniform_coord == 1
            and use_2cta_instrs
            and self.cluster_shape_mn == (2, 1)
        ):
            bidx = cute.arch.make_warp_uniform(bidx)
            bidy = cute.arch.make_warp_uniform(bidy)
            mma_tile_coord_v = bidx & 1
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        if warp_idx == self.tma_warp_id:
            if cutlass.const_expr(
                self.first_inst_pdl_launch
                and self.early_pdl_launch
                and self.use_pdl
                and (not self.pdl_safe)
            ):
                cute.arch.griddepcontrol_launch_dependents()
            cpasync.prefetch_descriptor(tma_atom_a)
            if cutlass.const_expr(self.cta_desc_pf and self.fuse_b):
                if mma_tile_coord_v == 0:
                    cpasync.prefetch_descriptor(tma_atom_b1)
                else:
                    cpasync.prefetch_descriptor(tma_atom_b2)
            else:
                cpasync.prefetch_descriptor(tma_atom_b1)
                cpasync.prefetch_descriptor(tma_atom_b2)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb1)
            cpasync.prefetch_descriptor(tma_atom_sfb2)
            if cutlass.const_expr(
                (not self.first_inst_pdl_launch)
                and self.early_pdl_launch
                and self.use_pdl
                and (not self.pdl_safe)
            ):
                cute.arch.griddepcontrol_launch_dependents()
        if warp_idx == self.epilog_warp_id[1]:
            cpasync.prefetch_descriptor(tma_atom_c)
        if cutlass.const_expr(self.split_c_store_warps):
            if warp_idx == self.epilog_warp_id[2]:
                cpasync.prefetch_descriptor(tma_atom_c)

        if cutlass.const_expr(
            self.uniform_coord == 1
            and use_2cta_instrs
            and self.cluster_shape_mn == (2, 1)
        ):
            cta_rank_in_cluster = mma_tile_coord_v
            block_in_cluster_coord_vmnk = (cta_rank_in_cluster, 0, 0, 0)
            block_in_cluster_coord_sfb_vmnk = (0, cta_rank_in_cluster, 0, 0)
        elif cutlass.const_expr(
            self.uniform_coord == 2
            and use_2cta_instrs
            and self.cluster_shape_mn == (2, 1)
        ):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            block_in_cluster_coord_vmnk = (cta_rank_in_cluster, 0, 0, 0)
            block_in_cluster_coord_sfb_vmnk = (0, cta_rank_in_cluster, 0, 0)
        else:
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

        if cutlass.const_expr(
            self.uniform_coord == 1
            and use_2cta_instrs
            and self.cluster_shape_mn == (2, 1)
        ):
            mma_tile_coord_mnl = (
                0 if cute.size(mC_mnl, mode=[0]) <= 256 else (bidx >> 1),
                bidy,
                0 if cute.size(mC_mnl, mode=[2]) == 1 else bidz,
            )
        else:
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
        ab_base_ptr = storage.ab_full_mbar_ptr.data_ptr()
        acc_mbar_ptr = storage.acc_full_mbar_ptr.data_ptr()
        if cutlass.const_expr(self.single_mbar_init and self.direct_acc_mbar):
            if cutlass.const_expr(self.alg_mbar_mode == 1 and self.num_ab_stage == 5):
                if warp_idx < 5:
                    with cute.arch.elect_one():
                        w_base = ab_base_ptr + warp_idx * 2
                        for mb_i in cutlass.range_constexpr(2):
                            cute.arch.mbarrier_init(w_base + mb_i, 1)
                        if warp_idx == 0:
                            cute.arch.mbarrier_init(acc_mbar_ptr, 1)
                            if cutlass.const_expr(use_2cta_instrs):
                                if cutlass.const_expr(self.pre_fence_dealloc_arv):
                                    cute.arch.mbarrier_init(
                                        storage.tmem_dealloc_mbar_ptr, 4
                                    )
                                else:
                                    cute.arch.mbarrier_init(
                                        storage.tmem_dealloc_mbar_ptr, 32
                                    )
            elif cutlass.const_expr(self.alg_mbar_mode == 2 and self.num_ab_stage == 5):
                if warp_idx < 5:
                    with cute.arch.elect_one():
                        w_base = ab_base_ptr + warp_idx * 2
                        for mb_i in cutlass.range_constexpr(2):
                            cute.arch.mbarrier_init(w_base + mb_i, 1)
                        if warp_idx == 4:
                            cute.arch.mbarrier_init(acc_mbar_ptr, 1)
                            if cutlass.const_expr(use_2cta_instrs):
                                if cutlass.const_expr(self.pre_fence_dealloc_arv):
                                    cute.arch.mbarrier_init(
                                        storage.tmem_dealloc_mbar_ptr, 4
                                    )
                                else:
                                    cute.arch.mbarrier_init(
                                        storage.tmem_dealloc_mbar_ptr, 32
                                    )
            elif cutlass.const_expr(self.alg_mbar_mode == 3 and self.num_ab_stage == 5):
                if warp_idx < 5:
                    with cute.arch.elect_one():
                        w_base = ab_base_ptr + warp_idx * 2
                        for mb_i in cutlass.range_constexpr(2):
                            cute.arch.mbarrier_init(w_base + mb_i, 1)
                else:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(acc_mbar_ptr, 1)
                        if cutlass.const_expr(use_2cta_instrs):
                            if cutlass.const_expr(self.pre_fence_dealloc_arv):
                                cute.arch.mbarrier_init(
                                    storage.tmem_dealloc_mbar_ptr, 4
                                )
                            else:
                                cute.arch.mbarrier_init(
                                    storage.tmem_dealloc_mbar_ptr, 32
                                )
            else:
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        for mb_i in cutlass.range_constexpr(2 * self.num_ab_stage):
                            cute.arch.mbarrier_init(ab_base_ptr + mb_i, 1)
                        cute.arch.mbarrier_init(acc_mbar_ptr, 1)
                        if cutlass.const_expr(use_2cta_instrs):
                            if cutlass.const_expr(self.pre_fence_dealloc_arv):
                                cute.arch.mbarrier_init(
                                    storage.tmem_dealloc_mbar_ptr, 4
                                )
                            else:
                                cute.arch.mbarrier_init(
                                    storage.tmem_dealloc_mbar_ptr, 32
                                )
            cute.arch.mbarrier_init_fence()
            _orig_mb_init = pipeline.MbarrierArray.mbarrier_init
            pipeline.MbarrierArray.mbarrier_init = lambda *a, **k: None
            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=ab_base_ptr,
                num_stages=self.num_ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            pipeline.MbarrierArray.mbarrier_init = _orig_mb_init
            if cutlass.const_expr(
                self.static_cmask
                and use_2cta_instrs
                and self.cluster_shape_mn == (2, 1)
            ):
                object.__setattr__(ab_pipeline, "consumer_mask", cutlass.Int16(3))
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=self.tmem_alloc_barrier,
                allocator_warp_id=self.epilog_warp_id[0],
                is_two_cta=use_2cta_instrs,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
                initialize_mbarrier=False,
            )
        else:
            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=ab_base_ptr,
                num_stages=self.num_ab_stage,
                producer_group=ab_pipeline_producer_group,
                consumer_group=ab_pipeline_consumer_group,
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=self.defer_pipeline_sync,
            )
            if cutlass.const_expr(
                self.static_cmask
                and use_2cta_instrs
                and self.cluster_shape_mn == (2, 1)
            ):
                object.__setattr__(ab_pipeline, "consumer_mask", cutlass.Int16(3))

            if cutlass.const_expr(self.direct_acc_mbar):
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_init(acc_mbar_ptr, 1)
                        if cutlass.const_expr(
                            self.pre_fence_dealloc_arv and use_2cta_instrs
                        ):
                            cute.arch.mbarrier_init(storage.tmem_dealloc_mbar_ptr, 4)
                if cutlass.const_expr(self.pre_fence_dealloc_arv and use_2cta_instrs):
                    cute.arch.mbarrier_init_fence()
            else:
                acc_pipeline_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread
                )
                num_acc_consumer_threads = len(self.epilog_warp_id) * (
                    2 if use_2cta_instrs else 1
                )
                acc_pipeline_consumer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, num_acc_consumer_threads
                )
                acc_pipeline = pipeline.PipelineUmmaAsync.create(
                    barrier_storage=acc_mbar_ptr,
                    num_stages=self.num_acc_stage,
                    producer_group=acc_pipeline_producer_group,
                    consumer_group=acc_pipeline_consumer_group,
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=self.defer_pipeline_sync,
                )

            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=self.tmem_alloc_barrier,
                allocator_warp_id=self.epilog_warp_id[0],
                is_two_cta=use_2cta_instrs,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
                initialize_mbarrier=(not self.pre_fence_dealloc_arv),
            )
        if cutlass.const_expr(self.defer_pipeline_sync and (not use_2cta_instrs)):
            cute.arch.mbarrier_init_fence()

        if cutlass.const_expr(self.const_cluster_sync):
            if cutlass.const_expr(cute.size(self.cluster_shape_mn) > 1):
                cute.arch.cluster_arrive_relaxed()
        else:
            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_arrive_relaxed()

        if cutlass.const_expr(self.early_tmem_alloc):
            tmem.allocate(self.num_tmem_alloc_cols)
            if cutlass.const_expr(self.early_relinquish):
                tmem.relinquish_alloc_permit()

        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB1 = storage.sB1.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB1 = storage.sSFB1.get_tensor(sfb_smem_layout_staged)
        if cutlass.const_expr(self.fuse_b):
            sB2 = sB1
            h_l = sfb_half_smem_layout_staged
            f_l = sfb_smem_layout_staged
            sfb_half_view = cute.make_layout(
                h_l.shape,
                stride=(h_l.stride[0], h_l.stride[1], h_l.stride[2], f_l.stride[3]),
            )
            sfb2_offset = f_l.stride[0][0][1]
            sSFB1_ld = cute.make_tensor(sSFB1.iterator, sfb_half_view)
            sSFB2_ld = cute.make_tensor(sSFB1.iterator + sfb2_offset, sfb_half_view)
        else:
            sB2 = storage.sB2.get_tensor(
                b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
            )
            sSFB2 = storage.sSFB2.get_tensor(sfb_smem_layout_staged)
            sSFB1_ld = sSFB1
            sSFB2_ld = sSFB2

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            if cutlass.const_expr(
                self.uniform_coord in (1, 2)
                and use_2cta_instrs
                and self.cluster_shape_mn == (2, 1)
            ):
                sfb_full_mcast_mask = cutlass.Int16(3)
            else:
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
                    cluster_layout_sfb_vmnk,
                    block_in_cluster_coord_sfb_vmnk,
                    mcast_mode=1,
                )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler_f, (None, 0, None)), (None, None, None)
        )
        gB1_nkl = cute.local_tile(
            mB1_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gB2_nkl = cute.local_tile(
            mB2_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        if cutlass.const_expr(self.fuse_b):
            g1l = gB1_nkl.layout
            gB1_nkl = cute.make_tensor(
                gB1_nkl.iterator,
                cute.make_layout(
                    (
                        (g1l.shape[0], 2),
                        g1l.shape[1],
                        g1l.shape[2],
                        g1l.shape[3],
                        g1l.shape[4],
                    ),
                    stride=(
                        (g1l.stride[0], 0),
                        g1l.stride[1],
                        g1l.stride[2],
                        g1l.stride[3],
                        g1l.stride[4],
                    ),
                ),
            )
            g2l = gB2_nkl.layout
            gB2_nkl = cute.make_tensor(
                gB2_nkl.iterator,
                cute.make_layout(
                    (
                        (g2l.shape[0], 2),
                        g2l.shape[1],
                        g2l.shape[2],
                        g2l.shape[3],
                        g2l.shape[4],
                    ),
                    stride=(
                        (g2l.stride[0], 0),
                        g2l.stride[1],
                        g2l.stride[2],
                        g2l.stride[3],
                        g2l.stride[4],
                    ),
                ),
            )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler_f, (None, 0, None)), (None, None, None)
        )
        gSFB1_nkl = cute.local_tile(
            mSFB1_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        gSFB2_nkl = cute.local_tile(
            mSFB2_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        if cutlass.const_expr(not self.clean_c_prologue):
            gC_mnl = cute.local_tile(
                mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
            )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_f = tiled_mma_f.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma_f.partition_A(gA_mkl)
        tCgB1 = thr_mma_f.partition_B(gB1_nkl)
        tCgB2 = thr_mma_f.partition_B(gB2_nkl)
        tCgSFA = thr_mma_f.partition_A(gSFA_mkl)
        tCgSFB1 = thr_mma_sfb.partition_B(gSFB1_nkl)
        tCgSFB2 = thr_mma_sfb.partition_B(gSFB2_nkl)
        if cutlass.const_expr(not self.clean_c_prologue):
            tCgC = thr_mma.partition_C(gC_mnl)

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
        tBsB1, tBgB1 = cpasync.tma_partition(
            tma_atom_b1,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB1, 0, 3),
            cute.group_modes(tCgB1, 0, 3),
        )
        tBsB2, tBgB2 = cpasync.tma_partition(
            tma_atom_b2,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB2, 0, 3),
            cute.group_modes(tCgB2, 0, 3),
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
        tBsSFB1, tBgSFB1 = cpasync.tma_partition(
            tma_atom_sfb1,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB1_ld, 0, 3),
            cute.group_modes(tCgSFB1, 0, 3),
        )
        tBsSFB1 = cute.filter_zeros(tBsSFB1)
        tBgSFB1 = cute.filter_zeros(tBgSFB1)
        tBsSFB2, tBgSFB2 = cpasync.tma_partition(
            tma_atom_sfb2,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB2_ld, 0, 3),
            cute.group_modes(tCgSFB2, 0, 3),
        )
        tBsSFB2 = cute.filter_zeros(tBsSFB2)
        tBgSFB2 = cute.filter_zeros(tBgSFB2)

        if cutlass.const_expr((not self.warp_scoped_ab_frag) or (not self.fuse_b)):
            tCrA = tiled_mma_f.make_fragment_A(sA)
            tCrB1 = tiled_mma_f.make_fragment_B(sB1)
            if cutlass.const_expr(
                (not (self.prune_dead_smem and self.fuse_b))
                and ((not self.warp_spec_parts) or (not self.fuse_b))
            ):
                tCrB2 = tiled_mma_f.make_fragment_B(sB2)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        if cutlass.const_expr((not self.warp_scoped_ab_frag) or (not self.fuse_b)):
            acc_shape_f = tiled_mma_f.partition_shape_C(self.mma_tiler_f[:2])
            tCtAcc_fake_f = tiled_mma_f.make_fragment_C(acc_shape_f)

        pf_lim = k_tile_cnt
        if cutlass.const_expr(self.pf_row0_only):
            if mma_tile_coord_mnl[0] != 0:
                pf_lim = cutlass.Int32(0)

        n_k_tiles = cute.size(gA_mkl, mode=[3])
        m_clusters = cute.size(gA_mkl, mode=[1])

        if cutlass.const_expr(self.pf_head > 0):
            if cutlass.const_expr(self.pf_a_bidy0):
                pf_a_wid = (
                    self.epilog_warp_id[0]
                    if self.pf_a_warp0
                    else self.epilog_warp_id[1]
                )
                if cutlass.const_expr(self.split_bidy_pf_a_mode == 3):
                    if warp_idx == pf_a_wid:
                        if bidy == 0:
                            hA = tAgA[
                                (
                                    None,
                                    mma_tile_coord_mnl[0],
                                    None,
                                    mma_tile_coord_mnl[2],
                                )
                            ]
                            for t in cutlass.range_constexpr(
                                min(2, self.pf_head, n_k_tiles)
                                if self.const_prolog_pf
                                else min(2, self.pf_head)
                            ):
                                if cutlass.const_expr(self.const_prolog_pf):
                                    cute.prefetch(tma_atom_a, hA[(None, t)])
                                else:
                                    if t < k_tile_cnt:
                                        cute.prefetch(tma_atom_a, hA[(None, t)])
                        elif bidy == 1:
                            hSFA = tAgSFA[
                                (
                                    None,
                                    mma_tile_coord_mnl[0],
                                    None,
                                    mma_tile_coord_mnl[2],
                                )
                            ]
                            for t in cutlass.range_constexpr(
                                min(2, self.pf_head, n_k_tiles)
                                if self.const_prolog_pf
                                else min(2, self.pf_head)
                            ):
                                if cutlass.const_expr(self.const_prolog_pf):
                                    cute.prefetch(tma_atom_sfa, hSFA[(None, t)])
                                else:
                                    if t < k_tile_cnt:
                                        cute.prefetch(tma_atom_sfa, hSFA[(None, t)])
                        elif cutlass.const_expr(
                            self.pf_a_prolog_mode == 2 and self.pf_head > 2
                        ):
                            if bidy == 2:
                                hA = tAgA[
                                    (
                                        None,
                                        mma_tile_coord_mnl[0],
                                        None,
                                        mma_tile_coord_mnl[2],
                                    )
                                ]
                                for t in cutlass.range_constexpr(2, self.pf_head):
                                    if t < k_tile_cnt:
                                        cute.prefetch(tma_atom_a, hA[(None, t)])
                            elif bidy == 3:
                                hSFA = tAgSFA[
                                    (
                                        None,
                                        mma_tile_coord_mnl[0],
                                        None,
                                        mma_tile_coord_mnl[2],
                                    )
                                ]
                                for t in cutlass.range_constexpr(2, self.pf_head):
                                    if t < k_tile_cnt:
                                        cute.prefetch(tma_atom_sfa, hSFA[(None, t)])
                else:
                    if warp_idx == pf_a_wid and bidy == 0:
                        hA = tAgA[
                            (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                        ]
                        hSFA = tAgSFA[
                            (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                        ]
                        for t in cutlass.range_constexpr(min(2, self.pf_head)):
                            if t < k_tile_cnt:
                                cute.prefetch(tma_atom_a, hA[(None, t)])
                                cute.prefetch(tma_atom_sfa, hSFA[(None, t)])

            if cutlass.const_expr(self.pf_head_mode == 1):
                if cutlass.const_expr(self.interleave_m_row_pf and self.pf_row0_only):
                    pf_t_base = mma_tile_coord_mnl[0] * 2
                    if warp_idx == self.epilog_warp_id[2]:
                        if cutlass.const_expr(
                            (self.hoist_cta_prolog_b_pf or self.const_prolog_pf)
                            and self.fuse_b
                        ):
                            if is_leader_cta:
                                hB1 = tBgB1[
                                    (
                                        None,
                                        mma_tile_coord_mnl[1],
                                        None,
                                        mma_tile_coord_mnl[2],
                                    )
                                ]
                                for t in cutlass.range_constexpr(2):
                                    t_idx = pf_t_base + t
                                    if cutlass.const_expr(
                                        self.const_prolog_pf
                                        and (m_clusters * 2 <= n_k_tiles)
                                    ):
                                        cute.prefetch(tma_atom_b1, hB1[(None, t_idx)])
                                    else:
                                        if t_idx < k_tile_cnt:
                                            cute.prefetch(
                                                tma_atom_b1, hB1[(None, t_idx)]
                                            )
                            else:
                                hB2 = tBgB2[
                                    (
                                        None,
                                        mma_tile_coord_mnl[1],
                                        None,
                                        mma_tile_coord_mnl[2],
                                    )
                                ]
                                for t in cutlass.range_constexpr(2):
                                    t_idx = pf_t_base + t
                                    if cutlass.const_expr(
                                        self.const_prolog_pf
                                        and (m_clusters * 2 <= n_k_tiles)
                                    ):
                                        cute.prefetch(tma_atom_b2, hB2[(None, t_idx)])
                                    else:
                                        if t_idx < k_tile_cnt:
                                            cute.prefetch(
                                                tma_atom_b2, hB2[(None, t_idx)]
                                            )
                        else:
                            hB1 = tBgB1[
                                (
                                    None,
                                    mma_tile_coord_mnl[1],
                                    None,
                                    mma_tile_coord_mnl[2],
                                )
                            ]
                            hB2 = tBgB2[
                                (
                                    None,
                                    mma_tile_coord_mnl[1],
                                    None,
                                    mma_tile_coord_mnl[2],
                                )
                            ]
                            for t in cutlass.range_constexpr(2):
                                t_idx = pf_t_base + t
                                if t_idx < k_tile_cnt:
                                    if cutlass.const_expr(self.fuse_b):
                                        if mma_tile_coord_v == 0:
                                            cute.prefetch(
                                                tma_atom_b1, hB1[(None, t_idx)]
                                            )
                                        else:
                                            cute.prefetch(
                                                tma_atom_b2, hB2[(None, t_idx)]
                                            )
                                    else:
                                        cute.prefetch(tma_atom_b1, hB1[(None, t_idx)])
                                        cute.prefetch(tma_atom_b2, hB2[(None, t_idx)])
                    if warp_idx == self.epilog_warp_id[3]:
                        hn = mma_tile_coord_mnl[1]
                        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                            hn = mma_tile_coord_mnl[1] // 2
                        hSFB1 = tBgSFB1[(None, hn, None, mma_tile_coord_mnl[2])]
                        hSFB2 = tBgSFB2[(None, hn, None, mma_tile_coord_mnl[2])]
                        for t in cutlass.range_constexpr(2):
                            t_idx = pf_t_base + t
                            if cutlass.const_expr(
                                self.const_prolog_pf and (m_clusters * 2 <= n_k_tiles)
                            ):
                                cute.prefetch(tma_atom_sfb1, hSFB1[(None, t_idx)])
                                cute.prefetch(tma_atom_sfb2, hSFB2[(None, t_idx)])
                            else:
                                if t_idx < k_tile_cnt:
                                    cute.prefetch(tma_atom_sfb1, hSFB1[(None, t_idx)])
                                    cute.prefetch(tma_atom_sfb2, hSFB2[(None, t_idx)])
                else:
                    if warp_idx == self.epilog_warp_id[2]:
                        if cutlass.const_expr(
                            (self.hoist_cta_prolog_b_pf or self.const_prolog_pf)
                            and self.fuse_b
                        ):
                            if is_leader_cta:
                                hB1 = tBgB1[
                                    (
                                        None,
                                        mma_tile_coord_mnl[1],
                                        None,
                                        mma_tile_coord_mnl[2],
                                    )
                                ]
                                for t in cutlass.range_constexpr(
                                    min(self.pf_head, n_k_tiles)
                                    if (
                                        self.const_prolog_pf and (not self.pf_row0_only)
                                    )
                                    else self.pf_head
                                ):
                                    if cutlass.const_expr(
                                        self.const_prolog_pf and (not self.pf_row0_only)
                                    ):
                                        cute.prefetch(tma_atom_b1, hB1[(None, t)])
                                    else:
                                        if t < pf_lim:
                                            cute.prefetch(tma_atom_b1, hB1[(None, t)])
                            else:
                                hB2 = tBgB2[
                                    (
                                        None,
                                        mma_tile_coord_mnl[1],
                                        None,
                                        mma_tile_coord_mnl[2],
                                    )
                                ]
                                for t in cutlass.range_constexpr(
                                    min(self.pf_head, n_k_tiles)
                                    if (
                                        self.const_prolog_pf and (not self.pf_row0_only)
                                    )
                                    else self.pf_head
                                ):
                                    if cutlass.const_expr(
                                        self.const_prolog_pf and (not self.pf_row0_only)
                                    ):
                                        cute.prefetch(tma_atom_b2, hB2[(None, t)])
                                    else:
                                        if t < pf_lim:
                                            cute.prefetch(tma_atom_b2, hB2[(None, t)])
                        else:
                            hB1 = tBgB1[
                                (
                                    None,
                                    mma_tile_coord_mnl[1],
                                    None,
                                    mma_tile_coord_mnl[2],
                                )
                            ]
                            hB2 = tBgB2[
                                (
                                    None,
                                    mma_tile_coord_mnl[1],
                                    None,
                                    mma_tile_coord_mnl[2],
                                )
                            ]
                            for t in cutlass.range_constexpr(self.pf_head):
                                if t < pf_lim:
                                    if cutlass.const_expr(self.fuse_b):
                                        if mma_tile_coord_v == 0:
                                            cute.prefetch(tma_atom_b1, hB1[(None, t)])
                                        else:
                                            cute.prefetch(tma_atom_b2, hB2[(None, t)])
                                    else:
                                        cute.prefetch(tma_atom_b1, hB1[(None, t)])
                                        cute.prefetch(tma_atom_b2, hB2[(None, t)])
                    if warp_idx == self.epilog_warp_id[3]:
                        hn = mma_tile_coord_mnl[1]
                        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                            hn = mma_tile_coord_mnl[1] // 2
                        hSFB1 = tBgSFB1[(None, hn, None, mma_tile_coord_mnl[2])]
                        hSFB2 = tBgSFB2[(None, hn, None, mma_tile_coord_mnl[2])]
                        for t in cutlass.range_constexpr(
                            min(self.pf_head, n_k_tiles)
                            if (self.const_prolog_pf and (not self.pf_row0_only))
                            else self.pf_head
                        ):
                            if cutlass.const_expr(
                                self.const_prolog_pf and (not self.pf_row0_only)
                            ):
                                cute.prefetch(tma_atom_sfb1, hSFB1[(None, t)])
                                cute.prefetch(tma_atom_sfb2, hSFB2[(None, t)])
                            else:
                                if t < pf_lim:
                                    cute.prefetch(tma_atom_sfb1, hSFB1[(None, t)])
                                    cute.prefetch(tma_atom_sfb2, hSFB2[(None, t)])
            elif cutlass.const_expr(self.pf_head_mode == 2):
                if warp_idx == self.epilog_warp_id[2]:
                    hB1 = tBgB1[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    hB2 = tBgB2[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    hn = mma_tile_coord_mnl[1]
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        hn = mma_tile_coord_mnl[1] // 2
                    hSFB1 = tBgSFB1[(None, hn, None, mma_tile_coord_mnl[2])]
                    hSFB2 = tBgSFB2[(None, hn, None, mma_tile_coord_mnl[2])]
                    for t in cutlass.range_constexpr(0, self.pf_head, 2):
                        if t < pf_lim:
                            if cutlass.const_expr(self.fuse_b):
                                if mma_tile_coord_v == 0:
                                    cute.prefetch(tma_atom_b1, hB1[(None, t)])
                                else:
                                    cute.prefetch(tma_atom_b2, hB2[(None, t)])
                            else:
                                cute.prefetch(tma_atom_b1, hB1[(None, t)])
                                cute.prefetch(tma_atom_b2, hB2[(None, t)])
                            cute.prefetch(tma_atom_sfb1, hSFB1[(None, t)])
                            cute.prefetch(tma_atom_sfb2, hSFB2[(None, t)])
                if warp_idx == self.epilog_warp_id[3]:
                    hB1 = tBgB1[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    hB2 = tBgB2[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    hn = mma_tile_coord_mnl[1]
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        hn = mma_tile_coord_mnl[1] // 2
                    hSFB1 = tBgSFB1[(None, hn, None, mma_tile_coord_mnl[2])]
                    hSFB2 = tBgSFB2[(None, hn, None, mma_tile_coord_mnl[2])]
                    for t in cutlass.range_constexpr(1, self.pf_head, 2):
                        if t < pf_lim:
                            if cutlass.const_expr(self.fuse_b):
                                if mma_tile_coord_v == 0:
                                    cute.prefetch(tma_atom_b1, hB1[(None, t)])
                                else:
                                    cute.prefetch(tma_atom_b2, hB2[(None, t)])
                            else:
                                cute.prefetch(tma_atom_b1, hB1[(None, t)])
                                cute.prefetch(tma_atom_b2, hB2[(None, t)])
                            cute.prefetch(tma_atom_sfb1, hSFB1[(None, t)])
                            cute.prefetch(tma_atom_sfb2, hSFB2[(None, t)])
            else:
                pf_head_wid = (
                    self.epilog_warp_id[2] if self.pf_head_warp2 else self.tma_warp_id
                )
                if warp_idx == pf_head_wid:
                    hB1 = tBgB1[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    hB2 = tBgB2[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    hn = mma_tile_coord_mnl[1]
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        hn = mma_tile_coord_mnl[1] // 2
                    hSFB1 = tBgSFB1[(None, hn, None, mma_tile_coord_mnl[2])]
                    hSFB2 = tBgSFB2[(None, hn, None, mma_tile_coord_mnl[2])]
                    for t in cutlass.range_constexpr(self.pf_head):
                        if t < pf_lim:
                            if cutlass.const_expr(self.fuse_b):
                                if mma_tile_coord_v == 0:
                                    cute.prefetch(tma_atom_b1, hB1[(None, t)])
                                else:
                                    cute.prefetch(tma_atom_b2, hB2[(None, t)])
                            else:
                                cute.prefetch(tma_atom_b1, hB1[(None, t)])
                                cute.prefetch(tma_atom_b2, hB2[(None, t)])
                            cute.prefetch(tma_atom_sfb1, hSFB1[(None, t)])
                            cute.prefetch(tma_atom_sfb2, hSFB2[(None, t)])

        if cutlass.const_expr(self.const_cluster_sync):
            if cutlass.const_expr(cute.size(self.cluster_shape_mn) > 1):
                cute.arch.cluster_wait()
            else:
                self.cta_sync_barrier.arrive_and_wait()
        else:
            if cute.size(self.cluster_shape_mn) > 1:
                cute.arch.cluster_wait()
            else:
                self.cta_sync_barrier.arrive_and_wait()

        #
        # 1. Specialized TMA Load Warp
        #
        if warp_idx == self.tma_warp_id:
            if cutlass.const_expr(
                self.use_pdl and ((not self.early_pdl_launch) or self.pdl_safe)
            ):
                cute.arch.griddepcontrol_launch_dependents()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tAgA_slice = tAgA[
                (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
            ]
            tBgB1_slice = tBgB1[
                (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
            ]
            tBgB2_slice = tBgB2[
                (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
            ]
            tAgSFA_slice = tAgSFA[
                (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
            ]
            slice_n = mma_tile_coord_mnl[1]
            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                slice_n = mma_tile_coord_mnl[1] // 2
            tBgSFB1_slice = tBgSFB1[(None, slice_n, None, mma_tile_coord_mnl[2])]
            tBgSFB2_slice = tBgSFB2[(None, slice_n, None, mma_tile_coord_mnl[2])]

            ab_producer_state.reset_count()
            peek_ab_empty_status = cutlass.Boolean(1)
            if cutlass.const_expr(not self.peel_tma_init_try):
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

            if cutlass.const_expr(self.pf_prolog and self.pf2 > self.num_ab_stage):
                for pf0 in cutlass.range_constexpr(self.num_ab_stage, self.pf2):
                    if pf0 < pf_lim:
                        if cutlass.const_expr(self.fuse_b):
                            if mma_tile_coord_v == 0:
                                cute.prefetch(tma_atom_b1, tBgB1_slice[(None, pf0)])
                            else:
                                cute.prefetch(tma_atom_b2, tBgB2_slice[(None, pf0)])
                        else:
                            cute.prefetch(tma_atom_b1, tBgB1_slice[(None, pf0)])
                            cute.prefetch(tma_atom_b2, tBgB2_slice[(None, pf0)])
                        cute.prefetch(tma_atom_sfb1, tBgSFB1_slice[(None, pf0)])
                        cute.prefetch(tma_atom_sfb2, tBgSFB2_slice[(None, pf0)])
                        if cutlass.const_expr(self.pf_a):
                            cute.prefetch(tma_atom_a, tAgA_slice[(None, pf0)])
                            cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, pf0)])

            b_pol = None
            if cutlass.const_expr(self.hint_b):
                b_pol = cutlass.Int64(0x12F0000000000000)
            elif cutlass.const_expr(self.hint_b_last_row >= 0):
                b_pol = cutlass.Int64(0x1000000000000000)
                if mma_tile_coord_mnl[0] == self.hint_b_last_row:
                    b_pol = cutlass.Int64(0x12F0000000000000)
            a_pol = None
            if cutlass.const_expr(self.hint_a):
                a_pol = cutlass.Int64(0x14F0000000000000)

            if cutlass.const_expr(self.pdl_safe):
                n_pre = cutlass.Int32(self.num_ab_stage)
                if k_tile_cnt < n_pre:
                    n_pre = k_tile_cnt
                for k_tile in cutlass.range(0, n_pre, 1, unroll=1):
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cnt = ab_producer_state.count
                    stg = ab_producer_state.index
                    if cutlass.const_expr(self.fuse_b):
                        if mma_tile_coord_v == 0:
                            cute.copy(
                                tma_atom_b1,
                                tBgB1_slice[(None, cnt)],
                                tBsB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                        else:
                            cute.copy(
                                tma_atom_b2,
                                tBgB2_slice[(None, cnt)],
                                tBsB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                    else:
                        cute.copy(
                            tma_atom_b1,
                            tBgB1_slice[(None, cnt)],
                            tBsB1[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=b_full_mcast_mask,
                            cache_policy=b_pol,
                        )
                        cute.copy(
                            tma_atom_b2,
                            tBgB2_slice[(None, cnt)],
                            tBsB2[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=b_full_mcast_mask,
                            cache_policy=b_pol,
                        )
                    cute.copy(
                        tma_atom_sfb1,
                        tBgSFB1_slice[(None, cnt)],
                        tBsSFB1[(None, stg)],
                        tma_bar_ptr=bar,
                        mcast_mask=sfb_full_mcast_mask,
                        cache_policy=b_pol,
                    )
                    cute.copy(
                        tma_atom_sfb2,
                        tBgSFB2_slice[(None, cnt)],
                        tBsSFB2[(None, stg)],
                        tma_bar_ptr=bar,
                        mcast_mask=sfb_full_mcast_mask,
                        cache_policy=b_pol,
                    )
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()

                a_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                for k_tile in cutlass.range(0, n_pre, 1, unroll=1):
                    bar = ab_pipeline.producer_get_barrier(a_state)
                    cnt = a_state.count
                    stg = a_state.index
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
                    a_state.advance()

                k_start = n_pre
            else:
                k_start = cutlass.Int32(0)

            if cutlass.const_expr(
                self.tma_cta_split_m256
                and self.fuse_b
                and ((not self.pf_row0_only) or self.tma_cta_split_all_m)
            ):
                if is_leader_cta:
                    if cutlass.const_expr(
                        self.tma_loop_split_mode == 1
                        and (not self.pdl_safe)
                        and (self.pf2 > 0)
                        and (n_k_tiles > self.pf2)
                    ):
                        for k_tile in cutlass.range(
                            k_start, k_tile_cnt - self.pf2, 1, unroll=1
                        ):
                            if not peek_ab_empty_status:
                                ab_pipeline.sync_object_empty.wait(
                                    ab_producer_state.index, ab_producer_state.phase
                                )
                            ab_pipeline.sync_object_full.arrive_and_expect_tx(
                                ab_producer_state.index, self.num_tma_load_bytes
                            )
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cnt = ab_producer_state.count
                            stg = ab_producer_state.index
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_b1,
                                tBgB1_slice[(None, cnt)],
                                tBsB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_sfb1,
                                tBgSFB1_slice[(None, cnt)],
                                tBsSFB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfb2,
                                tBgSFB2_slice[(None, cnt)],
                                tBsSFB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            pf = cnt + self.pf2
                            if cutlass.const_expr(
                                self.tma_cta_split_all_m and self.pf_row0_only
                            ):
                                if (cnt & 1) == mma_tile_coord_mnl[0]:
                                    cute.prefetch(tma_atom_b1, tBgB1_slice[(None, pf)])
                                    cute.prefetch(
                                        tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                    )
                                if cutlass.const_expr(self.pf_a_bidy0_ml):
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                    elif bidy == 1:
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )
                            else:
                                cute.prefetch(tma_atom_b1, tBgB1_slice[(None, pf)])
                                cute.prefetch(tma_atom_sfb1, tBgSFB1_slice[(None, pf)])
                                cute.prefetch(tma_atom_sfb2, tBgSFB2_slice[(None, pf)])
                                if cutlass.const_expr(self.pf_a_bidy0_ml):
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                    elif bidy == 1:
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )
                            ab_producer_state.advance()
                            peek_ab_empty_status = cutlass.Boolean(1)
                            if ab_producer_state.count >= self.num_ab_stage:
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )
                        for k_tile in cutlass.range(
                            k_tile_cnt - self.pf2, k_tile_cnt, 1, unroll=1
                        ):
                            if not peek_ab_empty_status:
                                ab_pipeline.sync_object_empty.wait(
                                    ab_producer_state.index, ab_producer_state.phase
                                )
                            ab_pipeline.sync_object_full.arrive_and_expect_tx(
                                ab_producer_state.index, self.num_tma_load_bytes
                            )
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cnt = ab_producer_state.count
                            stg = ab_producer_state.index
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_b1,
                                tBgB1_slice[(None, cnt)],
                                tBsB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_sfb1,
                                tBgSFB1_slice[(None, cnt)],
                                tBsSFB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfb2,
                                tBgSFB2_slice[(None, cnt)],
                                tBsSFB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            ab_producer_state.advance()
                            peek_ab_empty_status = cutlass.Boolean(1)
                            if ab_producer_state.count < k_tile_cnt:
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )
                    elif cutlass.const_expr(
                        self.tma_loop_split_mode in (2, 3)
                        and (not self.pdl_safe)
                        and (self.pf2 > 0)
                        and (n_k_tiles > self.num_ab_stage + self.pf2)
                    ):
                        for k_tile in cutlass.range(0, self.num_ab_stage, 1, unroll=1):
                            ab_pipeline.sync_object_full.arrive_and_expect_tx(
                                ab_producer_state.index, self.num_tma_load_bytes
                            )
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cnt = ab_producer_state.count
                            stg = ab_producer_state.index
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_b1,
                                tBgB1_slice[(None, cnt)],
                                tBsB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_sfb1,
                                tBgSFB1_slice[(None, cnt)],
                                tBsSFB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfb2,
                                tBgSFB2_slice[(None, cnt)],
                                tBsSFB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            pf = cnt + self.pf2
                            if cutlass.const_expr(
                                self.tma_cta_split_all_m and self.pf_row0_only
                            ):
                                if (cnt & 1) == mma_tile_coord_mnl[0]:
                                    cute.prefetch(tma_atom_b1, tBgB1_slice[(None, pf)])
                                    cute.prefetch(
                                        tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                    )
                                if cutlass.const_expr(self.pf_a_bidy0_ml):
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                    elif bidy == 1:
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )
                            else:
                                cute.prefetch(tma_atom_b1, tBgB1_slice[(None, pf)])
                                cute.prefetch(tma_atom_sfb1, tBgSFB1_slice[(None, pf)])
                                cute.prefetch(tma_atom_sfb2, tBgSFB2_slice[(None, pf)])
                                if cutlass.const_expr(self.pf_a_bidy0_ml):
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                    elif bidy == 1:
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )
                            ab_producer_state.advance()
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )
                        if cutlass.const_expr(self.tma_loop_split_mode == 3):
                            for k_tile in cutlass.range(
                                self.num_ab_stage, k_tile_cnt - self.pf2, 1, unroll=1
                            ):
                                if not peek_ab_empty_status:
                                    ab_pipeline.sync_object_empty.wait(
                                        ab_producer_state.index, ab_producer_state.phase
                                    )
                                ab_pipeline.sync_object_full.arrive_and_expect_tx(
                                    ab_producer_state.index, self.num_tma_load_bytes
                                )
                                bar = ab_pipeline.producer_get_barrier(
                                    ab_producer_state
                                )
                                cnt = ab_producer_state.count
                                stg = ab_producer_state.index
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, cnt)],
                                    tAsA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=a_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_b1,
                                    tBgB1_slice[(None, cnt)],
                                    tBsB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA_slice[(None, cnt)],
                                    tAsSFA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfa_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb1,
                                    tBgSFB1_slice[(None, cnt)],
                                    tBsSFB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb2,
                                    tBgSFB2_slice[(None, cnt)],
                                    tBsSFB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                pf = cnt + self.pf2
                                if cutlass.const_expr(
                                    self.tma_cta_split_all_m and self.pf_row0_only
                                ):
                                    if (cnt & 1) == mma_tile_coord_mnl[0]:
                                        cute.prefetch(
                                            tma_atom_b1, tBgB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                        )
                                    if cutlass.const_expr(self.pf_a_bidy0_ml):
                                        if bidy == 0:
                                            cute.prefetch(
                                                tma_atom_a, tAgA_slice[(None, pf)]
                                            )
                                        elif bidy == 1:
                                            cute.prefetch(
                                                tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                            )
                                else:
                                    cute.prefetch(tma_atom_b1, tBgB1_slice[(None, pf)])
                                    cute.prefetch(
                                        tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                    )
                                    if cutlass.const_expr(self.pf_a_bidy0_ml):
                                        if bidy == 0:
                                            cute.prefetch(
                                                tma_atom_a, tAgA_slice[(None, pf)]
                                            )
                                        elif bidy == 1:
                                            cute.prefetch(
                                                tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                            )
                                ab_producer_state.advance()
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )
                            for k_tile in cutlass.range(
                                k_tile_cnt - self.pf2, k_tile_cnt, 1, unroll=1
                            ):
                                if not peek_ab_empty_status:
                                    ab_pipeline.sync_object_empty.wait(
                                        ab_producer_state.index, ab_producer_state.phase
                                    )
                                ab_pipeline.sync_object_full.arrive_and_expect_tx(
                                    ab_producer_state.index, self.num_tma_load_bytes
                                )
                                bar = ab_pipeline.producer_get_barrier(
                                    ab_producer_state
                                )
                                cnt = ab_producer_state.count
                                stg = ab_producer_state.index
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, cnt)],
                                    tAsA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=a_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_b1,
                                    tBgB1_slice[(None, cnt)],
                                    tBsB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA_slice[(None, cnt)],
                                    tAsSFA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfa_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb1,
                                    tBgSFB1_slice[(None, cnt)],
                                    tBsSFB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb2,
                                    tBgSFB2_slice[(None, cnt)],
                                    tBsSFB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                ab_producer_state.advance()
                                peek_ab_empty_status = cutlass.Boolean(1)
                                if ab_producer_state.count < k_tile_cnt:
                                    peek_ab_empty_status = (
                                        ab_pipeline.producer_try_acquire(
                                            ab_producer_state
                                        )
                                    )
                        else:
                            for k_tile in cutlass.range(
                                self.num_ab_stage, k_tile_cnt, 1, unroll=1
                            ):
                                if not peek_ab_empty_status:
                                    ab_pipeline.sync_object_empty.wait(
                                        ab_producer_state.index, ab_producer_state.phase
                                    )
                                ab_pipeline.sync_object_full.arrive_and_expect_tx(
                                    ab_producer_state.index, self.num_tma_load_bytes
                                )
                                bar = ab_pipeline.producer_get_barrier(
                                    ab_producer_state
                                )
                                cnt = ab_producer_state.count
                                stg = ab_producer_state.index
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, cnt)],
                                    tAsA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=a_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_b1,
                                    tBgB1_slice[(None, cnt)],
                                    tBsB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA_slice[(None, cnt)],
                                    tAsSFA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfa_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb1,
                                    tBgSFB1_slice[(None, cnt)],
                                    tBsSFB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb2,
                                    tBgSFB2_slice[(None, cnt)],
                                    tBsSFB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                pf = cnt + self.pf2
                                if cutlass.const_expr(
                                    self.tma_cta_split_all_m and self.pf_row0_only
                                ):
                                    if pf < k_tile_cnt:
                                        if (cnt & 1) == mma_tile_coord_mnl[0]:
                                            cute.prefetch(
                                                tma_atom_b1, tBgB1_slice[(None, pf)]
                                            )
                                            cute.prefetch(
                                                tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                            )
                                            cute.prefetch(
                                                tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                            )
                                        if cutlass.const_expr(self.pf_a_bidy0_ml):
                                            if bidy == 0:
                                                cute.prefetch(
                                                    tma_atom_a, tAgA_slice[(None, pf)]
                                                )
                                            elif bidy == 1:
                                                cute.prefetch(
                                                    tma_atom_sfa,
                                                    tAgSFA_slice[(None, pf)],
                                                )
                                else:
                                    if pf < pf_lim:
                                        cute.prefetch(
                                            tma_atom_b1, tBgB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                        )
                                        if cutlass.const_expr(self.pf_a_bidy0_ml):
                                            if bidy == 0:
                                                cute.prefetch(
                                                    tma_atom_a, tAgA_slice[(None, pf)]
                                                )
                                            elif bidy == 1:
                                                cute.prefetch(
                                                    tma_atom_sfa,
                                                    tAgSFA_slice[(None, pf)],
                                                )
                                ab_producer_state.advance()
                                peek_ab_empty_status = cutlass.Boolean(1)
                                if ab_producer_state.count < k_tile_cnt:
                                    peek_ab_empty_status = (
                                        ab_pipeline.producer_try_acquire(
                                            ab_producer_state
                                        )
                                    )
                    else:
                        for k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
                            if not peek_ab_empty_status:
                                ab_pipeline.sync_object_empty.wait(
                                    ab_producer_state.index, ab_producer_state.phase
                                )
                            ab_pipeline.sync_object_full.arrive_and_expect_tx(
                                ab_producer_state.index, self.num_tma_load_bytes
                            )
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cnt = ab_producer_state.count
                            stg = ab_producer_state.index
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_b1,
                                tBgB1_slice[(None, cnt)],
                                tBsB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_sfb1,
                                tBgSFB1_slice[(None, cnt)],
                                tBsSFB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfb2,
                                tBgSFB2_slice[(None, cnt)],
                                tBsSFB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            if cutlass.const_expr(self.pf2 > 0):
                                pf = cnt + self.pf2
                                if cutlass.const_expr(
                                    self.tma_cta_split_all_m and self.pf_row0_only
                                ):
                                    if pf < k_tile_cnt:
                                        if cutlass.const_expr(
                                            self.split_mrow_ml_pf_mode == 2
                                        ):
                                            if (cnt & 1) == mma_tile_coord_mnl[0]:
                                                cute.prefetch(
                                                    tma_atom_b1, tBgB1_slice[(None, pf)]
                                                )
                                                cute.prefetch(
                                                    tma_atom_sfb1,
                                                    tBgSFB1_slice[(None, pf)],
                                                )
                                                cute.prefetch(
                                                    tma_atom_sfb2,
                                                    tBgSFB2_slice[(None, pf)],
                                                )
                                        else:
                                            if mma_tile_coord_mnl[0] == 0:
                                                cute.prefetch(
                                                    tma_atom_b1, tBgB1_slice[(None, pf)]
                                                )
                                                cute.prefetch(
                                                    tma_atom_sfb1,
                                                    tBgSFB1_slice[(None, pf)],
                                                )
                                                cute.prefetch(
                                                    tma_atom_sfb2,
                                                    tBgSFB2_slice[(None, pf)],
                                                )
                                        if cutlass.const_expr(self.pf_a_bidy0_ml):
                                            if cutlass.const_expr(
                                                self.split_bidy_pf_a_mode in (2, 3)
                                            ):
                                                if bidy == 0:
                                                    cute.prefetch(
                                                        tma_atom_a,
                                                        tAgA_slice[(None, pf)],
                                                    )
                                                elif bidy == 1:
                                                    cute.prefetch(
                                                        tma_atom_sfa,
                                                        tAgSFA_slice[(None, pf)],
                                                    )
                                            else:
                                                if bidy == 0:
                                                    cute.prefetch(
                                                        tma_atom_a,
                                                        tAgA_slice[(None, pf)],
                                                    )
                                                    cute.prefetch(
                                                        tma_atom_sfa,
                                                        tAgSFA_slice[(None, pf)],
                                                    )
                                else:
                                    if pf < pf_lim:
                                        cute.prefetch(
                                            tma_atom_b1, tBgB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                        )
                                        if cutlass.const_expr(self.pf_a_bidy0_ml):
                                            if cutlass.const_expr(
                                                self.split_bidy_pf_a_mode in (1, 2, 3)
                                            ):
                                                if bidy == 0:
                                                    cute.prefetch(
                                                        tma_atom_a,
                                                        tAgA_slice[(None, pf)],
                                                    )
                                                elif bidy == 1:
                                                    cute.prefetch(
                                                        tma_atom_sfa,
                                                        tAgSFA_slice[(None, pf)],
                                                    )
                                            else:
                                                if bidy == 0:
                                                    cute.prefetch(
                                                        tma_atom_a,
                                                        tAgA_slice[(None, pf)],
                                                    )
                                                    cute.prefetch(
                                                        tma_atom_sfa,
                                                        tAgSFA_slice[(None, pf)],
                                                    )
                            ab_producer_state.advance()
                            peek_ab_empty_status = cutlass.Boolean(1)
                            if ab_producer_state.count < k_tile_cnt:
                                if ab_producer_state.count >= self.num_ab_stage:
                                    peek_ab_empty_status = (
                                        ab_pipeline.producer_try_acquire(
                                            ab_producer_state
                                        )
                                    )
                else:
                    if cutlass.const_expr(
                        self.tma_loop_split_mode == 1
                        and (not self.pdl_safe)
                        and (self.pf2 > 0)
                        and (n_k_tiles > self.pf2)
                    ):
                        for k_tile in cutlass.range(
                            k_start, k_tile_cnt - self.pf2, 1, unroll=1
                        ):
                            if not peek_ab_empty_status:
                                ab_pipeline.sync_object_empty.wait(
                                    ab_producer_state.index, ab_producer_state.phase
                                )
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cnt = ab_producer_state.count
                            stg = ab_producer_state.index
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_b2,
                                tBgB2_slice[(None, cnt)],
                                tBsB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_sfb1,
                                tBgSFB1_slice[(None, cnt)],
                                tBsSFB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfb2,
                                tBgSFB2_slice[(None, cnt)],
                                tBsSFB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            pf = cnt + self.pf2
                            if cutlass.const_expr(
                                self.tma_cta_split_all_m and self.pf_row0_only
                            ):
                                if (cnt & 1) == mma_tile_coord_mnl[0]:
                                    cute.prefetch(tma_atom_b2, tBgB2_slice[(None, pf)])
                                    cute.prefetch(
                                        tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                    )
                                if cutlass.const_expr(self.pf_a_bidy0_ml):
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                    elif bidy == 1:
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )
                            else:
                                cute.prefetch(tma_atom_b2, tBgB2_slice[(None, pf)])
                                cute.prefetch(tma_atom_sfb1, tBgSFB1_slice[(None, pf)])
                                cute.prefetch(tma_atom_sfb2, tBgSFB2_slice[(None, pf)])
                                if cutlass.const_expr(self.pf_a_bidy0_ml):
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                    elif bidy == 1:
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )
                            ab_producer_state.advance()
                            peek_ab_empty_status = cutlass.Boolean(1)
                            if ab_producer_state.count >= self.num_ab_stage:
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )
                        for k_tile in cutlass.range(
                            k_tile_cnt - self.pf2, k_tile_cnt, 1, unroll=1
                        ):
                            if not peek_ab_empty_status:
                                ab_pipeline.sync_object_empty.wait(
                                    ab_producer_state.index, ab_producer_state.phase
                                )
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cnt = ab_producer_state.count
                            stg = ab_producer_state.index
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_b2,
                                tBgB2_slice[(None, cnt)],
                                tBsB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_sfb1,
                                tBgSFB1_slice[(None, cnt)],
                                tBsSFB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfb2,
                                tBgSFB2_slice[(None, cnt)],
                                tBsSFB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            ab_producer_state.advance()
                            peek_ab_empty_status = cutlass.Boolean(1)
                            if ab_producer_state.count < k_tile_cnt:
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )
                    elif cutlass.const_expr(
                        self.tma_loop_split_mode in (2, 3)
                        and (not self.pdl_safe)
                        and (self.pf2 > 0)
                        and (n_k_tiles > self.num_ab_stage + self.pf2)
                    ):
                        for k_tile in cutlass.range(0, self.num_ab_stage, 1, unroll=1):
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cnt = ab_producer_state.count
                            stg = ab_producer_state.index
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_b2,
                                tBgB2_slice[(None, cnt)],
                                tBsB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_sfb1,
                                tBgSFB1_slice[(None, cnt)],
                                tBsSFB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfb2,
                                tBgSFB2_slice[(None, cnt)],
                                tBsSFB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            pf = cnt + self.pf2
                            if cutlass.const_expr(
                                self.tma_cta_split_all_m and self.pf_row0_only
                            ):
                                if (cnt & 1) == mma_tile_coord_mnl[0]:
                                    cute.prefetch(tma_atom_b2, tBgB2_slice[(None, pf)])
                                    cute.prefetch(
                                        tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                    )
                                if cutlass.const_expr(self.pf_a_bidy0_ml):
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                    elif bidy == 1:
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )
                            else:
                                cute.prefetch(tma_atom_b2, tBgB2_slice[(None, pf)])
                                cute.prefetch(tma_atom_sfb1, tBgSFB1_slice[(None, pf)])
                                cute.prefetch(tma_atom_sfb2, tBgSFB2_slice[(None, pf)])
                                if cutlass.const_expr(self.pf_a_bidy0_ml):
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                    elif bidy == 1:
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )
                            ab_producer_state.advance()
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )
                        if cutlass.const_expr(self.tma_loop_split_mode == 3):
                            for k_tile in cutlass.range(
                                self.num_ab_stage, k_tile_cnt - self.pf2, 1, unroll=1
                            ):
                                if not peek_ab_empty_status:
                                    ab_pipeline.sync_object_empty.wait(
                                        ab_producer_state.index, ab_producer_state.phase
                                    )
                                bar = ab_pipeline.producer_get_barrier(
                                    ab_producer_state
                                )
                                cnt = ab_producer_state.count
                                stg = ab_producer_state.index
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, cnt)],
                                    tAsA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=a_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_b2,
                                    tBgB2_slice[(None, cnt)],
                                    tBsB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA_slice[(None, cnt)],
                                    tAsSFA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfa_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb1,
                                    tBgSFB1_slice[(None, cnt)],
                                    tBsSFB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb2,
                                    tBgSFB2_slice[(None, cnt)],
                                    tBsSFB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                pf = cnt + self.pf2
                                if cutlass.const_expr(
                                    self.tma_cta_split_all_m and self.pf_row0_only
                                ):
                                    if (cnt & 1) == mma_tile_coord_mnl[0]:
                                        cute.prefetch(
                                            tma_atom_b2, tBgB2_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                        )
                                    if cutlass.const_expr(self.pf_a_bidy0_ml):
                                        if bidy == 0:
                                            cute.prefetch(
                                                tma_atom_a, tAgA_slice[(None, pf)]
                                            )
                                        elif bidy == 1:
                                            cute.prefetch(
                                                tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                            )
                                else:
                                    cute.prefetch(tma_atom_b2, tBgB2_slice[(None, pf)])
                                    cute.prefetch(
                                        tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                    )
                                    if cutlass.const_expr(self.pf_a_bidy0_ml):
                                        if bidy == 0:
                                            cute.prefetch(
                                                tma_atom_a, tAgA_slice[(None, pf)]
                                            )
                                        elif bidy == 1:
                                            cute.prefetch(
                                                tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                            )
                                ab_producer_state.advance()
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )
                            for k_tile in cutlass.range(
                                k_tile_cnt - self.pf2, k_tile_cnt, 1, unroll=1
                            ):
                                if not peek_ab_empty_status:
                                    ab_pipeline.sync_object_empty.wait(
                                        ab_producer_state.index, ab_producer_state.phase
                                    )
                                bar = ab_pipeline.producer_get_barrier(
                                    ab_producer_state
                                )
                                cnt = ab_producer_state.count
                                stg = ab_producer_state.index
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, cnt)],
                                    tAsA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=a_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_b2,
                                    tBgB2_slice[(None, cnt)],
                                    tBsB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA_slice[(None, cnt)],
                                    tAsSFA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfa_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb1,
                                    tBgSFB1_slice[(None, cnt)],
                                    tBsSFB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb2,
                                    tBgSFB2_slice[(None, cnt)],
                                    tBsSFB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                ab_producer_state.advance()
                                peek_ab_empty_status = cutlass.Boolean(1)
                                if ab_producer_state.count < k_tile_cnt:
                                    peek_ab_empty_status = (
                                        ab_pipeline.producer_try_acquire(
                                            ab_producer_state
                                        )
                                    )
                        else:
                            for k_tile in cutlass.range(
                                self.num_ab_stage, k_tile_cnt, 1, unroll=1
                            ):
                                if not peek_ab_empty_status:
                                    ab_pipeline.sync_object_empty.wait(
                                        ab_producer_state.index, ab_producer_state.phase
                                    )
                                bar = ab_pipeline.producer_get_barrier(
                                    ab_producer_state
                                )
                                cnt = ab_producer_state.count
                                stg = ab_producer_state.index
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_slice[(None, cnt)],
                                    tAsA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=a_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_b2,
                                    tBgB2_slice[(None, cnt)],
                                    tBsB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfa,
                                    tAgSFA_slice[(None, cnt)],
                                    tAsSFA[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfa_full_mcast_mask,
                                    cache_policy=a_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb1,
                                    tBgSFB1_slice[(None, cnt)],
                                    tBsSFB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                cute.copy(
                                    tma_atom_sfb2,
                                    tBgSFB2_slice[(None, cnt)],
                                    tBsSFB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=sfb_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                                pf = cnt + self.pf2
                                if cutlass.const_expr(
                                    self.tma_cta_split_all_m and self.pf_row0_only
                                ):
                                    if pf < k_tile_cnt:
                                        if (cnt & 1) == mma_tile_coord_mnl[0]:
                                            cute.prefetch(
                                                tma_atom_b2, tBgB2_slice[(None, pf)]
                                            )
                                            cute.prefetch(
                                                tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                            )
                                            cute.prefetch(
                                                tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                            )
                                        if cutlass.const_expr(self.pf_a_bidy0_ml):
                                            if bidy == 0:
                                                cute.prefetch(
                                                    tma_atom_a, tAgA_slice[(None, pf)]
                                                )
                                            elif bidy == 1:
                                                cute.prefetch(
                                                    tma_atom_sfa,
                                                    tAgSFA_slice[(None, pf)],
                                                )
                                else:
                                    if pf < pf_lim:
                                        cute.prefetch(
                                            tma_atom_b2, tBgB2_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                        )
                                        if cutlass.const_expr(self.pf_a_bidy0_ml):
                                            if bidy == 0:
                                                cute.prefetch(
                                                    tma_atom_a, tAgA_slice[(None, pf)]
                                                )
                                            elif bidy == 1:
                                                cute.prefetch(
                                                    tma_atom_sfa,
                                                    tAgSFA_slice[(None, pf)],
                                                )
                                ab_producer_state.advance()
                                peek_ab_empty_status = cutlass.Boolean(1)
                                if ab_producer_state.count < k_tile_cnt:
                                    peek_ab_empty_status = (
                                        ab_pipeline.producer_try_acquire(
                                            ab_producer_state
                                        )
                                    )
                    else:
                        for k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
                            if not peek_ab_empty_status:
                                ab_pipeline.sync_object_empty.wait(
                                    ab_producer_state.index, ab_producer_state.phase
                                )
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cnt = ab_producer_state.count
                            stg = ab_producer_state.index
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, cnt)],
                                tAsA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=a_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_b2,
                                tBgB2_slice[(None, cnt)],
                                tBsB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, cnt)],
                                tAsSFA[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfa_full_mcast_mask,
                                cache_policy=a_pol,
                            )
                            cute.copy(
                                tma_atom_sfb1,
                                tBgSFB1_slice[(None, cnt)],
                                tBsSFB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_sfb2,
                                tBgSFB2_slice[(None, cnt)],
                                tBsSFB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=sfb_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            if cutlass.const_expr(self.pf2 > 0):
                                pf = cnt + self.pf2
                                if cutlass.const_expr(
                                    self.tma_cta_split_all_m and self.pf_row0_only
                                ):
                                    if pf < k_tile_cnt:
                                        if cutlass.const_expr(
                                            self.split_mrow_ml_pf_mode == 2
                                        ):
                                            if (cnt & 1) == mma_tile_coord_mnl[0]:
                                                cute.prefetch(
                                                    tma_atom_b2, tBgB2_slice[(None, pf)]
                                                )
                                                cute.prefetch(
                                                    tma_atom_sfb1,
                                                    tBgSFB1_slice[(None, pf)],
                                                )
                                                cute.prefetch(
                                                    tma_atom_sfb2,
                                                    tBgSFB2_slice[(None, pf)],
                                                )
                                        else:
                                            if mma_tile_coord_mnl[0] == 0:
                                                cute.prefetch(
                                                    tma_atom_b2, tBgB2_slice[(None, pf)]
                                                )
                                                cute.prefetch(
                                                    tma_atom_sfb1,
                                                    tBgSFB1_slice[(None, pf)],
                                                )
                                                cute.prefetch(
                                                    tma_atom_sfb2,
                                                    tBgSFB2_slice[(None, pf)],
                                                )
                                        if cutlass.const_expr(self.pf_a_bidy0_ml):
                                            if cutlass.const_expr(
                                                self.split_bidy_pf_a_mode in (2, 3)
                                            ):
                                                if bidy == 0:
                                                    cute.prefetch(
                                                        tma_atom_a,
                                                        tAgA_slice[(None, pf)],
                                                    )
                                                elif bidy == 1:
                                                    cute.prefetch(
                                                        tma_atom_sfa,
                                                        tAgSFA_slice[(None, pf)],
                                                    )
                                            else:
                                                if bidy == 0:
                                                    cute.prefetch(
                                                        tma_atom_a,
                                                        tAgA_slice[(None, pf)],
                                                    )
                                                    cute.prefetch(
                                                        tma_atom_sfa,
                                                        tAgSFA_slice[(None, pf)],
                                                    )
                                else:
                                    if pf < pf_lim:
                                        cute.prefetch(
                                            tma_atom_b2, tBgB2_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                        )
                                        if cutlass.const_expr(self.pf_a_bidy0_ml):
                                            if cutlass.const_expr(
                                                self.split_bidy_pf_a_mode in (1, 2, 3)
                                            ):
                                                if bidy == 0:
                                                    cute.prefetch(
                                                        tma_atom_a,
                                                        tAgA_slice[(None, pf)],
                                                    )
                                                elif bidy == 1:
                                                    cute.prefetch(
                                                        tma_atom_sfa,
                                                        tAgSFA_slice[(None, pf)],
                                                    )
                                            else:
                                                if bidy == 0:
                                                    cute.prefetch(
                                                        tma_atom_a,
                                                        tAgA_slice[(None, pf)],
                                                    )
                                                    cute.prefetch(
                                                        tma_atom_sfa,
                                                        tAgSFA_slice[(None, pf)],
                                                    )
                            ab_producer_state.advance()
                            peek_ab_empty_status = cutlass.Boolean(1)
                            if ab_producer_state.count < k_tile_cnt:
                                if ab_producer_state.count >= self.num_ab_stage:
                                    peek_ab_empty_status = (
                                        ab_pipeline.producer_try_acquire(
                                            ab_producer_state
                                        )
                                    )
            else:
                for k_tile in cutlass.range(k_start, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cnt = ab_producer_state.count
                    stg = ab_producer_state.index

                    if cutlass.const_expr(self.tma_order == 1):
                        if cutlass.const_expr(self.fuse_b):
                            if mma_tile_coord_v == 0:
                                cute.copy(
                                    tma_atom_b1,
                                    tBgB1_slice[(None, cnt)],
                                    tBsB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                            else:
                                cute.copy(
                                    tma_atom_b2,
                                    tBgB2_slice[(None, cnt)],
                                    tBsB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                        else:
                            cute.copy(
                                tma_atom_b1,
                                tBgB1_slice[(None, cnt)],
                                tBsB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_b2,
                                tBgB2_slice[(None, cnt)],
                                tBsB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                        cute.copy(
                            tma_atom_sfb1,
                            tBgSFB1_slice[(None, cnt)],
                            tBsSFB1[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                            cache_policy=b_pol,
                        )
                        cute.copy(
                            tma_atom_sfb2,
                            tBgSFB2_slice[(None, cnt)],
                            tBsSFB2[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                            cache_policy=b_pol,
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
                    elif cutlass.const_expr(self.tma_order == 2):
                        if cutlass.const_expr(self.fuse_b):
                            if mma_tile_coord_v == 0:
                                cute.copy(
                                    tma_atom_b1,
                                    tBgB1_slice[(None, cnt)],
                                    tBsB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                            else:
                                cute.copy(
                                    tma_atom_b2,
                                    tBgB2_slice[(None, cnt)],
                                    tBsB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                        else:
                            cute.copy(
                                tma_atom_b1,
                                tBgB1_slice[(None, cnt)],
                                tBsB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_b2,
                                tBgB2_slice[(None, cnt)],
                                tBsB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, cnt)],
                            tAsA[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb1,
                            tBgSFB1_slice[(None, cnt)],
                            tBsSFB1[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                            cache_policy=b_pol,
                        )
                        cute.copy(
                            tma_atom_sfb2,
                            tBgSFB2_slice[(None, cnt)],
                            tBsSFB2[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                            cache_policy=b_pol,
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
                            cache_policy=a_pol,
                        )
                        if cutlass.const_expr(self.fuse_b):
                            if mma_tile_coord_v == 0:
                                cute.copy(
                                    tma_atom_b1,
                                    tBgB1_slice[(None, cnt)],
                                    tBsB1[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                            else:
                                cute.copy(
                                    tma_atom_b2,
                                    tBgB2_slice[(None, cnt)],
                                    tBsB2[(None, stg)],
                                    tma_bar_ptr=bar,
                                    mcast_mask=b_full_mcast_mask,
                                    cache_policy=b_pol,
                                )
                        else:
                            cute.copy(
                                tma_atom_b1,
                                tBgB1_slice[(None, cnt)],
                                tBsB1[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                            cute.copy(
                                tma_atom_b2,
                                tBgB2_slice[(None, cnt)],
                                tBsB2[(None, stg)],
                                tma_bar_ptr=bar,
                                mcast_mask=b_full_mcast_mask,
                                cache_policy=b_pol,
                            )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, cnt)],
                            tAsSFA[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfa_full_mcast_mask,
                            cache_policy=a_pol,
                        )
                        cute.copy(
                            tma_atom_sfb1,
                            tBgSFB1_slice[(None, cnt)],
                            tBsSFB1[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                            cache_policy=b_pol,
                        )
                        cute.copy(
                            tma_atom_sfb2,
                            tBgSFB2_slice[(None, cnt)],
                            tBsSFB2[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb_full_mcast_mask,
                            cache_policy=b_pol,
                        )

                    if cutlass.const_expr(self.pf2 > 0):
                        pf = cnt + self.pf2
                        if cutlass.const_expr(
                            self.split_mrow_ml_pf_mode == 1 and self.pf_row0_only
                        ):
                            if pf < k_tile_cnt:
                                if mma_tile_coord_mnl[0] == 0:
                                    if cutlass.const_expr(self.fuse_b):
                                        if mma_tile_coord_v == 0:
                                            cute.prefetch(
                                                tma_atom_b1, tBgB1_slice[(None, pf)]
                                            )
                                        else:
                                            cute.prefetch(
                                                tma_atom_b2, tBgB2_slice[(None, pf)]
                                            )
                                    else:
                                        cute.prefetch(
                                            tma_atom_b1, tBgB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_b2, tBgB2_slice[(None, pf)]
                                        )
                                else:
                                    cute.prefetch(
                                        tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                    )
                        elif cutlass.const_expr(
                            self.split_mrow_ml_pf_mode == 2 and self.pf_row0_only
                        ):
                            if pf < k_tile_cnt:
                                if (cnt & 1) == mma_tile_coord_mnl[0]:
                                    if cutlass.const_expr(self.fuse_b):
                                        if mma_tile_coord_v == 0:
                                            cute.prefetch(
                                                tma_atom_b1, tBgB1_slice[(None, pf)]
                                            )
                                        else:
                                            cute.prefetch(
                                                tma_atom_b2, tBgB2_slice[(None, pf)]
                                            )
                                    else:
                                        cute.prefetch(
                                            tma_atom_b1, tBgB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_b2, tBgB2_slice[(None, pf)]
                                        )
                                    cute.prefetch(
                                        tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                    )
                                    cute.prefetch(
                                        tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                    )
                        elif cutlass.const_expr(
                            self.split_mrow_ml_pf_mode == 3 and self.pf_row0_only
                        ):
                            if pf < k_tile_cnt:
                                if mma_tile_coord_mnl[0] == 0:
                                    if cutlass.const_expr(self.fuse_b):
                                        if mma_tile_coord_v == 0:
                                            cute.prefetch(
                                                tma_atom_b1, tBgB1_slice[(None, pf)]
                                            )
                                        else:
                                            cute.prefetch(
                                                tma_atom_b2, tBgB2_slice[(None, pf)]
                                            )
                                    else:
                                        cute.prefetch(
                                            tma_atom_b1, tBgB1_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_b2, tBgB2_slice[(None, pf)]
                                        )
                                else:
                                    if mma_tile_coord_v == 0:
                                        cute.prefetch(
                                            tma_atom_sfb1, tBgSFB1_slice[(None, pf)]
                                        )
                                    else:
                                        cute.prefetch(
                                            tma_atom_sfb2, tBgSFB2_slice[(None, pf)]
                                        )
                        else:
                            if pf < pf_lim:
                                if cutlass.const_expr(self.fuse_b):
                                    if mma_tile_coord_v == 0:
                                        cute.prefetch(
                                            tma_atom_b1, tBgB1_slice[(None, pf)]
                                        )
                                    else:
                                        cute.prefetch(
                                            tma_atom_b2, tBgB2_slice[(None, pf)]
                                        )
                                else:
                                    cute.prefetch(tma_atom_b1, tBgB1_slice[(None, pf)])
                                    cute.prefetch(tma_atom_b2, tBgB2_slice[(None, pf)])
                                cute.prefetch(tma_atom_sfb1, tBgSFB1_slice[(None, pf)])
                                cute.prefetch(tma_atom_sfb2, tBgSFB2_slice[(None, pf)])
                                if cutlass.const_expr(self.pf_a):
                                    cute.prefetch(tma_atom_a, tAgA_slice[(None, pf)])
                                    cute.prefetch(
                                        tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                    )
                                elif cutlass.const_expr(
                                    self.pf_a_bidy0_ml and (not self.pf_a_all_m)
                                ):
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )
                        if cutlass.const_expr(self.pf_a_all_m and self.pf_a_bidy0_ml):
                            if pf < k_tile_cnt:
                                if cutlass.const_expr(
                                    self.split_bidy_pf_a_mode in (2, 3)
                                ):
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                    elif bidy == 1:
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )
                                else:
                                    if bidy == 0:
                                        cute.prefetch(
                                            tma_atom_a, tAgA_slice[(None, pf)]
                                        )
                                        cute.prefetch(
                                            tma_atom_sfa, tAgSFA_slice[(None, pf)]
                                        )

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        if cutlass.const_expr(self.peel_tma_init_try):
                            if ab_producer_state.count >= self.num_ab_stage:
                                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                    ab_producer_state
                                )
                        else:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state
                            )

        #
        # 2. Specialized MMA Warp
        #
        if cutlass.const_expr(self.fuse_b):
            if warp_idx == self.mma_warp_id:
                if cutlass.const_expr(self.warp_scoped_ab_frag):
                    tCrA_m = tiled_mma_f.make_fragment_A(sA)
                    tCrB1_m = tiled_mma_f.make_fragment_B(sB1)
                    acc_shape_f_m = tiled_mma_f.partition_shape_C(self.mma_tiler_f[:2])
                    tCtAcc_fake_f_m = tiled_mma_f.make_fragment_C(acc_shape_f_m)
                else:
                    tCrA_m = tCrA
                    tCrB1_m = tCrB1
                    tCtAcc_fake_f_m = tCtAcc_fake_f
                if cutlass.const_expr(
                    (not self.const_tmem_bar)
                    and self.leader_tmem_bar
                    and self.mma_leader_only
                ):
                    if not is_leader_cta:
                        self.tmem_alloc_barrier.arrive_unaligned()
                if (
                    not (self.leader_tmem_bar and self.mma_leader_only)
                ) or is_leader_cta:
                    if cutlass.const_expr(self.const_tmem_bar):
                        if cutlass.const_expr(
                            (not self.defer_mma_tmem_bar)
                            and (not self.mma_tmem_bar_after_k0_wait)
                        ):
                            self.tmem_alloc_barrier.arrive_and_wait()
                        acc_tmem_ptr = cute.make_ptr(
                            self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16
                        )
                    else:
                        tmem.wait_for_alloc()
                        acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                    tCtAccF = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake_f_m.layout)
                    accf_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtAccF)
                    sfa_tmem_ptr = cute.recast_ptr(
                        acc_tmem_ptr + accf_col_offset, dtype=self.sf_dtype
                    )
                    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                        tiled_mma_f,
                        self.mma_tiler_f,
                        self.sf_vec_size,
                        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                    )
                    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
                    sfa_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                    sfb_tmem_ptr = cute.recast_ptr(
                        acc_tmem_ptr + accf_col_offset + sfa_col_offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                        tiled_mma_f,
                        self.mma_tiler_f,
                        self.sf_vec_size,
                        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                    )
                    tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
                    sfb_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtSFB)

                    tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                        self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
                    )
                    tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                        self.mainloop_s2t_copy_and_partition(sSFB1, tCtSFB)
                    )

                    if cutlass.const_expr(self.db_tmem_sf):
                        sfa_tmem_ptr1 = cute.recast_ptr(
                            acc_tmem_ptr
                            + accf_col_offset
                            + sfa_col_offset
                            + sfb_col_offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFA1 = cute.make_tensor(sfa_tmem_ptr1, tCtSFA_layout)
                        sfb_tmem_ptr1 = cute.recast_ptr(
                            acc_tmem_ptr
                            + accf_col_offset
                            + sfa_col_offset * 2
                            + sfb_col_offset,
                            dtype=self.sf_dtype,
                        )
                        tCtSFB1_db = cute.make_tensor(sfb_tmem_ptr1, tCtSFB_layout)
                        _, _, tCtSFA1_compact_s2t = (
                            self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA1)
                        )
                        _, _, tCtSFB1_compact_s2t = (
                            self.mainloop_s2t_copy_and_partition(sSFB1, tCtSFB1_db)
                        )

                    if cutlass.const_expr(self.mma_leader_only):
                        if is_leader_cta:
                            ab_consumer_state = pipeline.make_pipeline_state(
                                pipeline.PipelineUserType.Consumer, self.num_ab_stage
                            )
                            if cutlass.const_expr(not self.direct_acc_mbar):
                                acc_producer_state = pipeline.make_pipeline_state(
                                    pipeline.PipelineUserType.Producer,
                                    self.num_acc_stage,
                                )
                            ab_consumer_state.reset_count()
                            if cutlass.const_expr(self.peel_mma_init_try):
                                peek_ab_full_status = cutlass.Boolean(0)
                            else:
                                peek_ab_full_status = cutlass.Boolean(1)
                                if ab_consumer_state.count < k_tile_cnt:
                                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                        ab_consumer_state
                                    )

                            if cutlass.const_expr(not self.direct_acc_mbar):
                                acc_pipeline.producer_acquire(acc_producer_state)
                            if cutlass.const_expr(
                                self.unified_mma_loop
                                and self.s2t_interleave_mode == 3
                                and self.peel_acc_init
                            ):
                                tiled_mma_f.set(tcgen05.Field.ACCUMULATE, False)
                                if cutlass.const_expr(
                                    self.const_tmem_bar
                                    and self.defer_mma_tmem_bar
                                    and (not self.mma_tmem_bar_after_k0_wait)
                                ):
                                    self.tmem_alloc_barrier.arrive_and_wait()
                                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                                    ab_pipeline.consumer_wait(
                                        ab_consumer_state, peek_ab_full_status
                                    )
                                    if cutlass.const_expr(
                                        self.const_tmem_bar
                                        and self.mma_tmem_bar_after_k0_wait
                                    ):
                                        if k_tile == 0:
                                            self.tmem_alloc_barrier.arrive_and_wait()
                                    for kblock_idx in cutlass.range_constexpr(
                                        self.k_blocks
                                    ):
                                        s2t_kb_coord = (
                                            None,
                                            None,
                                            None,
                                            kblock_idx,
                                            ab_consumer_state.index,
                                        )
                                        t_kb_coord = (None, None, None, kblock_idx)
                                        cute.copy(
                                            tiled_copy_s2t_sfa,
                                            tCsSFA_compact_s2t[s2t_kb_coord],
                                            tCtSFA_compact_s2t[t_kb_coord],
                                        )
                                        cute.copy(
                                            tiled_copy_s2t_sfb,
                                            tCsSFB_compact_s2t[s2t_kb_coord],
                                            tCtSFB_compact_s2t[t_kb_coord],
                                        )
                                        kblock_coord = (
                                            None,
                                            None,
                                            kblock_idx,
                                            ab_consumer_state.index,
                                        )
                                        sf_kblock_coord = (None, None, kblock_idx)
                                        tiled_mma_f.set(
                                            tcgen05.Field.SFA,
                                            tCtSFA[sf_kblock_coord].iterator,
                                        )
                                        tiled_mma_f.set(
                                            tcgen05.Field.SFB,
                                            tCtSFB[sf_kblock_coord].iterator,
                                        )
                                        cute.gemm(
                                            tiled_mma_f,
                                            tCtAccF,
                                            tCrA_m[kblock_coord],
                                            tCrB1_m[kblock_coord],
                                            tCtAccF,
                                        )
                                        if cutlass.const_expr(kblock_idx == 0):
                                            tiled_mma_f.set(
                                                tcgen05.Field.ACCUMULATE, True
                                            )
                                    if cutlass.const_expr(
                                        self.fused_mma_tail_mode == 1
                                    ):
                                        rel_idx = ab_consumer_state.index
                                        ab_consumer_state.advance()
                                        peek_ab_full_status = cutlass.Boolean(1)
                                        if ab_consumer_state.count < k_tile_cnt:
                                            ab_pipeline.sync_object_empty.arrive(
                                                rel_idx,
                                                ab_pipeline.consumer_mask,
                                                self.cta_group,
                                            )
                                            peek_ab_full_status = (
                                                ab_pipeline.consumer_try_wait(
                                                    ab_consumer_state
                                                )
                                            )
                                    elif cutlass.const_expr(
                                        self.fused_mma_tail_mode == 2
                                    ):
                                        rel_idx = ab_consumer_state.index
                                        ab_consumer_state.advance()
                                        peek_ab_full_status = cutlass.Boolean(1)
                                        if ab_consumer_state.count < k_tile_cnt:
                                            if (
                                                ab_consumer_state.count
                                                <= k_tile_cnt - self.num_ab_stage
                                            ):
                                                ab_pipeline.sync_object_empty.arrive(
                                                    rel_idx,
                                                    ab_pipeline.consumer_mask,
                                                    self.cta_group,
                                                )
                                            peek_ab_full_status = (
                                                ab_pipeline.consumer_try_wait(
                                                    ab_consumer_state
                                                )
                                            )
                                    else:
                                        if cutlass.const_expr(
                                            self.elide_mma_tail_release == 1
                                        ):
                                            if k_tile < k_tile_cnt - 1:
                                                ab_pipeline.consumer_release(
                                                    ab_consumer_state
                                                )
                                        elif cutlass.const_expr(
                                            self.elide_mma_tail_release == 2
                                        ):
                                            if k_tile < k_tile_cnt - self.num_ab_stage:
                                                ab_pipeline.consumer_release(
                                                    ab_consumer_state
                                                )
                                        else:
                                            ab_pipeline.consumer_release(
                                                ab_consumer_state
                                            )
                                        ab_consumer_state.advance()
                                        peek_ab_full_status = cutlass.Boolean(1)
                                        if ab_consumer_state.count < k_tile_cnt:
                                            peek_ab_full_status = (
                                                ab_pipeline.consumer_try_wait(
                                                    ab_consumer_state
                                                )
                                            )
                            else:
                                ab_pipeline.consumer_wait(
                                    ab_consumer_state, peek_ab_full_status
                                )
                                if cutlass.const_expr(
                                    self.s2t_interleave_mode in (1, 3)
                                    and self.peel_acc_init
                                ):
                                    tiled_mma_f.set(tcgen05.Field.ACCUMULATE, False)
                                    for kblock_idx in cutlass.range_constexpr(
                                        self.k_blocks
                                    ):
                                        s2t_kb_coord0 = (
                                            None,
                                            None,
                                            None,
                                            kblock_idx,
                                            ab_consumer_state.index,
                                        )
                                        t_kb_coord = (None, None, None, kblock_idx)
                                        cute.copy(
                                            tiled_copy_s2t_sfa,
                                            tCsSFA_compact_s2t[s2t_kb_coord0],
                                            tCtSFA_compact_s2t[t_kb_coord],
                                        )
                                        cute.copy(
                                            tiled_copy_s2t_sfb,
                                            tCsSFB_compact_s2t[s2t_kb_coord0],
                                            tCtSFB_compact_s2t[t_kb_coord],
                                        )
                                        kblock_coord = (
                                            None,
                                            None,
                                            kblock_idx,
                                            ab_consumer_state.index,
                                        )
                                        sf_kblock_coord = (None, None, kblock_idx)
                                        tiled_mma_f.set(
                                            tcgen05.Field.SFA,
                                            tCtSFA[sf_kblock_coord].iterator,
                                        )
                                        tiled_mma_f.set(
                                            tcgen05.Field.SFB,
                                            tCtSFB[sf_kblock_coord].iterator,
                                        )
                                        cute.gemm(
                                            tiled_mma_f,
                                            tCtAccF,
                                            tCrA[kblock_coord],
                                            tCrB1[kblock_coord],
                                            tCtAccF,
                                        )
                                        if cutlass.const_expr(kblock_idx == 0):
                                            tiled_mma_f.set(
                                                tcgen05.Field.ACCUMULATE, True
                                            )
                                elif cutlass.const_expr(
                                    self.s2t_interleave_mode == 2 and self.peel_acc_init
                                ):
                                    tiled_mma_f.set(tcgen05.Field.ACCUMULATE, False)
                                    s2t_kb0 = (
                                        None,
                                        None,
                                        None,
                                        0,
                                        ab_consumer_state.index,
                                    )
                                    t_kb0 = (None, None, None, 0)
                                    cute.copy(
                                        tiled_copy_s2t_sfa,
                                        tCsSFA_compact_s2t[s2t_kb0],
                                        tCtSFA_compact_s2t[t_kb0],
                                    )
                                    cute.copy(
                                        tiled_copy_s2t_sfb,
                                        tCsSFB_compact_s2t[s2t_kb0],
                                        tCtSFB_compact_s2t[t_kb0],
                                    )
                                    tiled_mma_f.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA[(None, None, 0)].iterator,
                                    )
                                    tiled_mma_f.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB[(None, None, 0)].iterator,
                                    )
                                    cute.gemm(
                                        tiled_mma_f,
                                        tCtAccF,
                                        tCrA[(None, None, 0, ab_consumer_state.index)],
                                        tCrB1[(None, None, 0, ab_consumer_state.index)],
                                        tCtAccF,
                                    )
                                    tiled_mma_f.set(tcgen05.Field.ACCUMULATE, True)
                                    for kblock_idx in cutlass.range_constexpr(
                                        1, self.k_blocks
                                    ):
                                        s2t_kb_coord0 = (
                                            None,
                                            None,
                                            None,
                                            kblock_idx,
                                            ab_consumer_state.index,
                                        )
                                        t_kb_coord = (None, None, None, kblock_idx)
                                        cute.copy(
                                            tiled_copy_s2t_sfa,
                                            tCsSFA_compact_s2t[s2t_kb_coord0],
                                            tCtSFA_compact_s2t[t_kb_coord],
                                        )
                                        cute.copy(
                                            tiled_copy_s2t_sfb,
                                            tCsSFB_compact_s2t[s2t_kb_coord0],
                                            tCtSFB_compact_s2t[t_kb_coord],
                                        )
                                    for kblock_idx in cutlass.range_constexpr(
                                        1, self.k_blocks
                                    ):
                                        kblock_coord = (
                                            None,
                                            None,
                                            kblock_idx,
                                            ab_consumer_state.index,
                                        )
                                        sf_kblock_coord = (None, None, kblock_idx)
                                        tiled_mma_f.set(
                                            tcgen05.Field.SFA,
                                            tCtSFA[sf_kblock_coord].iterator,
                                        )
                                        tiled_mma_f.set(
                                            tcgen05.Field.SFB,
                                            tCtSFB[sf_kblock_coord].iterator,
                                        )
                                        cute.gemm(
                                            tiled_mma_f,
                                            tCtAccF,
                                            tCrA[kblock_coord],
                                            tCrB1[kblock_coord],
                                            tCtAccF,
                                        )
                                else:
                                    s2t_stage_coord0 = (
                                        None,
                                        None,
                                        None,
                                        None,
                                        ab_consumer_state.index,
                                    )
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
                                    if cutlass.const_expr(self.peel_acc_init):
                                        tiled_mma_f.set(tcgen05.Field.ACCUMULATE, False)
                                        for kblock_idx in cutlass.range_constexpr(
                                            self.k_blocks
                                        ):
                                            kblock_coord = (
                                                None,
                                                None,
                                                kblock_idx,
                                                ab_consumer_state.index,
                                            )
                                            sf_kblock_coord = (None, None, kblock_idx)
                                            tiled_mma_f.set(
                                                tcgen05.Field.SFA,
                                                tCtSFA[sf_kblock_coord].iterator,
                                            )
                                            tiled_mma_f.set(
                                                tcgen05.Field.SFB,
                                                tCtSFB[sf_kblock_coord].iterator,
                                            )
                                            cute.gemm(
                                                tiled_mma_f,
                                                tCtAccF,
                                                tCrA[kblock_coord],
                                                tCrB1[kblock_coord],
                                                tCtAccF,
                                            )
                                            if cutlass.const_expr(kblock_idx == 0):
                                                tiled_mma_f.set(
                                                    tcgen05.Field.ACCUMULATE, True
                                                )
                                    else:
                                        for kblock_idx in cutlass.range_constexpr(
                                            self.k_blocks
                                        ):
                                            tiled_mma_f.set(
                                                tcgen05.Field.ACCUMULATE, kblock_idx > 0
                                            )
                                            kblock_coord = (
                                                None,
                                                None,
                                                kblock_idx,
                                                ab_consumer_state.index,
                                            )
                                            sf_kblock_coord = (None, None, kblock_idx)
                                            tiled_mma_f.set(
                                                tcgen05.Field.SFA,
                                                tCtSFA[sf_kblock_coord].iterator,
                                            )
                                            tiled_mma_f.set(
                                                tcgen05.Field.SFB,
                                                tCtSFB[sf_kblock_coord].iterator,
                                            )
                                            cute.gemm(
                                                tiled_mma_f,
                                                tCtAccF,
                                                tCrA[kblock_coord],
                                                tCrB1[kblock_coord],
                                                tCtAccF,
                                            )
                                        tiled_mma_f.set(tcgen05.Field.ACCUMULATE, True)
                                ab_pipeline.consumer_release(ab_consumer_state)

                                ab_consumer_state.advance()
                                peek_ab_full_status = cutlass.Boolean(1)
                                if ab_consumer_state.count < k_tile_cnt:
                                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                        ab_consumer_state
                                    )

                                for k_tile in cutlass.range(1, k_tile_cnt, 1, unroll=1):
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
                                    if cutlass.const_expr(self.db_tmem_sf):
                                        if (k_tile & 1) != 0:
                                            cute.copy(
                                                tiled_copy_s2t_sfa,
                                                tCsSFA_compact_s2t[s2t_stage_coord],
                                                tCtSFA1_compact_s2t,
                                            )
                                            cute.copy(
                                                tiled_copy_s2t_sfb,
                                                tCsSFB_compact_s2t[s2t_stage_coord],
                                                tCtSFB1_compact_s2t,
                                            )
                                            for kblock_idx in cutlass.range_constexpr(
                                                self.k_blocks
                                            ):
                                                kblock_coord = (
                                                    None,
                                                    None,
                                                    kblock_idx,
                                                    ab_consumer_state.index,
                                                )
                                                sf_kblock_coord = (
                                                    None,
                                                    None,
                                                    kblock_idx,
                                                )
                                                tiled_mma_f.set(
                                                    tcgen05.Field.SFA,
                                                    tCtSFA1[sf_kblock_coord].iterator,
                                                )
                                                tiled_mma_f.set(
                                                    tcgen05.Field.SFB,
                                                    tCtSFB1_db[
                                                        sf_kblock_coord
                                                    ].iterator,
                                                )
                                                cute.gemm(
                                                    tiled_mma_f,
                                                    tCtAccF,
                                                    tCrA[kblock_coord],
                                                    tCrB1[kblock_coord],
                                                    tCtAccF,
                                                )
                                        else:
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
                                            for kblock_idx in cutlass.range_constexpr(
                                                self.k_blocks
                                            ):
                                                kblock_coord = (
                                                    None,
                                                    None,
                                                    kblock_idx,
                                                    ab_consumer_state.index,
                                                )
                                                sf_kblock_coord = (
                                                    None,
                                                    None,
                                                    kblock_idx,
                                                )
                                                tiled_mma_f.set(
                                                    tcgen05.Field.SFA,
                                                    tCtSFA[sf_kblock_coord].iterator,
                                                )
                                                tiled_mma_f.set(
                                                    tcgen05.Field.SFB,
                                                    tCtSFB[sf_kblock_coord].iterator,
                                                )
                                                cute.gemm(
                                                    tiled_mma_f,
                                                    tCtAccF,
                                                    tCrA[kblock_coord],
                                                    tCrB1[kblock_coord],
                                                    tCtAccF,
                                                )
                                    elif cutlass.const_expr(
                                        self.s2t_interleave_mode == 3
                                    ):
                                        for kblock_idx in cutlass.range_constexpr(
                                            self.k_blocks
                                        ):
                                            s2t_kb_coord = (
                                                None,
                                                None,
                                                None,
                                                kblock_idx,
                                                ab_consumer_state.index,
                                            )
                                            t_kb_coord = (None, None, None, kblock_idx)
                                            cute.copy(
                                                tiled_copy_s2t_sfa,
                                                tCsSFA_compact_s2t[s2t_kb_coord],
                                                tCtSFA_compact_s2t[t_kb_coord],
                                            )
                                            cute.copy(
                                                tiled_copy_s2t_sfb,
                                                tCsSFB_compact_s2t[s2t_kb_coord],
                                                tCtSFB_compact_s2t[t_kb_coord],
                                            )
                                            kblock_coord = (
                                                None,
                                                None,
                                                kblock_idx,
                                                ab_consumer_state.index,
                                            )
                                            sf_kblock_coord = (None, None, kblock_idx)
                                            tiled_mma_f.set(
                                                tcgen05.Field.SFA,
                                                tCtSFA[sf_kblock_coord].iterator,
                                            )
                                            tiled_mma_f.set(
                                                tcgen05.Field.SFB,
                                                tCtSFB[sf_kblock_coord].iterator,
                                            )
                                            cute.gemm(
                                                tiled_mma_f,
                                                tCtAccF,
                                                tCrA[kblock_coord],
                                                tCrB1[kblock_coord],
                                                tCtAccF,
                                            )
                                    elif cutlass.const_expr(
                                        self.s2t_interleave_mode == 2
                                    ):
                                        s2t_kb0 = (
                                            None,
                                            None,
                                            None,
                                            0,
                                            ab_consumer_state.index,
                                        )
                                        t_kb0 = (None, None, None, 0)
                                        cute.copy(
                                            tiled_copy_s2t_sfa,
                                            tCsSFA_compact_s2t[s2t_kb0],
                                            tCtSFA_compact_s2t[t_kb0],
                                        )
                                        cute.copy(
                                            tiled_copy_s2t_sfb,
                                            tCsSFB_compact_s2t[s2t_kb0],
                                            tCtSFB_compact_s2t[t_kb0],
                                        )
                                        tiled_mma_f.set(
                                            tcgen05.Field.SFA,
                                            tCtSFA[(None, None, 0)].iterator,
                                        )
                                        tiled_mma_f.set(
                                            tcgen05.Field.SFB,
                                            tCtSFB[(None, None, 0)].iterator,
                                        )
                                        cute.gemm(
                                            tiled_mma_f,
                                            tCtAccF,
                                            tCrA[
                                                (None, None, 0, ab_consumer_state.index)
                                            ],
                                            tCrB1[
                                                (None, None, 0, ab_consumer_state.index)
                                            ],
                                            tCtAccF,
                                        )
                                        for kblock_idx in cutlass.range_constexpr(
                                            1, self.k_blocks
                                        ):
                                            s2t_kb_coord = (
                                                None,
                                                None,
                                                None,
                                                kblock_idx,
                                                ab_consumer_state.index,
                                            )
                                            t_kb_coord = (None, None, None, kblock_idx)
                                            cute.copy(
                                                tiled_copy_s2t_sfa,
                                                tCsSFA_compact_s2t[s2t_kb_coord],
                                                tCtSFA_compact_s2t[t_kb_coord],
                                            )
                                            cute.copy(
                                                tiled_copy_s2t_sfb,
                                                tCsSFB_compact_s2t[s2t_kb_coord],
                                                tCtSFB_compact_s2t[t_kb_coord],
                                            )
                                        for kblock_idx in cutlass.range_constexpr(
                                            1, self.k_blocks
                                        ):
                                            kblock_coord = (
                                                None,
                                                None,
                                                kblock_idx,
                                                ab_consumer_state.index,
                                            )
                                            sf_kblock_coord = (None, None, kblock_idx)
                                            tiled_mma_f.set(
                                                tcgen05.Field.SFA,
                                                tCtSFA[sf_kblock_coord].iterator,
                                            )
                                            tiled_mma_f.set(
                                                tcgen05.Field.SFB,
                                                tCtSFB[sf_kblock_coord].iterator,
                                            )
                                            cute.gemm(
                                                tiled_mma_f,
                                                tCtAccF,
                                                tCrA[kblock_coord],
                                                tCrB1[kblock_coord],
                                                tCtAccF,
                                            )
                                    else:
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
                                        for kblock_idx in cutlass.range_constexpr(
                                            self.k_blocks
                                        ):
                                            kblock_coord = (
                                                None,
                                                None,
                                                kblock_idx,
                                                ab_consumer_state.index,
                                            )
                                            sf_kblock_coord = (None, None, kblock_idx)
                                            tiled_mma_f.set(
                                                tcgen05.Field.SFA,
                                                tCtSFA[sf_kblock_coord].iterator,
                                            )
                                            tiled_mma_f.set(
                                                tcgen05.Field.SFB,
                                                tCtSFB[sf_kblock_coord].iterator,
                                            )
                                            cute.gemm(
                                                tiled_mma_f,
                                                tCtAccF,
                                                tCrA[kblock_coord],
                                                tCrB1[kblock_coord],
                                                tCtAccF,
                                            )
                                    ab_pipeline.consumer_release(ab_consumer_state)

                                    ab_consumer_state.advance()
                                    peek_ab_full_status = cutlass.Boolean(1)
                                    if ab_consumer_state.count < k_tile_cnt:
                                        peek_ab_full_status = (
                                            ab_pipeline.consumer_try_wait(
                                                ab_consumer_state
                                            )
                                        )

                            if cutlass.const_expr(self.direct_acc_mbar):
                                if cutlass.const_expr(use_2cta_instrs):
                                    if cutlass.const_expr(
                                        self.static_cmask
                                        and self.cluster_shape_mn == (2, 1)
                                    ):
                                        acc_mcast_mask = cutlass.Int16(3)
                                    else:
                                        acc_mcast_mask = cute.make_layout_image_mask(
                                            cluster_layout_vmnk,
                                            block_in_cluster_coord_vmnk,
                                            mode=0,
                                        )
                                    with cute.arch.elect_one():
                                        cute.nvgpu.tcgen05.commit(
                                            acc_mbar_ptr, acc_mcast_mask, self.cta_group
                                        )
                                else:
                                    with cute.arch.elect_one():
                                        cute.nvgpu.tcgen05.commit(acc_mbar_ptr)
                            else:
                                acc_pipeline.producer_commit(acc_producer_state)
                    else:
                        ab_consumer_state = pipeline.make_pipeline_state(
                            pipeline.PipelineUserType.Consumer, self.num_ab_stage
                        )
                        if cutlass.const_expr(not self.direct_acc_mbar):
                            acc_producer_state = pipeline.make_pipeline_state(
                                pipeline.PipelineUserType.Producer, self.num_acc_stage
                            )

                        ab_consumer_state.reset_count()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                        if is_leader_cta and cutlass.const_expr(
                            not self.direct_acc_mbar
                        ):
                            acc_pipeline.producer_acquire(acc_producer_state)

                        if is_leader_cta:
                            ab_pipeline.consumer_wait(
                                ab_consumer_state, peek_ab_full_status
                            )
                            s2t_stage_coord0 = (
                                None,
                                None,
                                None,
                                None,
                                ab_consumer_state.index,
                            )
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
                            for kblock_idx in cutlass.range_constexpr(self.k_blocks):
                                tiled_mma_f.set(
                                    tcgen05.Field.ACCUMULATE, kblock_idx > 0
                                )
                                kblock_coord = (
                                    None,
                                    None,
                                    kblock_idx,
                                    ab_consumer_state.index,
                                )
                                sf_kblock_coord = (None, None, kblock_idx)
                                tiled_mma_f.set(
                                    tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator
                                )
                                tiled_mma_f.set(
                                    tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator
                                )
                                cute.gemm(
                                    tiled_mma_f,
                                    tCtAccF,
                                    tCrA[kblock_coord],
                                    tCrB1[kblock_coord],
                                    tCtAccF,
                                )
                            ab_pipeline.consumer_release(ab_consumer_state)

                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                    ab_consumer_state
                                )

                        tiled_mma_f.set(tcgen05.Field.ACCUMULATE, True)
                        for k_tile in cutlass.range(1, k_tile_cnt, 1, unroll=1):
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
                                for kblock_idx in cutlass.range_constexpr(
                                    self.k_blocks
                                ):
                                    kblock_coord = (
                                        None,
                                        None,
                                        kblock_idx,
                                        ab_consumer_state.index,
                                    )
                                    sf_kblock_coord = (None, None, kblock_idx)
                                    tiled_mma_f.set(
                                        tcgen05.Field.SFA,
                                        tCtSFA[sf_kblock_coord].iterator,
                                    )
                                    tiled_mma_f.set(
                                        tcgen05.Field.SFB,
                                        tCtSFB[sf_kblock_coord].iterator,
                                    )
                                    cute.gemm(
                                        tiled_mma_f,
                                        tCtAccF,
                                        tCrA[kblock_coord],
                                        tCrB1[kblock_coord],
                                        tCtAccF,
                                    )
                                ab_pipeline.consumer_release(ab_consumer_state)

                            ab_consumer_state.advance()
                            peek_ab_full_status = cutlass.Boolean(1)
                            if ab_consumer_state.count < k_tile_cnt:
                                if is_leader_cta:
                                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                        ab_consumer_state
                                    )

                        if is_leader_cta:
                            if cutlass.const_expr(self.direct_acc_mbar):
                                if cutlass.const_expr(use_2cta_instrs):
                                    acc_mcast_mask = cute.make_layout_image_mask(
                                        cluster_layout_vmnk,
                                        block_in_cluster_coord_vmnk,
                                        mode=0,
                                    )
                                    with cute.arch.elect_one():
                                        cute.nvgpu.tcgen05.commit(
                                            acc_mbar_ptr, acc_mcast_mask, self.cta_group
                                        )
                                else:
                                    with cute.arch.elect_one():
                                        cute.nvgpu.tcgen05.commit(acc_mbar_ptr)
                            else:
                                acc_pipeline.producer_commit(acc_producer_state)
        else:
            if warp_idx == self.mma_warp_id:
                tmem.wait_for_alloc()
                acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc1 = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
                acc_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtAcc1)
                acc_tmem_ptr2 = cute.recast_ptr(
                    acc_tmem_ptr + acc_col_offset, dtype=self.acc_dtype
                )
                tCtAcc2 = cute.make_tensor(acc_tmem_ptr2, tCtAcc_fake.layout)

                sfa_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + acc_col_offset * 2, dtype=self.sf_dtype
                )
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

                sfa_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                sfb1_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + acc_col_offset * 2 + sfa_col_offset,
                    dtype=self.sf_dtype,
                )
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.mma_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFB1 = cute.make_tensor(sfb1_tmem_ptr, tCtSFB_layout)
                sfb_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtSFB1)
                sfb2_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + acc_col_offset * 2 + sfa_col_offset + sfb_col_offset,
                    dtype=self.sf_dtype,
                )
                tCtSFB2 = cute.make_tensor(sfb2_tmem_ptr, tCtSFB_layout)

                tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
                )
                tiled_copy_s2t_sfb1, tCsSFB1_compact_s2t, tCtSFB1_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFB1, tCtSFB1)
                )
                tiled_copy_s2t_sfb2, tCsSFB2_compact_s2t, tCtSFB2_compact_s2t = (
                    self.mainloop_s2t_copy_and_partition(sSFB2, tCtSFB2)
                )

                ab_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                if cutlass.const_expr(not self.direct_acc_mbar):
                    acc_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.num_acc_stage
                    )

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                if is_leader_cta and cutlass.const_expr(not self.direct_acc_mbar):
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB1_mma = tCtSFB1
                tCtSFB2_mma = tCtSFB2
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr1 = cute.recast_ptr(
                        acc_tmem_ptr + acc_col_offset * 2 + sfa_col_offset + offset,
                        dtype=self.sf_dtype,
                    )
                    shifted_ptr2 = cute.recast_ptr(
                        acc_tmem_ptr
                        + acc_col_offset * 2
                        + sfa_col_offset
                        + sfb_col_offset
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB1_mma = cute.make_tensor(shifted_ptr1, tCtSFB_layout)
                    tCtSFB2_mma = cute.make_tensor(shifted_ptr2, tCtSFB_layout)

                if is_leader_cta:
                    ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                    s2t_stage_coord0 = (None, None, None, None, ab_consumer_state.index)
                    cute.copy(
                        tiled_copy_s2t_sfa,
                        tCsSFA_compact_s2t[s2t_stage_coord0],
                        tCtSFA_compact_s2t,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfb1,
                        tCsSFB1_compact_s2t[s2t_stage_coord0],
                        tCtSFB1_compact_s2t,
                    )
                    for kblock_idx in cutlass.range_constexpr(self.k_blocks):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, kblock_idx > 0)
                        kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                        sf_kblock_coord = (None, None, kblock_idx)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB1_mma[sf_kblock_coord].iterator
                        )
                        cute.gemm(
                            tiled_mma,
                            tCtAcc1,
                            tCrA[kblock_coord],
                            tCrB1[kblock_coord],
                            tCtAcc1,
                        )
                    cute.copy(
                        tiled_copy_s2t_sfb2,
                        tCsSFB2_compact_s2t[s2t_stage_coord0],
                        tCtSFB2_compact_s2t,
                    )
                    for kblock_idx in cutlass.range_constexpr(self.k_blocks):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, kblock_idx > 0)
                        kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                        sf_kblock_coord = (None, None, kblock_idx)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB2_mma[sf_kblock_coord].iterator
                        )
                        cute.gemm(
                            tiled_mma,
                            tCtAcc2,
                            tCrA[kblock_coord],
                            tCrB2[kblock_coord],
                            tCtAcc2,
                        )
                    ab_pipeline.consumer_release(ab_consumer_state)

                ab_consumer_state.advance()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt:
                    if is_leader_cta:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )

                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                for k_tile in cutlass.range(1, k_tile_cnt, 1, unroll=1):
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
                            tiled_copy_s2t_sfb1,
                            tCsSFB1_compact_s2t[s2t_stage_coord],
                            tCtSFB1_compact_s2t,
                        )
                        for kblock_idx in cutlass.range_constexpr(self.k_blocks):
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
                                tcgen05.Field.SFB, tCtSFB1_mma[sf_kblock_coord].iterator
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc1,
                                tCrA[kblock_coord],
                                tCrB1[kblock_coord],
                                tCtAcc1,
                            )
                        cute.copy(
                            tiled_copy_s2t_sfb2,
                            tCsSFB2_compact_s2t[s2t_stage_coord],
                            tCtSFB2_compact_s2t,
                        )
                        for kblock_idx in cutlass.range_constexpr(self.k_blocks):
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
                                tcgen05.Field.SFB, tCtSFB2_mma[sf_kblock_coord].iterator
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc2,
                                tCrA[kblock_coord],
                                tCrB2[kblock_coord],
                                tCtAcc2,
                            )

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                if is_leader_cta:
                    if cutlass.const_expr(self.direct_acc_mbar):
                        if cutlass.const_expr(use_2cta_instrs):
                            acc_mcast_mask = cute.make_layout_image_mask(
                                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0
                            )
                            with cute.arch.elect_one():
                                cute.nvgpu.tcgen05.commit(
                                    acc_mbar_ptr, acc_mcast_mask, self.cta_group
                                )
                        else:
                            with cute.arch.elect_one():
                                cute.nvgpu.tcgen05.commit(acc_mbar_ptr)
                    else:
                        acc_pipeline.producer_commit(acc_producer_state)

        #
        # 3. Specialized Epilogue Warps
        #
        if warp_idx < self.mma_warp_id:
            if cutlass.const_expr(
                self.coalesce_warp0_tmem
                and (not self.early_tmem_alloc)
                and self.const_tmem_bar
                and self.early_relinquish
            ):
                if warp_idx == self.epilog_warp_id[0]:
                    cute.arch.alloc_tmem(
                        self.num_tmem_alloc_cols,
                        storage.tmem_holding_buf,
                        is_two_cta=use_2cta_instrs,
                        arch="sm_100",
                    )
                    if is_leader_cta:
                        self.tmem_alloc_barrier.arrive_unaligned()
                    cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            else:
                if cutlass.const_expr(not self.early_tmem_alloc):
                    tmem.allocate(self.num_tmem_alloc_cols)
                if cutlass.const_expr(self.const_tmem_bar):
                    if warp_idx == self.epilog_warp_id[0] and is_leader_cta:
                        self.tmem_alloc_barrier.arrive_unaligned()
                else:
                    tmem.wait_for_alloc()
                if cutlass.const_expr(
                    self.early_relinquish and (not self.early_tmem_alloc)
                ):
                    tmem.relinquish_alloc_permit()

            if cutlass.const_expr(self.const_tmem_bar):
                acc_tmem_ptr = cute.make_ptr(
                    self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16
                )
            else:
                acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc1 = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            acc_col_offset = tcgen05.find_tmem_tensor_col_offset(tCtAcc1)
            acc_tmem_ptr2 = cute.recast_ptr(
                acc_tmem_ptr + acc_col_offset, dtype=self.acc_dtype
            )
            tCtAcc2 = cute.make_tensor(acc_tmem_ptr2, tCtAcc_fake.layout)

            if cutlass.const_expr(self.t2r_rep_x32):
                copy_atom_t2r = cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), self.acc_dtype
                )
            else:
                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    self.cta_tile_shape_mnk,
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    epi_tile,
                    use_2cta_instrs,
                )
            tAcc1_epi = cute.flat_divide(tCtAcc1[((None, None), 0, 0)], epi_tile)
            tAcc2_epi = cute.flat_divide(tCtAcc2[((None, None), 0, 0)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc1_epi[(None, None, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc1 = thr_copy_t2r.partition_S(tAcc1_epi)
            tTR_tAcc2 = thr_copy_t2r.partition_S(tAcc2_epi)

            if cutlass.const_expr(self.clean_c_prologue):
                gC_mnl_c = cute.local_tile(
                    mC_mnl,
                    cute.slice_(self.mma_tiler, (None, None, 0)),
                    (None, None, None),
                )
                tCgC_c = thr_mma.partition_C(gC_mnl_c)
                gC_mnl_epi = cute.flat_divide(
                    tCgC_c[((None, None), 0, 0, None, None, None)], epi_tile
                )
            else:
                gC_mnl_epi = cute.flat_divide(
                    tCgC[((None, None), 0, 0, None, None, None)], epi_tile
                )
            tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
            tTR_rAcc1 = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
            )
            tTR_rAcc2 = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc1.shape, self.c_dtype)

            copy_atom_r2s = sm100_utils.get_smem_store_op(
                self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)

            if cutlass.const_expr(not self.warp1_only_tma_c_part):
                sC_for_tma_partition = cute.group_modes(sC, 0, 2)
                gC_for_tma_partition = cute.group_modes(gC_mnl_epi, 0, 2)
                bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sC_for_tma_partition,
                    gC_for_tma_partition,
                )

            if cutlass.const_expr(not self.direct_acc_mbar):
                acc_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_acc_stage
                )
            if cutlass.const_expr(not self.warp1_only_tma_c_part):
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
            tTR_tAcc1 = cute.group_modes(tTR_tAcc1, 3, cute.rank(tTR_tAcc1))
            tTR_tAcc2 = cute.group_modes(tTR_tAcc2, 3, cute.rank(tTR_tAcc2))

            c_pol = None
            if cutlass.const_expr(self.hint_c):
                c_pol = cutlass.Int64(0x12F0000000000000)

            if cutlass.const_expr(self.inline_swz_r2s):
                sc_base_ptr = storage.sC.data_ptr()
                if cutlass.const_expr(self.single_swz_base):
                    swz_addrs = [
                        _swz_sc_addr_fast(sc_base_ptr, tidx, g_idx * 2)
                        for g_idx in range(0, 64, 8)
                    ]
                else:
                    swz_addrs = [
                        _swz_sc_addr(sc_base_ptr, tidx, g_idx * 2)
                        for g_idx in range(0, 64, 8)
                    ]

            if cutlass.const_expr(
                self.pre_wait_pdl and self.use_pdl and (not self.pdl_safe)
            ):
                if warp_idx == self.epilog_warp_id[1]:
                    cute.arch.griddepcontrol_wait()

            alpha_val = cutlass.Float32(1.0)
            if cutlass.const_expr(self.has_alpha):
                alpha_val = cutlass.Float32(mAlpha[0])

            if cutlass.const_expr(self.direct_acc_mbar):
                cute.arch.mbarrier_wait(acc_mbar_ptr, 0)
            else:
                acc_pipeline.consumer_wait(acc_consumer_state)

            LOG2_E = cutlass.Float32(1.4426950408889634)
            T_MAX = cutlass.Float32(126.0)
            subtile_cnt = cute.size(tTR_tAcc1.shape, mode=[3])
            if cutlass.const_expr(
                self.t2r_interleave_mode == 1
                and self.t2r_rep_x32
                and cute.size(tTR_rAcc1, mode=[2]) == 2
            ):
                cute.copy(
                    tiled_copy_t2r,
                    tTR_tAcc1[(None, None, 0, 0)],
                    tTR_rAcc1[(None, None, 0)],
                )
                cute.copy(
                    tiled_copy_t2r,
                    tTR_tAcc2[(None, None, 0, 0)],
                    tTR_rAcc2[(None, None, 0)],
                )
                cute.copy(
                    tiled_copy_t2r,
                    tTR_tAcc1[(None, None, 1, 0)],
                    tTR_rAcc1[(None, None, 1)],
                )
                cute.copy(
                    tiled_copy_t2r,
                    tTR_tAcc2[(None, None, 1, 0)],
                    tTR_rAcc2[(None, None, 1)],
                )
                acc_vec1 = tTR_rAcc1.load()
                acc_vec2 = tTR_rAcc2.load()
                if cutlass.const_expr(self.has_alpha):
                    acc_vec1 = acc_vec1 * alpha_val
                    acc_vec2 = acc_vec2 * alpha_val
            elif cutlass.const_expr(
                not (
                    self.t2r_interleave_mode == 3
                    and self.t2r_rep_x32
                    and cute.size(tTR_rAcc1, mode=[2]) == 2
                )
            ):
                cute.copy(tiled_copy_t2r, tTR_tAcc1[(None, None, None, 0)], tTR_rAcc1)
                cute.copy(tiled_copy_t2r, tTR_tAcc2[(None, None, None, 0)], tTR_rAcc2)
                acc_vec1 = tTR_rAcc1.load()
                acc_vec2 = tTR_rAcc2.load()
                if cutlass.const_expr(self.has_alpha):
                    acc_vec1 = acc_vec1 * alpha_val
                    acc_vec2 = acc_vec2 * alpha_val

            for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                if cutlass.const_expr(
                    self.t2r_interleave_mode == 3
                    and self.t2r_rep_x32
                    and cute.size(tTR_rAcc1, mode=[2]) == 2
                ):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc1[(None, None, 0, subtile_idx)],
                        tTR_rAcc1[(None, None, 0)],
                    )
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc2[(None, None, 0, subtile_idx)],
                        tTR_rAcc2[(None, None, 0)],
                    )
                    acc_vec1 = tTR_rAcc1[(None, None, 0)].load()
                    acc_vec2 = tTR_rAcc2[(None, None, 0)].load()
                    if cutlass.const_expr(self.has_alpha):
                        acc_vec1 = acc_vec1 * alpha_val
                        acc_vec2 = acc_vec2 * alpha_val
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc1[(None, None, 1, subtile_idx)],
                        tTR_rAcc1[(None, None, 1)],
                    )
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc2[(None, None, 1, subtile_idx)],
                        tTR_rAcc2[(None, None, 1)],
                    )
                elif cutlass.const_expr(subtile_idx > 0):
                    if cutlass.const_expr(
                        (not self.preload_subtile1_t2r) and (self.sub1_ldtm_mode == 0)
                    ):
                        if cutlass.const_expr(
                            self.t2r_interleave_mode in (1, 2)
                            and self.t2r_rep_x32
                            and cute.size(tTR_rAcc1, mode=[2]) == 2
                        ):
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc1[(None, None, 0, subtile_idx)],
                                tTR_rAcc1[(None, None, 0)],
                            )
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc2[(None, None, 0, subtile_idx)],
                                tTR_rAcc2[(None, None, 0)],
                            )
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc1[(None, None, 1, subtile_idx)],
                                tTR_rAcc1[(None, None, 1)],
                            )
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc2[(None, None, 1, subtile_idx)],
                                tTR_rAcc2[(None, None, 1)],
                            )
                        else:
                            tTR_tAcc1_mn = tTR_tAcc1[(None, None, None, subtile_idx)]
                            tTR_tAcc2_mn = tTR_tAcc2[(None, None, None, subtile_idx)]
                            cute.copy(tiled_copy_t2r, tTR_tAcc1_mn, tTR_rAcc1)
                            cute.copy(tiled_copy_t2r, tTR_tAcc2_mn, tTR_rAcc2)
                    elif cutlass.const_expr(self.sub1_ldtm_mode == 1):
                        tTR_tAcc2_mn = tTR_tAcc2[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc2_mn, tTR_rAcc2)
                    acc_vec1 = tTR_rAcc1.load()
                    acc_vec2 = tTR_rAcc2.load()
                    if cutlass.const_expr(self.has_alpha):
                        acc_vec1 = acc_vec1 * alpha_val
                        acc_vec2 = acc_vec2 * alpha_val
                    if cutlass.const_expr(
                        self.pre_fence_dealloc_arv
                        and use_2cta_instrs
                        and self.early_peer_dealloc_mode == 2
                        and (subtile_idx == subtile_cnt - 1)
                    ):
                        _arrive_peer_dealloc_warp_wait_ld(
                            tmem._two_cta_tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                        )
                    if cutlass.const_expr(
                        self.split_tmem_dealloc and (subtile_idx == subtile_cnt - 1)
                    ):
                        self.epilog_sub1_barrier.arrive_and_wait()
                        if warp_idx == self.epilog_warp_id[0]:
                            if cutlass.const_expr(use_2cta_instrs):
                                cute.arch.mbarrier_arrive(
                                    tmem._two_cta_tmem_dealloc_mbar_ptr,
                                    cta_rank_in_cluster ^ 1,
                                )

                if cutlass.const_expr(
                    self.early_tmem_free and (subtile_idx == subtile_cnt - 1)
                ):
                    if cutlass.const_expr(not self.early_relinquish):
                        tmem.relinquish_alloc_permit()
                    self.epilog_sync_barrier.arrive_and_wait()
                    tmem.free(acc_tmem_ptr)

                if cutlass.const_expr(self.swiglu_group4 and self.nr_steps == 2):
                    for g in cutlass.range_constexpr(0, cute.size(tTR_rAcc1), 8):
                        if cutlass.const_expr(
                            self.t2r_interleave_mode == 3
                            and self.t2r_rep_x32
                            and cute.size(tTR_rAcc1, mode=[2]) == 2
                            and g == 32
                        ):
                            acc_vec1 = tTR_rAcc1[(None, None, 1)].load()
                            acc_vec2 = tTR_rAcc2[(None, None, 1)].load()
                            if cutlass.const_expr(self.has_alpha):
                                acc_vec1 = acc_vec1 * alpha_val
                                acc_vec2 = acc_vec2 * alpha_val
                        gi = (
                            (g - 32)
                            if (
                                self.t2r_interleave_mode == 3
                                and self.t2r_rep_x32
                                and cute.size(tTR_rAcc1, mode=[2]) == 2
                                and g >= 32
                            )
                            else g
                        )

                        tl0 = cute.arch.mul_packed_f32x2(
                            (acc_vec1[gi], acc_vec1[gi + 1]), (-LOG2_E, -LOG2_E)
                        )
                        tl1 = cute.arch.mul_packed_f32x2(
                            (acc_vec1[gi + 2], acc_vec1[gi + 3]), (-LOG2_E, -LOG2_E)
                        )
                        tl2 = cute.arch.mul_packed_f32x2(
                            (acc_vec1[gi + 4], acc_vec1[gi + 5]), (-LOG2_E, -LOG2_E)
                        )
                        tl3 = cute.arch.mul_packed_f32x2(
                            (acc_vec1[gi + 6], acc_vec1[gi + 7]), (-LOG2_E, -LOG2_E)
                        )

                        if cutlass.const_expr(
                            (self.rcp_pair_mode in (9, 10))
                            or (self.rcp_pair_mode == 11 and ((g // 2) % 3 == 0))
                            or (
                                self.rcp_pair_mode == 0
                                and self.hybrid_rcp_mod > 0
                                and (
                                    (g // 2) % self.hybrid_rcp_mod
                                    == self.hybrid_rcp_offset
                                )
                            )
                        ):
                            t0_0, t1_0 = tl0[0], tl0[1]
                        else:
                            t0_0 = cute.arch.fmin(tl0[0], T_MAX)
                            t1_0 = cute.arch.fmin(tl0[1], T_MAX)
                        if cutlass.const_expr(
                            (self.rcp_pair_mode == 11 and ((g // 2 + 1) % 3 == 0))
                            or (
                                self.rcp_pair_mode == 0
                                and self.hybrid_rcp_mod > 0
                                and (
                                    (g // 2 + 1) % self.hybrid_rcp_mod
                                    == self.hybrid_rcp_offset
                                )
                            )
                        ):
                            t0_1, t1_1 = tl1[0], tl1[1]
                        else:
                            t0_1 = cute.arch.fmin(tl1[0], T_MAX)
                            t1_1 = cute.arch.fmin(tl1[1], T_MAX)
                        if cutlass.const_expr(
                            (self.rcp_pair_mode == 9 and g == 0)
                            or (self.rcp_pair_mode == 10 and (g == 0 or g == 32))
                            or (self.rcp_pair_mode == 11 and ((g // 2 + 2) % 3 == 0))
                            or (
                                self.rcp_pair_mode == 0
                                and self.hybrid_rcp_mod > 0
                                and (
                                    (g // 2 + 2) % self.hybrid_rcp_mod
                                    == self.hybrid_rcp_offset
                                )
                            )
                        ):
                            t0_2, t1_2 = tl2[0], tl2[1]
                        else:
                            t0_2 = cute.arch.fmin(tl2[0], T_MAX)
                            t1_2 = cute.arch.fmin(tl2[1], T_MAX)
                        if cutlass.const_expr(
                            (self.rcp_pair_mode == 11 and ((g // 2 + 3) % 3 == 0))
                            or (
                                self.rcp_pair_mode == 0
                                and self.hybrid_rcp_mod > 0
                                and (
                                    (g // 2 + 3) % self.hybrid_rcp_mod
                                    == self.hybrid_rcp_offset
                                )
                            )
                        ):
                            t0_3, t1_3 = tl3[0], tl3[1]
                        else:
                            t0_3 = cute.arch.fmin(tl3[0], T_MAX)
                            t1_3 = cute.arch.fmin(tl3[1], T_MAX)

                        e0_0 = cute.math.exp2(t0_0, fastmath=True)
                        e1_0 = cute.math.exp2(t1_0, fastmath=True)
                        e0_1 = cute.math.exp2(t0_1, fastmath=True)
                        e1_1 = cute.math.exp2(t1_1, fastmath=True)
                        e0_2 = cute.math.exp2(t0_2, fastmath=True)
                        e1_2 = cute.math.exp2(t1_2, fastmath=True)
                        e0_3 = cute.math.exp2(t0_3, fastmath=True)
                        e1_3 = cute.math.exp2(t1_3, fastmath=True)

                        p0_0, p1_0 = cute.arch.mul_packed_f32x2(
                            (acc_vec1[gi], acc_vec1[gi + 1]),
                            (acc_vec2[gi], acc_vec2[gi + 1]),
                        )
                        p0_1, p1_1 = cute.arch.mul_packed_f32x2(
                            (acc_vec1[gi + 2], acc_vec1[gi + 3]),
                            (acc_vec2[gi + 2], acc_vec2[gi + 3]),
                        )
                        p0_2, p1_2 = cute.arch.mul_packed_f32x2(
                            (acc_vec1[gi + 4], acc_vec1[gi + 5]),
                            (acc_vec2[gi + 4], acc_vec2[gi + 5]),
                        )
                        p0_3, p1_3 = cute.arch.mul_packed_f32x2(
                            (acc_vec1[gi + 6], acc_vec1[gi + 7]),
                            (acc_vec2[gi + 6], acc_vec2[gi + 7]),
                        )

                        if cutlass.const_expr(
                            (self.rcp_pair_mode in (9, 10))
                            or (self.rcp_pair_mode == 11 and ((g // 2) % 3 == 0))
                            or (
                                self.rcp_pair_mode == 0
                                and self.hybrid_rcp_mod > 0
                                and (
                                    (g // 2) % self.hybrid_rcp_mod
                                    == self.hybrid_rcp_offset
                                )
                            )
                        ):
                            d0_0, d1_0 = cute.arch.add_packed_f32x2(
                                (e0_0, e1_0), (1.0, 1.0)
                            )
                            s0_0 = cute.arch.rcp_approx(d0_0)
                            s1_0 = cute.arch.rcp_approx(d1_0)
                            tTR_rAcc1[g], tTR_rAcc1[g + 1] = cute.arch.mul_packed_f32x2(
                                (s0_0, s1_0), (p0_0, p1_0)
                            )
                        else:
                            nd0_0, nd1_0 = cute.arch.fma_packed_f32x2(
                                (e0_0, e1_0), (-1.0, -1.0), (-1.0, -1.0)
                            )
                            s0_0 = _rcp_seed_neg(nd0_0)
                            s1_0 = _rcp_seed_neg(nd1_0)
                            u0_0, u1_0 = cute.arch.fma_packed_f32x2(
                                (nd0_0, nd1_0), (s0_0, s1_0), (2.0, 2.0)
                            )
                            s0_0, s1_0 = cute.arch.mul_packed_f32x2(
                                (s0_0, s1_0), (u0_0, u1_0)
                            )
                            v0_0, v1_0 = cute.arch.fma_packed_f32x2(
                                (nd0_0, nd1_0), (s0_0, s1_0), (2.0, 2.0)
                            )
                            sp0_0, sp1_0 = cute.arch.mul_packed_f32x2(
                                (s0_0, s1_0), (p0_0, p1_0)
                            )
                            tTR_rAcc1[g], tTR_rAcc1[g + 1] = cute.arch.mul_packed_f32x2(
                                (sp0_0, sp1_0), (v0_0, v1_0)
                            )

                        if cutlass.const_expr(
                            (self.rcp_pair_mode == 11 and ((g // 2 + 1) % 3 == 0))
                            or (
                                self.rcp_pair_mode == 0
                                and self.hybrid_rcp_mod > 0
                                and (
                                    (g // 2 + 1) % self.hybrid_rcp_mod
                                    == self.hybrid_rcp_offset
                                )
                            )
                        ):
                            d0_1, d1_1 = cute.arch.add_packed_f32x2(
                                (e0_1, e1_1), (1.0, 1.0)
                            )
                            s0_1 = cute.arch.rcp_approx(d0_1)
                            s1_1 = cute.arch.rcp_approx(d1_1)
                            tTR_rAcc1[g + 2], tTR_rAcc1[g + 3] = (
                                cute.arch.mul_packed_f32x2((s0_1, s1_1), (p0_1, p1_1))
                            )
                        else:
                            nd0_1, nd1_1 = cute.arch.fma_packed_f32x2(
                                (e0_1, e1_1), (-1.0, -1.0), (-1.0, -1.0)
                            )
                            s0_1 = _rcp_seed_neg(nd0_1)
                            s1_1 = _rcp_seed_neg(nd1_1)
                            u0_1, u1_1 = cute.arch.fma_packed_f32x2(
                                (nd0_1, nd1_1), (s0_1, s1_1), (2.0, 2.0)
                            )
                            s0_1, s1_1 = cute.arch.mul_packed_f32x2(
                                (s0_1, s1_1), (u0_1, u1_1)
                            )
                            v0_1, v1_1 = cute.arch.fma_packed_f32x2(
                                (nd0_1, nd1_1), (s0_1, s1_1), (2.0, 2.0)
                            )
                            sp0_1, sp1_1 = cute.arch.mul_packed_f32x2(
                                (s0_1, s1_1), (p0_1, p1_1)
                            )
                            tTR_rAcc1[g + 2], tTR_rAcc1[g + 3] = (
                                cute.arch.mul_packed_f32x2((sp0_1, sp1_1), (v0_1, v1_1))
                            )

                        if cutlass.const_expr(
                            (self.rcp_pair_mode == 9 and g == 0)
                            or (self.rcp_pair_mode == 10 and (g == 0 or g == 32))
                            or (self.rcp_pair_mode == 11 and ((g // 2 + 2) % 3 == 0))
                            or (
                                self.rcp_pair_mode == 0
                                and self.hybrid_rcp_mod > 0
                                and (
                                    (g // 2 + 2) % self.hybrid_rcp_mod
                                    == self.hybrid_rcp_offset
                                )
                            )
                        ):
                            d0_2, d1_2 = cute.arch.add_packed_f32x2(
                                (e0_2, e1_2), (1.0, 1.0)
                            )
                            s0_2 = cute.arch.rcp_approx(d0_2)
                            s1_2 = cute.arch.rcp_approx(d1_2)
                            tTR_rAcc1[g + 4], tTR_rAcc1[g + 5] = (
                                cute.arch.mul_packed_f32x2((s0_2, s1_2), (p0_2, p1_2))
                            )
                        else:
                            nd0_2, nd1_2 = cute.arch.fma_packed_f32x2(
                                (e0_2, e1_2), (-1.0, -1.0), (-1.0, -1.0)
                            )
                            s0_2 = _rcp_seed_neg(nd0_2)
                            s1_2 = _rcp_seed_neg(nd1_2)
                            u0_2, u1_2 = cute.arch.fma_packed_f32x2(
                                (nd0_2, nd1_2), (s0_2, s1_2), (2.0, 2.0)
                            )
                            s0_2, s1_2 = cute.arch.mul_packed_f32x2(
                                (s0_2, s1_2), (u0_2, u1_2)
                            )
                            v0_2, v1_2 = cute.arch.fma_packed_f32x2(
                                (nd0_2, nd1_2), (s0_2, s1_2), (2.0, 2.0)
                            )
                            sp0_2, sp1_2 = cute.arch.mul_packed_f32x2(
                                (s0_2, s1_2), (p0_2, p1_2)
                            )
                            tTR_rAcc1[g + 4], tTR_rAcc1[g + 5] = (
                                cute.arch.mul_packed_f32x2((sp0_2, sp1_2), (v0_2, v1_2))
                            )

                        if cutlass.const_expr(
                            (self.rcp_pair_mode == 11 and ((g // 2 + 3) % 3 == 0))
                            or (
                                self.rcp_pair_mode == 0
                                and self.hybrid_rcp_mod > 0
                                and (
                                    (g // 2 + 3) % self.hybrid_rcp_mod
                                    == self.hybrid_rcp_offset
                                )
                            )
                        ):
                            d0_3, d1_3 = cute.arch.add_packed_f32x2(
                                (e0_3, e1_3), (1.0, 1.0)
                            )
                            s0_3 = cute.arch.rcp_approx(d0_3)
                            s1_3 = cute.arch.rcp_approx(d1_3)
                            tTR_rAcc1[g + 6], tTR_rAcc1[g + 7] = (
                                cute.arch.mul_packed_f32x2((s0_3, s1_3), (p0_3, p1_3))
                            )
                        else:
                            nd0_3, nd1_3 = cute.arch.fma_packed_f32x2(
                                (e0_3, e1_3), (-1.0, -1.0), (-1.0, -1.0)
                            )
                            s0_3 = _rcp_seed_neg(nd0_3)
                            s1_3 = _rcp_seed_neg(nd1_3)
                            u0_3, u1_3 = cute.arch.fma_packed_f32x2(
                                (nd0_3, nd1_3), (s0_3, s1_3), (2.0, 2.0)
                            )
                            s0_3, s1_3 = cute.arch.mul_packed_f32x2(
                                (s0_3, s1_3), (u0_3, u1_3)
                            )
                            v0_3, v1_3 = cute.arch.fma_packed_f32x2(
                                (nd0_3, nd1_3), (s0_3, s1_3), (2.0, 2.0)
                            )
                            sp0_3, sp1_3 = cute.arch.mul_packed_f32x2(
                                (s0_3, s1_3), (p0_3, p1_3)
                            )
                            tTR_rAcc1[g + 6], tTR_rAcc1[g + 7] = (
                                cute.arch.mul_packed_f32x2((sp0_3, sp1_3), (v0_3, v1_3))
                            )
                        if cutlass.const_expr(self.inline_swz_r2s):
                            _pack_st_f16x8(
                                swz_addrs[g // 8],
                                tTR_rAcc1[g],
                                tTR_rAcc1[g + 1],
                                tTR_rAcc1[g + 2],
                                tTR_rAcc1[g + 3],
                                tTR_rAcc1[g + 4],
                                tTR_rAcc1[g + 5],
                                tTR_rAcc1[g + 6],
                                tTR_rAcc1[g + 7],
                                subtile_idx,
                            )
                        if cutlass.const_expr(
                            self.pre_fence_dealloc_arv
                            and use_2cta_instrs
                            and (
                                self.early_peer_dealloc_mode == 1
                                or self.relaxed_peer_dealloc == 2
                            )
                            and subtile_idx == 1
                            and g == 0
                        ):
                            if cutlass.const_expr(self.relaxed_peer_dealloc == 2):
                                _arrive_peer_dealloc_warp_relaxed(
                                    tmem._two_cta_tmem_dealloc_mbar_ptr,
                                    cta_rank_in_cluster ^ 1,
                                )
                            else:
                                _arrive_peer_dealloc_warp(
                                    tmem._two_cta_tmem_dealloc_mbar_ptr,
                                    cta_rank_in_cluster ^ 1,
                                )
                else:
                    for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc1), 2):
                        pair_idx = i // 2
                        t_log2e = cute.arch.mul_packed_f32x2(
                            (acc_vec1[i], acc_vec1[i + 1]),
                            (-LOG2_E, -LOG2_E),
                        )
                        if cutlass.const_expr(
                            self.hybrid_rcp_mod > 0
                            and (pair_idx % self.hybrid_rcp_mod == 0)
                        ):
                            e0 = cute.math.exp2(t_log2e[0], fastmath=True)
                            e1 = cute.math.exp2(t_log2e[1], fastmath=True)
                            prod0, prod1 = cute.arch.mul_packed_f32x2(
                                (acc_vec1[i], acc_vec1[i + 1]),
                                (acc_vec2[i], acc_vec2[i + 1]),
                            )
                            denom0, denom1 = cute.arch.add_packed_f32x2(
                                (e0, e1),
                                (1.0, 1.0),
                            )
                            sig0 = cute.arch.rcp_approx(denom0)
                            sig1 = cute.arch.rcp_approx(denom1)
                            tTR_rAcc1[i], tTR_rAcc1[i + 1] = cute.arch.mul_packed_f32x2(
                                (sig0, sig1),
                                (prod0, prod1),
                            )
                        elif cutlass.const_expr(self.nr_steps > 0):
                            if cutlass.const_expr(self.no_fmin_clamp):
                                t0, t1 = t_log2e[0], t_log2e[1]
                            else:
                                t0 = cute.arch.fmin(t_log2e[0], T_MAX)
                                t1 = cute.arch.fmin(t_log2e[1], T_MAX)
                            e0 = cute.math.exp2(t0, fastmath=True)
                            e1 = cute.math.exp2(t1, fastmath=True)
                            if cutlass.const_expr(
                                self.swiglu_ilp and self.nr_steps == 2
                            ):
                                prod0, prod1 = cute.arch.mul_packed_f32x2(
                                    (acc_vec1[i], acc_vec1[i + 1]),
                                    (acc_vec2[i], acc_vec2[i + 1]),
                                )
                                nd0, nd1 = cute.arch.fma_packed_f32x2(
                                    (e0, e1),
                                    (-1.0, -1.0),
                                    (-1.0, -1.0),
                                )
                                sig0 = _rcp_seed_neg(nd0)
                                sig1 = _rcp_seed_neg(nd1)
                                u0_0, u1_0 = cute.arch.fma_packed_f32x2(
                                    (nd0, nd1), (sig0, sig1), (2.0, 2.0)
                                )
                                sig0, sig1 = cute.arch.mul_packed_f32x2(
                                    (sig0, sig1), (u0_0, u1_0)
                                )
                                u0_1, u1_1 = cute.arch.fma_packed_f32x2(
                                    (nd0, nd1), (sig0, sig1), (2.0, 2.0)
                                )
                                sp0, sp1 = cute.arch.mul_packed_f32x2(
                                    (sig0, sig1), (prod0, prod1)
                                )
                                tTR_rAcc1[i], tTR_rAcc1[i + 1] = (
                                    cute.arch.mul_packed_f32x2((sp0, sp1), (u0_1, u1_1))
                                )
                            else:
                                nd0, nd1 = cute.arch.fma_packed_f32x2(
                                    (e0, e1),
                                    (-1.0, -1.0),
                                    (-1.0, -1.0),
                                )
                                sig0 = _rcp_seed_neg(nd0)
                                sig1 = _rcp_seed_neg(nd1)
                                for _nr in cutlass.range_constexpr(self.nr_steps):
                                    u0, u1 = cute.arch.fma_packed_f32x2(
                                        (nd0, nd1), (sig0, sig1), (2.0, 2.0)
                                    )
                                    sig0, sig1 = cute.arch.mul_packed_f32x2(
                                        (sig0, sig1), (u0, u1)
                                    )
                                silu0, silu1 = cute.arch.mul_packed_f32x2(
                                    (sig0, sig1),
                                    (acc_vec1[i], acc_vec1[i + 1]),
                                )
                                tTR_rAcc1[i], tTR_rAcc1[i + 1] = (
                                    cute.arch.mul_packed_f32x2(
                                        (silu0, silu1),
                                        (acc_vec2[i], acc_vec2[i + 1]),
                                    )
                                )
                        else:
                            denom0, denom1 = cute.arch.add_packed_f32x2(
                                (
                                    cute.math.exp2(t_log2e[0], fastmath=True),
                                    cute.math.exp2(t_log2e[1], fastmath=True),
                                ),
                                (1.0, 1.0),
                            )
                            sig0 = cute.arch.rcp_approx(denom0)
                            sig1 = cute.arch.rcp_approx(denom1)
                            silu0, silu1 = cute.arch.mul_packed_f32x2(
                                (sig0, sig1),
                                (acc_vec1[i], acc_vec1[i + 1]),
                            )
                            tTR_rAcc1[i], tTR_rAcc1[i + 1] = cute.arch.mul_packed_f32x2(
                                (silu0, silu1),
                                (acc_vec2[i], acc_vec2[i + 1]),
                            )

                if cutlass.const_expr(not self.inline_swz_r2s):
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc1).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))
                    cute.copy(
                        tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, subtile_idx)]
                    )

                if cutlass.const_expr(
                    self.preload_subtile1_t2r
                    and (subtile_cnt > 1)
                    and (subtile_idx == 0)
                ):
                    cute.copy(
                        tiled_copy_t2r, tTR_tAcc1[(None, None, None, 1)], tTR_rAcc1
                    )
                    cute.copy(
                        tiled_copy_t2r, tTR_tAcc2[(None, None, None, 1)], tTR_rAcc2
                    )

                if cutlass.const_expr(
                    self.pipe_subtile_store and (subtile_cnt > 1) and (subtile_idx == 0)
                ):
                    sub0_c_wid = (
                        self.epilog_warp_id[2]
                        if self.split_c_store_warps
                        else self.epilog_warp_id[1]
                    )
                    if cutlass.const_expr(
                        self.pre_bar_pdl_wait
                        and self.use_pdl
                        and (not self.pdl_safe)
                        and (not self.pre_wait_pdl)
                    ):
                        if warp_idx == sub0_c_wid:
                            cute.arch.griddepcontrol_wait()
                    if cutlass.const_expr(self.epilog_fence_mode != 2):
                        cute.arch.fence_view_async_shared()
                    if cutlass.const_expr(self.async_epilog_bar):
                        if warp_idx == sub0_c_wid:
                            self.epilog_sync_barrier.arrive_and_wait()
                        else:
                            self.epilog_sync_barrier.arrive_unaligned()
                    else:
                        self.epilog_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(self.sub1_ldtm_mode == 1):
                        cute.copy(
                            tiled_copy_t2r, tTR_tAcc1[(None, None, None, 1)], tTR_rAcc1
                        )
                    elif cutlass.const_expr(self.sub1_ldtm_mode == 2):
                        cute.copy(
                            tiled_copy_t2r, tTR_tAcc1[(None, None, None, 1)], tTR_rAcc1
                        )
                        cute.copy(
                            tiled_copy_t2r, tTR_tAcc2[(None, None, None, 1)], tTR_rAcc2
                        )
                    if warp_idx == sub0_c_wid:
                        if cutlass.const_expr(self.epilog_fence_mode == 2):
                            cute.arch.fence_view_async_shared()
                        if cutlass.const_expr(
                            (not self.pre_bar_pdl_wait)
                            and self.use_pdl
                            and (not self.pdl_safe)
                            and (not self.pre_wait_pdl)
                            and ((self.pdl_wait_mode == 0) or (not self.pf_row0_only))
                        ):
                            cute.arch.griddepcontrol_wait()
                        if cutlass.const_expr(self.warp1_only_tma_c_part):
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
                            cute.copy(
                                tma_atom_c,
                                bSG_sC_w0[(None, 0)],
                                bSG_gC_w0[(None, 0)],
                                cache_policy=c_pol,
                            )
                        else:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, 0)],
                                bSG_gC[(None, 0)],
                                cache_policy=c_pol,
                            )

            if cutlass.const_expr(
                self.pre_fence_dealloc_arv
                and use_2cta_instrs
                and self.early_peer_dealloc_mode == 0
                and self.relaxed_peer_dealloc != 2
            ):
                if cutlass.const_expr(self.relaxed_peer_dealloc == 1):
                    _arrive_peer_dealloc_warp_relaxed(
                        tmem._two_cta_tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                else:
                    _arrive_peer_dealloc_warp(
                        tmem._two_cta_tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
            if cutlass.const_expr(self.epilog_fence_mode == 0):
                cute.arch.fence_view_async_shared()
            if cutlass.const_expr(not self.early_tmem_free):
                if cutlass.const_expr(not self.early_relinquish):
                    tmem.relinquish_alloc_permit()
                if cutlass.const_expr(
                    self.split_tmem_dealloc
                    and self.pipe_subtile_store
                    and (subtile_cnt > 1)
                ):
                    if warp_idx == self.epilog_warp_id[1]:
                        self.epilog_sync_barrier.arrive_and_wait()
                    else:
                        self.epilog_sync_barrier.arrive_unaligned()
                    if warp_idx == self.epilog_warp_id[0]:
                        if cutlass.const_expr(use_2cta_instrs):
                            cute.arch.mbarrier_wait(
                                tmem._two_cta_tmem_dealloc_mbar_ptr, 0
                            )
                        cute.arch.dealloc_tmem(
                            acc_tmem_ptr,
                            self.num_tmem_alloc_cols,
                            is_two_cta=use_2cta_instrs,
                            arch="sm_100",
                        )
                else:
                    if cutlass.const_expr(
                        self.async_epilog_bar
                        and self.pipe_subtile_store
                        and (subtile_cnt > 1)
                    ):
                        if warp_idx <= self.epilog_warp_id[1]:
                            self.epilog_sub1_barrier.arrive_and_wait()
                        else:
                            self.epilog_sub1_barrier.arrive_unaligned()
                    elif cutlass.const_expr(
                        self.sub1_bar_mode == 1
                        and self.pipe_subtile_store
                        and (subtile_cnt > 1)
                    ):
                        if warp_idx <= self.epilog_warp_id[1]:
                            self.epilog_sync_barrier.arrive_and_wait()
                        else:
                            self.epilog_sync_barrier.arrive_unaligned()
                    else:
                        self.epilog_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(self.pre_fence_dealloc_arv):
                        if warp_idx == self.epilog_warp_id[0]:
                            if cutlass.const_expr(use_2cta_instrs):
                                cute.arch.mbarrier_wait(
                                    tmem._two_cta_tmem_dealloc_mbar_ptr, 0
                                )
                            cute.arch.dealloc_tmem(
                                acc_tmem_ptr,
                                self.num_tmem_alloc_cols,
                                is_two_cta=use_2cta_instrs,
                                arch="sm_100",
                            )
                    else:
                        tmem.free(acc_tmem_ptr)
            else:
                self.epilog_sync_barrier.arrive_and_wait()

            if warp_idx == self.epilog_warp_id[1]:
                if cutlass.const_expr(self.epilog_fence_mode in (1, 2)):
                    cute.arch.fence_view_async_shared()
                if cutlass.const_expr(
                    self.use_pdl
                    and (not self.pdl_safe)
                    and (not self.pre_wait_pdl)
                    and (
                        (not (self.pipe_subtile_store and (subtile_cnt > 1)))
                        or ((self.pdl_wait_mode == 1) and self.pf_row0_only)
                    )
                ):
                    cute.arch.griddepcontrol_wait()
                start_sub = 1 if (self.pipe_subtile_store and (subtile_cnt > 1)) else 0
                if cutlass.const_expr(self.warp1_only_tma_c_part):
                    sC_for_tma_w1 = cute.group_modes(sC, 0, 2)
                    gC_for_tma_w1 = cute.group_modes(gC_mnl_epi, 0, 2)
                    bSG_sC_w1, bSG_gC_part_w1 = cpasync.tma_partition(
                        tma_atom_c, 0, cute.make_layout(1), sC_for_tma_w1, gC_for_tma_w1
                    )
                    bSG_gC_slice_w1 = bSG_gC_part_w1[
                        (None, None, None, *mma_tile_coord_mnl)
                    ]
                    bSG_gC_w1 = cute.group_modes(
                        bSG_gC_slice_w1, 1, cute.rank(bSG_gC_slice_w1)
                    )
                    for subtile_idx in cutlass.range_constexpr(start_sub, subtile_cnt):
                        cute.copy(
                            tma_atom_c,
                            bSG_sC_w1[(None, subtile_idx)],
                            bSG_gC_w1[(None, subtile_idx)],
                            cache_policy=c_pol,
                        )
                else:
                    for subtile_idx in cutlass.range_constexpr(start_sub, subtile_cnt):
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, subtile_idx)],
                            bSG_gC[(None, subtile_idx)],
                            cache_policy=c_pol,
                        )
                if cutlass.const_expr(self.store_drain):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

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


_FAST_CACHE = {}
_KEEPALIVE = []


def _to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = input_matrix.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    if rows != n_row_blocks * 128 or cols != n_col_blocks * 4:
        padded = torch.zeros(
            (n_row_blocks * 128, n_col_blocks * 4),
            dtype=input_matrix.dtype,
            device=input_matrix.device,
        )
        padded[:rows, :cols] = input_matrix
        input_matrix = padded
    blocks = input_matrix.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


# CANDIDATE_CONFIG_JSON: {"alg_mbar_mode": 0, "async_epilog_bar": false, "clean_c_prologue": true, "coalesce_warp0_tmem": false, "const_cluster_sync": false, "const_mma_fields": true, "const_prolog_pf": false, "const_tmem_bar": true, "cta_desc_pf": true, "db_tmem_sf": false, "defer_mma_tmem_bar": true, "defer_pipeline_sync": true, "direct_acc_mbar": true, "early_pdl_launch": true, "early_peer_dealloc_mode": 0, "early_relinquish": true, "early_tmem_alloc": false, "early_tmem_free": false, "elide_mma_tail_release": 0, "epilog_fence_mode": 2, "first_inst_pdl_all_m": false, "first_inst_pdl_launch": false, "first_inst_pdl_m512": true, "fused_mma_tail_mode": 2, "hint_a_mode": 1, "hint_b_last_row_mode": false, "hint_c": true, "hoist_cta_prolog_b_pf": false, "hybrid_rcp_mod": 4, "hybrid_rcp_offset": 0, "inline_swz_r2s": true, "interleave_m_row_pf": true, "leader_tmem_bar": true, "mma_leader_only": true, "mma_tmem_bar_after_k0_wait": false, "no_fmin_clamp": false, "pdl_wait_mode": 2, "peel_acc_init": true, "peel_mma_init_try": true, "peel_tma_init_try": true, "pf_a_all_m": true, "pf_a_bidy0": true, "pf_a_bidy0_ml": true, "pf_a_prolog_mode": 0, "pf_a_warp0": false, "pf_head_mode": 1, "pf_head_warp2": true, "pipe_subtile_store": true, "pre_bar_pdl_wait": false, "pre_fence_dealloc_arv": true, "pre_wait_pdl": false, "preload_subtile1_t2r": false, "prune_dead_smem": false, "rcp_pair_mode": 0, "relaxed_peer_dealloc": 1, "s2t_interleave_mode": 3, "shape_pf_mode": 0, "single_mbar_init": true, "single_swz_base": true, "split_bidy_pf_a_mode": 3, "split_c_store_warps": false, "split_mrow_ml_pf_mode": 2, "split_tmem_dealloc": false, "static_cmask_mode": 1, "sub1_bar_mode": 0, "sub1_ldtm_mode": 0, "swiglu_group4": true, "swiglu_ilp": true, "t2r_interleave_mode": 0, "t2r_rep_x32": true, "tma_cta_split_all_m": true, "tma_cta_split_m256": true, "tma_loop_split_mode": 0, "tma_order": 0, "unified_mma_loop": true, "uniform_coord_mode": 2, "warp1_only_tma_c_part": false, "warp_scoped_ab_frag": false, "warp_spec_parts": false}
def _get_fast_runner(shape: Tuple[int, int, int, int]):
    runner = _FAST_CACHE.get(shape)
    if runner is not None:
        return runner

    m, n, k, l = shape
    if (m % 256 == 0) and (n % 128 == 0):
        cfg = dict(
            tiler=(256, 128),
            cluster=(2, 1),
            stages=5,
            c_stages=2,
            fuse_b=True,
            k_tile=256,
            hint_b=(m <= 256),
            hint_b_last_row=-1,
            hint_c=True,
            pf_dist=(3 if m <= 256 else 4),
            pf_prolog=False,
            pf_head=(3 if m <= 256 else 4),
            pf_row0_only=(m > 256),
            pf_head_warp2=True,
            cta_desc_pf=True,
            early_pdl_launch=True,
            tma_order=0,
            mma_leader_only=True,
            peel_acc_init=True,
            db_tmem_sf=False,
            early_relinquish=True,
            pre_wait_pdl=False,
            pipe_subtile_store=True,
            early_tmem_free=False,
            defer_pipeline_sync=True,
            first_inst_pdl_launch=(m > 256),
            early_tmem_alloc=False,
            pf_head_mode=1,
            pf_a_bidy0=True,
            async_epilog_bar=False,
            preload_subtile1_t2r=False,
            swiglu_ilp=True,
            no_fmin_clamp=False,
            warp_spec_parts=False,
            split_tmem_dealloc=False,
            hybrid_rcp_mod=4,
            direct_acc_mbar=True,
            const_mma_fields=True,
            swiglu_group4=True,
            pre_bar_pdl_wait=False,
            pf_a_bidy0_ml=True,
            peel_tma_init_try=True,
            leader_tmem_bar=True,
            pf_a_warp0=False,
            s2t_interleave_mode=3,
            t2r_rep_x32=True,
            inline_swz_r2s=True,
            pf_a_all_m=True,
            tma_cta_split_m256=True,
            single_mbar_init=True,
            const_tmem_bar=True,
            pre_fence_dealloc_arv=True,
            prune_dead_smem=False,
            pdl_wait_mode=2,
            unified_mma_loop=True,
            peel_mma_init_try=True,
            tma_cta_split_all_m=True,
            interleave_m_row_pf=True,
            split_bidy_pf_a_mode=3,
            single_swz_base=True,
            coalesce_warp0_tmem=False,
            split_mrow_ml_pf_mode=2,
            hoist_cta_prolog_b_pf=False,
            pf_a_prolog_mode=0,
            hybrid_rcp_offset=0,
            early_peer_dealloc_mode=0,
            sub1_bar_mode=0,
            hint_a=(m > 256),
            rcp_pair_mode=0,
            t2r_interleave_mode=0,
            const_prolog_pf=False,
            defer_mma_tmem_bar=True,
            split_c_store_warps=False,
            alg_mbar_mode=0,
            const_cluster_sync=False,
            tma_loop_split_mode=0,
            warp_scoped_ab_frag=False,
            mma_tmem_bar_after_k0_wait=False,
            elide_mma_tail_release=0,
            clean_c_prologue=True,
            warp1_only_tma_c_part=False,
            relaxed_peer_dealloc=1,
            epilog_fence_mode=2,
            fused_mma_tail_mode=2,
            static_cmask=True,
            uniform_coord=(1 if m > 256 else 0),
            sub1_ldtm_mode=0,
            nr_steps=2,
            store_drain=False,
            pdl_safe=False,
        )
    elif (m % 256 == 0) and (n % 64 == 0):
        cfg = dict(tiler=(256, 64), cluster=(2, 1), stages=5, c_stages=1)
    elif (m % 128 == 0) and (n % 128 == 0):
        cfg = dict(tiler=(128, 128), cluster=(1, 1), stages=3, c_stages=2)
    else:
        cfg = dict(tiler=(128, 64), cluster=(1, 1), stages=4, c_stages=1)

    kern = Sm100Nvfp4DualGemmFast(
        mma_tiler_mn=cfg["tiler"],
        cluster_shape_mn=cfg["cluster"],
        num_ab_stage=cfg["stages"],
        num_c_stage=cfg["c_stages"],
        use_pdl=True,
        fuse_b=cfg.get("fuse_b", False),
        k_tile=cfg.get("k_tile", 256),
        hint_b=cfg.get("hint_b", False),
        hint_b_last_row=cfg.get("hint_b_last_row", -1),
        hint_c=cfg.get("hint_c", False),
        pf_dist=cfg.get("pf_dist", 0),
        pf_prolog=cfg.get("pf_prolog", False),
        pf_a=cfg.get("pf_a", False),
        pf_head=cfg.get("pf_head", 0),
        pf_row0_only=cfg.get("pf_row0_only", False),
        pf_head_warp2=cfg.get("pf_head_warp2", False),
        cta_desc_pf=cfg.get("cta_desc_pf", False),
        early_pdl_launch=cfg.get("early_pdl_launch", False),
        tma_order=cfg.get("tma_order", 0),
        mma_leader_only=cfg.get("mma_leader_only", False),
        peel_acc_init=cfg.get("peel_acc_init", False),
        db_tmem_sf=cfg.get("db_tmem_sf", False),
        early_relinquish=cfg.get("early_relinquish", False),
        pre_wait_pdl=cfg.get("pre_wait_pdl", False),
        pipe_subtile_store=cfg.get("pipe_subtile_store", False),
        early_tmem_free=cfg.get("early_tmem_free", False),
        defer_pipeline_sync=cfg.get("defer_pipeline_sync", False),
        first_inst_pdl_launch=cfg.get("first_inst_pdl_launch", False),
        early_tmem_alloc=cfg.get("early_tmem_alloc", False),
        pf_head_mode=cfg.get("pf_head_mode", 0),
        pf_a_bidy0=cfg.get("pf_a_bidy0", False),
        async_epilog_bar=cfg.get("async_epilog_bar", False),
        preload_subtile1_t2r=cfg.get("preload_subtile1_t2r", False),
        swiglu_ilp=cfg.get("swiglu_ilp", False),
        no_fmin_clamp=cfg.get("no_fmin_clamp", False),
        warp_spec_parts=cfg.get("warp_spec_parts", False),
        split_tmem_dealloc=cfg.get("split_tmem_dealloc", False),
        hybrid_rcp_mod=cfg.get("hybrid_rcp_mod", 0),
        direct_acc_mbar=cfg.get("direct_acc_mbar", False),
        const_mma_fields=cfg.get("const_mma_fields", False),
        swiglu_group4=cfg.get("swiglu_group4", False),
        pre_bar_pdl_wait=cfg.get("pre_bar_pdl_wait", False),
        pf_a_bidy0_ml=cfg.get("pf_a_bidy0_ml", False),
        peel_tma_init_try=cfg.get("peel_tma_init_try", False),
        leader_tmem_bar=cfg.get("leader_tmem_bar", False),
        pf_a_warp0=cfg.get("pf_a_warp0", False),
        s2t_interleave_mode=cfg.get("s2t_interleave_mode", 0),
        t2r_rep_x32=cfg.get("t2r_rep_x32", False),
        inline_swz_r2s=cfg.get("inline_swz_r2s", False),
        pf_a_all_m=cfg.get("pf_a_all_m", False),
        tma_cta_split_m256=cfg.get("tma_cta_split_m256", False),
        single_mbar_init=cfg.get("single_mbar_init", False),
        const_tmem_bar=cfg.get("const_tmem_bar", False),
        pre_fence_dealloc_arv=cfg.get("pre_fence_dealloc_arv", False),
        prune_dead_smem=cfg.get("prune_dead_smem", False),
        pdl_wait_mode=cfg.get("pdl_wait_mode", 0),
        unified_mma_loop=cfg.get("unified_mma_loop", False),
        peel_mma_init_try=cfg.get("peel_mma_init_try", False),
        tma_cta_split_all_m=cfg.get("tma_cta_split_all_m", False),
        interleave_m_row_pf=cfg.get("interleave_m_row_pf", False),
        split_bidy_pf_a_mode=cfg.get("split_bidy_pf_a_mode", 0),
        single_swz_base=cfg.get("single_swz_base", False),
        coalesce_warp0_tmem=cfg.get("coalesce_warp0_tmem", False),
        split_mrow_ml_pf_mode=cfg.get("split_mrow_ml_pf_mode", 0),
        hoist_cta_prolog_b_pf=cfg.get("hoist_cta_prolog_b_pf", False),
        pf_a_prolog_mode=cfg.get("pf_a_prolog_mode", 0),
        hybrid_rcp_offset=cfg.get("hybrid_rcp_offset", 0),
        early_peer_dealloc_mode=cfg.get("early_peer_dealloc_mode", 0),
        sub1_bar_mode=cfg.get("sub1_bar_mode", 0),
        hint_a=cfg.get("hint_a", False),
        rcp_pair_mode=cfg.get("rcp_pair_mode", 0),
        t2r_interleave_mode=cfg.get("t2r_interleave_mode", 0),
        const_prolog_pf=cfg.get("const_prolog_pf", False),
        defer_mma_tmem_bar=cfg.get("defer_mma_tmem_bar", False),
        split_c_store_warps=cfg.get("split_c_store_warps", False),
        alg_mbar_mode=cfg.get("alg_mbar_mode", 0),
        const_cluster_sync=cfg.get("const_cluster_sync", False),
        tma_loop_split_mode=cfg.get("tma_loop_split_mode", 0),
        warp_scoped_ab_frag=cfg.get("warp_scoped_ab_frag", False),
        mma_tmem_bar_after_k0_wait=cfg.get("mma_tmem_bar_after_k0_wait", False),
        elide_mma_tail_release=cfg.get("elide_mma_tail_release", 0),
        clean_c_prologue=cfg.get("clean_c_prologue", False),
        warp1_only_tma_c_part=cfg.get("warp1_only_tma_c_part", False),
        relaxed_peer_dealloc=cfg.get("relaxed_peer_dealloc", 0),
        epilog_fence_mode=cfg.get("epilog_fence_mode", 0),
        fused_mma_tail_mode=cfg.get("fused_mma_tail_mode", 0),
        static_cmask=cfg.get("static_cmask", False),
        uniform_coord=cfg.get("uniform_coord", 0),
        sub1_ldtm_mode=cfg.get("sub1_ldtm_mode", 0),
        nr_steps=cfg.get("nr_steps", 2),
        store_drain=cfg.get("store_drain", False),
        pdl_safe=cfg.get("pdl_safe", False),
    )
    a_p = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    b1_p = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    b2_p = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_p = make_ptr(cutlass.Float16, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_p = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb1_p = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb2_p = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32)

    fn = cute.compile(kern, a_p, b1_p, b2_p, sfa_p, sfb1_p, sfb2_p, c_p, shape)

    def _init_and_run(a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, c):
        fn(
            make_ptr(
                cutlass.Float4E2M1FN,
                a.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float4E2M1FN,
                b1.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float4E2M1FN,
                b2.data_ptr(),
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
                sfb1_perm.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                sfb2_perm.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
        )
        ex = fn._default_executor
        packed_args = ex._tls.packed_args
        ptr_buf = (ctypes.c_uint64 * 7)()
        base_addr = ctypes.addressof(ptr_buf)
        for i in range(7):
            packed_args[i] = base_addr + i * 8
        capi_func = ex.capi_func
        _KEEPALIVE.append((kern, fn, ex, packed_args, ptr_buf))

        def _fast_call(a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, c):
            ptr_buf[0] = a.data_ptr()
            ptr_buf[1] = b1.data_ptr()
            ptr_buf[2] = b2.data_ptr()
            ptr_buf[3] = sfa_perm.data_ptr()
            ptr_buf[4] = sfb1_perm.data_ptr()
            ptr_buf[5] = sfb2_perm.data_ptr()
            ptr_buf[6] = c.data_ptr()
            capi_func(packed_args)

        _FAST_CACHE[shape] = _fast_call

    _FAST_CACHE[shape] = _init_and_run
    return _init_and_run


def compile_kernel():
    for shape in [
        (256, 4096, 7168, 1),
        (512, 4096, 7168, 1),
        (256, 3072, 4096, 1),
        (512, 3072, 7168, 1),
    ]:
        _get_fast_runner(shape)


Sm100Nvfp4DualGemmKernel = Sm100Nvfp4DualGemmFast
_COMPILED_RUNNERS = {}


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


def _get_cutlass_c_dtype(out_dtype: torch.dtype):
    if out_dtype == torch.float16:
        return cutlass.Float16
    if out_dtype == torch.bfloat16:
        return cutlass.BFloat16
    raise ValueError(
        f"Unsupported output dtype {out_dtype}; expected float16 or bfloat16"
    )


def _get_sglang_runner(
    shape: Tuple[int, int, int, int],
    out_dtype: torch.dtype,
    enable_pdl: bool,
    pdl_safe: bool,
    has_alpha: bool = False,
    hybrid_rcp_mod_override: Optional[int] = None,
    nr_steps_override: Optional[int] = None,
):
    key = (
        shape,
        out_dtype,
        enable_pdl,
        pdl_safe,
        has_alpha,
        hybrid_rcp_mod_override,
        nr_steps_override,
    )
    runner = _COMPILED_RUNNERS.get(key)
    if runner is not None:
        return runner

    m, n, k, l = shape
    if (m % 256 == 0) and (n % 128 == 0):
        cfg = dict(
            tiler=(256, 128),
            cluster=(2, 1),
            stages=5,
            c_stages=2,
            fuse_b=True,
            k_tile=256,
            hint_b=(m <= 256),
            hint_b_last_row=-1,
            hint_c=True,
            pf_dist=(3 if m <= 256 else 4),
            pf_prolog=False,
            pf_head=(3 if m <= 256 else 4),
            pf_row0_only=(m > 256),
            pf_head_warp2=True,
            cta_desc_pf=True,
            early_pdl_launch=True,
            tma_order=0,
            mma_leader_only=True,
            peel_acc_init=True,
            db_tmem_sf=False,
            early_relinquish=True,
            pre_wait_pdl=False,
            pipe_subtile_store=True,
            early_tmem_free=False,
            defer_pipeline_sync=True,
            first_inst_pdl_launch=(m > 256),
            early_tmem_alloc=False,
            pf_head_mode=1,
            pf_a_bidy0=True,
            async_epilog_bar=False,
            preload_subtile1_t2r=False,
            swiglu_ilp=True,
            no_fmin_clamp=False,
            warp_spec_parts=False,
            split_tmem_dealloc=False,
            hybrid_rcp_mod=4,
            direct_acc_mbar=True,
            const_mma_fields=True,
            swiglu_group4=True,
            pre_bar_pdl_wait=False,
            pf_a_bidy0_ml=True,
            peel_tma_init_try=True,
            leader_tmem_bar=True,
            pf_a_warp0=False,
            s2t_interleave_mode=3,
            t2r_rep_x32=True,
            inline_swz_r2s=True,
            pf_a_all_m=True,
            tma_cta_split_m256=True,
            single_mbar_init=True,
            const_tmem_bar=True,
            pre_fence_dealloc_arv=True,
            prune_dead_smem=False,
            pdl_wait_mode=2,
            unified_mma_loop=True,
            peel_mma_init_try=True,
            tma_cta_split_all_m=True,
            interleave_m_row_pf=True,
            split_bidy_pf_a_mode=3,
            single_swz_base=True,
            coalesce_warp0_tmem=False,
            split_mrow_ml_pf_mode=2,
            hoist_cta_prolog_b_pf=False,
            pf_a_prolog_mode=0,
            hybrid_rcp_offset=0,
            early_peer_dealloc_mode=0,
            sub1_bar_mode=0,
            hint_a=(m > 256),
            rcp_pair_mode=0,
            t2r_interleave_mode=0,
            const_prolog_pf=False,
            defer_mma_tmem_bar=True,
            split_c_store_warps=False,
            alg_mbar_mode=0,
            const_cluster_sync=False,
            tma_loop_split_mode=0,
            warp_scoped_ab_frag=False,
            mma_tmem_bar_after_k0_wait=False,
            elide_mma_tail_release=0,
            clean_c_prologue=True,
            warp1_only_tma_c_part=False,
            relaxed_peer_dealloc=1,
            epilog_fence_mode=2,
            fused_mma_tail_mode=2,
            static_cmask=True,
            uniform_coord=(1 if m > 256 else 0),
            sub1_ldtm_mode=0,
            nr_steps=2,
            store_drain=False,
            pdl_safe=False,
        )
        if pdl_safe or (not enable_pdl) or (out_dtype != torch.float16):
            cfg["single_mbar_init"] = False
            cfg["const_tmem_bar"] = False
            cfg["pre_fence_dealloc_arv"] = False
            cfg["inline_swz_r2s"] = False
            cfg["fused_mma_tail_mode"] = 0
            cfg["epilog_fence_mode"] = 0
            cfg["sub1_ldtm_mode"] = 0
            cfg["store_drain"] = pdl_safe
            cfg["pdl_safe"] = pdl_safe
    elif (m % 256 == 0) and (n % 64 == 0):
        cfg = dict(
            tiler=(256, 64),
            cluster=(2, 1),
            stages=5,
            c_stages=1,
            swiglu_ilp=True,
            hybrid_rcp_mod=4,
            nr_steps=2,
            store_drain=pdl_safe,
            pdl_safe=pdl_safe,
        )
    elif (m % 128 == 0) and (n % 128 == 0):
        cfg = dict(
            tiler=(128, 128),
            cluster=(1, 1),
            stages=3,
            c_stages=2,
            fuse_b=True,
            pf_dist=2,
            pf_head=2,
            hint_b=True,
            hint_c=True,
            pipe_subtile_store=True,
            swiglu_ilp=True,
            swiglu_group4=True,
            t2r_rep_x32=True,
            hybrid_rcp_mod=4,
            nr_steps=2,
            store_drain=pdl_safe,
            pdl_safe=pdl_safe,
        )
    else:
        cfg = dict(
            tiler=(128, 64),
            cluster=(1, 1),
            stages=4,
            c_stages=1,
            swiglu_ilp=True,
            hybrid_rcp_mod=4,
            nr_steps=2,
            store_drain=pdl_safe,
            pdl_safe=pdl_safe,
        )

    if hybrid_rcp_mod_override is not None:
        cfg["hybrid_rcp_mod"] = hybrid_rcp_mod_override
    if nr_steps_override is not None:
        cfg["nr_steps"] = nr_steps_override

    c_dtype_cutlass = _get_cutlass_c_dtype(out_dtype)
    kern = Sm100Nvfp4DualGemmFast(
        mma_tiler_mn=cfg["tiler"],
        cluster_shape_mn=cfg["cluster"],
        num_ab_stage=cfg["stages"],
        num_c_stage=cfg["c_stages"],
        c_dtype=c_dtype_cutlass,
        use_pdl=enable_pdl,
        fuse_b=cfg.get("fuse_b", False),
        k_tile=cfg.get("k_tile", 256),
        hint_b=cfg.get("hint_b", False),
        hint_b_last_row=cfg.get("hint_b_last_row", -1),
        hint_c=cfg.get("hint_c", False),
        pf_dist=cfg.get("pf_dist", 0),
        pf_prolog=cfg.get("pf_prolog", False),
        pf_a=cfg.get("pf_a", False),
        pf_head=cfg.get("pf_head", 0),
        pf_row0_only=cfg.get("pf_row0_only", False),
        pf_head_warp2=cfg.get("pf_head_warp2", False),
        cta_desc_pf=cfg.get("cta_desc_pf", False),
        early_pdl_launch=cfg.get("early_pdl_launch", False),
        tma_order=cfg.get("tma_order", 0),
        mma_leader_only=cfg.get("mma_leader_only", False),
        peel_acc_init=cfg.get("peel_acc_init", False),
        db_tmem_sf=cfg.get("db_tmem_sf", False),
        early_relinquish=cfg.get("early_relinquish", False),
        pre_wait_pdl=cfg.get("pre_wait_pdl", False),
        pipe_subtile_store=cfg.get("pipe_subtile_store", False),
        early_tmem_free=cfg.get("early_tmem_free", False),
        defer_pipeline_sync=cfg.get("defer_pipeline_sync", False),
        first_inst_pdl_launch=cfg.get("first_inst_pdl_launch", False),
        early_tmem_alloc=cfg.get("early_tmem_alloc", False),
        pf_head_mode=cfg.get("pf_head_mode", 0),
        pf_a_bidy0=cfg.get("pf_a_bidy0", False),
        async_epilog_bar=cfg.get("async_epilog_bar", False),
        preload_subtile1_t2r=cfg.get("preload_subtile1_t2r", False),
        swiglu_ilp=cfg.get("swiglu_ilp", False),
        no_fmin_clamp=cfg.get("no_fmin_clamp", False),
        warp_spec_parts=cfg.get("warp_spec_parts", False),
        split_tmem_dealloc=cfg.get("split_tmem_dealloc", False),
        hybrid_rcp_mod=cfg.get("hybrid_rcp_mod", 0),
        direct_acc_mbar=cfg.get("direct_acc_mbar", False),
        const_mma_fields=cfg.get("const_mma_fields", False),
        swiglu_group4=cfg.get("swiglu_group4", False),
        pre_bar_pdl_wait=cfg.get("pre_bar_pdl_wait", False),
        pf_a_bidy0_ml=cfg.get("pf_a_bidy0_ml", False),
        peel_tma_init_try=cfg.get("peel_tma_init_try", False),
        leader_tmem_bar=cfg.get("leader_tmem_bar", False),
        pf_a_warp0=cfg.get("pf_a_warp0", False),
        s2t_interleave_mode=cfg.get("s2t_interleave_mode", 0),
        t2r_rep_x32=cfg.get("t2r_rep_x32", False),
        inline_swz_r2s=cfg.get("inline_swz_r2s", False),
        pf_a_all_m=cfg.get("pf_a_all_m", False),
        tma_cta_split_m256=cfg.get("tma_cta_split_m256", False),
        single_mbar_init=cfg.get("single_mbar_init", False),
        const_tmem_bar=cfg.get("const_tmem_bar", False),
        pre_fence_dealloc_arv=cfg.get("pre_fence_dealloc_arv", False),
        prune_dead_smem=cfg.get("prune_dead_smem", False),
        pdl_wait_mode=cfg.get("pdl_wait_mode", 0),
        unified_mma_loop=cfg.get("unified_mma_loop", False),
        peel_mma_init_try=cfg.get("peel_mma_init_try", False),
        tma_cta_split_all_m=cfg.get("tma_cta_split_all_m", False),
        interleave_m_row_pf=cfg.get("interleave_m_row_pf", False),
        split_bidy_pf_a_mode=cfg.get("split_bidy_pf_a_mode", 0),
        single_swz_base=cfg.get("single_swz_base", False),
        coalesce_warp0_tmem=cfg.get("coalesce_warp0_tmem", False),
        split_mrow_ml_pf_mode=cfg.get("split_mrow_ml_pf_mode", 0),
        hoist_cta_prolog_b_pf=cfg.get("hoist_cta_prolog_b_pf", False),
        pf_a_prolog_mode=cfg.get("pf_a_prolog_mode", 0),
        hybrid_rcp_offset=cfg.get("hybrid_rcp_offset", 0),
        early_peer_dealloc_mode=cfg.get("early_peer_dealloc_mode", 0),
        sub1_bar_mode=cfg.get("sub1_bar_mode", 0),
        hint_a=cfg.get("hint_a", False),
        rcp_pair_mode=cfg.get("rcp_pair_mode", 0),
        t2r_interleave_mode=cfg.get("t2r_interleave_mode", 0),
        const_prolog_pf=cfg.get("const_prolog_pf", False),
        defer_mma_tmem_bar=cfg.get("defer_mma_tmem_bar", False),
        split_c_store_warps=cfg.get("split_c_store_warps", False),
        alg_mbar_mode=cfg.get("alg_mbar_mode", 0),
        const_cluster_sync=cfg.get("const_cluster_sync", False),
        tma_loop_split_mode=cfg.get("tma_loop_split_mode", 0),
        warp_scoped_ab_frag=cfg.get("warp_scoped_ab_frag", False),
        mma_tmem_bar_after_k0_wait=cfg.get("mma_tmem_bar_after_k0_wait", False),
        elide_mma_tail_release=cfg.get("elide_mma_tail_release", 0),
        clean_c_prologue=cfg.get("clean_c_prologue", False),
        warp1_only_tma_c_part=cfg.get("warp1_only_tma_c_part", False),
        relaxed_peer_dealloc=cfg.get("relaxed_peer_dealloc", 0),
        epilog_fence_mode=cfg.get("epilog_fence_mode", 0),
        fused_mma_tail_mode=cfg.get("fused_mma_tail_mode", 0),
        static_cmask=cfg.get("static_cmask", False),
        uniform_coord=cfg.get("uniform_coord", 0),
        sub1_ldtm_mode=cfg.get("sub1_ldtm_mode", 0),
        nr_steps=cfg.get("nr_steps", 2),
        store_drain=cfg.get("store_drain", False),
        pdl_safe=cfg.get("pdl_safe", False),
        has_alpha=has_alpha,
    )
    a_p = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    b1_p = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    b2_p = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_p = make_ptr(c_dtype_cutlass, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_p = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb1_p = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb2_p = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32)
    s_p = cuda.CUstream(0)

    if has_alpha:
        alpha_p = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4)
        fn = cute.compile(
            kern.call_with_stream_alpha,
            a_p,
            b1_p,
            b2_p,
            sfa_p,
            sfb1_p,
            sfb2_p,
            c_p,
            alpha_p,
            s_p,
            shape,
        )

        def _init_and_run_alpha(
            a_ptr_i,
            b1_ptr_i,
            b2_ptr_i,
            sfa_ptr_i,
            sfb1_ptr_i,
            sfb2_ptr_i,
            c_ptr_i,
            alpha_ptr_i,
            stream_i,
        ):
            fn(
                make_ptr(
                    cutlass.Float4E2M1FN,
                    a_ptr_i,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    cutlass.Float4E2M1FN,
                    b1_ptr_i,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    cutlass.Float4E2M1FN,
                    b2_ptr_i,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    cutlass.Float8E4M3FN,
                    sfa_ptr_i,
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                ),
                make_ptr(
                    cutlass.Float8E4M3FN,
                    sfb1_ptr_i,
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                ),
                make_ptr(
                    cutlass.Float8E4M3FN,
                    sfb2_ptr_i,
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                ),
                make_ptr(
                    c_dtype_cutlass, c_ptr_i, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Float32,
                    alpha_ptr_i,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                cuda.CUstream(stream_i),
            )
            ex = fn._default_executor
            packed_args = ex._tls.packed_args
            ptr_buf = (ctypes.c_uint64 * 9)()
            base_addr = ctypes.addressof(ptr_buf)
            for i in range(9):
                packed_args[i] = base_addr + i * 8
            capi_func = ex.capi_func
            _KEEPALIVE.append((kern, fn, ex, packed_args, ptr_buf))

            def _fast_call_alpha(
                a_ptr_i,
                b1_ptr_i,
                b2_ptr_i,
                sfa_ptr_i,
                sfb1_ptr_i,
                sfb2_ptr_i,
                c_ptr_i,
                alpha_ptr_i,
                stream_i,
            ):
                ptr_buf[0] = a_ptr_i
                ptr_buf[1] = b1_ptr_i
                ptr_buf[2] = b2_ptr_i
                ptr_buf[3] = sfa_ptr_i
                ptr_buf[4] = sfb1_ptr_i
                ptr_buf[5] = sfb2_ptr_i
                ptr_buf[6] = c_ptr_i
                ptr_buf[7] = alpha_ptr_i
                ptr_buf[8] = stream_i
                capi_func(packed_args)

            _COMPILED_RUNNERS[key] = _fast_call_alpha

        _COMPILED_RUNNERS[key] = _init_and_run_alpha
        return _init_and_run_alpha

    fn = cute.compile(
        kern.call_with_stream, a_p, b1_p, b2_p, sfa_p, sfb1_p, sfb2_p, c_p, s_p, shape
    )

    def _init_and_run(
        a_ptr_i,
        b1_ptr_i,
        b2_ptr_i,
        sfa_ptr_i,
        sfb1_ptr_i,
        sfb2_ptr_i,
        c_ptr_i,
        stream_i,
    ):
        fn(
            make_ptr(
                cutlass.Float4E2M1FN, a_ptr_i, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float4E2M1FN, b1_ptr_i, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float4E2M1FN, b2_ptr_i, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                sfa_ptr_i,
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                sfb1_ptr_i,
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                cutlass.Float8E4M3FN,
                sfb2_ptr_i,
                cute.AddressSpace.gmem,
                assumed_align=32,
            ),
            make_ptr(
                c_dtype_cutlass, c_ptr_i, cute.AddressSpace.gmem, assumed_align=16
            ),
            cuda.CUstream(stream_i),
        )
        ex = fn._default_executor
        packed_args = ex._tls.packed_args
        ptr_buf = (ctypes.c_uint64 * 8)()
        base_addr = ctypes.addressof(ptr_buf)
        for i in range(8):
            packed_args[i] = base_addr + i * 8
        capi_func = ex.capi_func
        _KEEPALIVE.append((kern, fn, ex, packed_args, ptr_buf))

        def _fast_call(
            a_ptr_i,
            b1_ptr_i,
            b2_ptr_i,
            sfa_ptr_i,
            sfb1_ptr_i,
            sfb2_ptr_i,
            c_ptr_i,
            stream_i,
        ):
            ptr_buf[0] = a_ptr_i
            ptr_buf[1] = b1_ptr_i
            ptr_buf[2] = b2_ptr_i
            ptr_buf[3] = sfa_ptr_i
            ptr_buf[4] = sfb1_ptr_i
            ptr_buf[5] = sfb2_ptr_i
            ptr_buf[6] = c_ptr_i
            ptr_buf[7] = stream_i
            capi_func(packed_args)

        _COMPILED_RUNNERS[key] = _fast_call

    _COMPILED_RUNNERS[key] = _init_and_run
    return _init_and_run


def nvfp4_dual_gemm_swiglu(
    a: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    sfa: torch.Tensor,
    sfb1: torch.Tensor,
    sfb2: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    alpha: Optional[torch.Tensor] = None,
    swizzled_scales: bool = False,
    out_dtype: Optional[torch.dtype] = None,
    enable_pdl: bool = True,
    pdl_safe: bool = False,
    hybrid_rcp_mod_override: Optional[int] = None,
    nr_steps_override: Optional[int] = None,
) -> torch.Tensor:
    if a.device.type != "cuda" or b1.device != a.device or b2.device != a.device:
        raise ValueError(
            "nvfp4_dual_gemm_swiglu requires CUDA tensors on the same device"
        )

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

    if b1.shape != b2.shape or b1.ndim != a.ndim:
        raise ValueError(
            f"Expected b1 and b2 to have matching shapes and rank as a, "
            f"got b1={tuple(b1.shape)}, b2={tuple(b2.shape)}, a={tuple(a.shape)}"
        )
    if b1.shape[1] != k_half or (not is_2d and b1.shape[2] != l):
        raise ValueError(
            f"K/L dimension mismatch between a {tuple(a.shape)} and b1 {tuple(b1.shape)}"
        )
    if a.stride(1) != 1 or b1.stride(1) != 1 or b2.stride(1) != 1:
        raise ValueError("Input tensors a, b1, b2 must be K-major (stride(1) == 1)")

    n = b1.shape[0]
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
            f"Unsupported NVFP4 dual GEMM shape (M={m}, N={n}, K={k}, L={l}): "
            "M must be a multiple of 128, N a multiple of 64, and K a multiple of 256."
        )

    if out is not None:
        expected_out_shape = (m, n) if is_2d else (m, n, l)
        if out.shape != expected_out_shape or out.device != a.device:
            raise ValueError(
                f"Expected out shape {expected_out_shape} on {a.device}, got {tuple(out.shape)} on {out.device}"
            )
        if not (
            (is_2d and out.stride(1) == 1 and out.stride(0) >= n)
            or (not is_2d and out.stride(1) == 1 and out.stride(0) >= n)
        ):
            raise ValueError("Output tensor out must be row-major (stride(1) == 1)")
        if out_dtype is not None and out.dtype != out_dtype:
            raise ValueError(
                f"out.dtype ({out.dtype}) does not match out_dtype ({out_dtype})"
            )
        out_dtype = out.dtype
    elif out_dtype is None:
        out_dtype = torch.float16

    _get_cutlass_c_dtype(out_dtype)

    if not swizzled_scales:
        if sfa.ndim != 6:
            sfa = permute_nvfp4_scale(sfa)
        if sfb1.ndim != 6:
            sfb1 = permute_nvfp4_scale(sfb1)
        if sfb2.ndim != 6:
            sfb2 = permute_nvfp4_scale(sfb2)

    if out is None:
        if is_2d:
            out = torch.empty((m, n), dtype=out_dtype, device=a.device)
        else:
            out = torch.empty((l, m, n), dtype=out_dtype, device=a.device).permute(
                1, 2, 0
            )

    shape = (m, n, k, l)
    stream_i = torch.cuda.current_stream(a.device).cuda_stream
    if (
        stream_i == 0
        and out_dtype == torch.float16
        and enable_pdl
        and (not pdl_safe)
        and (alpha is None)
        and (hybrid_rcp_mod_override is None)
        and (nr_steps_override is None)
    ):
        _get_fast_runner(shape)(a, b1, b2, sfa, sfb1, sfb2, out)
        return out

    has_alpha = alpha is not None
    runner = _get_sglang_runner(
        shape,
        out_dtype,
        enable_pdl,
        pdl_safe,
        has_alpha=has_alpha,
        hybrid_rcp_mod_override=hybrid_rcp_mod_override,
        nr_steps_override=nr_steps_override,
    )
    if has_alpha:
        runner(
            a.data_ptr(),
            b1.data_ptr(),
            b2.data_ptr(),
            sfa.data_ptr(),
            sfb1.data_ptr(),
            sfb2.data_ptr(),
            out.data_ptr(),
            alpha.data_ptr(),
            stream_i,
        )
    else:
        runner(
            a.data_ptr(),
            b1.data_ptr(),
            b2.data_ptr(),
            sfa.data_ptr(),
            sfb1.data_ptr(),
            sfb2.data_ptr(),
            out.data_ptr(),
            stream_i,
        )
    return out


__all__ = [
    "Sm100Nvfp4DualGemmFast",
    "Sm100Nvfp4DualGemmKernel",
    "compile_kernel",
    "nvfp4_dual_gemm_swiglu",
    "permute_nvfp4_scale",
    "_ceil_div",
]
