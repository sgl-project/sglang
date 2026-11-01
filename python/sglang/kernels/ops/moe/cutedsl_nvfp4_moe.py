from __future__ import annotations

import ctypes
from typing import Dict, List, Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
import triton
import triton.language as tl
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from sglang.kernels.ops.gemm.cutedsl_nvfp4_group_gemm import (
    _KEEPALIVE,
    _ceil_div,
    _load_in_mem_module,
    nvfp4_masked_grouped_gemm,
)

_OP_K_MAJOR = (
    cute.nvgpu.OperandMajorMode.K
    if hasattr(cute.nvgpu, "OperandMajorMode")
    else tcgen05.OperandMajorMode.K
)

_DUAL_MASKED_CACHE: Dict[Tuple, object] = {}
_DUAL_QUANT_CACHE: Dict[Tuple, object] = {}
_MOE_WS_CACHE: Dict[Tuple, Tuple[torch.Tensor, ...]] = {}
_ROUTED_WS_CACHE: Dict[Tuple, Tuple[torch.Tensor, ...]] = {}
_C_QUANT_FN = None


def _get_c_quant_fn():
    global _C_QUANT_FN
    if _C_QUANT_FN is not None:
        return _C_QUANT_FN
    try:
        from flashinfer.quantization import get_fp4_quantization_module

        mod = get_fp4_quantization_module("100")
        fn = mod.scaled_fp4_grouped_quant_sm100
        for cell in fn.__closure__ or ():
            if hasattr(
                cell.cell_contents, "silu_and_mul_scaled_nvfp4_experts_quantize"
            ):
                _C_QUANT_FN = (
                    cell.cell_contents.silu_and_mul_scaled_nvfp4_experts_quantize
                )
                return _C_QUANT_FN
    except Exception:
        pass
    return None


def _build_masked_dual_gemm_swiglu_1cta(
    G: int,
    m_max: int,
    n: int,
    k: int,
    num_ab_stage: int = 4,
    order_mn: bool = True,
    c_dtype_str: str = "BFloat16",
):
    key = ("1cta_dual_masked", G, m_max, n, k, num_ab_stage, order_mn, c_dtype_str)
    if key in _DUAL_MASKED_CACHE:
        return _DUAL_MASKED_CACHE[key]

    tile_m = 128
    tile_n = 128
    tile_k = 256
    subtile_cnt = 2
    cm = _ceil_div(m_max, tile_m)
    cn = _ceil_div(n, tile_n)
    tiles_per_group = cm * cn
    total_ctas = G * tiles_per_group
    k_tiles = k // tile_k
    init_st = min(num_ab_stage, k_tiles)

    ptr_args_jit = (
        ", ".join(
            f"a_ptr_{g}: cute.Pointer, b1_ptr_{g}: cute.Pointer, b2_ptr_{g}: cute.Pointer, "
            f"sfa_ptr_{g}: cute.Pointer, sfb1_ptr_{g}: cute.Pointer, sfb2_ptr_{g}: cute.Pointer, "
            f"c_ptr_{g}: cute.Pointer, alpha_ptr_{g}: cute.Pointer"
            for g in range(G)
        )
        + ", masked_m_ptr: cute.Pointer, stream: cuda.CUstream"
    )
    kern_args_def = (
        ", ".join(
            f"tma_atom_a_{g}: cute.CopyAtom, mA_{g}: cute.Tensor, "
            f"tma_atom_b1_{g}: cute.CopyAtom, mB1_{g}: cute.Tensor, "
            f"tma_atom_b2_{g}: cute.CopyAtom, mB2_{g}: cute.Tensor, "
            f"tma_atom_sfa_{g}: cute.CopyAtom, mSFA_{g}: cute.Tensor, "
            f"tma_atom_sfb1_{g}: cute.CopyAtom, mSFB1_{g}: cute.Tensor, "
            f"tma_atom_sfb2_{g}: cute.CopyAtom, mSFB2_{g}: cute.Tensor, "
            f"tma_atom_c_{g}: cute.CopyAtom, mC_{g}: cute.Tensor, mAlpha_{g}: cute.Tensor"
            for g in range(G)
        )
        + ", mMaskedM: cute.Tensor"
    )
    kern_args_call = (
        ", ".join(
            f"tma_atom_a_{g}, tma_tensor_a_{g}, "
            f"tma_atom_b1_{g}, tma_tensor_b1_{g}, "
            f"tma_atom_b2_{g}, tma_tensor_b2_{g}, "
            f"tma_atom_sfa_{g}, tma_tensor_sfa_{g}, "
            f"tma_atom_sfb1_{g}, tma_tensor_sfb1_{g}, "
            f"tma_atom_sfb2_{g}, tma_tensor_sfb2_{g}, "
            f"tma_atom_c_{g}, tma_tensor_c_{g}, alpha_t_{g}"
            for g in range(G)
        )
        + ", masked_m_t"
    )

    host_setup = []
    tma_branches = []
    epi_branches = []
    alpha_branches = []

    for g in range(G):
        kw = "if" if g == 0 else "elif"
        alpha_branches.append(
            f"                {kw} group_idx == {g}:\n                    alpha_val = cutlass.Float32(mAlpha_{g}[0])\n"
        )
        host_setup.append(f"""
        a_t_{g} = cute.make_tensor(a_ptr_{g}, cute.make_layout(({m_max}, cute.assume({k}, 32), 1), stride=(cute.assume({k}, 32), 1, cute.assume({m_max * k}, 32))))
        b1_t_{g} = cute.make_tensor(b1_ptr_{g}, cute.make_layout(({n}, cute.assume({k}, 32), 1), stride=(cute.assume({k}, 32), 1, cute.assume({n * k}, 32))))
        b2_t_{g} = cute.make_tensor(b2_ptr_{g}, cute.make_layout(({n}, cute.assume({k}, 32), 1), stride=(cute.assume({k}, 32), 1, cute.assume({n * k}, 32))))
        c_t_{g} = cute.make_tensor(c_ptr_{g}, cute.make_layout(({m_max}, cute.assume({n}, 32), 1), stride=(cute.assume({n}, 32), 1, cute.assume({m_max * n}, 32))))
        sfa_t_{g} = cute.make_tensor(sfa_ptr_{g}, blockscaled_utils.tile_atom_to_shape_SF(a_t_{g}.shape, 16))
        sfb1_t_{g} = cute.make_tensor(sfb1_ptr_{g}, blockscaled_utils.tile_atom_to_shape_SF(b1_t_{g}.shape, 16))
        sfb2_t_{g} = cute.make_tensor(sfb2_ptr_{g}, blockscaled_utils.tile_atom_to_shape_SF(b2_t_{g}.shape, 16))
        alpha_t_{g} = cute.make_tensor(alpha_ptr_{g}, cute.make_layout((1,), stride=(1,)))
        tma_atom_a_{g}, tma_tensor_a_{g} = cute.nvgpu.make_tiled_tma_atom_A(a_op, a_t_{g}, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_atom_b1_{g}, tma_tensor_b1_{g} = cute.nvgpu.make_tiled_tma_atom_B(b_op, b1_t_{g}, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_atom_b2_{g}, tma_tensor_b2_{g} = cute.nvgpu.make_tiled_tma_atom_B(b_op, b2_t_{g}, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_atom_sfa_{g}, tma_tensor_sfa_{g} = cute.nvgpu.make_tiled_tma_atom_A(sfa_op, sfa_t_{g}, sfa_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16)
        tma_atom_sfb1_{g}, tma_tensor_sfb1_{g} = cute.nvgpu.make_tiled_tma_atom_B(sfb_op, sfb1_t_{g}, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)
        tma_atom_sfb2_{g}, tma_tensor_sfb2_{g} = cute.nvgpu.make_tiled_tma_atom_B(sfb_op, sfb2_t_{g}, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)
        tma_atom_c_{g}, tma_tensor_c_{g} = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c_t_{g}, epi_smem_layout, self.epi_tile)
""")
        tma_branches.append(f"""
                {kw} group_idx == {g}:
                    cpasync.prefetch_descriptor(tma_atom_a_{g})
                    cpasync.prefetch_descriptor(tma_atom_b1_{g})
                    cpasync.prefetch_descriptor(tma_atom_b2_{g})
                    cpasync.prefetch_descriptor(tma_atom_sfa_{g})
                    cpasync.prefetch_descriptor(tma_atom_sfb1_{g})
                    cpasync.prefetch_descriptor(tma_atom_sfb2_{g})
                    gA = cute.local_tile(mA_{g}, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                    gB1 = cute.local_tile(mB1_{g}, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                    gB2 = cute.local_tile(mB2_{g}, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                    gSFA = cute.local_tile(mSFA_{g}, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                    gSFB1 = cute.local_tile(mSFB1_{g}, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
                    gSFB2 = cute.local_tile(mSFB2_{g}, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
                    tCgA = thr_mma.partition_A(gA)
                    tCgB1 = thr_mma.partition_B(gB1)
                    tCgB2 = thr_mma.partition_B(gB2)
                    tCgSFA = thr_mma.partition_A(gSFA)
                    tAsA, tAgA = cpasync.tma_partition(tma_atom_a_{g}, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
                    tBsB1, tBgB1 = cpasync.tma_partition(tma_atom_b1_{g}, 0, cute.make_layout(1), cute.group_modes(sB1, 0, 3), cute.group_modes(tCgB1, 0, 3))
                    tBsB2, tBgB2 = cpasync.tma_partition(tma_atom_b2_{g}, 0, cute.make_layout(1), cute.group_modes(sB2, 0, 3), cute.group_modes(tCgB2, 0, 3))
                    tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa_{g}, 0, cute.make_layout(1), cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
                    tAsSFA = cute.filter_zeros(tAsSFA); tAgSFA = cute.filter_zeros(tAgSFA)
                    tBsSFB1, tBgSFB1 = cpasync.tma_partition(tma_atom_sfb1_{g}, 0, cute.make_layout(1), cute.group_modes(sSFB1, 0, 3), cute.group_modes(thr_mma_sfb.partition_B(gSFB1), 0, 3))
                    tBsSFB1 = cute.filter_zeros(tBsSFB1); tBgSFB1 = cute.filter_zeros(tBgSFB1)
                    tBsSFB2, tBgSFB2 = cpasync.tma_partition(tma_atom_sfb2_{g}, 0, cute.make_layout(1), cute.group_modes(sSFB2, 0, 3), cute.group_modes(thr_mma_sfb.partition_B(gSFB2), 0, 3))
                    tBsSFB2 = cute.filter_zeros(tBsSFB2); tBgSFB2 = cute.filter_zeros(tBgSFB2)
                    tAgA_s = tAgA[(None, coord_x, None, 0)]
                    tBgB1_s = tBgB1[(None, coord_y, None, 0)]
                    tBgB2_s = tBgB2[(None, coord_y, None, 0)]
                    tAgSFA_s = tAgSFA[(None, coord_x, None, 0)]
                    tBgSFB1_s = tBgSFB1[(None, coord_y, None, 0)]
                    tBgSFB2_s = tBgSFB2[(None, coord_y, None, 0)]
                    ab_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
                    ab_prod.reset_count()
                    for kt_p in cutlass.range_constexpr({init_st}):
                        ab_pipeline.producer_acquire(ab_prod, cutlass.Boolean(1))
                        bar_p = ab_pipeline.producer_get_barrier(ab_prod)
                        st_p = ab_prod.index
                        cute.copy(tma_atom_b1_{g}, tBgB1_s[(None, kt_p)], tBsB1[(None, st_p)], tma_bar_ptr=bar_p, cache_policy=cutlass.Int64(0x12F0000000000000))
                        cute.copy(tma_atom_b2_{g}, tBgB2_s[(None, kt_p)], tBsB2[(None, st_p)], tma_bar_ptr=bar_p, cache_policy=cutlass.Int64(0x12F0000000000000))
                        cute.copy(tma_atom_a_{g}, tAgA_s[(None, kt_p)], tAsA[(None, st_p)], tma_bar_ptr=bar_p, cache_policy=cutlass.Int64(0x14F0000000000000))
                        cute.copy(tma_atom_sfb1_{g}, tBgSFB1_s[(None, kt_p)], tBsSFB1[(None, st_p)], tma_bar_ptr=bar_p, cache_policy=cutlass.Int64(0x12F0000000000000))
                        cute.copy(tma_atom_sfb2_{g}, tBgSFB2_s[(None, kt_p)], tBsSFB2[(None, st_p)], tma_bar_ptr=bar_p, cache_policy=cutlass.Int64(0x12F0000000000000))
                        cute.copy(tma_atom_sfa_{g}, tAgSFA_s[(None, kt_p)], tAsSFA[(None, st_p)], tma_bar_ptr=bar_p, cache_policy=cutlass.Int64(0x14F0000000000000))
                        ab_prod.advance()
                    if cutlass.const_expr({k_tiles} > {init_st}):
                        peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
                        for kt in cutlass.range({init_st}, {k_tiles}, 1, unroll=1):
                            ab_pipeline.producer_acquire(ab_prod, peek_empty)
                            bar = ab_pipeline.producer_get_barrier(ab_prod)
                            st = ab_prod.index
                            cute.copy(tma_atom_b1_{g}, tBgB1_s[(None, kt)], tBsB1[(None, st)], tma_bar_ptr=bar, cache_policy=cutlass.Int64(0x12F0000000000000))
                            cute.copy(tma_atom_b2_{g}, tBgB2_s[(None, kt)], tBsB2[(None, st)], tma_bar_ptr=bar, cache_policy=cutlass.Int64(0x12F0000000000000))
                            cute.copy(tma_atom_a_{g}, tAgA_s[(None, kt)], tAsA[(None, st)], tma_bar_ptr=bar, cache_policy=cutlass.Int64(0x14F0000000000000))
                            cute.copy(tma_atom_sfb1_{g}, tBgSFB1_s[(None, kt)], tBsSFB1[(None, st)], tma_bar_ptr=bar, cache_policy=cutlass.Int64(0x12F0000000000000))
                            cute.copy(tma_atom_sfb2_{g}, tBgSFB2_s[(None, kt)], tBsSFB2[(None, st)], tma_bar_ptr=bar, cache_policy=cutlass.Int64(0x12F0000000000000))
                            cute.copy(tma_atom_sfa_{g}, tAgSFA_s[(None, kt)], tAsSFA[(None, st)], tma_bar_ptr=bar, cache_policy=cutlass.Int64(0x14F0000000000000))
                            ab_prod.advance()
                            peek_empty = cutlass.Boolean(1)
                            if ab_prod.count < {k_tiles}:
                                peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
""")
        epi_branches.append(f"""
                    {kw} group_idx == {g}:
                        gC = cute.local_tile(mC_{g}, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                        tCgC = thr_mma.partition_C(gC)
                        gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], self.epi_tile)
                        bSG_sC, bSG_gC_p = cpasync.tma_partition(tma_atom_c_{g}, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gC_epi, 0, 2))
                        bSG_gC = cute.group_modes(bSG_gC_p[(None, None, None, coord_x, coord_y, 0)], 1, cute.rank(bSG_gC_p[(None, None, None, coord_x, coord_y, 0)]))
                        for sub_i in cutlass.range_constexpr({subtile_cnt}):
                            cute.copy(tma_atom_c_{g}, bSG_sC[(None, sub_i)], bSG_gC[(None, sub_i)], cache_policy=cutlass.Int64(0x12F0000000000000))
""")

    if order_mn:
        coord_code = f"        coord_x = rel % {cm}\n        coord_y = rel // {cm}"
    else:
        coord_code = f"        coord_x = rel // {cn}\n        coord_y = rel % {cn}"

    src = f"""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

class Sm100Nvfp4MaskedDualGemmSwiGLU1CTA:
    def __init__(self):
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.{c_dtype_str}
        self.sf_vec_size = 16
        self.mma_tiler = (128, 128, 256)
        self.mma_tiler_sfb = (128, 128, 256)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192
        self.num_ab_stage = {num_ab_stage}
        self.epi_tile = (128, 64)
        self.num_c_stage = 2
        self.num_tmem_alloc_cols = 512
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=160)

    @cute.jit
    def __call__(self, {ptr_args_jit}):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            self.sf_dtype, self.sf_vec_size, self.cta_group, (128, 128)
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            self.sf_dtype, self.sf_vec_size, tcgen05.CtaGroup.ONE, (128, 128)
        )
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma_sfb.thr_id.shape,))
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.c_dtype, utils.LayoutEnum.ROW_MAJOR, self.epi_tile, self.num_c_stage)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)

        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.ab_dtype, a_smem_layout)
            + cute.size_in_bytes(self.ab_dtype, b_smem_layout) * 2
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout) * 2
        )

        masked_m_t = cute.make_tensor(masked_m_ptr, cute.make_layout(({G},), stride=(1,)))
        {"".join(host_setup)}

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)], 1024]
            sB1: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)], 1024]
            sB2: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)], 1024]
            sSFA: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)], 1024]
            sSFB1: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)], 1024]
            sSFB2: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)], 1024]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma, tiled_mma_sfb,
            {kern_args_call},
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
            self.c_smem_layout_staged
        ).launch(
            grid=({total_ctas}, 1, 1),
            block=[192, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        {kern_args_def},
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()

        group_idx = bidx // {tiles_per_group}
        rel = bidx % {tiles_per_group}
{coord_code}
        k_tile_cnt = cutlass.Int32({k_tiles})

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=None,
        )
        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        active_m = mMaskedM[group_idx]
        if coord_x * 128 < active_m:
            sC_ptr = cute.recast_ptr(storage.sA.data_ptr(), c_smem_layout_staged.inner, dtype=self.c_dtype)
            sC = cute.make_tensor(sC_ptr, c_smem_layout_staged.outer)
            sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
            sB1 = storage.sB1.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
            sB2 = storage.sB2.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
            sSFB1 = storage.sSFB1.get_tensor(sfb_smem_layout_staged)
            sSFB2 = storage.sSFB2.get_tensor(sfb_smem_layout_staged)

            thr_mma = tiled_mma.get_slice(0)
            thr_mma_sfb = tiled_mma_sfb.get_slice(0)

            if warp_idx == self.tma_warp_id:
                cute.arch.griddepcontrol_wait()
{"".join(tma_branches)}
                cute.arch.griddepcontrol_launch_dependents()

            if warp_idx == self.mma_warp_id:
                tmem.wait_for_alloc()
                acc1_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
                tCtAcc1_base = cute.make_tensor(acc1_tmem_ptr, tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((128, 128))).layout)
                acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAcc1_base)
                acc2_tmem_ptr = acc1_tmem_ptr + acc_cols
                tCtAcc2_base = cute.make_tensor(acc2_tmem_ptr, tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((128, 128))).layout)

                sfa_tmem_ptr = cute.recast_ptr(acc1_tmem_ptr + acc_cols * 2, dtype=self.sf_dtype)
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)))
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
                sf_cols = tcgen05.find_tmem_tensor_col_offset(tCtSFA)

                sfb1_tmem_ptr = cute.recast_ptr(acc1_tmem_ptr + acc_cols * 2 + sf_cols, dtype=self.sf_dtype)
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)))
                tCtSFB1 = cute.make_tensor(sfb1_tmem_ptr, tCtSFB_layout)
                sfb_cols = tcgen05.find_tmem_tensor_col_offset(tCtSFB1)

                sfb2_tmem_ptr = cute.recast_ptr(acc1_tmem_ptr + acc_cols * 2 + sf_cols + sfb_cols, dtype=self.sf_dtype)
                tCtSFB2 = cute.make_tensor(sfb2_tmem_ptr, tCtSFB_layout)

                copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), self.sf_dtype)
                tCsSFA_c = cute.filter_zeros(sSFA); tCtSFA_c = cute.filter_zeros(tCtSFA)
                tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_c)
                thr_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
                tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfa, thr_s2t_sfa.partition_S(tCsSFA_c))
                tCtSFA_s2t = thr_s2t_sfa.partition_D(tCtSFA_c)

                tCsSFB1_c = cute.filter_zeros(sSFB1); tCtSFB1_c = cute.filter_zeros(tCtSFB1)
                tiled_copy_s2t_sfb1 = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB1_c)
                thr_s2t_sfb1 = tiled_copy_s2t_sfb1.get_slice(0)
                tCsSFB1_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfb1, thr_s2t_sfb1.partition_S(tCsSFB1_c))
                tCtSFB1_s2t = thr_s2t_sfb1.partition_D(tCtSFB1_c)

                tCsSFB2_c = cute.filter_zeros(sSFB2); tCtSFB2_c = cute.filter_zeros(tCtSFB2)
                tiled_copy_s2t_sfb2 = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB2_c)
                thr_s2t_sfb2 = tiled_copy_s2t_sfb2.get_slice(0)
                tCsSFB2_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfb2, thr_s2t_sfb2.partition_S(tCsSFB2_c))
                tCtSFB2_s2t = thr_s2t_sfb2.partition_D(tCtSFB2_c)

                tCrA = tiled_mma.make_fragment_A(sA)
                tCrB1 = tiled_mma.make_fragment_B(sB1)
                tCrB2 = tiled_mma.make_fragment_B(sB2)
                tCtAcc1 = tCtAcc1_base[(None, None, None)]
                tCtAcc2 = tCtAcc2_base[(None, None, None)]

                ab_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
                acc_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                ab_cons.reset_count()
                peek_full = ab_pipeline.consumer_try_wait(ab_cons)

                ab_pipeline.consumer_wait(ab_cons, peek_full)
                st0 = ab_cons.index
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb0 in cutlass.range_constexpr(4):
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t[(None, None, None, kb0, st0)], tCtSFA_s2t[(None, None, None, kb0)])
                    cute.copy(tiled_copy_s2t_sfb1, tCsSFB1_s2t[(None, None, None, kb0, st0)], tCtSFB1_s2t[(None, None, None, kb0)])
                    cute.copy(tiled_copy_s2t_sfb2, tCsSFB2_s2t[(None, None, None, kb0, st0)], tCtSFB2_s2t[(None, None, None, kb0)])
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb0)].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB1[(None, None, kb0)].iterator)
                    cute.gemm(tiled_mma, tCtAcc1, tCrA[(None, None, kb0, st0)], tCrB1[(None, None, kb0, st0)], tCtAcc1)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB2[(None, None, kb0)].iterator)
                    cute.gemm(tiled_mma, tCtAcc2, tCrA[(None, None, kb0, st0)], tCrB2[(None, None, kb0, st0)], tCtAcc2)
                    if cutlass.const_expr(kb0 == 0):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                rel0 = ab_cons.index
                ab_cons.advance()
                peek_full = cutlass.Boolean(1)
                if ab_cons.count < k_tile_cnt:
                    if ab_cons.count <= k_tile_cnt - self.num_ab_stage:
                        ab_pipeline.sync_object_empty.arrive(rel0, cutlass.Int16(1), self.cta_group)
                    peek_full = ab_pipeline.consumer_try_wait(ab_cons)

                for kt in cutlass.range(1, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.consumer_wait(ab_cons, peek_full)
                    st = ab_cons.index
                    for kb in cutlass.range_constexpr(4):
                        cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t[(None, None, None, kb, st)], tCtSFA_s2t[(None, None, None, kb)])
                        cute.copy(tiled_copy_s2t_sfb1, tCsSFB1_s2t[(None, None, None, kb, st)], tCtSFB1_s2t[(None, None, None, kb)])
                        cute.copy(tiled_copy_s2t_sfb2, tCsSFB2_s2t[(None, None, None, kb, st)], tCtSFB2_s2t[(None, None, None, kb)])
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB1[(None, None, kb)].iterator)
                        cute.gemm(tiled_mma, tCtAcc1, tCrA[(None, None, kb, st)], tCrB1[(None, None, kb, st)], tCtAcc1)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB2[(None, None, kb)].iterator)
                        cute.gemm(tiled_mma, tCtAcc2, tCrA[(None, None, kb, st)], tCrB2[(None, None, kb, st)], tCtAcc2)
                    rel_i = ab_cons.index
                    ab_cons.advance()
                    peek_full = cutlass.Boolean(1)
                    if ab_cons.count < k_tile_cnt:
                        if ab_cons.count <= k_tile_cnt - self.num_ab_stage:
                            ab_pipeline.sync_object_empty.arrive(rel_i, cutlass.Int16(1), self.cta_group)
                        peek_full = ab_pipeline.consumer_try_wait(ab_cons)
                acc_pipeline.producer_commit(acc_prod)

            if warp_idx < 4:
                tmem.allocate(self.num_tmem_alloc_cols)
                tmem.wait_for_alloc()
                acc1_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
                tCtAcc1_base = cute.make_tensor(acc1_tmem_ptr, tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((128, 128))).layout)
                acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAcc1_base)
                acc2_tmem_ptr = acc1_tmem_ptr + acc_cols
                tCtAcc2_base = cute.make_tensor(acc2_tmem_ptr, tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((128, 128))).layout)

                epi_tidx = tidx
                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    (128, 128, 256), utils.LayoutEnum.ROW_MAJOR, self.c_dtype, self.acc_dtype, self.epi_tile, False
                )
                tCtAcc1_epi = cute.flat_divide(tCtAcc1_base[((None, None), 0, 0)], self.epi_tile)
                tCtAcc2_epi = cute.flat_divide(tCtAcc2_base[((None, None), 0, 0)], self.epi_tile)
                tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc1_epi[(None, None, 0, 0)])
                thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
                tTR_tAcc1 = cute.group_modes(thr_copy_t2r.partition_S(tCtAcc1_epi), 3, cute.rank(thr_copy_t2r.partition_S(tCtAcc1_epi)))
                tTR_tAcc2 = cute.group_modes(thr_copy_t2r.partition_S(tCtAcc2_epi), 3, cute.rank(thr_copy_t2r.partition_S(tCtAcc2_epi)))

                copy_atom_r2s = sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
                tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
                thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
                tRS_sC = thr_copy_r2s.partition_D(sC)

                c_ident = cute.make_identity_tensor((128, 128))
                c_ident_epi = cute.flat_divide(c_ident, self.epi_tile)
                tTR_cC = thr_copy_t2r.partition_D(c_ident_epi)
                tTR_rAcc1 = cute.make_rmem_tensor(tTR_cC[(None, None, None, 0, 0)].shape, self.acc_dtype)
                tTR_rAcc2 = cute.make_rmem_tensor(tTR_cC[(None, None, None, 0, 0)].shape, self.acc_dtype)
                tRS_rC = cute.make_rmem_tensor(thr_copy_r2s.retile(tTR_rAcc1).shape, self.c_dtype)

                tmem.relinquish_alloc_permit()
                alpha_val = cutlass.Float32(1.0)
{"".join(alpha_branches)}
                acc_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                acc_pipeline.consumer_wait(acc_cons)
                for sub_i in cutlass.range_constexpr(2):
                    cute.copy(tiled_copy_t2r, tTR_tAcc1[(None, None, None, sub_i)], tTR_rAcc1)
                    cute.copy(tiled_copy_t2r, tTR_tAcc2[(None, None, None, sub_i)], tTR_rAcc2)
                    g_val = (tTR_rAcc1.load() * alpha_val).to(self.c_dtype).to(cutlass.Float32)
                    u_val = (tTR_rAcc2.load() * alpha_val).to(self.c_dtype).to(cutlass.Float32)
                    denom = cutlass.Float32(1.0) + cute.math.exp2(g_val * cutlass.Float32(-1.4426950408889634), fastmath=True)
                    tRS_rC.store(((g_val / denom) * u_val).to(self.c_dtype))
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, sub_i)])
                cute.arch.fence_view_async_tmem_load()
                self.epilog_sync_barrier.arrive_and_wait()
                tmem.free(acc1_tmem_ptr)
                if warp_idx == 0:
                    cute.arch.fence_view_async_shared()
{"".join(epi_branches)}
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            if warp_idx == self.tma_warp_id:
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
"""
    mod = _load_in_mem_module(
        f"mdual1cta_{G}_{m_max}_{n}_{k}_{num_ab_stage}_{order_mn}_{c_dtype_str}",
        src,
    )
    kern = mod.Sm100Nvfp4MaskedDualGemmSwiGLU1CTA()
    c_dtype_cutlass = getattr(cutlass, c_dtype_str)

    fake_args = []
    for _ in range(G):
        fake_args.extend(
            [
                make_ptr(
                    cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32
                ),
                make_ptr(
                    cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32
                ),
                make_ptr(
                    cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32
                ),
                make_ptr(c_dtype_cutlass, 0, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
            ]
        )
    fake_args.append(
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4)
    )
    fake_args.append(cuda.CUstream(0))
    fn = cute.compile(kern, *fake_args)

    total_slots = G * 8 + 2
    capi_func = None
    packed_args = None
    ptr_buf = None

    def runner(ptr_list: List[int], masked_m_ptr: int, stream_i: int):
        nonlocal capi_func, packed_args, ptr_buf
        if capi_func is None:
            real_args = []
            idx = 0
            for _ in range(G):
                real_args.extend(
                    [
                        make_ptr(
                            cutlass.Float4E2M1FN,
                            ptr_list[idx],
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        make_ptr(
                            cutlass.Float4E2M1FN,
                            ptr_list[idx + 1],
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        make_ptr(
                            cutlass.Float4E2M1FN,
                            ptr_list[idx + 2],
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        make_ptr(
                            cutlass.Float8E4M3FN,
                            ptr_list[idx + 3],
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        ),
                        make_ptr(
                            cutlass.Float8E4M3FN,
                            ptr_list[idx + 4],
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        ),
                        make_ptr(
                            cutlass.Float8E4M3FN,
                            ptr_list[idx + 5],
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        ),
                        make_ptr(
                            c_dtype_cutlass,
                            ptr_list[idx + 6],
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        make_ptr(
                            cutlass.Float32,
                            ptr_list[idx + 7],
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        ),
                    ]
                )
                idx += 8
            real_args.append(
                make_ptr(
                    cutlass.Int32, masked_m_ptr, cute.AddressSpace.gmem, assumed_align=4
                )
            )
            real_args.append(cuda.CUstream(stream_i))
            fn(*real_args)
            ex = fn._default_executor
            packed_args = ex._tls.packed_args
            ptr_buf = (ctypes.c_uint64 * total_slots)()
            base = ctypes.addressof(ptr_buf)
            for i in range(total_slots):
                packed_args[i] = base + i * 8
            capi_func = ex.capi_func
            _KEEPALIVE.append((kern, fn, ex, packed_args, ptr_buf))
        for i in range(G * 8):
            ptr_buf[i] = ptr_list[i]
        ptr_buf[G * 8] = masked_m_ptr
        ptr_buf[G * 8 + 1] = stream_i
        capi_func(packed_args)

    _DUAL_MASKED_CACHE[key] = runner
    return runner


@dsl_user_op
def _rcp_approx_ftz_f32(x: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [x.ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_markstein_f32(
    num: cutlass.Float32, den: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [num.ir_value(loc=loc, ip=ip), den.ir_value(loc=loc, ip=ip)],
            """{
            .reg .f32 r, e, nd, q;
            rcp.approx.ftz.f32 r, $2;
            neg.f32 nd, $2;
            fma.rn.f32 e, nd, r, 0f3F800000;
            fma.rn.f32 r, r, e, r;
            mul.rn.f32 q, $1, r;
            fma.rn.f32 e, nd, q, $1;
            fma.rn.f32 $0, e, r, q;
        }""",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _fast_silu_f32(g: cutlass.Float32) -> cutlass.Float32:
    g_safe = cute.arch.fmax(g, cutlass.Float32(-88.0))
    den = cutlass.Float32(1.0) + cute.math.exp2(
        g_safe * cutlass.Float32(-1.4426950408889634), fastmath=True
    )
    return _div_markstein_f32(g_safe, den)


@dsl_user_op
def _max16_abs_f32(
    y0,
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    y7,
    y8,
    y9,
    y10,
    y11,
    y12,
    y13,
    y14,
    y15,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                x.ir_value(loc=loc, ip=ip)
                for x in (
                    y0,
                    y1,
                    y2,
                    y3,
                    y4,
                    y5,
                    y6,
                    y7,
                    y8,
                    y9,
                    y10,
                    y11,
                    y12,
                    y13,
                    y14,
                    y15,
                )
            ],
            """{
            .reg .f32 m, t;
            abs.f32 m, $1;
            abs.f32 t, $2; max.f32 m, m, t;
            abs.f32 t, $3; max.f32 m, m, t;
            abs.f32 t, $4; max.f32 m, m, t;
            abs.f32 t, $5; max.f32 m, m, t;
            abs.f32 t, $6; max.f32 m, m, t;
            abs.f32 t, $7; max.f32 m, m, t;
            abs.f32 t, $8; max.f32 m, m, t;
            abs.f32 t, $9; max.f32 m, m, t;
            abs.f32 t, $10; max.f32 m, m, t;
            abs.f32 t, $11; max.f32 m, m, t;
            abs.f32 t, $12; max.f32 m, m, t;
            abs.f32 t, $13; max.f32 m, m, t;
            abs.f32 t, $14; max.f32 m, m, t;
            abs.f32 t, $15; max.f32 m, m, t;
            abs.f32 t, $16; max.f32 m, m, t;
            mov.f32 $0, m;
        }""",
            "=f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack8_f32_to_e2m1_u32(
    v0, v1, v2, v3, v4, v5, v6, v7, inv_sf, *, loc=None, ip=None
) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                x.ir_value(loc=loc, ip=ip)
                for x in (v0, v1, v2, v3, v4, v5, v6, v7, inv_sf)
            ],
            """{
            .reg .f32 q0, q1, q2, q3, q4, q5, q6, q7;
            .reg .b8 b0, b1, b2, b3;
            .reg .b16 h0, h1;
            mul.f32 q0, $1, $9; mul.f32 q1, $2, $9; mul.f32 q2, $3, $9; mul.f32 q3, $4, $9;
            mul.f32 q4, $5, $9; mul.f32 q5, $6, $9; mul.f32 q6, $7, $9; mul.f32 q7, $8, $9;
            cvt.rn.satfinite.e2m1x2.f32 b0, q1, q0;
            cvt.rn.satfinite.e2m1x2.f32 b1, q3, q2;
            cvt.rn.satfinite.e2m1x2.f32 b2, q5, q4;
            cvt.rn.satfinite.e2m1x2.f32 b3, q7, q6;
            mov.b16 h0, {b0, b1};
            mov.b16 h1, {b2, b3};
            mov.b32 $0, {h0, h1};
        }""",
            "=r,f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _pack16_f32_to_e2m1_u64(
    v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, inv_sf
):
    lo = _pack8_f32_to_e2m1_u32(v0, v1, v2, v3, v4, v5, v6, v7, inv_sf)
    hi = _pack8_f32_to_e2m1_u32(v8, v9, v10, v11, v12, v13, v14, v15, inv_sf)
    return (cutlass.Uint64(hi) << cutlass.Uint64(32)) | cutlass.Uint64(lo)


class Sm100Nvfp4MaskedDualGemmSwiGLUQuantV2:
    """Persistent 3D-TMA Fused Dual-B [B1|B2] + SwiGLU + SMEM-Coalesced NVFP4 Quantization."""

    def __init__(self, use_2cta: bool = True, tile_n: int = 128, pdl_safe: bool = True):
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = 16
        self.use_2cta = use_2cta
        self.tile_m = 256 if use_2cta else 128
        self.tile_n = tile_n
        self.cluster_shape_mn = (2, 1) if use_2cta else (1, 1)
        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta else tcgen05.CtaGroup.ONE
        self.mma_tiler_f = (256, 256, 256) if use_2cta else (128, tile_n, 256)
        self.mma_tiler_half = (256, 128, 256) if use_2cta else (128, tile_n, 256)
        self.mma_tiler_sfb_half = (128, 128, 256)
        self.epi_16 = (128, 16)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192
        self.num_ab_stage = 5 if (use_2cta or tile_n == 64) else 4
        self.num_acc_stage = 2 if ((not use_2cta) and tile_n == 64) else 1
        self.overlapping_accum = use_2cta
        self.num_sfa_tmem_cols = 16
        self.num_sfb_tmem_cols = 32
        self.num_sf_tmem_cols = 48
        self.num_accumulator_tmem_cols = (512 - 48) if use_2cta else 256
        self.num_tmem_alloc_cols = 512
        self.pdl_safe = pdl_safe
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=160)
        self.sched_sync_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=192)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b1_ptr: cute.Pointer,
        b2_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb1_ptr: cute.Pointer,
        sfb2_ptr: cute.Pointer,
        diq_ptr: cute.Pointer,
        diq_sf_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        masked_m_ptr: cute.Pointer,
        stream: cuda.CUstream,
        shape_cfg: cutlass.Constexpr[tuple],
    ):
        G, m_max, sfa_m, diq_rm, n, k = shape_cfg
        rk2 = _ceil_div(n // 16, 4)

        tiled_mma_f = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            _OP_K_MAJOR,
            _OP_K_MAJOR,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler_f[:2],
        )
        tiled_mma_half = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            _OP_K_MAJOR,
            _OP_K_MAJOR,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler_half[:2],
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            _OP_K_MAJOR,
            _OP_K_MAJOR,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            (128, 128),
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma_f.thr_id.shape,)
        )
        cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma_sfb.thr_id.shape,)
        )

        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_f, self.mma_tiler_f, self.ab_dtype, self.num_ab_stage
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_f, self.mma_tiler_f, self.ab_dtype, self.num_ab_stage
        )
        sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_f, self.mma_tiler_f, self.sf_vec_size, self.num_ab_stage
        )
        sfb_full_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_f, self.mma_tiler_f, self.sf_vec_size, self.num_ab_stage
        )
        sfb_half_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_half, self.mma_tiler_half, self.sf_vec_size, self.num_ab_stage
        )

        a_smem = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        b_smem = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        sfa_smem = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
        sfb_half_smem = cute.slice_(sfb_half_smem_layout_staged, (None, None, None, 0))

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m_max, cute.assume(k, 32), G),
                stride=(cute.assume(k, 32), 1, cute.assume(m_max * k, 32)),
            ),
        )
        b1_tensor = cute.make_tensor(
            b1_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), G),
                stride=(cute.assume(k, 32), 1, cute.assume(2 * n * k, 32)),
            ),
        )
        b2_tensor = cute.make_tensor(
            b2_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), G),
                stride=(cute.assume(k, 32), 1, cute.assume(2 * n * k, 32)),
            ),
        )
        sfa_tensor = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF((sfa_m, k, G), 16)
        )

        sfb_1g = blockscaled_utils.tile_atom_to_shape_SF((n, k, 1), 16)
        rn = n // 128
        rk1 = _ceil_div(k // 16, 4)
        sfb_exp_stride = 2 * rn * rk1 * 512
        sfb_3d_layout = cute.make_layout(
            (sfb_1g.shape[0], sfb_1g.shape[1], G),
            stride=(sfb_1g.stride[0], sfb_1g.stride[1], sfb_exp_stride),
        )
        sfb1_tensor = cute.make_tensor(sfb1_ptr, sfb_3d_layout)
        sfb2_tensor = cute.make_tensor(sfb2_ptr, sfb_3d_layout)

        diq_u64 = cute.make_tensor(
            diq_ptr,
            cute.make_layout(
                (G, m_max, n // 16), stride=(m_max * (n // 16), n // 16, 1)
            ),
        )
        diq_sf_u32 = cute.make_tensor(
            diq_sf_ptr,
            cute.make_layout(
                (G, diq_rm, rk2, 32, 4),
                stride=(diq_rm * rk2 * 128, rk2 * 128, 128, 4, 1),
            ),
        )

        alpha_tensor = cute.make_tensor(alpha_ptr, cute.make_layout((G,), stride=(1,)))
        gs_tensor = cute.make_tensor(gs_ptr, cute.make_layout((G,), stride=(1,)))
        masked_m_tensor = cute.make_tensor(
            masked_m_ptr, cute.make_layout((G,), stride=(1,))
        )

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma_f.thr_id
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma_f.thr_id
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma_f.thr_id
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma_f.thr_id
        )

        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem,
            self.mma_tiler_f,
            tiled_mma_f,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b1_tensor,
            b_smem,
            self.mma_tiler_f,
            tiled_mma_f,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b2_tensor,
            b_smem,
            self.mma_tiler_f,
            tiled_mma_f,
            cluster_layout_vmnk.shape,
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem,
            self.mma_tiler_f,
            tiled_mma_f,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb1_tensor,
            sfb_half_smem,
            self.mma_tiler_sfb_half,
            tiled_mma_sfb,
            cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb2_tensor,
            sfb_half_smem,
            self.mma_tiler_sfb_half,
            tiled_mma_sfb,
            cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        atom_thr_size = 2 if self.use_2cta else 1
        if cutlass.const_expr(self.use_2cta):
            self.num_tma_load_bytes = (
                cute.size_in_bytes(self.ab_dtype, a_smem)
                + cute.size_in_bytes(self.ab_dtype, b_smem)
                + cute.size_in_bytes(self.sf_dtype, sfa_smem)
                + cute.size_in_bytes(self.sf_dtype, sfb_half_smem) * 2
            ) * 2
        else:
            self.num_tma_load_bytes = (
                cute.size_in_bytes(self.ab_dtype, a_smem)
                + cute.size_in_bytes(self.ab_dtype, b_smem) * 2
                + cute.size_in_bytes(self.sf_dtype, sfa_smem)
                + cute.size_in_bytes(self.sf_dtype, sfb_half_smem) * 2
            )

        cn = n // self.tile_n
        max_units = 74 if self.use_2cta else 148
        worst_units = G * ((m_max + self.tile_m - 1) // self.tile_m) * cn
        active_units = min(worst_units, max_units)
        grid = (active_units * atom_thr_size, 1, 1)

        b2_cosize = 16 if self.use_2cta else cute.cosize(b_smem_layout_staged.outer)
        sfb2_cosize = 16 if self.use_2cta else cute.cosize(sfb_half_smem_layout_staged)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            s_cum: cute.struct.MemRange[cutlass.Int32, G + 1]
            s_mg: cute.struct.MemRange[cutlass.Int32, G]
            s_diq: cute.struct.Align[cute.struct.MemRange[cutlass.Uint64, 128 * 5], 128]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(a_smem_layout_staged.outer)
                ],
                1024,
            ]
            sB1: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(b_smem_layout_staged.outer)
                ],
                1024,
            ]
            sB2: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, b2_cosize], 1024]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(sfa_smem_layout_staged)
                ],
                1024,
            ]
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(sfb_full_smem_layout_staged)
                ],
                1024,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sfb2_cosize], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma_f,
            tiled_mma_half,
            tiled_mma_sfb,
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
            diq_u64,
            diq_sf_u32,
            alpha_tensor,
            gs_tensor,
            masked_m_tensor,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_full_smem_layout_staged,
            sfb_half_smem_layout_staged,
            G,
            m_max,
            diq_rm,
            cn,
            active_units,
        ).launch(
            grid=grid,
            block=[192, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma_f: cute.TiledMma,
        tiled_mma_half: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
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
        mDiqU64: cute.Tensor,
        mDiqSfU32: cute.Tensor,
        mAlpha: cute.Tensor,
        mGs: cute.Tensor,
        mMaskedM: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_full_smem_layout_staged: cute.Layout,
        sfb_half_smem_layout_staged: cute.Layout,
        G: cutlass.Constexpr[int],
        m_max: cutlass.Constexpr[int],
        diq_rm: cutlass.Constexpr[int],
        cn: cutlass.Constexpr[int],
        active_units: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b1)
            cpasync.prefetch_descriptor(tma_atom_b2)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb1)
            cpasync.prefetch_descriptor(tma_atom_sfb2)

        cta_rank = bidx % 2 if cutlass.const_expr(self.use_2cta) else 0
        unit_id = bidx // 2 if cutlass.const_expr(self.use_2cta) else bidx
        is_leader_cta = cta_rank == 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        s_cum = storage.s_cum.get_tensor(cute.make_layout((G + 1,), stride=(1,)))
        s_mg = storage.s_mg.get_tensor(cute.make_layout((G,), stride=(1,)))
        s_diq = storage.s_diq.get_tensor(cute.make_layout((128, 5), stride=(5, 1)))

        for g_load in cutlass.range(tidx, G, 192, unroll=1):
            s_mg[g_load] = mMaskedM[g_load]

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        num_acc_cons = 8 if self.use_2cta else 4
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_acc_cons
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=self.use_2cta,
            two_cta_tmem_dealloc_mbar_ptr=(
                storage.tmem_dealloc_mbar_ptr if self.use_2cta else None
            ),
        )
        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        if not self.use_2cta:
            self.sched_sync_barrier.arrive_and_wait()

        if tidx == 0:
            running = cutlass.Int32(0)
            s_cum[0] = running
            for g_i in cutlass.range(0, G, 1, unroll=1):
                mg_val = s_mg[g_i]
                cm_val = (
                    (mg_val + self.tile_m - 1) // self.tile_m
                    if mg_val > 0
                    else cutlass.Int32(0)
                )
                running = running + cm_val * cn
                s_cum[g_i + 1] = running
        self.sched_sync_barrier.arrive_and_wait()
        total_units = s_cum[G]

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB1 = storage.sB1.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB1 = storage.sSFB1.get_tensor(sfb_full_smem_layout_staged)

        if cutlass.const_expr(self.use_2cta):
            sB2 = sB1
            h_l = sfb_half_smem_layout_staged
            f_l = sfb_full_smem_layout_staged
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
            sSFB2 = storage.sSFB2.get_tensor(sfb_half_smem_layout_staged)
            sSFB1_ld = sSFB1
            sSFB2_ld = sSFB2

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler_f, (None, 0, None)), (None, None, None)
        )
        gB1_nkl = cute.local_tile(
            mB1_nkl,
            cute.slice_(self.mma_tiler_half, (0, None, None)),
            (None, None, None),
        )
        gB2_nkl = cute.local_tile(
            mB2_nkl,
            cute.slice_(self.mma_tiler_half, (0, None, None)),
            (None, None, None),
        )
        if cutlass.const_expr(self.use_2cta):
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
            cute.slice_(self.mma_tiler_sfb_half, (0, None, None)),
            (None, None, None),
        )
        gSFB2_nkl = cute.local_tile(
            mSFB2_nkl,
            cute.slice_(self.mma_tiler_sfb_half, (0, None, None)),
            (None, None, None),
        )
        k_block_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma_f = tiled_mma_f.get_slice(cta_rank)
        thr_mma_sfb = tiled_mma_sfb.get_slice(0)
        tCgA = thr_mma_f.partition_A(gA_mkl)
        tCgB1 = thr_mma_f.partition_B(gB1_nkl)
        tCgB2 = thr_mma_f.partition_B(gB2_nkl)
        tCgSFA = thr_mma_f.partition_A(gSFA_mkl)
        tCgSFB1 = thr_mma_sfb.partition_B(gSFB1_nkl)
        tCgSFB2 = thr_mma_sfb.partition_B(gSFB2_nkl)

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

        acc_shape_f = tiled_mma_f.partition_shape_C(self.mma_tiler_f[:2])
        if cutlass.const_expr(self.use_2cta):
            tCtAcc_fake_f = tiled_mma_f.make_fragment_C(cute.append(acc_shape_f, 2))
            tCtAcc_fake_f = cute.make_tensor(
                tCtAcc_fake_f.iterator,
                cute.make_layout(
                    tCtAcc_fake_f.shape,
                    stride=(
                        tCtAcc_fake_f.stride[0],
                        tCtAcc_fake_f.stride[1],
                        tCtAcc_fake_f.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake_f.stride[0][1],
                    ),
                ),
            )
        else:
            tCtAcc_fake_f = tiled_mma_f.make_fragment_C(acc_shape_f)

        # 1. TMA Warp
        if warp_idx == self.tma_warp_id:
            if cutlass.const_expr(self.pdl_safe):
                cute.arch.griddepcontrol_wait()
            else:
                cute.arch.griddepcontrol_launch_dependents()
            ab_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            work_u = cutlass.Int32(unit_id)
            cur_g = cutlass.Int32(0)
            while work_u < total_units:
                while s_cum[cur_g + 1] <= work_u:
                    cur_g = cur_g + 1
                rel = work_u - s_cum[cur_g]
                coord_m = rel // cn
                coord_n = rel % cn
                slice_n = (
                    coord_n // 2 if cutlass.const_expr(self.tile_n == 64) else coord_n
                )

                tAgA_s = tAgA[(None, coord_m, None, cur_g)]
                tBgB1_s = tBgB1[(None, coord_n, None, cur_g)]
                tBgB2_s = tBgB2[(None, coord_n, None, cur_g)]
                tAgSFA_s = tAgSFA[(None, coord_m, None, cur_g)]
                tBgSFB1_s = tBgSFB1[(None, slice_n, None, cur_g)]
                tBgSFB2_s = tBgSFB2[(None, slice_n, None, cur_g)]

                ab_prod.reset_count()
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
                    )
                    if cutlass.const_expr(self.use_2cta):
                        if is_leader_cta:
                            cute.copy(
                                tma_atom_b1,
                                tBgB1_s[(None, cnt)],
                                tBsB1[(None, stg)],
                                tma_bar_ptr=bar,
                            )
                        else:
                            cute.copy(
                                tma_atom_b2,
                                tBgB2_s[(None, cnt)],
                                tBsB1[(None, stg)],
                                tma_bar_ptr=bar,
                            )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_s[(None, cnt)],
                            tAsSFA[(None, stg)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_atom_sfb1,
                            tBgSFB1_s[(None, cnt)],
                            tBsSFB1[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=cutlass.Int16(3),
                        )
                        cute.copy(
                            tma_atom_sfb2,
                            tBgSFB2_s[(None, cnt)],
                            tBsSFB2[(None, stg)],
                            tma_bar_ptr=bar,
                            mcast_mask=cutlass.Int16(3),
                        )
                    else:
                        cute.copy(
                            tma_atom_b1,
                            tBgB1_s[(None, cnt)],
                            tBsB1[(None, stg)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_atom_b2,
                            tBgB2_s[(None, cnt)],
                            tBsB2[(None, stg)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_s[(None, cnt)],
                            tAsSFA[(None, stg)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_atom_sfb1,
                            tBgSFB1_s[(None, cnt)],
                            tBsSFB1[(None, stg)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_atom_sfb2,
                            tBgSFB2_s[(None, cnt)],
                            tBsSFB2[(None, stg)],
                            tma_bar_ptr=bar,
                        )
                    ab_prod.advance()
                    peek_empty = cutlass.Boolean(1)
                    if ab_prod.count < k_block_cnt:
                        peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
                work_u = work_u + active_units

            if cutlass.const_expr(self.use_2cta):
                ab_pipeline.producer_tail(ab_prod)
            if cutlass.const_expr(self.pdl_safe):
                cute.arch.griddepcontrol_launch_dependents()

        # 2. MMA Warp
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_f,
                self.mma_tiler_f,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            copy_atom_s2t = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype
            )
            tCsSFA_c = cute.filter_zeros(sSFA)
            tCtSFA_c = cute.filter_zeros(tCtSFA)
            tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_c)
            thr_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
            tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_sfa, thr_s2t_sfa.partition_S(tCsSFA_c)
            )
            tCtSFA_s2t = thr_s2t_sfa.partition_D(tCtSFA_c)

            tCrA = tiled_mma_f.make_fragment_A(sA)
            tCrB1 = tiled_mma_f.make_fragment_B(sB1)

            if cutlass.const_expr(self.use_2cta):
                tCtAccF_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake_f.layout)
                sfb_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr
                    + self.num_accumulator_tmem_cols
                    + self.num_sfa_tmem_cols,
                    dtype=self.sf_dtype,
                )
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma_f,
                    self.mma_tiler_f,
                    self.sf_vec_size,
                    cute.slice_(sfb_full_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
                tCsSFB_c = cute.filter_zeros(sSFB1)
                tCtSFB_c = cute.filter_zeros(tCtSFB)
                tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_c)
                thr_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
                tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
                    tiled_copy_s2t_sfb, thr_s2t_sfb.partition_S(tCsSFB_c)
                )
                tCtSFB_s2t = thr_s2t_sfb.partition_D(tCtSFB_c)
            else:
                tCrB2 = tiled_mma_f.make_fragment_B(sB2)
                sfb1_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + 256 + 16, dtype=self.sf_dtype
                )
                sfb2_tmem_ptr = cute.recast_ptr(
                    acc_tmem_ptr + 256 + 32, dtype=self.sf_dtype
                )
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma_f,
                    self.mma_tiler_f,
                    self.sf_vec_size,
                    cute.slice_(sfb_half_smem_layout_staged, (None, None, None, 0)),
                )
                tCtSFB1 = cute.make_tensor(sfb1_tmem_ptr, tCtSFB_layout)
                tCtSFB2 = cute.make_tensor(sfb2_tmem_ptr, tCtSFB_layout)
                tCsSFB1_c = cute.filter_zeros(sSFB1)
                tCtSFB1_c = cute.filter_zeros(tCtSFB1)
                tiled_copy_s2t_sfb1 = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB1_c)
                thr_s2t_sfb1 = tiled_copy_s2t_sfb1.get_slice(0)
                tCsSFB1_s2t = tcgen05.get_s2t_smem_desc_tensor(
                    tiled_copy_s2t_sfb1, thr_s2t_sfb1.partition_S(tCsSFB1_c)
                )
                tCtSFB1_s2t = thr_s2t_sfb1.partition_D(tCtSFB1_c)
                tCsSFB2_c = cute.filter_zeros(sSFB2)
                tCtSFB2_c = cute.filter_zeros(tCtSFB2)
                tiled_copy_s2t_sfb2 = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB2_c)
                thr_s2t_sfb2 = tiled_copy_s2t_sfb2.get_slice(0)
                tCsSFB2_s2t = tcgen05.get_s2t_smem_desc_tensor(
                    tiled_copy_s2t_sfb2, thr_s2t_sfb2.partition_S(tCsSFB2_c)
                )
                tCtSFB2_s2t = thr_s2t_sfb2.partition_D(tCtSFB2_c)

            ab_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            work_u_m = cutlass.Int32(unit_id)
            cur_g_m = cutlass.Int32(0)

            while work_u_m < total_units:
                while s_cum[cur_g_m + 1] <= work_u_m:
                    cur_g_m = cur_g_m + 1
                rel_m = work_u_m - s_cum[cur_g_m]
                coord_n_m = rel_m % cn

                ab_cons.reset_count()
                peek_full = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_full = ab_pipeline.consumer_try_wait(ab_cons)
                    acc_pipeline.producer_acquire(acc_prod)

                if cutlass.const_expr(self.use_2cta):
                    acc_stage_idx = acc_prod.phase ^ 1
                    tCtAccF = tCtAccF_base[(None, None, None, acc_stage_idx)]
                    tiled_mma_f.set(tcgen05.Field.ACCUMULATE, False)
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
                                tiled_mma_f.set(
                                    tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator
                                )
                                tiled_mma_f.set(
                                    tcgen05.Field.SFB, tCtSFB[(None, None, kb)].iterator
                                )
                                cute.gemm(
                                    tiled_mma_f,
                                    tCtAccF,
                                    tCrA[(None, None, kb, st)],
                                    tCrB1[(None, None, kb, st)],
                                    tCtAccF,
                                )
                                tiled_mma_f.set(tcgen05.Field.ACCUMULATE, True)
                            ab_pipeline.consumer_release(ab_cons)
                        ab_cons.advance()
                        peek_full = cutlass.Boolean(1)
                        if ab_cons.count < k_block_cnt and is_leader_cta:
                            peek_full = ab_pipeline.consumer_try_wait(ab_cons)
                else:
                    acc_base_col = acc_prod.index * (2 * self.tile_n)
                    tCtAcc1 = cute.make_tensor(
                        acc_tmem_ptr + acc_base_col, tCtAcc_fake_f.layout
                    )[(None, None, None)]
                    tCtAcc2 = cute.make_tensor(
                        acc_tmem_ptr + acc_base_col + self.tile_n, tCtAcc_fake_f.layout
                    )[(None, None, None)]
                    tCtSFB1_mma = tCtSFB1
                    tCtSFB2_mma = tCtSFB2
                    if cutlass.const_expr(self.tile_n == 64):
                        off = cutlass.Int32((coord_n_m % 2) * 2)
                        tCtSFB1_mma = cute.make_tensor(
                            cute.recast_ptr(
                                acc_tmem_ptr + 256 + 16 + off, dtype=self.sf_dtype
                            ),
                            tCtSFB_layout,
                        )
                        tCtSFB2_mma = cute.make_tensor(
                            cute.recast_ptr(
                                acc_tmem_ptr + 256 + 32 + off, dtype=self.sf_dtype
                            ),
                            tCtSFB_layout,
                        )

                    tiled_mma_f.set(tcgen05.Field.ACCUMULATE, False)
                    for kt in cutlass.range(0, k_block_cnt, 1, unroll=1):
                        ab_pipeline.consumer_wait(ab_cons, peek_full)
                        st = ab_cons.index
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_s2t[(None, None, None, None, st)],
                            tCtSFA_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb1,
                            tCsSFB1_s2t[(None, None, None, None, st)],
                            tCtSFB1_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb2,
                            tCsSFB2_s2t[(None, None, None, None, st)],
                            tCtSFB2_s2t,
                        )
                        for kb in cutlass.range(4, unroll_full=True):
                            tiled_mma_f.set(
                                tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator
                            )
                            tiled_mma_f.set(
                                tcgen05.Field.SFB,
                                tCtSFB1_mma[(None, None, kb)].iterator,
                            )
                            cute.gemm(
                                tiled_mma_f,
                                tCtAcc1,
                                tCrA[(None, None, kb, st)],
                                tCrB1[(None, None, kb, st)],
                                tCtAcc1,
                            )
                            tiled_mma_f.set(
                                tcgen05.Field.SFB,
                                tCtSFB2_mma[(None, None, kb)].iterator,
                            )
                            cute.gemm(
                                tiled_mma_f,
                                tCtAcc2,
                                tCrA[(None, None, kb, st)],
                                tCrB2[(None, None, kb, st)],
                                tCtAcc2,
                            )
                            tiled_mma_f.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipeline.consumer_release(ab_cons)
                        ab_cons.advance()
                        peek_full = cutlass.Boolean(1)
                        if ab_cons.count < k_block_cnt:
                            peek_full = ab_pipeline.consumer_try_wait(ab_cons)

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_prod)
                acc_prod.advance()
                work_u_m = work_u_m + active_units

            acc_pipeline.producer_tail(acc_prod)

        # 3. Epilogue Warps (0..3)
        if warp_idx < 4:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x16, tcgen05.Pack.NONE),
                cutlass.Float32,
            )
            acc_shape_half = tiled_mma_half.partition_shape_C(
                (self.tile_m, self.tile_n)
            )
            tCtAcc_proto = cute.make_tensor(
                acc_tmem_ptr, tiled_mma_half.make_fragment_C(acc_shape_half).layout
            )
            tAcc_epi16 = cute.flat_divide(
                tCtAcc_proto[((None, None), 0, 0)], self.epi_16
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi16[(None, None, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

            c_ident16 = cute.flat_divide(
                cute.make_identity_tensor((128, self.tile_n)), self.epi_16
            )
            tTR_cC16 = thr_copy_t2r.partition_D(c_ident16)
            rG = cute.make_rmem_tensor(
                tTR_cC16[(None, None, None, 0, 0)].shape, cutlass.Float32
            )
            rU = cute.make_rmem_tensor(
                tTR_cC16[(None, None, None, 0, 0)].shape, cutlass.Float32
            )
            rY = cute.make_rmem_tensor((16,), cutlass.Float32)
            rSF4 = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
            rSF4_u32 = cute.make_tensor(
                cute.recast_ptr(rSF4.iterator, dtype=cutlass.Uint32),
                cute.make_layout((1,)),
            )

            lane_idx = tidx & 31
            grp_row = tidx >> 2
            grp_col = tidx & 3
            num_64_subtiles = self.tile_n // 64

            acc_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            work_u_e = cutlass.Int32(unit_id)
            cur_g_e = cutlass.Int32(0)

            while work_u_e < total_units:
                while s_cum[cur_g_e + 1] <= work_u_e:
                    cur_g_e = cur_g_e + 1
                rel_e = work_u_e - s_cum[cur_g_e]
                coord_m_e = rel_e // cn
                coord_n_e = rel_e % cn

                alpha_val = cutlass.Float32(mAlpha[cur_g_e])
                gs_val = cutlass.Float32(mGs[cur_g_e])
                rcp6 = _rcp_approx_ftz_f32(cutlass.Float32(6.0))
                rcp_gs = _rcp_approx_ftz_f32(gs_val)
                mg_cur = s_mg[cur_g_e]
                rest_m = (
                    coord_m_e * 2 + cta_rank
                    if cutlass.const_expr(self.use_2cta)
                    else coord_m_e
                )
                row = rest_m * 128 + tidx

                acc_pipeline.consumer_wait(acc_cons)

                if cutlass.const_expr(self.use_2cta):
                    ph = acc_cons.phase
                    base_col = cutlass.Int32(0) if ph == 0 else cutlass.Int32(208)
                    tG_cur = cute.make_tensor(
                        acc_tmem_ptr + base_col, tCtAcc_proto.layout
                    )
                    tU_cur = cute.make_tensor(
                        acc_tmem_ptr + base_col + 128, tCtAcc_proto.layout
                    )
                else:
                    base_col_1c = acc_cons.index * (2 * self.tile_n)
                    tG_cur = cute.make_tensor(
                        acc_tmem_ptr + base_col_1c, tCtAcc_proto.layout
                    )
                    tU_cur = cute.make_tensor(
                        acc_tmem_ptr + base_col_1c + self.tile_n, tCtAcc_proto.layout
                    )

                tTR_G = cute.group_modes(
                    thr_copy_t2r.partition_S(
                        cute.flat_divide(tG_cur[((None, None), 0, 0)], self.epi_16)
                    ),
                    3,
                    5,
                )
                tTR_U = cute.group_modes(
                    thr_copy_t2r.partition_S(
                        cute.flat_divide(tU_cur[((None, None), 0, 0)], self.epi_16)
                    ),
                    3,
                    5,
                )

                for step_s in cutlass.range_constexpr(num_64_subtiles):
                    if cutlass.const_expr(self.use_2cta):
                        sub_i = (1 - step_s) if ph == 0 else step_s
                    else:
                        sub_i = step_s

                    for k4 in cutlass.range_constexpr(4):
                        blk16 = sub_i * 4 + k4
                        cute.copy(tiled_copy_t2r, tTR_G[(None, None, None, blk16)], rG)
                        cute.copy(tiled_copy_t2r, tTR_U[(None, None, None, blk16)], rU)
                        if cutlass.const_expr(self.use_2cta):
                            if step_s == 0 and k4 == 3:
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    acc_pipeline.consumer_release(acc_cons)
                                acc_cons.advance()
                        else:
                            if step_s == num_64_subtiles - 1 and k4 == 3:
                                cute.arch.fence_view_async_tmem_load()
                                with cute.arch.elect_one():
                                    acc_pipeline.consumer_release(acc_cons)
                                acc_cons.advance()

                        rG.store(
                            (rG.load() * alpha_val)
                            .to(cutlass.BFloat16)
                            .to(cutlass.Float32)
                        )
                        rU.store(
                            (rU.load() * alpha_val)
                            .to(cutlass.BFloat16)
                            .to(cutlass.Float32)
                        )
                        for j in cutlass.range_constexpr(16):
                            rY[j] = (
                                (_fast_silu_f32(rG[j]) * rU[j])
                                .to(cutlass.BFloat16)
                                .to(cutlass.Float32)
                            )

                        amax = _max16_abs_f32(
                            rY[0],
                            rY[1],
                            rY[2],
                            rY[3],
                            rY[4],
                            rY[5],
                            rY[6],
                            rY[7],
                            rY[8],
                            rY[9],
                            rY[10],
                            rY[11],
                            rY[12],
                            rY[13],
                            rY[14],
                            rY[15],
                        )
                        rSF4[k4] = (gs_val * (amax * rcp6)).to(cutlass.Float8E4M3FN)
                        sf_dec = rSF4[k4].to(cutlass.Float32)
                        inv_sf = (
                            _rcp_approx_ftz_f32(sf_dec * rcp_gs)
                            if amax > cutlass.Float32(0.0)
                            else cutlass.Float32(0.0)
                        )
                        q64 = _pack16_f32_to_e2m1_u64(
                            rY[0],
                            rY[1],
                            rY[2],
                            rY[3],
                            rY[4],
                            rY[5],
                            rY[6],
                            rY[7],
                            rY[8],
                            rY[9],
                            rY[10],
                            rY[11],
                            rY[12],
                            rY[13],
                            rY[14],
                            rY[15],
                            inv_sf,
                        )
                        if row < mg_cur:
                            s_diq[tidx, k4] = q64
                        else:
                            s_diq[tidx, k4] = cutlass.Uint64(0)

                    rk_idx = coord_n_e * num_64_subtiles + sub_i
                    if rest_m < diq_rm:
                        if row < mg_cur:
                            mDiqSfU32[cur_g_e, rest_m, rk_idx, lane_idx, warp_idx] = (
                                rSF4_u32[0]
                            )
                        else:
                            mDiqSfU32[cur_g_e, rest_m, rk_idx, lane_idx, warp_idx] = (
                                cutlass.Uint32(0)
                            )

                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id, number_of_threads=128
                    )
                    col_u64_base = (coord_n_e * num_64_subtiles + sub_i) * 4
                    for ir in cutlass.range_constexpr(4):
                        r_loc = ir * 32 + grp_row
                        g_row = rest_m * 128 + r_loc
                        if g_row < m_max:
                            mDiqU64[cur_g_e, g_row, col_u64_base + grp_col] = s_diq[
                                r_loc, grp_col
                            ]
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id, number_of_threads=128
                    )

                work_u_e = work_u_e + active_units

            tmem.relinquish_alloc_permit()
            if cutlass.const_expr(self.use_2cta):
                cute.arch.barrier(
                    barrier_id=self.epilog_sync_bar_id, number_of_threads=128
                )
            tmem.free(acc_tmem_ptr)


def _get_masked_dual_quant_runner(
    use_2cta: bool, tile_n: int, shape_cfg: Tuple[int, int, int, int, int, int]
):
    key = (use_2cta, tile_n, shape_cfg)
    runner = _DUAL_QUANT_CACHE.get(key)
    if runner is not None:
        return runner

    kern = Sm100Nvfp4MaskedDualGemmSwiGLUQuantV2(
        use_2cta=use_2cta, tile_n=tile_n, pdl_safe=True
    )
    fn = cute.compile(
        kern,
        make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Uint64, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4),
        cuda.CUstream(0),
        shape_cfg,
    )
    capi_func = None
    packed_args = None
    ptr_buf = None

    def runner(
        a_p: int,
        b1_p: int,
        b2_p: int,
        sfa_p: int,
        sfb1_p: int,
        sfb2_p: int,
        diq_p: int,
        diq_sf_p: int,
        al_p: int,
        gs_p: int,
        mm_p: int,
        st_i: int,
    ):
        nonlocal capi_func, packed_args, ptr_buf
        if capi_func is None:
            fn(
                make_ptr(
                    cutlass.Float4E2M1FN, a_p, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Float4E2M1FN, b1_p, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Float4E2M1FN, b2_p, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Float8E4M3FN,
                    sfa_p,
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                ),
                make_ptr(
                    cutlass.Float8E4M3FN,
                    sfb1_p,
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                ),
                make_ptr(
                    cutlass.Float8E4M3FN,
                    sfb2_p,
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                ),
                make_ptr(
                    cutlass.Uint64, diq_p, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Uint32, diq_sf_p, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Float32, al_p, cute.AddressSpace.gmem, assumed_align=4
                ),
                make_ptr(
                    cutlass.Float32, gs_p, cute.AddressSpace.gmem, assumed_align=4
                ),
                make_ptr(cutlass.Int32, mm_p, cute.AddressSpace.gmem, assumed_align=4),
                cuda.CUstream(st_i),
            )
            ex = fn._default_executor
            packed_args = ex._tls.packed_args
            ptr_buf = (ctypes.c_uint64 * 12)()
            base = ctypes.addressof(ptr_buf)
            for i in range(12):
                packed_args[i] = base + i * 8
            capi_func = ex.capi_func
            _KEEPALIVE.append((kern, fn, ex, packed_args, ptr_buf))
        ptr_buf[0] = a_p
        ptr_buf[1] = b1_p
        ptr_buf[2] = b2_p
        ptr_buf[3] = sfa_p
        ptr_buf[4] = sfb1_p
        ptr_buf[5] = sfb2_p
        ptr_buf[6] = diq_p
        ptr_buf[7] = diq_sf_p
        ptr_buf[8] = al_p
        ptr_buf[9] = gs_p
        ptr_buf[10] = mm_p
        ptr_buf[11] = st_i
        capi_func(packed_args)

    _DUAL_QUANT_CACHE[key] = runner
    return runner


def nvfp4_masked_grouped_moe(
    hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]],
    input_global_scale: Optional[torch.Tensor],
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2: torch.Tensor,
    a2_global_scale: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alpha: torch.Tensor,
    masked_m: torch.Tensor,
    down_start_event: Optional[torch.cuda.Event] = None,
    activation: str = "silu",
    use_fused_dual_gemm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Blackwell SM100 CuTe DSL 3D masked NVFP4 grouped MoE pipeline."""
    from flashinfer import (
        scaled_fp4_grouped_quantize,
        silu_and_mul_scaled_nvfp4_experts_quantize,
    )

    if isinstance(hidden_states, tuple):
        hs_x, hs_sf = hidden_states
    else:
        hs_x, hs_sf = hidden_states, None

    n = w2.shape[-1] * 2
    gated = activation == "silu"
    if not gated and activation != "relu2":
        raise ValueError(
            f"nvfp4_masked_grouped_moe supports 'silu' or 'relu2', got {activation!r}"
        )
    gemm1_out_dim = 2 * n if gated else n

    if hs_sf is not None:
        a_q = hs_x.view(torch.uint8)
        a_q_sf = hs_sf.view(torch.float8_e4m3fn)
        if a_q.stride(1) == 1:
            m_max, k_by_2, G = a_q.shape
        else:
            G, m_max, k_by_2 = a_q.shape
        k = k_by_2 * 2
    else:
        G, m_max, k = hs_x.shape

    if m_max == 0:
        return (
            out
            if out is not None
            else torch.empty((G, 0, k), device=hs_x.device, dtype=torch.bfloat16)
        )

    if input_global_scale is not None and input_global_scale.numel() == 1 and G > 1:
        input_global_scale = input_global_scale.expand(G).contiguous()
    if a2_global_scale.numel() == 1 and G > 1:
        a2_global_scale = a2_global_scale.expand(G).contiguous()
    if w1_alpha.numel() == 1 and G > 1:
        w1_alpha = w1_alpha.expand(G).contiguous()
    if w2_alpha.numel() == 1 and G > 1:
        w2_alpha = w2_alpha.expand(G).contiguous()

    c_quant_fn = _get_c_quant_fn()
    padded_m = _ceil_div(m_max, 128) * 128
    rk1 = _ceil_div(k // 16, 4)
    rk2 = _ceil_div(n // 16, 4)

    ws_key = (hs_x.device, G, m_max, n, k, gated, use_fused_dual_gemm)
    ws = _MOE_WS_CACHE.get(ws_key)
    if ws is None:
        inter_dim = n if (gated and use_fused_dual_gemm) else gemm1_out_dim
        ws = (
            torch.zeros((G, m_max, k // 2), device=hs_x.device, dtype=torch.uint8),
            torch.zeros((G, padded_m, rk1), device=hs_x.device, dtype=torch.int32),
            torch.zeros(
                (G, m_max, inter_dim), device=hs_x.device, dtype=torch.bfloat16
            ),
            torch.zeros((G, m_max, n // 2), device=hs_x.device, dtype=torch.uint8),
            torch.zeros((G, padded_m, rk2), device=hs_x.device, dtype=torch.int32),
        )
        _MOE_WS_CACHE[ws_key] = ws
    a_q_buf, a_q_sf_buf, inter_buf, diq_buf, diq_sf_buf = ws
    out_buf = (
        out
        if out is not None
        else torch.empty((G, m_max, k), device=hs_x.device, dtype=torch.bfloat16)
    )

    if hs_sf is None:
        if c_quant_fn is not None:
            c_quant_fn(
                a_q_buf.view(G * m_max, k // 2),
                a_q_sf_buf.view(G * padded_m, rk1),
                hs_x.view(G * m_max, k),
                input_global_scale,
                masked_m,
                False,
            )
            a_q = a_q_buf
            a_q_sf = a_q_sf_buf
        else:
            a_q, a_q_sf = scaled_fp4_grouped_quantize(
                hs_x, masked_m, input_global_scale
            )

    if gated and use_fused_dual_gemm and n % 128 == 0:
        k_half = k // 2
        if a_q.stride(1) == 1:
            step_a = a_q.stride(2) * a_q.element_size()
            a_row_ok = a_q.stride(0) == k_half
        else:
            step_a = a_q.stride(0) * a_q.element_size()
            a_row_ok = a_q.stride(1) == k_half and a_q.stride(2) == 1

        if w1.stride(1) == 1:
            step_b = w1.stride(2) * w1.element_size()
            w1_row_ok = w1.stride(0) == k_half
        else:
            step_b = w1.stride(0) * w1.element_size()
            w1_row_ok = w1.stride(1) == k_half and w1.stride(2) == 1

        b2_off = n * k_half
        step_sfa = (
            a_q_sf.stride(5) * a_q_sf.element_size()
            if a_q_sf.ndim == 6
            else a_q_sf.stride(0) * a_q_sf.element_size()
        )
        step_sfb = (
            w1_blockscale.stride(5) * w1_blockscale.element_size()
            if w1_blockscale.ndim == 6
            else w1_blockscale.stride(0) * w1_blockscale.element_size()
        )
        sfb2_off = (n // 128) * rk1 * 512

        if a_q_sf.ndim == 6:
            sfa_m = a_q_sf.shape[2] * 128
        elif G > 1 and step_sfa > 0:
            sfa_m = step_sfa // (rk1 * 4)
        else:
            sfa_m = a_q_sf.shape[1]

        can_use_v2_dual_quant = (
            (k % 256 == 0)
            and a_row_ok
            and w1_row_ok
            and (G == 1 or step_a == m_max * k_half)
            and (G == 1 or step_b == 2 * n * k_half)
            and (sfa_m >= padded_m)
            and (sfa_m % 128 == 0)
            and (G == 1 or step_sfa == (sfa_m // 128) * rk1 * 512)
            and (G == 1 or step_sfb == 2 * (n // 128) * rk1 * 512)
            and w1_alpha.is_contiguous()
            and a2_global_scale.is_contiguous()
        )

        ba = a_q.data_ptr()
        bb = w1.data_ptr()
        bsfa = a_q_sf.data_ptr()
        bsfb = w1_blockscale.data_ptr()
        balpha = w1_alpha.data_ptr()
        stream_i = torch.cuda.current_stream(hs_x.device).cuda_stream

        if can_use_v2_dual_quant:
            use_2cta_w1 = (m_max >= 512) and (m_max % 256 == 0) and (sfa_m % 256 == 0)
            tile_n_w1 = 128 if use_2cta_w1 else 64
            diq_rm = padded_m // 128
            dual_q_runner = _get_masked_dual_quant_runner(
                use_2cta=use_2cta_w1,
                tile_n=tile_n_w1,
                shape_cfg=(G, m_max, sfa_m, diq_rm, n, k),
            )
            dual_q_runner(
                ba,
                bb,
                bb + b2_off,
                bsfa,
                bsfb,
                bsfb + sfb2_off,
                diq_buf.data_ptr(),
                diq_sf_buf.data_ptr(),
                balpha,
                a2_global_scale.data_ptr(),
                masked_m.data_ptr(),
                stream_i,
            )
            diq, diq_sf = diq_buf, diq_sf_buf
        else:
            step_c = inter_buf.stride(0) * inter_buf.element_size()
            bc = inter_buf.data_ptr()
            dual_ptrs = []
            for g in range(G):
                dual_ptrs.extend(
                    [
                        ba + g * step_a,
                        bb + g * step_b,
                        bb + g * step_b + b2_off,
                        bsfa + g * step_sfa,
                        bsfb + g * step_sfb,
                        bsfb + g * step_sfb + sfb2_off,
                        bc + g * step_c,
                        balpha + g * 4,
                    ]
                )
            dual_runner = _build_masked_dual_gemm_swiglu_1cta(
                G, m_max, n, k, num_ab_stage=4, order_mn=True, c_dtype_str="BFloat16"
            )
            dual_runner(dual_ptrs, masked_m.data_ptr(), stream_i)
            if c_quant_fn is not None:
                c_quant_fn(
                    diq_buf.view(G * m_max, n // 2),
                    diq_sf_buf.view(G * padded_m, rk2),
                    inter_buf.view(G * m_max, n),
                    a2_global_scale,
                    masked_m,
                    False,
                )
                diq, diq_sf = diq_buf, diq_sf_buf
            else:
                diq, diq_sf = scaled_fp4_grouped_quantize(
                    inter_buf, masked_m, a2_global_scale
                )
    else:
        nvfp4_masked_grouped_gemm(
            a_q,
            a_q_sf,
            w1,
            w1_blockscale,
            masked_m,
            w1_alpha,
            out=inter_buf,
            out_dtype=torch.bfloat16,
            pdl_safe=True,
        )
        if gated:
            if c_quant_fn is not None:
                c_quant_fn(
                    diq_buf.view(G * m_max, n // 2),
                    diq_sf_buf.view(G * padded_m, rk2),
                    inter_buf.view(G * m_max, 2 * n),
                    a2_global_scale,
                    masked_m,
                    True,
                )
                diq, diq_sf = diq_buf, diq_sf_buf
            else:
                diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
                    inter_buf, masked_m, a2_global_scale
                )
        else:
            act = torch.relu(inter_buf).square().contiguous()
            if c_quant_fn is not None:
                c_quant_fn(
                    diq_buf.view(G * m_max, n // 2),
                    diq_sf_buf.view(G * padded_m, rk2),
                    act.view(G * m_max, n),
                    a2_global_scale,
                    masked_m,
                    False,
                )
                diq, diq_sf = diq_buf, diq_sf_buf
            else:
                diq, diq_sf = scaled_fp4_grouped_quantize(
                    act, masked_m, a2_global_scale
                )

    if down_start_event is not None:
        down_start_event.record()

    nvfp4_masked_grouped_gemm(
        diq,
        diq_sf,
        w2,
        w2_blockscale,
        masked_m,
        w2_alpha,
        out=out_buf,
        out_dtype=torch.bfloat16,
        pdl_safe=True,
    )
    return out_buf


@triton.jit
def _moe_scatter_tokens_to_experts_kernel(
    x_ptr,
    topk_ids_ptr,
    masked_m_ptr,
    token_slots_ptr,
    expert_x_ptr,
    G: tl.constexpr,
    topk: tl.constexpr,
    K: tl.constexpr,
    m_max,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid // topk
    exp_id = tl.load(topk_ids_ptr + pid).to(tl.int32)
    if exp_id < 0 or exp_id >= G:
        tl.store(token_slots_ptr + pid, -1)
        return

    slot = tl.atomic_add(masked_m_ptr + exp_id, 1)
    tl.store(token_slots_ptr + pid, slot)
    if slot >= m_max:
        return

    src_base = x_ptr + token_idx * K
    dst_base = expert_x_ptr + (exp_id.to(tl.int64) * m_max + slot.to(tl.int64)) * K
    for k0 in tl.static_range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        val = tl.load(src_base + k_offs, mask=mask, other=0.0)
        tl.store(dst_base + k_offs, val, mask=mask)


@triton.jit
def _moe_gather_weighted_reduce_kernel(
    expert_out_ptr,
    topk_ids_ptr,
    token_slots_ptr,
    topk_weights_ptr,
    out_ptr,
    M,
    G: tl.constexpr,
    K: tl.constexpr,
    m_max,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for j in tl.static_range(0, TOPK):
        idx = pid_m * TOPK + j
        exp_id = tl.load(topk_ids_ptr + idx).to(tl.int64)
        slot = tl.load(token_slots_ptr + idx).to(tl.int64)
        w = tl.load(topk_weights_ptr + idx).to(tl.float32)
        valid = (exp_id >= 0) & (exp_id < G) & (slot >= 0) & (slot < m_max)
        safe_exp = tl.where(valid, exp_id, 0)
        safe_slot = tl.where(valid, slot, 0)
        val = tl.load(
            expert_out_ptr + (safe_exp * m_max + safe_slot) * K + k_offs,
            mask=k_mask & valid,
            other=0.0,
        ).to(tl.float32)
        acc += w * val

    tl.store(out_ptr + pid_m * K + k_offs, acc.to(tl.bfloat16), mask=k_mask)


def nvfp4_routed_grouped_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_global_scale: torch.Tensor,
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2: torch.Tensor,
    a2_global_scale: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alpha: torch.Tensor,
    *,
    m_max: Optional[int] = None,
    activation: str = "silu",
    use_fused_dual_gemm: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """End-to-end SM100 routed Top-K NVFP4 MoE (scatter + masked grouped MoE + weighted reduce)."""
    M, K = hidden_states.shape
    if M == 0:
        return (
            out
            if out is not None
            else torch.empty((0, K), device=hidden_states.device, dtype=torch.bfloat16)
        )
    _, topk = topk_ids.shape
    G = w1.shape[0]
    if m_max is None:
        m_max = _ceil_div(M * topk, 128) * 128

    ws_key = (hidden_states.device, M, K, topk, G, m_max)
    ws = _ROUTED_WS_CACHE.get(ws_key)
    if ws is None:
        ws = (
            torch.empty((G,), device=hidden_states.device, dtype=torch.int32),
            torch.empty((M, topk), device=hidden_states.device, dtype=torch.int32),
            torch.empty(
                (G, m_max, K), device=hidden_states.device, dtype=torch.bfloat16
            ),
        )
        _ROUTED_WS_CACHE[ws_key] = ws
    masked_m, token_slots, expert_x = ws
    out_buf = (
        out
        if out is not None
        else torch.empty((M, K), device=hidden_states.device, dtype=torch.bfloat16)
    )

    masked_m.zero_()
    block_k = min(1024, triton.next_power_of_2(K))
    _moe_scatter_tokens_to_experts_kernel[(M * topk,)](
        hidden_states,
        topk_ids,
        masked_m,
        token_slots,
        expert_x,
        G=G,
        topk=topk,
        K=K,
        m_max=m_max,
        BLOCK_K=block_k,
        num_warps=4,
    )
    masked_m.clamp_(max=m_max)

    expert_out = nvfp4_masked_grouped_moe(
        (expert_x, None),
        input_global_scale,
        w1,
        w1_blockscale,
        w1_alpha,
        w2,
        a2_global_scale,
        w2_blockscale,
        w2_alpha,
        masked_m,
        activation=activation,
        use_fused_dual_gemm=use_fused_dual_gemm,
    )

    gather_bk = 256
    _moe_gather_weighted_reduce_kernel[(M, triton.cdiv(K, gather_bk))](
        expert_out,
        topk_ids,
        token_slots,
        topk_weights,
        out_buf,
        M,
        G=G,
        K=K,
        m_max=m_max,
        TOPK=topk,
        BLOCK_K=gather_bk,
        num_warps=4,
    )
    return out_buf


__all__ = [
    "Sm100Nvfp4MaskedDualGemmSwiGLUQuantV2",
    "nvfp4_masked_grouped_moe",
    "nvfp4_routed_grouped_moe",
    "_build_masked_dual_gemm_swiglu_1cta",
    "_get_masked_dual_quant_runner",
]
