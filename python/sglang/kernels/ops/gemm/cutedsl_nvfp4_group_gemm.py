import ctypes
import linecache
import os
import sys
import types
from typing import Dict, List, Optional, Sequence, Tuple, Union

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

_OP_K_MAJOR = (
    cute.nvgpu.OperandMajorMode.K
    if hasattr(cute.nvgpu, "OperandMajorMode")
    else tcgen05.OperandMajorMode.K
)

_DEFAULT_PDL_SAFE = os.environ.get("NVFP4_GROUP_GEMM_PDL_SAFE", "0") == "1"
_RUNNER_CACHE: Dict[Tuple, object] = {}
_SGLANG_GROUP_CACHE: Dict[Tuple, object] = {}
_KEEPALIVE: list = []


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


def _load_in_mem_module(name: str, src: str):
    fn = f"<mem_{name}.py>"
    linecache.cache[fn] = (len(src), None, src.splitlines(True), fn)
    mod = types.ModuleType(name)
    mod.__file__ = fn
    sys.modules[name] = mod
    exec(compile(src, fn, "exec"), mod.__dict__)
    return mod


def _build_fused_group_runner_1cta(
    problem_sizes: Tuple[Tuple[int, int, int, int], ...],
    tile_n: int = 128,
    num_ab_stage: int = 6,
    pf_dist: int = 6,
    order_mn: bool = True,
    l2_hint: bool = True,
    pdl_safe: bool = False,
    c_dtype_str: str = "Float16",
    has_alpha: bool = False,
    with_stream: bool = False,
):
    G = len(problem_sizes)
    tile_m = 128
    tile_k = 256
    subtile_cnt = tile_n // 64
    tmem_cols = 256 if tile_n == 128 else 128
    cta_mn = [
        ((m + tile_m - 1) // tile_m, (n + tile_n - 1) // tile_n)
        for (m, n, k, l) in problem_sizes
    ]
    cum_tiles = [0]
    for cm, cn in cta_mn:
        cum_tiles.append(cum_tiles[-1] + cm * cn)
    total_ctas = cum_tiles[-1]

    alpha_jit = lambda g: f", alpha_ptr_{g}: cute.Pointer" if has_alpha else ""
    alpha_def = lambda g: f", mAlpha_{g}: cute.Tensor" if has_alpha else ""
    alpha_call = lambda g: f", alpha_t_{g}" if has_alpha else ""

    ptr_args_jit = ", ".join(
        f"a_ptr_{g}: cute.Pointer, b_ptr_{g}: cute.Pointer, sfa_ptr_{g}: cute.Pointer, sfb_ptr_{g}: cute.Pointer, c_ptr_{g}: cute.Pointer{alpha_jit(g)}"
        for g in range(G)
    )
    if with_stream:
        ptr_args_jit += ", stream: cuda.CUstream"
    kern_args_def = ", ".join(
        f"tma_atom_a_{g}: cute.CopyAtom, mA_{g}: cute.Tensor, "
        f"tma_atom_b_{g}: cute.CopyAtom, mB_{g}: cute.Tensor, "
        f"tma_atom_sfa_{g}: cute.CopyAtom, mSFA_{g}: cute.Tensor, "
        f"tma_atom_sfb_{g}: cute.CopyAtom, mSFB_{g}: cute.Tensor, "
        f"tma_atom_c_{g}: cute.CopyAtom, mC_{g}: cute.Tensor{alpha_def(g)}"
        for g in range(G)
    )
    kern_args_call = ", ".join(
        f"tma_atom_a_{g}, tma_tensor_a_{g}, "
        f"tma_atom_b_{g}, tma_tensor_b_{g}, "
        f"tma_atom_sfa_{g}, tma_tensor_sfa_{g}, "
        f"tma_atom_sfb_{g}, tma_tensor_sfb_{g}, "
        f"tma_atom_c_{g}, tma_tensor_c_{g}{alpha_call(g)}"
        for g in range(G)
    )

    host_setup = []
    decode_branches = []
    tma_branches = []
    epi_branches = []
    alpha_branches = []

    pol_a = ", cache_policy=cutlass.Int64(0x14F0000000000000)" if l2_hint else ""
    pol_b = ", cache_policy=cutlass.Int64(0x12F0000000000000)" if l2_hint else ""
    pol_c = ", cache_policy=cutlass.Int64(0x12F0000000000000)" if l2_hint else ""

    tma_pdl_prologue = (
        "                cute.arch.griddepcontrol_wait()\n" if pdl_safe else ""
    )
    tma_pdl_epilogue = (
        "                cute.arch.griddepcontrol_launch_dependents()\n"
        if pdl_safe
        else ""
    )
    tma_top_launch = (
        "" if pdl_safe else "            cute.arch.griddepcontrol_launch_dependents()\n"
    )
    epi_pdl_wait = (
        "" if pdl_safe else "                cute.arch.griddepcontrol_wait()\n"
    )
    epi_bulk_wait = (
        "                cute.arch.cp_async_bulk_wait_group(0, read=True)\n"
        if pdl_safe
        else ""
    )

    for g in range(G):
        m_g, n_g, k_g, l_g = problem_sizes[g]
        cm_g, cn_g = cta_mn[g]
        c_start, c_end = cum_tiles[g], cum_tiles[g + 1]
        k_tiles_g = k_g // tile_k
        pf0_g = min(pf_dist, k_tiles_g)
        init_st_g = min(num_ab_stage, k_tiles_g)
        kw = "if" if g == 0 else "elif"
        alpha_host_line = (
            f"        alpha_t_{g} = cute.make_tensor(alpha_ptr_{g}, cute.make_layout((1,), stride=(1,)))\n"
            if has_alpha
            else ""
        )
        if has_alpha:
            alpha_branches.append(
                f"            {kw} group_idx == {g}:\n                alpha_val = cutlass.Float32(mAlpha_{g}[0])\n"
            )

        host_setup.append(f"""
        a_t_{g} = cute.make_tensor(a_ptr_{g}, cute.make_layout(({m_g}, cute.assume({k_g}, 32), {l_g}), stride=(cute.assume({k_g}, 32), 1, cute.assume({m_g * k_g}, 32))))
        b_t_{g} = cute.make_tensor(b_ptr_{g}, cute.make_layout(({n_g}, cute.assume({k_g}, 32), {l_g}), stride=(cute.assume({k_g}, 32), 1, cute.assume({n_g * k_g}, 32))))
        c_t_{g} = cute.make_tensor(c_ptr_{g}, cute.make_layout(({m_g}, cute.assume({n_g}, 32), {l_g}), stride=(cute.assume({n_g}, 32), 1, cute.assume({m_g * n_g}, 32))))
        sfa_t_{g} = cute.make_tensor(sfa_ptr_{g}, blockscaled_utils.tile_atom_to_shape_SF(a_t_{g}.shape, 16))
        sfb_t_{g} = cute.make_tensor(sfb_ptr_{g}, blockscaled_utils.tile_atom_to_shape_SF(b_t_{g}.shape, 16))
{alpha_host_line}        tma_atom_a_{g}, tma_tensor_a_{g} = cute.nvgpu.make_tiled_tma_atom_A(a_op, a_t_{g}, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_atom_b_{g}, tma_tensor_b_{g} = cute.nvgpu.make_tiled_tma_atom_B(b_op, b_t_{g}, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_atom_sfa_{g}, tma_tensor_sfa_{g} = cute.nvgpu.make_tiled_tma_atom_A(sfa_op, sfa_t_{g}, sfa_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16)
        tma_atom_sfb_{g}, tma_tensor_sfb_{g} = cute.nvgpu.make_tiled_tma_atom_B(sfb_op, sfb_t_{g}, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)
        tma_atom_c_{g}, tma_tensor_c_{g} = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c_t_{g}, epi_smem_layout, self.epi_tile)
""")

        if order_mn:
            coord_code = f"coord_x = rel % {cm_g}\n            coord_y = rel // {cm_g}"
        else:
            coord_code = f"coord_x = rel // {cn_g}\n            coord_y = rel % {cn_g}"

        decode_branches.append(f"""
        {kw} bidx < {c_end}:
            group_idx = cutlass.Int32({g})
            rel = bidx - {c_start}
            {coord_code}
            k_tile_cnt = cutlass.Int32({k_tiles_g})
""")

        sfb_coord_expr = "coord_y // 2" if tile_n == 64 else "coord_y"

        tma_branches.append(f"""
            {kw} group_idx == {g}:
                cpasync.prefetch_descriptor(tma_atom_a_{g})
                cpasync.prefetch_descriptor(tma_atom_b_{g})
                cpasync.prefetch_descriptor(tma_atom_sfa_{g})
                cpasync.prefetch_descriptor(tma_atom_sfb_{g})
                gA = cute.local_tile(mA_{g}, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB = cute.local_tile(mB_{g}, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                gSFA = cute.local_tile(mSFA_{g}, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gSFB = cute.local_tile(mSFB_{g}, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
                tCgA = thr_mma.partition_A(gA)
                tCgB = thr_mma.partition_B(gB)
                tCgSFA = thr_mma.partition_A(gSFA)
                tCgSFB = thr_mma_sfb.partition_B(gSFB)
                tAsA, tAgA = cpasync.tma_partition(tma_atom_a_{g}, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
                tBsB, tBgB = cpasync.tma_partition(tma_atom_b_{g}, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
                tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa_{g}, 0, cute.make_layout(1), cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
                tAsSFA = cute.filter_zeros(tAsSFA); tAgSFA = cute.filter_zeros(tAgSFA)
                tBsSFB, tBgSFB = cpasync.tma_partition(tma_atom_sfb_{g}, 0, cute.make_layout(1), cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
                tBsSFB = cute.filter_zeros(tBsSFB); tBgSFB = cute.filter_zeros(tBgSFB)
                tAgA_s = tAgA[(None, coord_x, None, 0)]
                tBgB_s = tBgB[(None, coord_y, None, 0)]
                tAgSFA_s = tAgSFA[(None, coord_x, None, 0)]
                tBgSFB_s = tBgSFB[(None, {sfb_coord_expr}, None, 0)]
                for pf_i in cutlass.range_constexpr({pf0_g}):
                    cute.prefetch(tma_atom_b_{g}, tBgB_s[(None, pf_i)])
                    cute.prefetch(tma_atom_sfb_{g}, tBgSFB_s[(None, pf_i)])
                if coord_y < {k_tiles_g}:
                    cute.prefetch(tma_atom_a_{g}, tAgA_s[(None, coord_y)])
                    cute.prefetch(tma_atom_sfa_{g}, tAgSFA_s[(None, coord_y)])
{tma_pdl_prologue}                ab_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
                ab_prod.reset_count()
                for kt_p in cutlass.range_constexpr({init_st_g}):
                    ab_pipeline.producer_acquire(ab_prod, cutlass.Boolean(1))
                    bar_p = ab_pipeline.producer_get_barrier(ab_prod)
                    st_p = ab_prod.index
                    cute.copy(tma_atom_b_{g}, tBgB_s[(None, kt_p)], tBsB[(None, st_p)], tma_bar_ptr=bar_p{pol_b})
                    cute.copy(tma_atom_a_{g}, tAgA_s[(None, kt_p)], tAsA[(None, st_p)], tma_bar_ptr=bar_p{pol_a})
                    cute.copy(tma_atom_sfb_{g}, tBgSFB_s[(None, kt_p)], tBsSFB[(None, st_p)], tma_bar_ptr=bar_p{pol_b})
                    cute.copy(tma_atom_sfa_{g}, tAgSFA_s[(None, kt_p)], tAsSFA[(None, st_p)], tma_bar_ptr=bar_p{pol_a})
                    if cutlass.const_expr(kt_p + {pf0_g} < {k_tiles_g}):
                        cute.prefetch(tma_atom_b_{g}, tBgB_s[(None, kt_p + {pf0_g})])
                        cute.prefetch(tma_atom_sfb_{g}, tBgSFB_s[(None, kt_p + {pf0_g})])
                    ab_prod.advance()
                if cutlass.const_expr({k_tiles_g} > {init_st_g}):
                    peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
                    for kt in cutlass.range({init_st_g}, {k_tiles_g}, 1, unroll=1):
                        ab_pipeline.producer_acquire(ab_prod, peek_empty)
                        bar = ab_pipeline.producer_get_barrier(ab_prod)
                        st = ab_prod.index
                        cute.copy(tma_atom_b_{g}, tBgB_s[(None, kt)], tBsB[(None, st)], tma_bar_ptr=bar{pol_b})
                        cute.copy(tma_atom_a_{g}, tAgA_s[(None, kt)], tAsA[(None, st)], tma_bar_ptr=bar{pol_a})
                        cute.copy(tma_atom_sfb_{g}, tBgSFB_s[(None, kt)], tBsSFB[(None, st)], tma_bar_ptr=bar{pol_b})
                        cute.copy(tma_atom_sfa_{g}, tAgSFA_s[(None, kt)], tAsSFA[(None, st)], tma_bar_ptr=bar{pol_a})
                        if kt + {pf0_g} < {k_tiles_g}:
                            cute.prefetch(tma_atom_b_{g}, tBgB_s[(None, kt + {pf0_g})])
                            cute.prefetch(tma_atom_sfb_{g}, tBgSFB_s[(None, kt + {pf0_g})])
                        ab_prod.advance()
                        peek_empty = cutlass.Boolean(1)
                        if ab_prod.count < {k_tiles_g}:
                            peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
{tma_pdl_epilogue}""")

        epi_branches.append(f"""
                {kw} group_idx == {g}:
                    gC = cute.local_tile(mC_{g}, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                    gC_epi = cute.flat_divide(gC, self.epi_tile)
                    bSG_sC, bSG_gC_p = cpasync.tma_partition(tma_atom_c_{g}, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gC_epi, 0, 2))
                    bSG_gC = cute.group_modes(bSG_gC_p[(None, None, None, coord_x, coord_y, 0)], 1, cute.rank(bSG_gC_p[(None, None, None, coord_x, coord_y, 0)]))
                    for sub_i in cutlass.range_constexpr({subtile_cnt}):
                        cute.copy(tma_atom_c_{g}, bSG_sC[(None, sub_i)], bSG_gC[(None, sub_i)]{pol_c})
""")

    if tile_n == 64:
        sfb_mma_setup = "            offset_sfb = cutlass.Int32((coord_y & 1) * 2)\n            sfb_mma_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base) + tcgen05.find_tmem_tensor_col_offset(tCtSFA) + offset_sfb, dtype=self.sf_dtype)\n            tCtSFB_mma = cute.make_tensor(sfb_mma_ptr, tCtSFB_layout)"
    else:
        sfb_mma_setup = "            tCtSFB_mma = tCtSFB"

    launch_stream_1cta = "            stream=stream,\n" if with_stream else ""
    alpha_setup_1cta = (
        "            alpha_val = cutlass.Float32(1.0)\n" + "".join(alpha_branches)
        if has_alpha
        else ""
    )
    store_expr_1cta = (
        "tRS_rC.store((tTR_rAcc.load() * alpha_val).to(self.c_dtype))"
        if has_alpha
        else "tRS_rC.store(tTR_rAcc.load().to(self.c_dtype))"
    )
    cuda_imp_1cta = "import cuda.bindings.driver as cuda\n" if with_stream else ""

    src = f"""
import ctypes
{cuda_imp_1cta}import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import dsl_user_op

class _NoInitMbarrierArray(pipeline.MbarrierArray):
    @dsl_user_op
    def mbarrier_init(self, *, loc=None, ip=None) -> None:
        pass
    def __new_from_mlir_values__(self, values: list[object]) -> "_NoInitMbarrierArray":
        res = object.__new__(_NoInitMbarrierArray)
        res.barrier_storage = values[0]
        res.op_type = self.op_type
        res.cg = self.cg
        res.num_stages = self.num_stages
        res.tx_count = self.tx_count
        res.arrive_count = self.arrive_count
        res.mbarrier_base = values[0]
        res.mbarrier_layout = self.mbarrier_layout
        res.name = self.name
        return res

class Sm100Nvfp4GroupGemm1CTA:
    def __init__(self):
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.{c_dtype_str}
        self.sf_vec_size = 16
        self.mma_tiler = (128, {tile_n}, 256)
        self.mma_tiler_sfb = (128, {max(tile_n, 128)}, 256)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192
        self.num_ab_stage = {num_ab_stage}
        self.num_acc_stage = 1
        self.num_c_stage = {subtile_cnt}
        self.epi_tile = (128, 64)
        self.num_tmem_alloc_cols = {tmem_cols}
        self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=192)
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=160)

    @cute.jit
    def __call__(self, {ptr_args_jit}):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            self.sf_dtype, self.sf_vec_size, self.cta_group, (128, {tile_n})
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            self.sf_dtype, self.sf_vec_size, tcgen05.CtaGroup.ONE, (128, {max(tile_n, 128)})
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
            + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )

        {"".join(host_setup)}

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)], 1024]
            sSFA: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)], 1024]
            sSFB: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)], 1024]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma, tiled_mma_sfb,
            {kern_args_call},
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
            self.c_smem_layout_staged
        ).launch(
            grid=({total_ctas}, 1, 1),
            block=[192, 1, 1],
            cluster=(1, 1, 1),
{launch_stream_1cta}            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        {kern_args_def},
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()

        group_idx = cutlass.Int32(0)
        coord_x = cutlass.Int32(0)
        coord_y = cutlass.Int32(0)
        k_tile_cnt = cutlass.Int32(1)
        {"".join(decode_branches)}

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        _orig_mbar = pipeline.MbarrierArray
        pipeline.MbarrierArray = _NoInitMbarrierArray
        try:
            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=self.num_tma_load_bytes,
                defer_sync=True,
            )
            acc_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_mbar_ptr.data_ptr(),
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
                defer_sync=True,
            )
        finally:
            pipeline.MbarrierArray = _orig_mbar

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.num_ab_stage):
                    cute.arch.mbarrier_init(ab_pipeline.sync_object_full.get_barrier(i), 1)
        elif warp_idx == 1:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.num_ab_stage):
                    cute.arch.mbarrier_init(ab_pipeline.sync_object_empty.get_barrier(i), 1)
        elif warp_idx == 2:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(acc_pipeline.sync_object_full.get_barrier(0), 1)
                cute.arch.mbarrier_init(acc_pipeline.sync_object_empty.get_barrier(0), 1)

        tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=self.tmem_alloc_barrier, is_two_cta=False)
        self.cta_sync_barrier.arrive_and_wait()

        sC_ptr = cute.recast_ptr(storage.sA.data_ptr(), c_smem_layout_staged.inner, dtype=self.c_dtype)
        sC = cute.make_tensor(sC_ptr, c_smem_layout_staged.outer)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        thr_mma = tiled_mma.get_slice(0)
        thr_mma_sfb = tiled_mma_sfb.get_slice(0)

        if warp_idx == self.tma_warp_id:
{tma_top_launch}            {"".join(tma_branches)}

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((128, {tile_n}))).layout)
            sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base), dtype=self.sf_dtype)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)))
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base) + tcgen05.find_tmem_tensor_col_offset(tCtSFA), dtype=self.sf_dtype)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)))
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
{sfb_mma_setup}

            copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), self.sf_dtype)
            tCsSFA_c = cute.filter_zeros(sSFA); tCtSFA_c = cute.filter_zeros(tCtSFA)
            tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_c)
            thr_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
            tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfa, thr_s2t_sfa.partition_S(tCsSFA_c))
            tCtSFA_s2t = thr_s2t_sfa.partition_D(tCtSFA_c)

            tCsSFB_c = cute.filter_zeros(sSFB); tCtSFB_c = cute.filter_zeros(tCtSFB)
            tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_c)
            thr_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
            tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfb, thr_s2t_sfb.partition_S(tCsSFB_c))
            tCtSFB_s2t = thr_s2t_sfb.partition_D(tCtSFB_c)

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            tCtAcc = tCtAcc_base[(None, None, None)]

            ab_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            ab_cons.reset_count()
            peek_full = ab_pipeline.consumer_try_wait(ab_cons)

            ab_pipeline.consumer_wait(ab_cons, peek_full)
            st0 = ab_cons.index
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kb0 in cutlass.range_constexpr(4):
                cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t[(None, None, None, kb0, st0)], tCtSFA_s2t[(None, None, None, kb0)])
                cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t[(None, None, None, kb0, st0)], tCtSFB_s2t[(None, None, None, kb0)])
                tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb0)].iterator)
                tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[(None, None, kb0)].iterator)
                cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb0, st0)], tCrB[(None, None, kb0, st0)], tCtAcc)
                if cutlass.const_expr(kb0 == 0):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            rel0 = ab_cons.index
            ab_cons.advance()
            peek_full = cutlass.Boolean(1)
            if ab_cons.count < k_tile_cnt:
                if ab_cons.count <= k_tile_cnt - self.num_ab_stage:
                    ab_pipeline.sync_object_empty.arrive(rel0, ab_pipeline.consumer_mask, self.cta_group)
                peek_full = ab_pipeline.consumer_try_wait(ab_cons)

            for kt in cutlass.range(1, k_tile_cnt, 1, unroll=1):
                ab_pipeline.consumer_wait(ab_cons, peek_full)
                st = ab_cons.index
                for kb in cutlass.range_constexpr(4):
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t[(None, None, None, kb, st)], tCtSFA_s2t[(None, None, None, kb)])
                    cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t[(None, None, None, kb, st)], tCtSFB_s2t[(None, None, None, kb)])
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[(None, None, kb)].iterator)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb, st)], tCrB[(None, None, kb, st)], tCtAcc)
                rel_i = ab_cons.index
                ab_cons.advance()
                peek_full = cutlass.Boolean(1)
                if ab_cons.count < k_tile_cnt:
                    if ab_cons.count <= k_tile_cnt - self.num_ab_stage:
                        ab_pipeline.sync_object_empty.arrive(rel_i, ab_pipeline.consumer_mask, self.cta_group)
                    peek_full = ab_pipeline.consumer_try_wait(ab_cons)
            acc_pipeline.producer_commit(acc_prod)

        if warp_idx < 4:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((128, {tile_n}))).layout)
            epi_tidx = tidx
            copy_atom_t2r = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), self.acc_dtype)
            tCtAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0)], self.epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

            copy_atom_r2s = sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)

            c_ident = cute.make_identity_tensor((128, {tile_n}))
            c_ident_epi = cute.flat_divide(c_ident, self.epi_tile)
            tTR_cC = thr_copy_t2r.partition_D(c_ident_epi)
            tTR_rAcc = cute.make_rmem_tensor(tTR_cC[(None, None, None, 0, 0)].shape, self.acc_dtype)
            tRS_rC = cute.make_rmem_tensor(thr_copy_r2s.retile(tTR_rAcc).shape, self.c_dtype)

            tmem.relinquish_alloc_permit()
{alpha_setup_1cta}            acc_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            acc_pipeline.consumer_wait(acc_cons)
            for sub_i in cutlass.range_constexpr({subtile_cnt}):
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, sub_i)], tTR_rAcc)
                {store_expr_1cta}
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, sub_i)])
            tmem.free(acc_tmem_ptr)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == 0:
                cute.arch.fence_view_async_shared()
{epi_pdl_wait}                {"".join(epi_branches)}
                cute.arch.cp_async_bulk_commit_group()
{epi_bulk_wait}"""
    mod = _load_in_mem_module(
        f"fg1cta_{tile_n}_{num_ab_stage}_{order_mn}_{l2_hint}_{pdl_safe}_{c_dtype_str}_{has_alpha}_{with_stream}_{hash(problem_sizes) & 0xFFFFFFFF}",
        src,
    )
    kern = mod.Sm100Nvfp4GroupGemm1CTA()
    if with_stream:
        return _compile_and_wrap_group_kernel_sglang(
            kern, G, getattr(cutlass, c_dtype_str), has_alpha
        )
    return _compile_and_wrap_group_kernel(kern, G)


def _build_fused_group_runner_2cta(
    problem_sizes: Tuple[Tuple[int, int, int, int], ...],
    tile_n: int = 128,
    num_ab_stage: int = 7,
    pf_dist: int = 6,
    order_mn: bool = True,
    l2_hint: bool = True,
    pdl_safe: bool = False,
    c_dtype_str: str = "Float16",
    has_alpha: bool = False,
    with_stream: bool = False,
):
    G = len(problem_sizes)
    cluster_m = 256
    tile_k = 256
    subtile_cnt = tile_n // 64
    tmem_cols = 512 if tile_n == 256 else 256
    sfb_n = ((tile_n + 127) // 128) * 128
    clusters_mn = [
        ((m + cluster_m - 1) // cluster_m, (n + tile_n - 1) // tile_n)
        for (m, n, k, l) in problem_sizes
    ]
    cum_clusters = [0]
    for cm, cn in clusters_mn:
        cum_clusters.append(cum_clusters[-1] + cm * cn)
    total_clusters = cum_clusters[-1]
    total_ctas = total_clusters * 2

    alpha_jit = lambda g: f", alpha_ptr_{g}: cute.Pointer" if has_alpha else ""
    alpha_def = lambda g: f", mAlpha_{g}: cute.Tensor" if has_alpha else ""
    alpha_call = lambda g: f", alpha_t_{g}" if has_alpha else ""

    ptr_args_jit = ", ".join(
        f"a_ptr_{g}: cute.Pointer, b_ptr_{g}: cute.Pointer, sfa_ptr_{g}: cute.Pointer, sfb_ptr_{g}: cute.Pointer, c_ptr_{g}: cute.Pointer{alpha_jit(g)}"
        for g in range(G)
    )
    if with_stream:
        ptr_args_jit += ", stream: cuda.CUstream"
    kern_args_def = ", ".join(
        f"tma_atom_a_{g}: cute.CopyAtom, mA_{g}: cute.Tensor, "
        f"tma_atom_b_{g}: cute.CopyAtom, mB_{g}: cute.Tensor, "
        f"tma_atom_sfa_{g}: cute.CopyAtom, mSFA_{g}: cute.Tensor, "
        f"tma_atom_sfb_{g}: cute.CopyAtom, mSFB_{g}: cute.Tensor, "
        f"tma_atom_c_{g}: cute.CopyAtom, mC_{g}: cute.Tensor{alpha_def(g)}"
        for g in range(G)
    )
    kern_args_call = ", ".join(
        f"tma_atom_a_{g}, tma_tensor_a_{g}, "
        f"tma_atom_b_{g}, tma_tensor_b_{g}, "
        f"tma_atom_sfa_{g}, tma_tensor_sfa_{g}, "
        f"tma_atom_sfb_{g}, tma_tensor_sfb_{g}, "
        f"tma_atom_c_{g}, tma_tensor_c_{g}{alpha_call(g)}"
        for g in range(G)
    )

    host_setup = []
    decode_branches = []
    tma_branches = []
    epi_branches = []
    alpha_branches = []

    pol_a = ", cache_policy=cutlass.Int64(0x14F0000000000000)" if l2_hint else ""
    pol_b = ", cache_policy=cutlass.Int64(0x12F0000000000000)" if l2_hint else ""
    pol_c = ", cache_policy=cutlass.Int64(0x12F0000000000000)" if l2_hint else ""

    tma_pdl_prologue = (
        "                cute.arch.griddepcontrol_wait()\n" if pdl_safe else ""
    )
    tma_pdl_epilogue = (
        "                cute.arch.griddepcontrol_launch_dependents()\n"
        if pdl_safe
        else ""
    )
    tma_top_launch = (
        "" if pdl_safe else "            cute.arch.griddepcontrol_launch_dependents()\n"
    )
    epi_pdl_wait = (
        "" if pdl_safe else "                cute.arch.griddepcontrol_wait()\n"
    )
    epi_bulk_wait = (
        "                cute.arch.cp_async_bulk_wait_group(0, read=True)\n"
        if pdl_safe
        else ""
    )

    for g in range(G):
        m_g, n_g, k_g, l_g = problem_sizes[g]
        cm_g, cn_g = clusters_mn[g]
        c_start, c_end = cum_clusters[g], cum_clusters[g + 1]
        k_tiles_g = k_g // tile_k
        pf0_g = min(pf_dist, k_tiles_g) if pf_dist > 0 else 0
        init_st_g = min(num_ab_stage, k_tiles_g)
        kw = "if" if g == 0 else "elif"
        alpha_host_line = (
            f"        alpha_t_{g} = cute.make_tensor(alpha_ptr_{g}, cute.make_layout((1,), stride=(1,)))\n"
            if has_alpha
            else ""
        )
        if has_alpha:
            alpha_branches.append(
                f"            {kw} group_idx == {g}:\n                alpha_val = cutlass.Float32(mAlpha_{g}[0])\n"
            )

        host_setup.append(f"""
        a_t_{g} = cute.make_tensor(a_ptr_{g}, cute.make_layout(({m_g}, cute.assume({k_g}, 32), {l_g}), stride=(cute.assume({k_g}, 32), 1, cute.assume({m_g * k_g}, 32))))
        b_t_{g} = cute.make_tensor(b_ptr_{g}, cute.make_layout(({n_g}, cute.assume({k_g}, 32), {l_g}), stride=(cute.assume({k_g}, 32), 1, cute.assume({n_g * k_g}, 32))))
        c_t_{g} = cute.make_tensor(c_ptr_{g}, cute.make_layout(({m_g}, cute.assume({n_g}, 32), {l_g}), stride=(cute.assume({n_g}, 32), 1, cute.assume({m_g * n_g}, 32))))
        sfa_t_{g} = cute.make_tensor(sfa_ptr_{g}, blockscaled_utils.tile_atom_to_shape_SF(a_t_{g}.shape, 16))
        sfb_t_{g} = cute.make_tensor(sfb_ptr_{g}, blockscaled_utils.tile_atom_to_shape_SF(b_t_{g}.shape, 16))
{alpha_host_line}        tma_atom_a_{g}, tma_tensor_a_{g} = cute.nvgpu.make_tiled_tma_atom_A(a_op, a_t_{g}, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_atom_b_{g}, tma_tensor_b_{g} = cute.nvgpu.make_tiled_tma_atom_B(b_op, b_t_{g}, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_atom_sfa_{g}, tma_tensor_sfa_{g} = cute.nvgpu.make_tiled_tma_atom_A(sfa_op, sfa_t_{g}, sfa_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16)
        tma_atom_sfb_{g}, tma_tensor_sfb_{g} = cute.nvgpu.make_tiled_tma_atom_B(sfb_op, sfb_t_{g}, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)
        tma_atom_c_{g}, tma_tensor_c_{g} = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c_t_{g}, epi_smem_layout, self.epi_tile)
""")

        if order_mn:
            coord_code_2c = (
                f"coord_x = rel % {cm_g}\n            coord_y = rel // {cm_g}"
            )
        else:
            coord_code_2c = (
                f"coord_x = rel // {cn_g}\n            coord_y = rel % {cn_g}"
            )

        decode_branches.append(f"""
        {kw} cl_idx < {c_end}:
            group_idx = cutlass.Int32({g})
            rel = cl_idx - {c_start}
            {coord_code_2c}
            k_tile_cnt = cutlass.Int32({k_tiles_g})
""")

        sfa_slice_expr = (
            "0"
            if m_g <= 128
            else (
                f"(0 if (coord_x == {cm_g - 1} and cta_rank == 1) else cta_rank)"
                if (m_g % 256 != 0 and m_g % 256 <= 128)
                else "cta_rank"
            )
        )

        pf_prologue = ""
        pf_steady_1 = ""
        pf_steady_2 = ""
        if pf0_g > 0:
            pf_prologue = f"""                for pf_i in cutlass.range_constexpr({pf0_g}):
                    cute.prefetch(tma_atom_b_{g}, tBgB_s[(None, pf_i)])
                    if cta_rank == 0:
                        cute.prefetch(tma_atom_sfb_{g}, tBgSFB_s[(None, pf_i)])
                if coord_y < {k_tiles_g}:
                    cute.prefetch(tma_atom_a_{g}, tAgA_s[(None, coord_y)])
                    cute.prefetch(tma_atom_sfa_{g}, tAgSFA_s[(None, coord_y)])
"""
            pf_steady_1 = f"""                    if cutlass.const_expr(kt_p + {pf0_g} < {k_tiles_g}):
                        cute.prefetch(tma_atom_b_{g}, tBgB_s[(None, kt_p + {pf0_g})])
                        if cta_rank == 0:
                            cute.prefetch(tma_atom_sfb_{g}, tBgSFB_s[(None, kt_p + {pf0_g})])
"""
            pf_steady_2 = f"""                        if kt + {pf0_g} < {k_tiles_g}:
                            cute.prefetch(tma_atom_b_{g}, tBgB_s[(None, kt + {pf0_g})])
                            if cta_rank == 0:
                                cute.prefetch(tma_atom_sfb_{g}, tBgSFB_s[(None, kt + {pf0_g})])
"""

        tma_branches.append(f"""
            {kw} group_idx == {g}:
                cpasync.prefetch_descriptor(tma_atom_a_{g})
                cpasync.prefetch_descriptor(tma_atom_b_{g})
                cpasync.prefetch_descriptor(tma_atom_sfa_{g})
                cpasync.prefetch_descriptor(tma_atom_sfb_{g})
                gA = cute.local_tile(mA_{g}, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB = cute.local_tile(mB_{g}, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                gSFA = cute.local_tile(mSFA_{g}, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gSFB = cute.local_tile(mSFB_{g}, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
                tCgA = thr_mma.partition_A(gA)
                tCgB = thr_mma.partition_B(gB)
                tCgSFA = tiled_mma.get_slice({sfa_slice_expr}).partition_A(gSFA)
                tCgSFB = thr_mma_sfb.partition_B(gSFB)
                tAsA, tAgA = cpasync.tma_partition(tma_atom_a_{g}, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
                tBsB, tBgB = cpasync.tma_partition(tma_atom_b_{g}, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
                tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa_{g}, 0, cute.make_layout(1), cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
                tAsSFA = cute.filter_zeros(tAsSFA); tAgSFA = cute.filter_zeros(tAgSFA)
                tBsSFB, tBgSFB = cpasync.tma_partition(tma_atom_sfb_{g}, cta_rank, cute.make_layout(2), cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
                tBsSFB = cute.filter_zeros(tBsSFB); tBgSFB = cute.filter_zeros(tBgSFB)
                tAgA_s = tAgA[(None, coord_x, None, 0)]
                tBgB_s = tBgB[(None, coord_y, None, 0)]
                tAgSFA_s = tAgSFA[(None, coord_x, None, 0)]
                tBgSFB_s = tBgSFB[(None, coord_y, None, 0)]
{pf_prologue}{tma_pdl_prologue}                ab_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
                ab_prod.reset_count()
                for kt_p in cutlass.range_constexpr({init_st_g}):
                    ab_pipeline.producer_acquire(ab_prod, cutlass.Boolean(1))
                    bar_p = ab_pipeline.producer_get_barrier(ab_prod)
                    st_p = ab_prod.index
                    cute.copy(tma_atom_b_{g}, tBgB_s[(None, kt_p)], tBsB[(None, st_p)], tma_bar_ptr=bar_p{pol_b})
                    cute.copy(tma_atom_a_{g}, tAgA_s[(None, kt_p)], tAsA[(None, st_p)], tma_bar_ptr=bar_p{pol_a})
                    cute.copy(tma_atom_sfb_{g}, tBgSFB_s[(None, kt_p)], tBsSFB[(None, st_p)], tma_bar_ptr=bar_p, mcast_mask=cutlass.Int16(3){pol_b})
                    cute.copy(tma_atom_sfa_{g}, tAgSFA_s[(None, kt_p)], tAsSFA[(None, st_p)], tma_bar_ptr=bar_p{pol_a})
{pf_steady_1}                    ab_prod.advance()
                if cutlass.const_expr({k_tiles_g} > {init_st_g}):
                    peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
                    for kt in cutlass.range({init_st_g}, {k_tiles_g}, 1, unroll=1):
                        ab_pipeline.producer_acquire(ab_prod, peek_empty)
                        bar = ab_pipeline.producer_get_barrier(ab_prod)
                        st = ab_prod.index
                        cute.copy(tma_atom_b_{g}, tBgB_s[(None, kt)], tBsB[(None, st)], tma_bar_ptr=bar{pol_b})
                        cute.copy(tma_atom_a_{g}, tAgA_s[(None, kt)], tAsA[(None, st)], tma_bar_ptr=bar{pol_a})
                        cute.copy(tma_atom_sfb_{g}, tBgSFB_s[(None, kt)], tBsSFB[(None, st)], tma_bar_ptr=bar, mcast_mask=cutlass.Int16(3){pol_b})
                        cute.copy(tma_atom_sfa_{g}, tAgSFA_s[(None, kt)], tAsSFA[(None, st)], tma_bar_ptr=bar{pol_a})
{pf_steady_2}                        ab_prod.advance()
                        peek_empty = cutlass.Boolean(1)
                        if ab_prod.count < {k_tiles_g}:
                            peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
{tma_pdl_epilogue}""")

        epi_branches.append(f"""
                {kw} group_idx == {g}:
                    if (coord_x * 2 + cta_rank) * 128 < {m_g}:
                        gC = cute.local_tile(mC_{g}, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
                        tCgC = thr_mma.partition_C(gC)
                        gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], self.epi_tile)
                        bSG_sC, bSG_gC_p = cpasync.tma_partition(tma_atom_c_{g}, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(gC_epi, 0, 2))
                        bSG_gC = cute.group_modes(bSG_gC_p[(None, None, None, coord_x, coord_y, 0)], 1, cute.rank(bSG_gC_p[(None, None, None, coord_x, coord_y, 0)]))
                        for sub_i in cutlass.range_constexpr({subtile_cnt}):
                            cute.copy(tma_atom_c_{g}, bSG_sC[(None, sub_i)], bSG_gC[(None, sub_i)]{pol_c})
""")

    launch_stream_2cta = "            stream=stream,\n" if with_stream else ""
    alpha_setup_2cta = (
        "            alpha_val = cutlass.Float32(1.0)\n" + "".join(alpha_branches)
        if has_alpha
        else ""
    )
    store_expr_2cta = (
        "tRS_rC.store((tTR_rAcc.load() * alpha_val).to(self.c_dtype))"
        if has_alpha
        else "tRS_rC.store(tTR_rAcc.load().to(self.c_dtype))"
    )
    cuda_imp_2cta = "import cuda.bindings.driver as cuda\n" if with_stream else ""

    src = f"""
import ctypes
{cuda_imp_2cta}import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr

_OP_K_MAJOR = cute.nvgpu.OperandMajorMode.K if hasattr(cute.nvgpu, "OperandMajorMode") else tcgen05.OperandMajorMode.K

class Sm100Nvfp4GroupGemm2CTA:
    def __init__(self):
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.{c_dtype_str}
        self.sf_vec_size = 16
        self.mma_tiler = (256, {tile_n}, 256)
        self.mma_tiler_sfb = (128, {sfb_n}, 256)
        self.cta_group = tcgen05.CtaGroup.TWO
        self.cluster_shape_mn = (2, 1)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192
        self.num_ab_stage = {num_ab_stage}
        self.num_acc_stage = 1
        self.num_c_stage = {subtile_cnt}
        self.epi_tile = (128, 64)
        self.num_tmem_alloc_cols = {tmem_cols}
        self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=192)
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=160)

    @cute.jit
    def __call__(self, {ptr_args_jit}):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype, _OP_K_MAJOR, _OP_K_MAJOR,
            self.sf_dtype, self.sf_vec_size, self.cta_group, (256, {tile_n})
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype, _OP_K_MAJOR, _OP_K_MAJOR,
            self.sf_dtype, self.sf_vec_size, tcgen05.CtaGroup.ONE, (128, {sfb_n})
        )
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (tiled_mma.thr_id.shape,))
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(cute.make_layout((2, 1, 1)), (tiled_mma_sfb.thr_id.shape,))
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
            + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        ) * 2

        {"".join(host_setup)}

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)], 1024]
            sSFA: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)], 1024]
            sSFB: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)], 1024]

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
            cluster=(2, 1, 1),
{launch_stream_2cta}            min_blocks_per_mp=1,
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
        cta_rank = bidx & 1
        cl_idx = bidx >> 1
        is_leader_cta = cta_rank == 0

        group_idx = cutlass.Int32(0)
        coord_x = cutlass.Int32(0)
        coord_y = cutlass.Int32(0)
        k_tile_cnt = cutlass.Int32(1)
        {"".join(decode_branches)}

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
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 256),
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

        sC_ptr = cute.recast_ptr(storage.sA.data_ptr(), c_smem_layout_staged.inner, dtype=self.c_dtype)
        sC = cute.make_tensor(sC_ptr, c_smem_layout_staged.outer)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        thr_mma = tiled_mma.get_slice(cta_rank)
        thr_mma_sfb = tiled_mma_sfb.get_slice(cta_rank)

        if warp_idx == self.tma_warp_id:
{tma_top_launch}            {"".join(tma_branches)}

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((256, {tile_n}))).layout)
            sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base), dtype=self.sf_dtype)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)))
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base) + tcgen05.find_tmem_tensor_col_offset(tCtSFA), dtype=self.sf_dtype)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)))
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO), self.sf_dtype)
            tCsSFA_c = cute.filter_zeros(sSFA); tCtSFA_c = cute.filter_zeros(tCtSFA)
            tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_c)
            thr_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
            tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfa, thr_s2t_sfa.partition_S(tCsSFA_c))
            tCtSFA_s2t = thr_s2t_sfa.partition_D(tCtSFA_c)

            tCsSFB_c = cute.filter_zeros(sSFB); tCtSFB_c = cute.filter_zeros(tCtSFB)
            tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_c)
            thr_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
            tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfb, thr_s2t_sfb.partition_S(tCsSFB_c))
            tCtSFB_s2t = thr_s2t_sfb.partition_D(tCtSFB_c)

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            tCtAcc = tCtAcc_base[(None, None, None)]

            if is_leader_cta:
                ab_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
                acc_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                ab_cons.reset_count()
                peek_full = ab_pipeline.consumer_try_wait(ab_cons)

                ab_pipeline.consumer_wait(ab_cons, peek_full)
                st0 = ab_cons.index
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb0 in cutlass.range_constexpr(4):
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t[(None, None, None, kb0, st0)], tCtSFA_s2t[(None, None, None, kb0)])
                    cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t[(None, None, None, kb0, st0)], tCtSFB_s2t[(None, None, None, kb0)])
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb0)].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kb0)].iterator)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb0, st0)], tCrB[(None, None, kb0, st0)], tCtAcc)
                    if cutlass.const_expr(kb0 == 0):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                rel0 = ab_cons.index
                ab_cons.advance()
                peek_full = cutlass.Boolean(1)
                if ab_cons.count < k_tile_cnt:
                    if ab_cons.count <= k_tile_cnt - self.num_ab_stage:
                        ab_pipeline.sync_object_empty.arrive(rel0, cutlass.Int16(3), self.cta_group)
                    peek_full = ab_pipeline.consumer_try_wait(ab_cons)

                for kt in cutlass.range(1, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.consumer_wait(ab_cons, peek_full)
                    st = ab_cons.index
                    for kb in cutlass.range_constexpr(4):
                        cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t[(None, None, None, kb, st)], tCtSFA_s2t[(None, None, None, kb)])
                        cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t[(None, None, None, kb, st)], tCtSFB_s2t[(None, None, None, kb)])
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kb)].iterator)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb, st)], tCrB[(None, None, kb, st)], tCtAcc)
                    rel_i = ab_cons.index
                    ab_cons.advance()
                    peek_full = cutlass.Boolean(1)
                    if ab_cons.count < k_tile_cnt:
                        if ab_cons.count <= k_tile_cnt - self.num_ab_stage:
                            ab_pipeline.sync_object_empty.arrive(rel_i, cutlass.Int16(3), self.cta_group)
                        peek_full = ab_pipeline.consumer_try_wait(ab_cons)
                acc_pipeline.producer_commit(acc_prod)

        if warp_idx < 4:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((256, {tile_n}))).layout)
            epi_tidx = tidx
            copy_atom_t2r = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), self.acc_dtype)
            tCtAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0)], self.epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

            copy_atom_r2s = sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)

            c_ident = cute.make_identity_tensor((128, {tile_n}))
            c_ident_epi = cute.flat_divide(c_ident, self.epi_tile)
            tTR_cC = thr_copy_t2r.partition_D(c_ident_epi)
            tTR_rAcc = cute.make_rmem_tensor(tTR_cC[(None, None, None, 0, 0)].shape, self.acc_dtype)
            tRS_rC = cute.make_rmem_tensor(thr_copy_r2s.retile(tTR_rAcc).shape, self.c_dtype)

            tmem.relinquish_alloc_permit()
{alpha_setup_2cta}            acc_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            acc_pipeline.consumer_wait(acc_cons)
            for sub_i in cutlass.range_constexpr({subtile_cnt}):
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, sub_i)], tTR_rAcc)
                {store_expr_2cta}
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, sub_i)])
            tmem.free(acc_tmem_ptr)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == 0:
                cute.arch.fence_view_async_shared()
{epi_pdl_wait}                {"".join(epi_branches)}
                cute.arch.cp_async_bulk_commit_group()
{epi_bulk_wait}"""
    mod = _load_in_mem_module(
        f"fg2cta_{tile_n}_{num_ab_stage}_{pf_dist}_{order_mn}_{l2_hint}_{pdl_safe}_{c_dtype_str}_{has_alpha}_{with_stream}_{hash(problem_sizes) & 0xFFFFFFFF}",
        src,
    )
    kern = mod.Sm100Nvfp4GroupGemm2CTA()
    if with_stream:
        return _compile_and_wrap_group_kernel_sglang(
            kern, G, getattr(cutlass, c_dtype_str), has_alpha
        )
    return _compile_and_wrap_group_kernel(kern, G)


def _compile_and_wrap_group_kernel(kern, G: int):
    dummy_ptrs = []
    for _ in range(G):
        dummy_ptrs.extend(
            [
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
                make_ptr(cutlass.Float16, 0, cute.AddressSpace.gmem, assumed_align=16),
            ]
        )
    fn = cute.compile(kern, *dummy_ptrs)

    state = {"fast": None}

    def _run(data: Tuple) -> List[torch.Tensor]:
        fast_fn = state["fast"]
        if fast_fn is not None:
            return fast_fn(data)
        abc_tensors, _, sfasfb_reordered_tensors, _ = data
        real_ptrs = []
        out = []
        for (a, b, c), (sfa_p, sfb_p) in zip(abc_tensors, sfasfb_reordered_tensors):
            real_ptrs.extend(
                [
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
                        sfa_p.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=32,
                    ),
                    make_ptr(
                        cutlass.Float8E4M3FN,
                        sfb_p.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=32,
                    ),
                    make_ptr(
                        cutlass.Float16,
                        c.data_ptr(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                ]
            )
            out.append(c)
        fn(*real_ptrs)
        ex = fn._default_executor
        packed_args = ex._tls.packed_args
        ptr_buf = (ctypes.c_uint64 * (5 * G))()
        base_addr = ctypes.addressof(ptr_buf)
        for i in range(5 * G):
            packed_args[i] = base_addr + i * 8
        capi_func = ex.capi_func
        _KEEPALIVE.append((kern, fn, ex, packed_args, ptr_buf))

        def _fast(data_inner: Tuple) -> List[torch.Tensor]:
            abc_t, _, sfasfb_t, _ = data_inner
            idx = 0
            res = []
            for (a, b, c), (sfa_p, sfb_p) in zip(abc_t, sfasfb_t):
                ptr_buf[idx] = a.data_ptr()
                ptr_buf[idx + 1] = b.data_ptr()
                ptr_buf[idx + 2] = sfa_p.data_ptr()
                ptr_buf[idx + 3] = sfb_p.data_ptr()
                ptr_buf[idx + 4] = c.data_ptr()
                idx += 5
                res.append(c)
            capi_func(packed_args)
            return res

        state["fast"] = _fast
        return out

    return _run


def _compile_and_wrap_group_kernel_sglang(
    kern, G: int, c_dtype_cutlass, has_alpha: bool
):
    ptrs_per_group = 6 if has_alpha else 5
    total_slots = ptrs_per_group * G + 1
    dummy_args = []
    for _ in range(G):
        dummy_args.extend(
            [
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
                make_ptr(c_dtype_cutlass, 0, cute.AddressSpace.gmem, assumed_align=16),
            ]
        )
        if has_alpha:
            dummy_args.append(
                make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4)
            )
    dummy_args.append(cuda.CUstream(0))
    fn = cute.compile(kern, *dummy_args)

    state = {"fast": None}

    def _run_sgl(ptr_list: List[int], stream_i: int):
        fast_fn = state["fast"]
        if fast_fn is not None:
            fast_fn(ptr_list, stream_i)
            return
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
                        cutlass.Float8E4M3FN,
                        ptr_list[idx + 2],
                        cute.AddressSpace.gmem,
                        assumed_align=32,
                    ),
                    make_ptr(
                        cutlass.Float8E4M3FN,
                        ptr_list[idx + 3],
                        cute.AddressSpace.gmem,
                        assumed_align=32,
                    ),
                    make_ptr(
                        c_dtype_cutlass,
                        ptr_list[idx + 4],
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                ]
            )
            if has_alpha:
                real_args.append(
                    make_ptr(
                        cutlass.Float32,
                        ptr_list[idx + 5],
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                )
            idx += ptrs_per_group
        real_args.append(cuda.CUstream(stream_i))
        fn(*real_args)
        ex = fn._default_executor
        packed_args = ex._tls.packed_args
        ptr_buf = (ctypes.c_uint64 * total_slots)()
        base_addr = ctypes.addressof(ptr_buf)
        for i in range(total_slots):
            packed_args[i] = base_addr + i * 8
        capi_func = ex.capi_func
        _KEEPALIVE.append((kern, fn, ex, packed_args, ptr_buf))

        def _fast_sgl(ptr_list_inner: List[int], stream_inner: int):
            for i, val in enumerate(ptr_list_inner):
                ptr_buf[i] = val
            ptr_buf[total_slots - 1] = stream_inner
            capi_func(packed_args)

        state["fast"] = _fast_sgl

    return _run_sgl


def _get_group_runner(
    problem_sizes: Tuple[Tuple[int, int, int, int], ...],
    pdl_safe: bool = _DEFAULT_PDL_SAFE,
):
    key = (problem_sizes, pdl_safe)
    runner = _RUNNER_CACHE.get(key)
    if runner is not None:
        return runner

    G = len(problem_sizes)
    if G <= 8:
        use_2cta = all(
            m > 128 and n % 128 == 0 and k % 256 == 0 and l == 1
            for m, n, k, l in problem_sizes
        )
        if use_2cta:
            sum_clusters_128 = sum(
                _ceil_div(m, 256) * (n // 128) for m, n, _, _ in problem_sizes
            )
            if sum_clusters_128 > 74 and all(
                n % 256 == 0 for _, n, _, _ in problem_sizes
            ):
                sum_clusters_256 = sum(
                    _ceil_div(m, 256) * (n // 256) for m, n, _, _ in problem_sizes
                )
                multi_wave_large = sum_clusters_256 > 200
                runner = _build_fused_group_runner_2cta(
                    problem_sizes,
                    tile_n=256,
                    num_ab_stage=5,
                    pf_dist=(0 if multi_wave_large else 5),
                    order_mn=(not multi_wave_large),
                    l2_hint=(not multi_wave_large),
                    pdl_safe=pdl_safe,
                )
            else:
                runner = _build_fused_group_runner_2cta(
                    problem_sizes,
                    tile_n=128,
                    num_ab_stage=7,
                    pf_dist=6,
                    order_mn=True,
                    l2_hint=True,
                    pdl_safe=pdl_safe,
                )
        else:
            tile_n = 128 if all(n % 128 == 0 for _, n, _, _ in problem_sizes) else 64
            runner = _build_fused_group_runner_1cta(
                problem_sizes,
                tile_n=tile_n,
                num_ab_stage=6,
                pf_dist=6,
                order_mn=True,
                l2_hint=True,
                pdl_safe=pdl_safe,
            )
    else:
        sub_runners = [
            _get_group_runner(problem_sizes[i : i + 8], pdl_safe=pdl_safe)
            for i in range(0, G, 8)
        ]

        def runner(data: Tuple) -> List[torch.Tensor]:
            abc_t, sfasfb_t, sfasfb_r, ps = data
            out = []
            for idx, sr in enumerate(sub_runners):
                s = idx * 8
                e = min(s + 8, G)
                out.extend(sr((abc_t[s:e], sfasfb_t[s:e], sfasfb_r[s:e], ps[s:e])))
            return out

    _RUNNER_CACHE[key] = runner
    return runner


def _get_sglang_group_runner(
    problem_sizes: Tuple[Tuple[int, int, int, int], ...],
    out_dtype: torch.dtype,
    has_alpha: bool,
    pdl_safe: bool,
):
    key = (problem_sizes, out_dtype, has_alpha, pdl_safe)
    runner = _SGLANG_GROUP_CACHE.get(key)
    if runner is not None:
        return runner

    c_dtype_str = "BFloat16" if out_dtype == torch.bfloat16 else "Float16"
    G = len(problem_sizes)
    if G <= 8:
        use_2cta = all(
            m > 128 and n % 128 == 0 and k % 256 == 0 and l == 1
            for m, n, k, l in problem_sizes
        )
        if use_2cta:
            sum_clusters_128 = sum(
                _ceil_div(m, 256) * (n // 128) for m, n, _, _ in problem_sizes
            )
            if sum_clusters_128 > 74 and all(
                n % 256 == 0 for _, n, _, _ in problem_sizes
            ):
                sum_clusters_256 = sum(
                    _ceil_div(m, 256) * (n // 256) for m, n, _, _ in problem_sizes
                )
                multi_wave_large = sum_clusters_256 > 200
                runner = _build_fused_group_runner_2cta(
                    problem_sizes,
                    tile_n=256,
                    num_ab_stage=5,
                    pf_dist=(0 if multi_wave_large else 5),
                    order_mn=(not multi_wave_large),
                    l2_hint=(not multi_wave_large),
                    pdl_safe=pdl_safe,
                    c_dtype_str=c_dtype_str,
                    has_alpha=has_alpha,
                    with_stream=True,
                )
            else:
                runner = _build_fused_group_runner_2cta(
                    problem_sizes,
                    tile_n=128,
                    num_ab_stage=7,
                    pf_dist=6,
                    order_mn=True,
                    l2_hint=True,
                    pdl_safe=pdl_safe,
                    c_dtype_str=c_dtype_str,
                    has_alpha=has_alpha,
                    with_stream=True,
                )
        else:
            tile_n = 128 if all(n % 128 == 0 for _, n, _, _ in problem_sizes) else 64
            runner = _build_fused_group_runner_1cta(
                problem_sizes,
                tile_n=tile_n,
                num_ab_stage=6,
                pf_dist=6,
                order_mn=True,
                l2_hint=True,
                pdl_safe=pdl_safe,
                c_dtype_str=c_dtype_str,
                has_alpha=has_alpha,
                with_stream=True,
            )
    else:
        ptrs_per_group = 6 if has_alpha else 5
        sub_runners = [
            _get_sglang_group_runner(
                problem_sizes[i : i + 8],
                out_dtype=out_dtype,
                has_alpha=has_alpha,
                pdl_safe=pdl_safe,
            )
            for i in range(0, G, 8)
        ]

        def runner(ptr_list: List[int], stream_i: int):
            for idx, sr in enumerate(sub_runners):
                s = idx * 8 * ptrs_per_group
                e = min((idx + 1) * 8, G) * ptrs_per_group
                sr(ptr_list[s:e], stream_i)

    _SGLANG_GROUP_CACHE[key] = runner
    return runner


def nvfp4_grouped_gemm(
    abc_tensors: Sequence[
        Union[
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor],
        ]
    ],
    sfasfb_tensors: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    problem_sizes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    *,
    alphas: Optional[Union[Sequence[torch.Tensor], torch.Tensor]] = None,
    swizzled_scales: bool = False,
    out_dtype: Optional[torch.dtype] = None,
    pdl_safe: bool = True,
) -> List[torch.Tensor]:
    """Fused single-launch Blackwell SM100 NVFP4 grouped GEMM."""
    G = len(abc_tensors)
    if G == 0:
        return []
    if len(sfasfb_tensors) != G:
        raise ValueError(
            f"Expected sfasfb_tensors length ({len(sfasfb_tensors)}) to match abc_tensors ({G})"
        )
    if problem_sizes is not None and len(problem_sizes) != G:
        raise ValueError(
            f"Expected problem_sizes length ({len(problem_sizes)}) to match abc_tensors ({G})"
        )

    has_alpha = alphas is not None
    if has_alpha and len(alphas) != G:
        raise ValueError(
            f"Expected alphas length ({len(alphas)}) to match abc_tensors ({G})"
        )

    first_a = abc_tensors[0][0]
    if first_a.device.type != "cuda":
        raise ValueError("nvfp4_grouped_gemm requires CUDA tensors")
    dev = first_a.device

    if out_dtype is None:
        if len(abc_tensors[0]) == 3 and abc_tensors[0][2] is not None:
            out_dtype = abc_tensors[0][2].dtype
        else:
            out_dtype = torch.float16
    if out_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"Unsupported output dtype {out_dtype}; expected float16 or bfloat16"
        )

    inferred_sizes: List[Tuple[int, int, int, int]] = []
    out_tensors: List[torch.Tensor] = []
    keepalive_scales: List[torch.Tensor] = []
    ptr_list: List[int] = []

    for g in range(G):
        abc_g = abc_tensors[g]
        if len(abc_g) == 2:
            a_g, b_g = abc_g
            c_g = None
        elif len(abc_g) == 3:
            a_g, b_g, c_g = abc_g
        else:
            raise ValueError(
                f"Group {g}: expected (a, b) or (a, b, c), got {len(abc_g)} tensors"
            )

        sfa_g, sfb_g = sfasfb_tensors[g]
        if (
            a_g.device != dev
            or b_g.device != dev
            or sfa_g.device != dev
            or sfb_g.device != dev
        ):
            raise ValueError(
                f"Group {g}: all tensors must be on the same CUDA device {dev}"
            )

        is_2d = a_g.ndim == 2
        if is_2d:
            m_g, k_half_g = a_g.shape
            l_g = 1
        elif a_g.ndim == 3:
            m_g, k_half_g, l_g = a_g.shape
        else:
            raise ValueError(
                f"Group {g}: expected 2D or 3D tensor a, got shape {tuple(a_g.shape)}"
            )

        if (
            b_g.ndim != a_g.ndim
            or b_g.shape[1] != k_half_g
            or (not is_2d and b_g.shape[2] != l_g)
        ):
            raise ValueError(
                f"Group {g}: shape mismatch between a {tuple(a_g.shape)} and b {tuple(b_g.shape)}"
            )
        if a_g.stride(1) != 1 or b_g.stride(1) != 1:
            raise ValueError(
                f"Group {g}: input tensors a and b must be K-major (stride(1) == 1)"
            )

        n_g = b_g.shape[0]
        k_g = k_half_g * 2
        if problem_sizes is not None:
            pm, pn, pk, pl = problem_sizes[g]
            if (pm, pn, pk, pl) != (m_g, n_g, k_g, l_g):
                raise ValueError(
                    f"Group {g}: problem_sizes {(pm, pn, pk, pl)} does not match tensor shapes {(m_g, n_g, k_g, l_g)}"
                )
        if (
            m_g < 0
            or n_g <= 0
            or k_g <= 0
            or l_g != 1
            or (n_g % 64 != 0)
            or (k_g % 256 != 0)
        ):
            raise ValueError(
                f"Group {g}: unsupported NVFP4 grouped GEMM shape (M={m_g}, N={n_g}, K={k_g}, L={l_g}); "
                "requires M >= 0, N % 64 == 0, K % 256 == 0, L == 1."
            )

        if c_g is not None:
            expected_c_shape = (m_g, n_g) if is_2d else (m_g, n_g, l_g)
            if c_g.shape != expected_c_shape or c_g.device != dev:
                raise ValueError(
                    f"Group {g}: expected c shape {expected_c_shape} on {dev}, got {tuple(c_g.shape)} on {c_g.device}"
                )
            if m_g > 0 and (c_g.stride(1) != 1 or c_g.stride(0) < n_g):
                raise ValueError(
                    f"Group {g}: output tensor c must be row-major (stride(1) == 1)"
                )
            if c_g.dtype != out_dtype:
                raise ValueError(
                    f"Group {g}: output tensor dtype ({c_g.dtype}) does not match out_dtype ({out_dtype})"
                )
        else:
            if is_2d:
                c_g = torch.empty((m_g, n_g), dtype=out_dtype, device=dev)
            else:
                c_g = torch.empty((l_g, m_g, n_g), dtype=out_dtype, device=dev).permute(
                    1, 2, 0
                )

        out_tensors.append(c_g)
        if m_g == 0:
            continue

        inferred_sizes.append((m_g, n_g, k_g, l_g))

        if not swizzled_scales:
            if sfa_g.ndim != 6:
                sfa_g = permute_nvfp4_scale(sfa_g)
            if sfb_g.ndim != 6:
                sfb_g = permute_nvfp4_scale(sfb_g)
        keepalive_scales.append(sfa_g)
        keepalive_scales.append(sfb_g)

        ptr_list.extend(
            [
                a_g.data_ptr(),
                b_g.data_ptr(),
                sfa_g.data_ptr(),
                sfb_g.data_ptr(),
                c_g.data_ptr(),
            ]
        )
        if has_alpha:
            alpha_g = alphas[g]
            if alpha_g.device != dev or alpha_g.dtype != torch.float32:
                raise ValueError(f"Group {g}: alpha must be a float32 tensor on {dev}")
            ptr_list.append(alpha_g.data_ptr())

    if not inferred_sizes:
        return out_tensors

    stream_i = torch.cuda.current_stream(dev).cuda_stream
    runner = _get_sglang_group_runner(
        tuple(inferred_sizes),
        out_dtype=out_dtype,
        has_alpha=has_alpha,
        pdl_safe=pdl_safe,
    )
    runner(ptr_list, stream_i)
    return out_tensors


_MASKED_GROUP_CACHE: Dict[Tuple, object] = {}


def _build_masked_group_runner_1cta(
    G: int,
    m_max: int,
    n: int,
    k: int,
    tile_n: int = 128,
    num_ab_stage: int = 6,
    pf_dist: int = 6,
    order_mn: bool = True,
    c_dtype_str: str = "BFloat16",
    pdl_safe: bool = True,
):
    key = (
        "1cta_masked",
        G,
        m_max,
        n,
        k,
        tile_n,
        num_ab_stage,
        pf_dist,
        order_mn,
        c_dtype_str,
        pdl_safe,
    )
    if key in _MASKED_GROUP_CACHE:
        return _MASKED_GROUP_CACHE[key]

    tile_m = 128
    tile_k = 256
    subtile_cnt = tile_n // 64
    tmem_cols = 256 if tile_n == 128 else 128
    cm = (m_max + tile_m - 1) // tile_m
    cn = (n + tile_n - 1) // tile_n
    tiles_per_group = cm * cn
    total_ctas = G * tiles_per_group
    k_tiles = k // tile_k
    init_st = min(num_ab_stage, k_tiles)

    ptr_args_jit = (
        ", ".join(
            f"a_ptr_{g}: cute.Pointer, b_ptr_{g}: cute.Pointer, sfa_ptr_{g}: cute.Pointer, sfb_ptr_{g}: cute.Pointer, c_ptr_{g}: cute.Pointer, alpha_ptr_{g}: cute.Pointer"
            for g in range(G)
        )
        + ", masked_m_ptr: cute.Pointer, stream: cuda.CUstream"
    )
    kern_args_def = (
        ", ".join(
            f"tma_atom_a_{g}: cute.CopyAtom, mA_{g}: cute.Tensor, "
            f"tma_atom_b_{g}: cute.CopyAtom, mB_{g}: cute.Tensor, "
            f"tma_atom_sfa_{g}: cute.CopyAtom, mSFA_{g}: cute.Tensor, "
            f"tma_atom_sfb_{g}: cute.CopyAtom, mSFB_{g}: cute.Tensor, "
            f"tma_atom_c_{g}: cute.CopyAtom, mC_{g}: cute.Tensor, mAlpha_{g}: cute.Tensor"
            for g in range(G)
        )
        + ", mMaskedM: cute.Tensor"
    )
    kern_args_call = (
        ", ".join(
            f"tma_atom_a_{g}, tma_tensor_a_{g}, "
            f"tma_atom_b_{g}, tma_tensor_b_{g}, "
            f"tma_atom_sfa_{g}, tma_tensor_sfa_{g}, "
            f"tma_atom_sfb_{g}, tma_tensor_sfb_{g}, "
            f"tma_atom_c_{g}, tma_tensor_c_{g}, alpha_t_{g}"
            for g in range(G)
        )
        + ", masked_m_t"
    )

    host_setup = []
    tma_branches = []
    epi_branches = []
    alpha_branches = []

    tma_pdl_prologue = (
        "                    cute.arch.griddepcontrol_wait()\n" if pdl_safe else ""
    )
    tma_pdl_epilogue = (
        "                    cute.arch.griddepcontrol_launch_dependents()\n"
        if pdl_safe
        else ""
    )

    for g in range(G):
        kw = "if" if g == 0 else "elif"
        alpha_branches.append(
            f"                {kw} group_idx == {g}:\n                    alpha_val = cutlass.Float32(mAlpha_{g}[0])\n"
        )
        host_setup.append(f"""
        a_t_{g} = cute.make_tensor(a_ptr_{g}, cute.make_layout(({m_max}, cute.assume({k}, 32), 1), stride=(cute.assume({k}, 32), 1, cute.assume({m_max * k}, 32))))
        b_t_{g} = cute.make_tensor(b_ptr_{g}, cute.make_layout(({n}, cute.assume({k}, 32), 1), stride=(cute.assume({k}, 32), 1, cute.assume({n * k}, 32))))
        c_t_{g} = cute.make_tensor(c_ptr_{g}, cute.make_layout(({m_max}, cute.assume({n}, 32), 1), stride=(cute.assume({n}, 32), 1, cute.assume({m_max * n}, 32))))
        sfa_t_{g} = cute.make_tensor(sfa_ptr_{g}, blockscaled_utils.tile_atom_to_shape_SF(a_t_{g}.shape, 16))
        sfb_t_{g} = cute.make_tensor(sfb_ptr_{g}, blockscaled_utils.tile_atom_to_shape_SF(b_t_{g}.shape, 16))
        alpha_t_{g} = cute.make_tensor(alpha_ptr_{g}, cute.make_layout((1,), stride=(1,)))
        tma_atom_a_{g}, tma_tensor_a_{g} = cute.nvgpu.make_tiled_tma_atom_A(a_op, a_t_{g}, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_atom_b_{g}, tma_tensor_b_{g} = cute.nvgpu.make_tiled_tma_atom_B(b_op, b_t_{g}, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_atom_sfa_{g}, tma_tensor_sfa_{g} = cute.nvgpu.make_tiled_tma_atom_A(sfa_op, sfa_t_{g}, sfa_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16)
        tma_atom_sfb_{g}, tma_tensor_sfb_{g} = cute.nvgpu.make_tiled_tma_atom_B(sfb_op, sfb_t_{g}, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)
        tma_atom_c_{g}, tma_tensor_c_{g} = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c_t_{g}, epi_smem_layout, self.epi_tile)
""")
        tma_branches.append(f"""
                {kw} group_idx == {g}:
                    cpasync.prefetch_descriptor(tma_atom_a_{g})
                    cpasync.prefetch_descriptor(tma_atom_b_{g})
                    cpasync.prefetch_descriptor(tma_atom_sfa_{g})
                    cpasync.prefetch_descriptor(tma_atom_sfb_{g})
                    gA = cute.local_tile(mA_{g}, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                    gB = cute.local_tile(mB_{g}, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                    gSFA = cute.local_tile(mSFA_{g}, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                    gSFB = cute.local_tile(mSFB_{g}, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
                    tCgA = thr_mma.partition_A(gA)
                    tCgB = thr_mma.partition_B(gB)
                    tCgSFA = thr_mma.partition_A(gSFA)
                    tAsA, tAgA = cpasync.tma_partition(tma_atom_a_{g}, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
                    tBsB, tBgB = cpasync.tma_partition(tma_atom_b_{g}, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
                    tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa_{g}, 0, cute.make_layout(1), cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
                    tAsSFA = cute.filter_zeros(tAsSFA); tAgSFA = cute.filter_zeros(tAgSFA)
                    tBsSFB, tBgSFB = cpasync.tma_partition(tma_atom_sfb_{g}, 0, cute.make_layout(1), cute.group_modes(sSFB, 0, 3), cute.group_modes(thr_mma_sfb.partition_B(gSFB), 0, 3))
                    tBsSFB = cute.filter_zeros(tBsSFB); tBgSFB = cute.filter_zeros(tBgSFB)
                    tAgA_s = tAgA[(None, coord_x, None, 0)]
                    tBgB_s = tBgB[(None, coord_y, None, 0)]
                    tAgSFA_s = tAgSFA[(None, coord_x, None, 0)]
                    tBgSFB_s = tBgSFB[(None, coord_y, None, 0)]
{tma_pdl_prologue}                    ab_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
                    ab_prod.reset_count()
                    for kt_p in cutlass.range_constexpr({init_st}):
                        ab_pipeline.producer_acquire(ab_prod, cutlass.Boolean(1))
                        bar_p = ab_pipeline.producer_get_barrier(ab_prod)
                        st_p = ab_prod.index
                        cute.copy(tma_atom_b_{g}, tBgB_s[(None, kt_p)], tBsB[(None, st_p)], tma_bar_ptr=bar_p, cache_policy=cutlass.Int64(0x12F0000000000000))
                        cute.copy(tma_atom_a_{g}, tAgA_s[(None, kt_p)], tAsA[(None, st_p)], tma_bar_ptr=bar_p, cache_policy=cutlass.Int64(0x14F0000000000000))
                        cute.copy(tma_atom_sfb_{g}, tBgSFB_s[(None, kt_p)], tBsSFB[(None, st_p)], tma_bar_ptr=bar_p, cache_policy=cutlass.Int64(0x12F0000000000000))
                        cute.copy(tma_atom_sfa_{g}, tAgSFA_s[(None, kt_p)], tAsSFA[(None, st_p)], tma_bar_ptr=bar_p, cache_policy=cutlass.Int64(0x14F0000000000000))
                        ab_prod.advance()
                    if cutlass.const_expr({k_tiles} > {init_st}):
                        peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
                        for kt in cutlass.range({init_st}, {k_tiles}, 1, unroll=1):
                            ab_pipeline.producer_acquire(ab_prod, peek_empty)
                            bar = ab_pipeline.producer_get_barrier(ab_prod)
                            st = ab_prod.index
                            cute.copy(tma_atom_b_{g}, tBgB_s[(None, kt)], tBsB[(None, st)], tma_bar_ptr=bar, cache_policy=cutlass.Int64(0x12F0000000000000))
                            cute.copy(tma_atom_a_{g}, tAgA_s[(None, kt)], tAsA[(None, st)], tma_bar_ptr=bar, cache_policy=cutlass.Int64(0x14F0000000000000))
                            cute.copy(tma_atom_sfb_{g}, tBgSFB_s[(None, kt)], tBsSFB[(None, st)], tma_bar_ptr=bar, cache_policy=cutlass.Int64(0x12F0000000000000))
                            cute.copy(tma_atom_sfa_{g}, tAgSFA_s[(None, kt)], tAsSFA[(None, st)], tma_bar_ptr=bar, cache_policy=cutlass.Int64(0x14F0000000000000))
                            ab_prod.advance()
                            peek_empty = cutlass.Boolean(1)
                            if ab_prod.count < {k_tiles}:
                                peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
{tma_pdl_epilogue}""")
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

class Sm100Nvfp4MaskedGroupGemm1CTA:
    def __init__(self):
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.{c_dtype_str}
        self.sf_vec_size = 16
        self.mma_tiler = (128, {tile_n}, 256)
        self.mma_tiler_sfb = (128, 128, 256)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192
        self.num_ab_stage = {num_ab_stage}
        self.epi_tile = (128, 64)
        self.num_c_stage = {subtile_cnt}
        self.num_tmem_alloc_cols = {tmem_cols}
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=160)

    @cute.jit
    def __call__(self, {ptr_args_jit}):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            self.sf_dtype, self.sf_vec_size, self.cta_group, (128, {tile_n})
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
            + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )

        masked_m_t = cute.make_tensor(masked_m_ptr, cute.make_layout(({G},), stride=(1,)))
        {"".join(host_setup)}

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.a_smem_layout_staged.outer)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[self.ab_dtype, cute.cosize(self.b_smem_layout_staged.outer)], 1024]
            sSFA: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)], 1024]
            sSFB: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)], 1024]

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
            sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
            sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
            sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

            thr_mma = tiled_mma.get_slice(0)
            thr_mma_sfb = tiled_mma_sfb.get_slice(0)

            if warp_idx == self.tma_warp_id:
{"".join(tma_branches)}

            if warp_idx == self.mma_warp_id:
                tmem.wait_for_alloc()
                acc_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
                tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((128, {tile_n}))).layout)
                sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base), dtype=self.sf_dtype)
                tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)))
                tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
                sfb_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base) + tcgen05.find_tmem_tensor_col_offset(tCtSFA), dtype=self.sf_dtype)
                tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)))
                tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

                copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), self.sf_dtype)
                tCsSFA_c = cute.filter_zeros(sSFA); tCtSFA_c = cute.filter_zeros(tCtSFA)
                tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_c)
                thr_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
                tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfa, thr_s2t_sfa.partition_S(tCsSFA_c))
                tCtSFA_s2t = thr_s2t_sfa.partition_D(tCtSFA_c)

                tCsSFB_c = cute.filter_zeros(sSFB); tCtSFB_c = cute.filter_zeros(tCtSFB)
                tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_c)
                thr_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
                tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfb, thr_s2t_sfb.partition_S(tCsSFB_c))
                tCtSFB_s2t = thr_s2t_sfb.partition_D(tCtSFB_c)

                tCrA = tiled_mma.make_fragment_A(sA)
                tCrB = tiled_mma.make_fragment_B(sB)
                tCtAcc = tCtAcc_base[(None, None, None)]

                ab_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
                acc_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                ab_cons.reset_count()
                peek_full = ab_pipeline.consumer_try_wait(ab_cons)

                ab_pipeline.consumer_wait(ab_cons, peek_full)
                st0 = ab_cons.index
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb0 in cutlass.range_constexpr(4):
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t[(None, None, None, kb0, st0)], tCtSFA_s2t[(None, None, None, kb0)])
                    cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t[(None, None, None, kb0, st0)], tCtSFB_s2t[(None, None, None, kb0)])
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb0)].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kb0)].iterator)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb0, st0)], tCrB[(None, None, kb0, st0)], tCtAcc)
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
                        cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t[(None, None, None, kb, st)], tCtSFB_s2t[(None, None, None, kb)])
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kb)].iterator)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb, st)], tCrB[(None, None, kb, st)], tCtAcc)
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
                acc_tmem_ptr = cute.make_ptr(self.acc_dtype, 0, cute.AddressSpace.tmem, assumed_align=16)
                tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((128, {tile_n}))).layout)
                epi_tidx = tidx
                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    (128, {tile_n}, 256), utils.LayoutEnum.ROW_MAJOR, self.c_dtype, self.acc_dtype, self.epi_tile, False
                )
                tCtAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0)], self.epi_tile)
                tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)])
                thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
                tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                copy_atom_r2s = sm100_utils.get_smem_store_op(utils.LayoutEnum.ROW_MAJOR, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
                tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
                thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
                tRS_sC = thr_copy_r2s.partition_D(sC)

                c_ident = cute.make_identity_tensor((128, {tile_n}))
                c_ident_epi = cute.flat_divide(c_ident, self.epi_tile)
                tTR_cC = thr_copy_t2r.partition_D(c_ident_epi)
                tTR_rAcc = cute.make_rmem_tensor(tTR_cC[(None, None, None, 0, 0)].shape, self.acc_dtype)
                tRS_rC = cute.make_rmem_tensor(thr_copy_r2s.retile(tTR_rAcc).shape, self.c_dtype)

                tmem.relinquish_alloc_permit()
                alpha_val = cutlass.Float32(1.0)
{"".join(alpha_branches)}
                acc_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
                acc_pipeline.consumer_wait(acc_cons)
                for sub_i in cutlass.range_constexpr({subtile_cnt}):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, sub_i)], tTR_rAcc)
                    tRS_rC.store((tTR_rAcc.load() * alpha_val).to(self.c_dtype))
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, sub_i)])
                tmem.free(acc_tmem_ptr)
                self.epilog_sync_barrier.arrive_and_wait()
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
        f"m1cta_{G}_{m_max}_{n}_{k}_{tile_n}_{num_ab_stage}_{order_mn}_{c_dtype_str}_{pdl_safe}",
        src,
    )
    kern = mod.Sm100Nvfp4MaskedGroupGemm1CTA()
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

    total_slots = G * 6 + 2
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
                            cutlass.Float8E4M3FN,
                            ptr_list[idx + 2],
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        ),
                        make_ptr(
                            cutlass.Float8E4M3FN,
                            ptr_list[idx + 3],
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        ),
                        make_ptr(
                            c_dtype_cutlass,
                            ptr_list[idx + 4],
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        make_ptr(
                            cutlass.Float32,
                            ptr_list[idx + 5],
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        ),
                    ]
                )
                idx += 6
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
        for i in range(G * 6):
            ptr_buf[i] = ptr_list[i]
        ptr_buf[G * 6] = masked_m_ptr
        ptr_buf[G * 6 + 1] = stream_i
        capi_func(packed_args)

    _MASKED_GROUP_CACHE[key] = runner
    return runner


class Sm100Nvfp4MaskedGroupGemmPersistent:
    """L2-optimal row-resident Persistent 3D-TMA Masked Grouped NVFP4 GEMM."""

    def __init__(
        self,
        use_2cta: bool = True,
        tile_n: int = 128,
        c_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        pdl_safe: bool = True,
    ):
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.c_dtype = c_dtype
        self.sf_vec_size = 16
        self.use_2cta = use_2cta
        self.tile_m = 256 if use_2cta else 128
        self.tile_n = tile_n
        self.cluster_shape_mn = (2, 1) if use_2cta else (1, 1)
        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta else tcgen05.CtaGroup.ONE
        self.mma_tiler = (self.tile_m, tile_n, 256)
        self.mma_tiler_sfb = (128, ((tile_n + 127) // 128) * 128, 256)
        self.cta_tile_shape_mnk = (128, tile_n, 256)
        self.epi_tile = (128, 64)
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192
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
        self.iter_acc_early_release = self.num_sf_tmem_cols // 64
        self.num_tmem_alloc_cols = 512
        self.pdl_safe = pdl_safe
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=160)
        self.sched_sync_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=192)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        masked_m_ptr: cute.Pointer,
        stream: cuda.CUstream,
        shape_cfg: cutlass.Constexpr[tuple],
    ):
        G, m_max, sfa_m, n, k = shape_cfg
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            _OP_K_MAJOR,
            _OP_K_MAJOR,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            (self.tile_m, self.tile_n),
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
        reserved = ((256 + (2 * G + 1) * 4 + 1023) // 1024) * 1024
        num_c_stage = 2
        self.num_ab_stage = (
            self.smem_capacity - (reserved + num_c_stage * c_bytes_per)
        ) // ab_bytes
        self.num_c_stage = (
            num_c_stage
            + (
                self.smem_capacity
                - ab_bytes * self.num_ab_stage
                - (reserved + num_c_stage * c_bytes_per)
            )
            // c_bytes_per
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma_sfb.thr_id.shape,)
        )

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

        a_smem = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        b_smem = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        sfa_smem = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem = cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
        epi_smem = cute.slice_(c_smem_layout_staged, (None, None, 0))

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m_max, cute.assume(k, 32), G),
                stride=(cute.assume(k, 32), 1, cute.assume(m_max * k, 32)),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), G),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout(
                (m_max, cute.assume(n, 32), G),
                stride=(cute.assume(n, 32), 1, cute.assume(m_max * n, 32)),
            ),
        )
        sfa_tensor = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF((sfa_m, k, G), 16)
        )
        sfb_tensor = cute.make_tensor(
            sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF((n, k, G), 16)
        )
        alpha_tensor = cute.make_tensor(alpha_ptr, cute.make_layout((G,), stride=(1,)))
        masked_m_tensor = cute.make_tensor(
            masked_m_ptr, cute.make_layout((G,), stride=(1,))
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
            a_op, a_tensor, a_smem, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b_tensor, b_smem, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c_tensor, epi_smem, self.epi_tile
        )

        atom_thr_size = 2 if self.use_2cta else 1
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.ab_dtype, a_smem)
            + cute.size_in_bytes(self.ab_dtype, b_smem)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem)
        ) * atom_thr_size

        cn = (n + self.tile_n - 1) // self.tile_n
        max_units = 74 if self.use_2cta else 148
        worst_units = G * ((m_max + self.tile_m - 1) // self.tile_m) * cn
        active_units = min(worst_units, max_units)
        grid = (active_units * atom_thr_size, 1, 1)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            s_cum: cute.struct.MemRange[cutlass.Int32, G + 1]
            s_mg: cute.struct.MemRange[cutlass.Int32, G]
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
            alpha_tensor,
            masked_m_tensor,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            c_smem_layout_staged,
            G,
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
        mMaskedM: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
        G: cutlass.Constexpr[int],
        cn: cutlass.Constexpr[int],
        active_units: cutlass.Constexpr[int],
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

        a_mcast_mask = None
        b_mcast_mask = None
        sfa_mcast_mask = None
        sfb_mcast_mask = None
        if cutlass.const_expr(self.use_2cta):
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

        thr_mma = tiled_mma.get_slice(cta_rank)
        thr_mma_sfb = tiled_mma_sfb.get_slice(cta_rank)
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
                tBgB_s = tBgB[(None, coord_n, None, cur_g)]
                tAgSFA_s = tAgSFA[(None, coord_m, None, cur_g)]
                tBgSFB_s = tBgSFB[(None, slice_n, None, cur_g)]

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
                    ab_prod.advance()
                    peek_empty = cutlass.Boolean(1)
                    if ab_prod.count < k_block_cnt:
                        peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
                work_u = work_u + active_units

            if cutlass.const_expr(self.use_2cta):
                ab_pipeline.producer_tail(ab_prod)
            if cutlass.const_expr(self.pdl_safe):
                cute.arch.griddepcontrol_launch_dependents()

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

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_idx = acc_prod.phase ^ 1
                else:
                    acc_stage_idx = acc_prod.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_idx)]

                ab_cons.reset_count()
                peek_full = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_full = ab_pipeline.consumer_try_wait(ab_cons)
                    acc_pipeline.producer_acquire(acc_prod)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.tile_n == 64):
                    offset = cutlass.Int32((coord_n_m % 2) * 2)
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
                work_u_m = work_u_m + active_units

            acc_pipeline.producer_tail(acc_prod)

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
                self.use_2cta,
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
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            )

            if cutlass.const_expr(not self.pdl_safe):
                if warp_idx == 0:
                    cute.arch.griddepcontrol_wait()

            acc_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            work_u_e = cutlass.Int32(unit_id)
            cur_g_e = cutlass.Int32(0)
            tiles_done = cutlass.Int32(0)

            while work_u_e < total_units:
                while s_cum[cur_g_e + 1] <= work_u_e:
                    cur_g_e = cur_g_e + 1
                rel_e = work_u_e - s_cum[cur_g_e]
                coord_m_e = rel_e // cn
                coord_n_e = rel_e % cn
                alpha_val = cutlass.Float32(mAlpha[cur_g_e])
                mg_cur = s_mg[cur_g_e]
                cta_row_base = (
                    (coord_m_e * 2 + cta_rank) * 128
                    if cutlass.const_expr(self.use_2cta)
                    else coord_m_e * 128
                )

                bSG_gC = bSG_gC_part[(None, None, None, coord_m_e, coord_n_e, cur_g_e)]
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
                num_prev_subtiles = tiles_done * subtile_cnt

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

                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load() * alpha_val
                    tRS_rC.store(acc_vec.to(self.c_dtype))
                    c_buf = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buf)])
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id, number_of_threads=128
                    )
                    if warp_idx == 0:
                        if cta_row_base < mg_cur:
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

                tiles_done = tiles_done + 1
                work_u_e = work_u_e + active_units

            tmem.relinquish_alloc_permit()
            c_pipeline.producer_tail()
            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=128)
            tmem.free(acc_tmem_ptr)


def _get_masked_group_persistent_runner(
    use_2cta: bool,
    tile_n: int,
    shape_cfg: Tuple[int, int, int, int, int],
    c_dtype_str: str = "BFloat16",
    pdl_safe: bool = True,
):
    key = ("pers_masked", use_2cta, tile_n, shape_cfg, c_dtype_str, pdl_safe)
    runner = _MASKED_GROUP_CACHE.get(key)
    if runner is not None:
        return runner

    c_dtype_cutlass = getattr(cutlass, c_dtype_str)
    kern = Sm100Nvfp4MaskedGroupGemmPersistent(
        use_2cta=use_2cta,
        tile_n=tile_n,
        c_dtype=c_dtype_cutlass,
        pdl_safe=pdl_safe,
    )
    fn = cute.compile(
        kern,
        make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=32),
        make_ptr(c_dtype_cutlass, 0, cute.AddressSpace.gmem, assumed_align=16),
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
        b_p: int,
        sfa_p: int,
        sfb_p: int,
        c_p: int,
        al_p: int,
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
                    cutlass.Float4E2M1FN, b_p, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Float8E4M3FN,
                    sfa_p,
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                ),
                make_ptr(
                    cutlass.Float8E4M3FN,
                    sfb_p,
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                ),
                make_ptr(
                    c_dtype_cutlass, c_p, cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    cutlass.Float32, al_p, cute.AddressSpace.gmem, assumed_align=4
                ),
                make_ptr(cutlass.Int32, mm_p, cute.AddressSpace.gmem, assumed_align=4),
                cuda.CUstream(st_i),
            )
            ex = fn._default_executor
            packed_args = ex._tls.packed_args
            ptr_buf = (ctypes.c_uint64 * 8)()
            base = ctypes.addressof(ptr_buf)
            for i in range(8):
                packed_args[i] = base + i * 8
            capi_func = ex.capi_func
            _KEEPALIVE.append((kern, fn, ex, packed_args, ptr_buf))
        ptr_buf[0] = a_p
        ptr_buf[1] = b_p
        ptr_buf[2] = sfa_p
        ptr_buf[3] = sfb_p
        ptr_buf[4] = c_p
        ptr_buf[5] = al_p
        ptr_buf[6] = mm_p
        ptr_buf[7] = st_i
        capi_func(packed_args)

    _MASKED_GROUP_CACHE[key] = runner
    return runner


def nvfp4_masked_grouped_gemm(
    a_q: torch.Tensor,
    a_q_sf: torch.Tensor,
    w: torch.Tensor,
    w_sf: torch.Tensor,
    masked_m: torch.Tensor,
    alpha: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    pdl_safe: bool = True,
) -> torch.Tensor:
    """Fast SM100 3D masked NVFP4 grouped GEMM with persistent dynamic masked_m prefix scheduling.

    Supports ``a_q`` in either ``(m_max, k // 2, G)`` (FlashInfer layout) or
    ``(G, m_max, k // 2)`` (expert-major layout), and ``w`` in either
    ``(n, k // 2, G)`` or ``(G, n, k // 2)``.
    """
    if out_dtype is None:
        out_dtype = out.dtype if out is not None else torch.bfloat16
    a_is_mkg = a_q.stride(1) == 1
    if a_is_mkg:
        m_max, k_half, G = a_q.shape
        step_a = a_q.stride(2) * a_q.element_size()
        a_row_ok = a_q.stride(0) == k_half
    else:
        G, m_max, k_half = a_q.shape
        step_a = a_q.stride(0) * a_q.element_size()
        a_row_ok = a_q.stride(1) == k_half and a_q.stride(2) == 1

    if w.stride(1) == 1:
        n, _, _ = w.shape
        step_b = w.stride(2) * w.element_size()
        w_row_ok = w.stride(0) == k_half
    else:
        _, n, _ = w.shape
        step_b = w.stride(0) * w.element_size()
        w_row_ok = w.stride(1) == k_half and w.stride(2) == 1

    k = k_half * 2
    if G <= 0 or m_max < 0 or n <= 0 or k <= 0 or (n % 64 != 0) or (k % 256 != 0):
        raise ValueError(
            f"Unsupported nvfp4_masked_grouped_gemm shape (G={G}, m_max={m_max}, N={n}, K={k}); "
            "requires G > 0, m_max >= 0, N % 64 == 0, K % 256 == 0."
        )

    if out is None:
        out = torch.empty((G, m_max, n), dtype=out_dtype, device=a_q.device)
        if a_is_mkg:
            out = out.permute(1, 2, 0)
    if m_max == 0:
        return out

    step_sfa = (
        a_q_sf.stride(5) * a_q_sf.element_size()
        if a_q_sf.ndim == 6
        else a_q_sf.stride(0) * a_q_sf.element_size()
    )
    step_sfb = (
        w_sf.stride(5) * w_sf.element_size()
        if w_sf.ndim == 6
        else w_sf.stride(0) * w_sf.element_size()
    )
    if out.stride(1) == 1:
        step_c = out.stride(2) * out.element_size()
        out_row_ok = out.stride(0) == n
    else:
        step_c = out.stride(0) * out.element_size()
        out_row_ok = out.stride(1) == n and out.stride(2) == 1

    ba = a_q.data_ptr()
    bb = w.data_ptr()
    bsfa = a_q_sf.data_ptr()
    bsfb = w_sf.data_ptr()
    bc = out.data_ptr()
    balpha = alpha.data_ptr()
    c_dtype_str = "BFloat16" if out.dtype == torch.bfloat16 else "Float16"
    stream_i = torch.cuda.current_stream(a_q.device).cuda_stream

    rk = _ceil_div(k // 16, 4)
    if a_q_sf.ndim == 6:
        sfa_m = a_q_sf.shape[2] * 128
    elif G > 1 and step_sfa > 0:
        sfa_m = step_sfa // (rk * 4)
    else:
        sfa_m = a_q_sf.shape[1]

    can_use_persistent_3d = (
        a_row_ok
        and w_row_ok
        and out_row_ok
        and (G == 1 or step_a == m_max * k_half)
        and (G == 1 or step_b == n * k_half)
        and (G == 1 or step_c == m_max * n * out.element_size())
        and (sfa_m >= _ceil_div(m_max, 128) * 128)
        and (sfa_m % 128 == 0)
        and (G == 1 or step_sfa == (sfa_m // 128) * rk * 512)
        and (G == 1 or step_sfb == _ceil_div(n, 128) * rk * 512)
        and alpha.is_contiguous()
        and alpha.numel() >= G
    )
    if can_use_persistent_3d:
        use_2cta = (
            (m_max >= 512)
            and (m_max % 256 == 0)
            and (sfa_m % 256 == 0)
            and (n % 256 == 0)
        )
        tile_n_p = 256 if use_2cta else (128 if n % 128 == 0 else 64)
        p_runner = _get_masked_group_persistent_runner(
            use_2cta=use_2cta,
            tile_n=tile_n_p,
            shape_cfg=(G, m_max, sfa_m, n, k),
            c_dtype_str=c_dtype_str,
            pdl_safe=pdl_safe,
        )
        p_runner(ba, bb, bsfa, bsfb, bc, balpha, masked_m.data_ptr(), stream_i)
        return out

    ptr_list = []
    for g in range(G):
        ptr_list.extend(
            [
                ba + g * step_a,
                bb + g * step_b,
                bsfa + g * step_sfa,
                bsfb + g * step_sfb,
                bc + g * step_c,
                balpha + g * 4,
            ]
        )

    tile_n = 128 if n % 128 == 0 else 64
    runner = _build_masked_group_runner_1cta(
        G,
        m_max,
        n,
        k,
        tile_n=tile_n,
        num_ab_stage=6,
        pf_dist=6,
        order_mn=True,
        c_dtype_str=c_dtype_str,
        pdl_safe=pdl_safe,
    )
    runner(ptr_list, masked_m.data_ptr(), stream_i)
    return out


__all__ = [
    "Sm100Nvfp4MaskedGroupGemmPersistent",
    "nvfp4_grouped_gemm",
    "nvfp4_masked_grouped_gemm",
    "permute_nvfp4_scale",
    "_ceil_div",
    "_build_masked_group_runner_1cta",
    "_get_masked_group_persistent_runner",
]
