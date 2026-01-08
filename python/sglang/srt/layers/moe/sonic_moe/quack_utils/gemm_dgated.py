# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import operator
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import quack.activation
import quack.sm90_utils as sm90_utils
import quack.utils as utils
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import (
    ArgumentsBase,
    ParamsBase,
    get_device_capacity,
    get_max_active_clusters,
    torch2cute_dtype_map,
)
from quack.gemm_act import GemmActMixin
from quack.gemm_default_epi import GemmDefaultEpiMixin
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_wrapper_utils import GemmWrapperBase
from quack.reduce import warp_reduce
from quack.sm90_utils import partition_for_epilogue
from quack.varlen_utils import VarlenManager
from torch import Tensor


class GemmDGatedMixin(GemmActMixin):
    # Different from GemmActMixin, here act_bwd_fn must take in 3 arguments (x, y, dout)
    # and return 3 arguments (dx, dy, out)
    @dataclass
    class EpilogueArguments(ArgumentsBase):
        mPostAct: cute.Tensor
        act_bwd_fn: cutlass.Constexpr[Callable]
        implicit_dtype: Type[cutlass.Numeric] = cute.BFloat16
        # We don't use alpha, beta, mRowVecBroadcast for now
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None

    @dataclass
    class EpilogueParams(ParamsBase):
        tma_atom_postact: cute.CopyAtom
        mPostAct_mnl: cute.Tensor
        epi_postact_smem_layout_staged: cute.ComposedLayout
        epi_tile_postact: cute.Tile
        act_bwd_fn: cutlass.Constexpr[Callable]
        implicit_dtype: Type[cutlass.Numeric]
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)
        # C and D are implicitly 2 16-bit elements packed into 32 bits, simply for the purpose
        # for reusing the existing load/store code.
        assert args.implicit_dtype.width == 16, "GemmDGated only supports 16bit for now"
        assert self.d_dtype.width == 32, "D storage type must be 32 bit"
        assert self.c_dtype.width == 32, "C storage type must be 32 bit"

        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        epi_tile_postact = self.epi_tile
        utils_cls = sm100_utils if self.arch == 100 else sm90_utils
        epi_postact_smem_layout_staged = utils_cls.make_smem_layout_epi(
            self.postact_dtype, self.postact_layout, epi_tile_postact, self.epi_stage
        )
        tma_atom_postact, tma_tensor_postact = self._make_tma_epi_atoms_and_tensors(
            args.mPostAct,
            epi_postact_smem_layout_staged,
            epi_tile_postact,
            op_type="store",
        )
        # Assume all strides are divisible by 32 bits except the last stride
        new_stride = lambda t: tuple(
            (
                cute.assume(s, divby=32 // t.element_type.width)
                if not cute.is_static(s)
                else s
            )
            for s in t.stride
        )
        mRowVecBroadcast, mColVecBroadcast, mColVecReduce = [
            (
                cute.make_tensor(
                    t.iterator, cute.make_layout(t.shape, stride=new_stride(t))
                )
                if t is not None
                else None
            )
            for t in (args.mRowVecBroadcast, args.mColVecBroadcast, args.mColVecReduce)
        ]
        return self.EpilogueParams(
            tma_atom_postact,
            tma_tensor_postact,
            epi_postact_smem_layout_staged,
            epi_tile_postact,
            args.act_bwd_fn,
            args.implicit_dtype,
            alpha=args.alpha,
            beta=args.beta,
            mRowVecBroadcast=mRowVecBroadcast,
            mColVecBroadcast=mColVecBroadcast,
            mColVecReduce=mColVecReduce,
        )

    @cute.jit
    def epi_begin(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tidx: Int32,
    ) -> Tuple[cute.Tensor, ...]:
        epi_tensors = GemmDefaultEpiMixin.epi_begin(
            self,
            params,
            epi_smem_tensors,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tile_coord_mnkl,
            varlen_manager,
            epilogue_barrier,
            tidx,
        )
        partition_for_epilogue_fn = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s,
            tidx=tidx,
            reference_src=tiled_copy_t2r is None,
        )
        tDrColVecReduce = None
        if const_expr(params.mColVecReduce is not None):
            colvec_mma_layout = cute.make_layout(
                self.cta_tile_shape_mnk[:2], stride=(1, 0)
            )
            tDrColVec_layout = partition_for_epilogue_fn(
                cute.make_fragment(colvec_mma_layout, Float32)
            ).layout
            tDrColVecReduce = cute.make_fragment(tDrColVec_layout, Float32)
            cute.filter_zeros(tDrColVecReduce).fill(0.0)
        return (*epi_tensors, tDrColVecReduce)

    def epi_begin_loop(
        self, params: EpilogueParams, epi_tensors, epi_coord: cute.Coord
    ):
        epi_tensors, tDrColVecReduce = epi_tensors[:-1], epi_tensors[-1]
        epi_loop_tensors = super().epi_begin_loop(params, epi_tensors, epi_coord)
        tDrColVecReduce_cur = None
        if const_expr(tDrColVecReduce is not None):
            tDrColVecReduce_cur = cute.group_modes(
                tDrColVecReduce, 3, cute.rank(tDrColVecReduce)
            )[None, None, None, epi_coord]
        return (*epi_loop_tensors, tDrColVecReduce_cur)

    @cute.jit
    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        alpha, beta, tDrRowVec, tDrColVec, tDrColVecReduce = epi_loop_tensors
        assert (
            alpha is None and beta is None and tDrRowVec is None
        )  # We don't use these for now
        assert tRS_rC is not None
        implicit_dtype = params.implicit_dtype
        assert implicit_dtype.width == 16, "GemmDGatedMixin only supports 16bit for now"
        tRS_rXY_f16x2 = cute.recast_tensor(tRS_rC, implicit_dtype)
        tRS_rXY_f32x2 = cute.make_fragment(tRS_rXY_f16x2.layout, Float32)
        tRS_rXY_f32x2.store(tRS_rXY_f16x2.load().to(Float32))
        tRS_rdXY_f32x2 = cute.make_fragment_like(tRS_rXY_f32x2, Float32)
        tRS_rOut = cute.make_fragment_like(tRS_rD, Float32)
        tRS_rD_scaled = cute.make_fragment_like(tRS_rD)
        if const_expr(tDrColVec is not None):  # Scale D by colvec
            if const_expr(self.arch < 100):
                tRS_rD_scaled.store(
                    tRS_rD.load() * tDrColVec.load().to(tRS_rD.element_type)
                )
            else:
                tDrColVec_mn = utils.convert_layout_zero_stride(
                    tDrColVec, tDrColVec.layout
                )
                tRS_rD_mn = utils.convert_layout_zero_stride(tRS_rD, tDrColVec.layout)
                tRS_rD_scaled_mn = utils.convert_layout_zero_stride(
                    tRS_rD_scaled, tDrColVec.layout
                )
                for m in cutlass.range(
                    cute.size(tDrColVec_mn, mode=[0]), unroll_full=True
                ):
                    for n in cutlass.range(
                        cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True
                    ):
                        (
                            tRS_rD_scaled_mn[m, 2 * n],
                            tRS_rD_scaled_mn[m, 2 * n + 1],
                        ) = utils.mul_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                        )
        else:
            tRS_rD_scaled.store(tRS_rD.load())
        if const_expr(self.arch < 100):
            for i in cutlass.range(cute.size(tRS_rD)):
                (
                    tRS_rdXY_f32x2[2 * i],
                    tRS_rdXY_f32x2[2 * i + 1],
                    tRS_rOut[i],
                ) = params.act_bwd_fn(
                    tRS_rXY_f32x2[2 * i], tRS_rXY_f32x2[2 * i + 1], tRS_rD_scaled[i]
                )
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2):
                (
                    (tRS_rdXY_f32x2[4 * i], tRS_rdXY_f32x2[4 * i + 2]),
                    (tRS_rdXY_f32x2[4 * i + 1], tRS_rdXY_f32x2[4 * i + 3]),
                    (tRS_rOut[2 * i], tRS_rOut[2 * i + 1]),
                ) = params.act_bwd_fn(
                    (tRS_rXY_f32x2[4 * i], tRS_rXY_f32x2[4 * i + 2]),
                    (tRS_rXY_f32x2[4 * i + 1], tRS_rXY_f32x2[4 * i + 3]),
                    (tRS_rD_scaled[2 * i], tRS_rD_scaled[2 * i + 1]),
                )
        if const_expr(tDrColVecReduce is not None):
            # Need to multiply before D is scaled by colvec_scale
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tDrColVecReduce), unroll_full=True):
                    tDrColVecReduce[i] += tRS_rOut[i] * tRS_rD[i]
            else:
                tDrColVecReduce_mn = utils.convert_layout_zero_stride(
                    tDrColVecReduce, tDrColVecReduce.layout
                )
                tRS_rD_mn = utils.convert_layout_zero_stride(
                    tRS_rD, tDrColVecReduce.layout
                )
                tRS_rOut_mn = utils.convert_layout_zero_stride(
                    tRS_rOut, tDrColVecReduce.layout
                )
                for m in cutlass.range(
                    cute.size(tDrColVecReduce_mn, mode=[0]), unroll_full=True
                ):
                    row_sum = utils.mul_packed_f32x2(
                        (tRS_rD_mn[m, 0], tRS_rD_mn[m, 1]),
                        (tRS_rOut_mn[m, 0], tRS_rOut_mn[m, 1]),
                    )
                    for n in cutlass.range(
                        1,
                        cute.size(tDrColVecReduce_mn, mode=[1]) // 2,
                        unroll_full=True,
                    ):
                        row_sum = utils.fma_packed_f32x2(
                            (tRS_rD_mn[m, 2 * n], tRS_rD_mn[m, 2 * n + 1]),
                            (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                            row_sum,
                        )
                    tDrColVecReduce_mn[m, 0] += row_sum[0] + row_sum[1]

        if const_expr(tDrColVec is not None):  # Scale Out by colvec
            if const_expr(self.arch < 100):
                tRS_rOut.store(
                    tRS_rOut.load() * tDrColVec.load().to(tRS_rD.element_type)
                )
            else:
                tDrColVec_mn = utils.convert_layout_zero_stride(
                    tDrColVec, tDrColVec.layout
                )
                tRS_rOut_mn = utils.convert_layout_zero_stride(
                    tRS_rOut, tDrColVec.layout
                )
                for m in cutlass.range(
                    cute.size(tDrColVec_mn, mode=[0]), unroll_full=True
                ):
                    for n in cutlass.range(
                        cute.size(tDrColVec_mn, mode=[1]) // 2, unroll_full=True
                    ):
                        tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1] = (
                            utils.mul_packed_f32x2(
                                (tRS_rOut_mn[m, 2 * n], tRS_rOut_mn[m, 2 * n + 1]),
                                (tDrColVec_mn[m, 0], tDrColVec_mn[m, 0]),
                            )
                        )
        # Type conversion
        tRS_rdXY_f16x2 = cute.make_fragment(tRS_rdXY_f32x2.layout, implicit_dtype)
        tRS_rdXY_f16x2.store(tRS_rdXY_f32x2.load().to(implicit_dtype))
        tRS_rD.store(cute.recast_tensor(tRS_rdXY_f16x2, Float32).load())
        tRS_rOut_cvt = cute.make_fragment_like(tRS_rOut, self.postact_dtype)
        tRS_rOut_cvt.store(tRS_rOut.load().to(self.postact_dtype))
        return tRS_rOut_cvt

    @cute.jit
    def epi_end(
        self,
        params: EpilogueParams,
        epi_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        tidx: Int32,
    ) -> None:
        partition_for_epilogue_fn = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s,
            tidx=tidx,
            reference_src=tiled_copy_t2r is None,
        )
        tDrColVecReduce = epi_tensors[-1]
        tile_M, tile_N = self.cta_tile_shape_mnk[:2]
        if const_expr(params.mColVecReduce is not None):
            tDrCVR_flt = cute.filter_zeros(tDrColVecReduce)
            if const_expr(self.arch != 100):
                for i in cutlass.range(cute.size(tDrCVR_flt), unroll_full=True):
                    tDrCVR_flt[i] = warp_reduce(tDrCVR_flt[i], operator.add, width=4)
            else:
                # Don't need warp_reduce since we load from tmem with one thread per row
                assert (
                    self.d_layout.is_n_major_c()
                ), "GemmDGated only supports n-major output for now"
            batch_idx = tile_coord_mnkl[3]
            limit_n = (
                params.mColVecReduce.shape[2]
                if not varlen_manager.varlen_m
                else params.mColVecReduce.shape[1]
            )
            if tile_coord_mnkl[1] < limit_n:
                if const_expr(not varlen_manager.varlen_m):
                    mColVec = params.mColVecReduce[batch_idx, None, tile_coord_mnkl[1]]
                else:
                    mColVec = cute.domain_offset(
                        (varlen_manager.params.cu_seqlens_m[batch_idx],),
                        params.mColVecReduce[None, tile_coord_mnkl[1]],
                    )
                gColVec = cute.local_tile(mColVec, (tile_M,), (tile_coord_mnkl[0],))
                limit_m = min(
                    varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * tile_M,
                    tile_M,
                )
                tDcCV = partition_for_epilogue_fn(
                    cute.make_identity_tensor((tile_M, tile_N))
                )
                tDrColVecReduce_m = utils.convert_layout_zero_stride(
                    tDrColVecReduce, tDrColVecReduce.layout
                )[None, 0]
                tDcCV_m = utils.convert_layout_zero_stride(
                    tDcCV, tDrColVecReduce.layout
                )[None, 0]
                if tDcCV_m[0][1] == 0:
                    for m in cutlass.range(cute.size(tDcCV_m, mode=[0])):
                        row_idx = tDcCV_m[m][0]
                        if row_idx < limit_m:
                            gColVec[row_idx] = tDrColVecReduce_m[m]


class GemmDGatedSm90(GemmDGatedMixin, GemmSm90):
    pass


class GemmDGatedSm100(GemmDGatedMixin, GemmSm100):
    pass


dgate_fn_map = {
    "swiglu": quack.activation.dswiglu,
    "swiglu_oai": quack.activation.dswiglu_oai,
    "reglu": quack.activation.dreglu,
    "geglu": quack.activation.dgeglu,
    "glu": quack.activation.dglu,
}


def gemm_dgated(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    Out: Tensor,  # (l, m, 2*n) if n_major or (l, 2*m, n) if m_major, or (total_m, 2*n) if varlen_m
    PreAct: Tensor,  # (l, m, 2*n) if n_major or (l, 2*m, n) if m_major, or (total_m, 2*n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = True,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    colvec_scale: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    # (l, m, ceildiv(n, tile_n)), or (total_m, ceildiv(n, tile_n)) if varlen_m
    colvec_reduce: Optional[Tensor] = None,
    cu_seqlens_m: Optional[
        Tensor
    ] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
) -> None:
    """If tile_count_semaphore is provided, it must already be zero'ed out."""
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert Out.stride(-1) == 1, "varlen_m requires Out to be n-major"
        assert PreAct.stride(-1) == 1, "varlen_m requires PreAct to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    gather_A = A_idx is not None
    if gather_A:
        assert (
            cu_seqlens_m is not None
        ), "gather_A requires varlen (cu_seqlens_m must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    assert activation in dgate_fn_map, f"Unsupported activation {activation}"

    # Special handling for Out and PreAct
    AB_swapped = not Out.stride(-1) == 1
    assert Out.dtype == PreAct.dtype
    implicit_dtype = torch2cute_dtype_map[Out.dtype]
    assert Out.element_size() == 2, "Out dtype must be fp16 or bf16"
    assert PreAct.element_size() == 2, "Preact dtype must be fp16 or bf16"
    # We pretend that Out is (M, N, L) of type fp32 instead of (M, 2N, L) of type f16.
    # Similarly we pretend that PreAct is (M, N, L) of type fp32 instead of (M, 2N, L) of type f16
    if cu_seqlens_m is not None or not AB_swapped:
        # varlen_m (always AB_swapped=False) or normal case with AB_swapped=False
        Out = Out.view(torch.float32)
        PreAct = PreAct.view(torch.float32)
    else:
        # Normal case with AB_swapped=True
        Out = Out.mT.view(torch.float32).mT
        PreAct = PreAct.mT.view(torch.float32).mT

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A,
        B,
        Out,
        PreAct,
        additional_tensors={"PostAct": PostAct},
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
    )
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=cu_seqlens_m is not None)
    GemmWrapperBase.extract_dtypes(tensor_infos)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
        "PostAct": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10], "Only SM90 and SM100 are supported"
    GemmCls = GemmDGatedSm100 if device_capacity[0] > 9 else GemmDGatedSm90

    acc_dtype = Float32
    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    if not GemmCls.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        acc_dtype,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Skipping due to unsupported combination of types and majors")

    max_active_clusters = (
        get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    )
    GemmWrapperBase.create_cute_tensors(tensor_infos, major_configs)
    act_fn = dgate_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(
        tensor_infos["PostAct"].cute_tensor,
        act_fn,
        implicit_dtype=implicit_dtype,
        mColVecBroadcast=(
            from_dlpack(colvec_scale.detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=1 if cu_seqlens_m is None else 0
            )
            if colvec_scale is not None
            else None
        ),
        mColVecReduce=(
            from_dlpack(colvec_reduce.detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=2 if cu_seqlens_m is None else 1
            )
            if colvec_reduce is not None
            else None
        ),
    )
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters, tile_count_semaphore
    )

    # Create varlen arguments if needed (assumes persistent=True when varlen_m)
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        None,  # cu_seqlens_k
        A_idx,
        max_active_clusters,
        cluster_shape_mnk,
        tensor_infos,
        GemmCls.num_epi_tensormaps,
        pingpong,
    )

    current_stream = cutlass_torch.current_stream()
    compile_key = GemmWrapperBase.get_compile_key(
        tensor_infos,
        activation,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        tile_count_semaphore is not None,
        device_capacity,
        max_swizzle_size,
        colvec_scale.dtype if colvec_scale is not None else None,
        colvec_reduce.dtype if colvec_reduce is not None else None,
        cu_seqlens_m is not None,
        A_idx is not None,
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_dgated.compile_cache
    if compile_key not in cache:
        if device_capacity[0] == 9:
            GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm_obj = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=gather_A,
        )
        cache[compile_key] = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,  # Out
            tensor_infos["C"].cute_tensor,  # PreAct
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
        )
    cache[compile_key](
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,  # Out
        tensor_infos["C"].cute_tensor,  # PreAct
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
    )


gemm_dgated.compile_cache = {}
