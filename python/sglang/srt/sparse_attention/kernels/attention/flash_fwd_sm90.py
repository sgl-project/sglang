# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of
# https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm80.h
# and https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm90.h
# from Cutlass C++ to Cute-DSL.
# Built on Cute-DSL example: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

import math
from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup

from sglang.srt.sparse_attention.kernels.attention import hopper_helpers as sm90_utils
from sglang.srt.sparse_attention.kernels.attention import pipeline, utils
from sglang.srt.sparse_attention.kernels.attention.block_info import BlockInfo
from sglang.srt.sparse_attention.kernels.attention.flash_fwd import (
    FlashAttentionForwardBase,
)
from sglang.srt.sparse_attention.kernels.attention.mask import AttentionMask
from sglang.srt.sparse_attention.kernels.attention.named_barrier import NamedBarrierFwd
from sglang.srt.sparse_attention.kernels.attention.pack_gqa import PackGQA
from sglang.srt.sparse_attention.kernels.attention.seqlen_info import SeqlenInfoQK
from sglang.srt.sparse_attention.kernels.attention.softmax import Softmax
from sglang.srt.sparse_attention.kernels.attention.tile_scheduler import (
    ParamsBase,
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FlashAttentionForwardSm90(FlashAttentionForwardBase):

    arch = 90

    def __init__(self, *args, intra_wg_overlap: bool = True, **kwargs):
        self.groupwise = kwargs.pop("groupwise", False)
        super().__init__(*args, **kwargs)
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = True

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim_padded
            ),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                cutlass.utils.LayoutEnum.ROW_MAJOR, self.dtype, self.head_dim_v_padded
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    cutlass.utils.LayoutEnum.ROW_MAJOR, self.dtype, self.n_block_size
                ),
                self.dtype,
            )
        else:
            sP_layout_atom = None
        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(
                self.m_block_size // 64,
                1,
                1,
            ),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.n_block_size),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(
                self.m_block_size // 64,
                1,
                1,
            ),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.head_dim_v_padded),
            a_source=(
                warpgroup.OperandSource.RMEM
                if self.mma_pv_is_rs
                else warpgroup.OperandSource.SMEM
            ),
        )
        tiled_mma_pv_rs = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(
                self.m_block_size // 64,
                1,
                1,
            ),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.head_dim_v_padded),
            a_source=warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_rs

    def _get_shared_storage_cls(self):
        # If we use cp.async to load Q, we want sQ to align to 1024 bytes
        sQ_alignment = 128 if const_expr(self.use_tma_Q) else 1024
        sK_alignment = 128
        sV_alignment = 128
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], alignment
            ]
            for layout, alignment in zip(
                (self.sQ_layout, self.sK_layout, self.sV_layout),
                (sQ_alignment, sK_alignment, sV_alignment),
            )
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQV], 1024
        ]
        cosize_sP = (
            cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        )
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        # 1 for Q, 1 for O, self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_QO_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr: mbar_ptr_QO_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr: mbar_ptr_QO_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        return (
            SharedStorageQKV
            if const_expr(not self.Q_in_regs)
            else SharedStorageSharedQV
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                )
            )
        )
        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        mQ, mK, mV, mO = [
            cute.make_tensor(
                t.iterator, cute.make_layout(t.shape, stride=new_stride(t))
            )
            for t in (mQ, mK, mV, mO)
        ]
        QO_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )  # T, H, D -> T, D, H
        mQ, mO = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=QO_layout_transpose)
            )
            for t in (mQ, mO)
        ]
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]  # PAGE_NUM, PAGE_SIZE, H, D -> PAGE_SIZE, D, H, PAGE_NUM
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = (
            cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
            if const_expr(mLSE is not None)
            else None
        )
        tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_rs = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_mma_warp_groups = (
            self.num_mma_threads // self.num_threads_per_warp_group
        )
        self.num_producer_threads = 32
        self.num_Q_load_threads = (
            self.num_mma_threads
        )  # If not TMA_Q, MMA threads load Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs = 240
        self.num_producer_regs = 24
        # self.num_mma_regs = 232
        # self.num_producer_regs = 40
        self.use_scheduler_barrier = (
            (self.num_mma_warp_groups >= 2 and self.head_dim_padded <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_mma_warp_groups == 2)
        )
        self.use_tma_Q = self.arch >= 90 and not (
            self.pack_gqa and self.m_block_size % self.qhead_per_kvhead != 0
        )
        self.use_tma_O = (
            self.arch >= 90
            and mCuSeqlensQ is None
            and mSeqUsedQ is None
            and not self.pack_gqa
        )
        # TODO: rescale_O_before_gemm
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        if const_expr(self.pack_gqa):
            shape_Q_packed = (
                (self.qhead_per_kvhead, mQ.shape[0]),
                mQ.shape[1],
                mK.shape[2],
                *mQ.shape[3:],
            )
            stride_Q_packed = (
                (mQ.stride[2], mQ.stride[0]),
                mQ.stride[1],
                mQ.stride[2] * self.qhead_per_kvhead,
                *mQ.stride[3:],
            )
            mQ = cute.make_tensor(
                mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed)
            )
            shape_O_packed = (
                (self.qhead_per_kvhead, mO.shape[0]),
                mK.shape[1],
                mK.shape[2],
                *mO.shape[3:],
            )
            stride_O_packed = (
                (mO.stride[2], mO.stride[0]),
                mO.stride[1],
                mO.stride[2] * self.qhead_per_kvhead,
                *mO.stride[3:],
            )
            mO = cute.make_tensor(
                mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed)
            )
            if const_expr(mLSE is not None):
                shape_LSE_packed = (
                    (self.qhead_per_kvhead, mLSE.shape[0]),
                    mK.shape[2],
                    *mLSE.shape[2:],
                )
                stride_LSE_packed = (
                    (mLSE.stride[1], mLSE.stride[0]),
                    mLSE.stride[1] * self.qhead_per_kvhead,
                    *mLSE.stride[2:],
                )
                mLSE = cute.make_tensor(
                    mLSE.iterator,
                    cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed),
                )

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_q_bytes = cute.size_in_bytes(
            mQ.element_type, cute.select(self.sQ_layout, mode=[0, 1])
        )
        self.tma_copy_k_bytes = cute.size_in_bytes(
            mK.element_type, cute.select(self.sK_layout, mode=[0, 1])
        )
        self.tma_copy_v_bytes = cute.size_in_bytes(
            mV.element_type, cute.select(self.sV_layout, mode=[0, 1])
        )
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_Q,
                mQ,
                self.sQ_layout,
                (self.m_block_size, self.head_dim_padded),  # No mcast
            )
        else:
            tma_atom_Q, tma_tensor_Q = None, None
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_padded),
            1,  # No mcast for now
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.n_block_size, self.head_dim_v_padded),
            1,  # No mcast for now
        )
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_O,
                mO,
                self.sO_layout,
                (self.m_block_size, self.head_dim_v_padded),  # No mcast
            )
        else:
            tma_atom_O = None
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.m_block_size),
            cute.size(mQ.shape[2]),
            (
                cute.size(mQ.shape[3])
                if const_expr(mCuSeqlensQ is None)
                else cute.size(mCuSeqlensQ.shape[0] - 1)
            ),
            (
                cute.size(mK.shape[0])
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mQ.shape[1],
            mV.shape[1],
            total_q=(
                cute.size(mQ.shape[0])
                if const_expr(mCuSeqlensQ is not None)
                else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3])
            ),
            tile_shape_mn=(self.m_block_size, self.n_block_size),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        # If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        # Right after this, we multiply by log2(e) before applying exp2.
        # To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        # (assigning it to softcap_val) and pre-multiply softcap_val * log2(e)
        # (assigning it to softmax_scale_log2).
        LOG2_E = math.log2(math.e)
        if const_expr(softcap is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softcap_val = None
        else:
            softmax_scale_log2 = softcap * LOG2_E
            softcap_val = Float32(softmax_scale / softcap)
        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)
        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K,
            tma_tensor_V,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softcap_val,
            window_size_left,
            window_size_right,
            learnable_sink,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_pv_rs,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            self.groupwise,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softcap_val: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        groupwise: bool,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            if const_expr(tma_atom_Q is not None):
                cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(self.use_tma_O):
                cpasync.prefetch_descriptor(tma_atom_O)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier init
        mbar_ptr_Q = storage.mbar_ptr.data_ptr()
        if warp_idx == 1:
            # if tidx < 2:
            #     # barrierO num threads should be self.num_mma_threads
            #     cute.arch.mbarrier_init(mbar_ptr_Q + tidx, 1 if tidx == 0 else self.num_mma_threads)
            cute.arch.mbarrier_init(
                mbar_ptr_Q, 1 if const_expr(self.use_tma_Q) else self.num_Q_load_threads
            )
            # cute.arch.mbarrier_init(mbar_ptr_Q + 1, self.num_mma_threads)
        # We rely on pipeline_k and pipeline_v to initialize the mbarrier fence and sync
        pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            self.num_mma_threads // self.num_threads_per_warp_group,
        )
        pipeline_k = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_k_bytes,
            init_wait=False,
        )
        pipeline_v = pipeline.PipelineTmaAsyncNoCluster.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_v_bytes,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        # TODO: how to get sQ_pi for cp.async if pack_gqa?
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        # Transpose view of V to tensor with layout (head_dim_v, n_block_size) for tiled mma
        sVt = utils.transpose_view(sV)
        if const_expr(sP_layout is not None):
            sP_pi = storage.sP.get_tensor(sP_layout)
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        else:
            sP, sP_pi = None, None
        # reuse sQ's data iterator
        sO_pi = storage.sQ.get_tensor(sO_layout)
        # TODO: idk why not using sO_pi is faster
        sO = cute.make_tensor(
            cute.recast_ptr(sO_pi.iterator, sO_layout.inner, dtype=sO_pi.element_type),
            sO_layout.outer,
        )

        block_info = BlockInfo(
            self.m_block_size,
            self.n_block_size,
            self.is_causal,
            self.is_local,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK,
            seqlen_q_static=(
                mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=(
                mK.shape[0]
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:  # Producer
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                groupwise,
            )

        else:  # Consumer
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                tiled_mma_pv_rs,
                mQ,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                gmem_tiled_copy_Q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softcap_val,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        groupwise: bool,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mQ_cur = mQ[None, None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q
                        if const_expr(not self.pack_gqa)
                        else (0, seqlen.offset_q)
                    )
                    mQ_cur = cute.domain_offset((offset, 0), mQ[None, None, head_idx])
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead
                    if const_expr(not self.pack_gqa)
                    else head_idx
                )
                if const_expr(mPageTable is None):
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        mK_cur, mV_cur = [
                            t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)
                        ]
                    else:
                        mK_cur, mV_cur = [
                            cute.domain_offset(
                                (seqlen.offset_k, 0), t[None, None, head_idx_kv]
                            )
                            for t in (mK, mV)
                        ]
                    gK = cute.local_tile(
                        mK_cur, (self.n_block_size, self.head_dim_padded), (None, 0)
                    )
                    gV = cute.local_tile(
                        mV_cur, (self.n_block_size, self.head_dim_v_padded), (None, 0)
                    )
                else:
                    mK_cur, mV_cur = [
                        t[None, None, head_idx_kv, None] for t in (mK, mV)
                    ]
                    gK = cute.local_tile(
                        mK_cur,
                        (self.n_block_size, self.head_dim_padded),
                        (None, 0, None),
                    )
                    gV = cute.local_tile(
                        mV_cur,
                        (self.n_block_size, self.head_dim_v_padded),
                        (0, None, None),
                    )
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(
                        mQ_cur, (self.m_block_size, self.head_dim_padded), (m_block, 0)
                    )
                    tQsQ, tQgQ = cpasync.tma_partition(
                        tma_atom_Q,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sQ, 0, 2),
                        cute.group_modes(gQ, 0, 2),
                    )
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 2),
                    cute.group_modes(gK, 0, 2),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 2),
                    cute.group_modes(gV, 0, 2),
                )
                load_K = partial(self.load_K, tma_atom_K, tKgK, tKsK, pipeline_k)
                load_V = partial(self.load_K, tma_atom_V, tVgV, tVsV, pipeline_v)
                # load_Q
                if const_expr(self.use_tma_Q):
                    # TODO: wait for Q to be empty
                    q_producer_phase ^= 1
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr_Q, self.tma_copy_q_bytes
                        )
                    cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=mbar_ptr_Q)
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block
                )
                # if cute.arch.thread_idx()[0] == 0:
                #     cute.printf("m_block = %d, n_block_min: %d, n_block_max: %d", m_block, n_block_min, n_block_max)
                for i in cutlass.range(n_block_max - n_block_min, unroll=2):
                    n_block = n_block_max - i - 1
                    # page_idx = mPageTable[batch_idx, n_block] if const_expr(mPageTable is not None) else None
                    page_idx = (
                        None
                        if const_expr(mPageTable is None)
                        else (
                            mPageTable[
                                batch_idx * tile_scheduler.params.num_head
                                + head_idx_kv,
                                n_block,
                            ]
                            if const_expr(self.groupwise)
                            else mPageTable[batch_idx, n_block]
                        )
                    )
                    load_K(
                        block=n_block,
                        producer_state=kv_producer_state,
                        page_idx=page_idx,
                    )
                    load_V(
                        block=n_block,
                        producer_state=kv_producer_state,
                        page_idx=page_idx,
                    )
                    kv_producer_state.advance()
                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        # softmax: Softmax,
        # acc_O: cute.Tensor,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softcap_val: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        tSrQ = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sK))
        if const_expr(self.mma_pv_is_rs):
            acc_S_shape = tiled_mma_qk.partition_shape_C(
                (self.m_block_size, self.n_block_size)
            )
            tOrP = cute.make_fragment(
                utils.convert_layout_acc_frgA(cute.make_layout(acc_S_shape)), self.dtype
            )
        else:
            tOrP = tiled_mma_pv.make_fragment_A(wg_mma_pv.partition_A(sP))
        tOrVt = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sVt))

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_P = utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_P = cute.make_tiled_copy_C(
            smem_copy_atom_P, tiled_mma_qk
        ).get_slice(tidx)
        # tPsP = smem_thr_copy_P.partition_D(sP_pi) if const_expr(sP_pi is not None) else None
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        # if cute.arch.thread_idx()[0] == 0:
        #     cute.printf(sP_pi.layout, sP_pi.iterator)
        #     cute.printf(sP.layout, sP.iterator)
        #     cute.printf(tPsP.layout, tPsP.iterator)

        self.mma_init()

        acc_shape_O = tiled_mma_pv.partition_shape_C(
            (self.m_block_size, self.head_dim_v_padded)
        )
        acc_O = cute.make_fragment(acc_shape_O, Float32)
        # group parameters for mma_one_n_block
        mma_params = SimpleNamespace(
            tSrQ=tSrQ, tSrK=tSrK, tOrP=tOrP, tOrVt=tOrVt, acc_O=acc_O
        )
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        mma_one_n_block_all = partial(
            (
                self.mma_one_n_block_intrawg_overlap
                if const_expr(self.intra_wg_overlap)
                else self.mma_one_n_block
            ),
            tiled_mma_qk=tiled_mma_qk,
            tiled_mma_pv=tiled_mma_pv,
            tiled_mma_pv_rs=tiled_mma_pv_rs,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            check_inf=True,
        )

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:
            # Softcapping needs to happen before masking since if we apply after masking, softcapping can turn
            # -inf to e.g. -50.0, which can affect the attention softmax.
            def scoremod_premask_fn(acc_S):
                if const_expr(softcap_val is not None):
                    acc_S.store(
                        cute.math.tanh(acc_S.load() * softcap_val, fastmath=True)
                    )

            # shape: (atom_v_m * rest_m)
            softmax = Softmax(
                softmax_scale_log2, num_rows=acc_O.shape[0][0] * acc_O.shape[1]
            )
            mma_one_n_block = partial(
                mma_one_n_block_all,
                softmax=softmax,
                scoremod_premask_fn=scoremod_premask_fn,
            )

            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            mask_fn = partial(
                mask.apply_mask,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
            )
            softmax.reset()
            # Load Q if not TMA_Q
            if const_expr(not self.use_tma_Q):
                pack_gqa = PackGQA(
                    self.m_block_size,
                    self.head_dim_padded,
                    self.check_hdim_oob,
                    self.qhead_per_kvhead,
                )
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mQ_cur = mQ[None, None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q
                        if const_expr(not self.pack_gqa)
                        else (0, seqlen.offset_q)
                    )
                    mQ_cur = cute.domain_offset((offset, 0), mQ[None, None, head_idx])
                # gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
                # gQ = cute.local_tile(mQ_cur, (self.m_block_size, self.head_dim_padded), (m_block, 0))
                # self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q,
                #             headdim=mQ.shape[1])
                pack_gqa.load_Q(
                    mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q
                )
                utils.cp_async_mbarrier_arrive_shared(mbar_ptr_Q, noinc=True)

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            cute.arch.mbarrier_wait(mbar_ptr_Q, phase=q_consumer_phase)
            q_consumer_phase ^= 1
            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of n_block_size.
            # We also need masking on S if it's causal, for the last several blocks.
            O_should_accumulate = False
            # First iteration with seqlen masking
            if const_expr(self.intra_wg_overlap):
                acc_S = cute.make_fragment(
                    tiled_mma_qk.partition_shape_C(
                        (self.m_block_size, self.n_block_size)
                    ),
                    Float32,
                )
                pipeline_k.consumer_wait(kv_consumer_state)
                sm90_utils.gemm(
                    tiled_mma_qk,
                    acc_S,
                    tSrQ,
                    tSrK[None, None, None, kv_consumer_state.index],
                    zero_init=True,
                    wg_wait=0,
                )
                pipeline_k.consumer_release(kv_consumer_state)
                scoremod_premask_fn(acc_S)
                # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
                mask_fn(acc_S, n_block=n_block_max - 1, mask_seqlen=True)
                # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
                softmax.online_softmax(acc_S, is_first=True)
                tOrP_acc = cute.make_tensor(
                    acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout)
                )
                tOrP = (
                    mma_params.tOrP
                    if const_expr(self.mma_pv_is_rs)
                    else cute.make_fragment_like(tOrP_acc, self.dtype)
                )
                # tOrP.store(tOrP_acc.load().to(self.dtype))
                # the "to(self.dtype)" conversion fails to vectorize for block sizes other
                # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
                # 2 elements. So we just call ptx directly.
                utils.cvt_f16(tOrP_acc, tOrP)
                if const_expr(not self.mma_pv_is_rs):
                    tPrP = smem_thr_copy_P.retile(tOrP)
                    cute.copy(smem_thr_copy_P, tPrP, tPsP)
                    # Fence and barrier to make sure smem store is visible to WGMMA
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
                # Need to initialize tOrO in the case of RescaleOBeforeGemm where we will scale tOrO even in the 1st iter
                # acc_O.fill(0.0)
            else:
                self.warp_scheduler_barrier_sync()
                kv_consumer_state = mma_one_n_block(
                    n_block_max - 1,
                    kv_consumer_state,
                    is_first_n_block=True,
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                    O_should_accumulate=False,
                )
                O_should_accumulate = True
            # if cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block_max = {}, n_block_min = {}", m_block, n_block_max, n_block_min)
            n_block_max -= 1
            # Next couple of iterations with causal masking
            if const_expr(self.is_causal or self.is_local):
                n_block_min_causal_local_mask = (
                    block_info.get_n_block_min_causal_local_mask(
                        seqlen, m_block, n_block_min
                    )
                )
                # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_causal_local_mask = {}", n_block_min_causal_local_mask)
                for n_tile in cutlass.range(
                    n_block_max - n_block_min_causal_local_mask, unroll=1
                ):
                    n_block = n_block_max - 1 - n_tile
                    kv_consumer_state = mma_one_n_block(
                        n_block,
                        kv_consumer_state,
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                        O_should_accumulate=O_should_accumulate,
                    )
                    O_should_accumulate = True
                n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
            # The remaining iterations have no masking
            n_block_min_before_local_mask = (
                block_info.get_n_block_min_before_local_mask(
                    seqlen, m_block, n_block_min
                )
            )
            # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block_min_before_local_mask = {}, n_block_min = {}", n_block_min_before_local_mask, n_block_min)
            for n_tile in cutlass.range(
                n_block_max - n_block_min_before_local_mask, unroll=1
            ):
                n_block = n_block_max - 1 - n_tile
                kv_consumer_state = mma_one_n_block(
                    n_block,
                    kv_consumer_state,
                    check_inf=True,
                    O_should_accumulate=O_should_accumulate,
                )
                O_should_accumulate = True
            # Separate iterations with local masking on the left
            if const_expr(self.is_local and block_info.window_size_left is not None):
                n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    kv_consumer_state = mma_one_n_block(
                        n_block,
                        kv_consumer_state,
                        check_inf=True,
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                        O_should_accumulate=O_should_accumulate,
                    )
                    O_should_accumulate = True
            # Last "half" iteration
            if const_expr(self.intra_wg_overlap):
                pipeline_v.consumer_wait(
                    kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state)
                )
                sm90_utils.gemm(
                    tiled_mma_pv,
                    mma_params.acc_O,
                    mma_params.tOrP,
                    mma_params.tOrVt[None, None, None, kv_consumer_state.index],
                    zero_init=not O_should_accumulate,
                    wg_wait=-1,
                )
                warpgroup.wait_group(0)
                pipeline_v.consumer_release(kv_consumer_state)
                kv_consumer_state.advance()
            else:
                self.warp_scheduler_barrier_arrive()

            # normalize acc_O by row_sum and calculate the lse
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:  # Each thread might have a different sink value due to different q_head
                    sink_val = cute.make_fragment_like(softmax.row_max, Float32)
                    cS = cute.make_identity_tensor(
                        (self.m_block_size, self.n_block_size)
                    )
                    tScS_mn = utils.make_acc_tensor_mn_view(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.m_block_size + tScS_mn[r][0]
                        q_head_idx = (
                            row % self.qhead_per_kvhead
                            + head_idx * self.qhead_per_kvhead
                        )
                        sink_val[r] = Float32(learnable_sink[q_head_idx])
            else:
                sink_val = None

            row_scale = softmax.finalize(sink_val=sink_val)
            softmax.rescale_O(acc_O, row_scale)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        scoremod_premask_fn: Callable,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        O_should_accumulate: cutlass.Boolean = True,
    ):
        acc_S = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size)),
            Float32,
        )
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        sm90_utils.gemm(
            tiled_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK[None, None, None, smem_pipe_read.index],
            zero_init=True,
            wg_wait=-1,
        )
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)
        scoremod_premask_fn(acc_S)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
        tOrP_acc = cute.make_tensor(
            acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout)
        )
        tOrP = (
            mma_params.tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_fragment_like(tOrP_acc, self.dtype)
        )
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        utils.cvt_f16(tOrP_acc, tOrP)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(mma_params.tOrP)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        sm90_utils.gemm(
            tiled_mma_pv,
            mma_params.acc_O,
            mma_params.tOrP,
            mma_params.tOrVt[None, None, None, smem_pipe_read.index],
            zero_init=not O_should_accumulate,
            wg_wait=0,
        )
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        n_block: Int32,
        smem_pipe_read: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        scoremod_premask_fn: Callable,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
        O_should_accumulate: cutlass.Boolean = True,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        acc_S = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.m_block_size, self.n_block_size)),
            Float32,
        )
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        sm90_utils.gemm(
            tiled_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK[None, None, None, smem_pipe_read.index],
            zero_init=True,
            wg_wait=-1,
        )
        pipeline_v.consumer_wait(
            smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v)
        )
        sm90_utils.gemm(
            tiled_mma_pv,
            mma_params.acc_O,
            mma_params.tOrP,
            mma_params.tOrVt[None, None, None, smem_pipe_read_v.index],
            zero_init=not O_should_accumulate,
            wg_wait=-1,
        )
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)
        scoremod_premask_fn(acc_S)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(utils.make_acc_tensor_mn_view(acc_S))
        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = cute.make_tensor(
            acc_S.iterator, utils.convert_layout_acc_frgA(acc_S.layout)
        )
        tOrP = (
            mma_params.tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_fragment_like(tOrP_acc, self.dtype)
        )
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        utils.cvt_f16(tOrP_acc, tOrP)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(mma_params.acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_mma_warp_groups in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            next_wg = (
                1 - cur_wg
                if const_expr(self.num_mma_warp_groups == 2)
                else (cur_wg + 1 if cur_wg < self.num_mma_warp_groups - 1 else 0)
            )
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    # @cute.jit
    def load_K(
        self,
        tma_atom: cute.CopyAtom,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        pipeline: cutlass.pipeline.PipelineAsync,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        page_idx: Optional[Int32] = None,
    ):
        # TODO: mcast
        # TODO check warp_idx if we have 128 producer threads
        pipeline.producer_acquire(producer_state)
        cute.copy(
            tma_atom,
            (
                tKgK[None, block]
                if const_expr(page_idx is None)
                else tKgK[None, 0, page_idx]
            ),
            tKsK[None, producer_state.index],
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
        )
