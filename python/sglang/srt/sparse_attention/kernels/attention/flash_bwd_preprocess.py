# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_bwd_preprocess_kernel.h
# from Cutlass C++ to Cute-DSL.
import math
import operator
from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from sglang.srt.sparse_attention.kernels.attention import utils


class FlashAttentionBackwardPreprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        m_block_size: int = 128,
        num_threads: int = 128,
    ):
        """
        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        """
        self.dtype = dtype
        self.m_block_size = m_block_size
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.num_threads = num_threads

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param num_threads: number of threads
        :type num_threads: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if num_threads < m_block_size:  # For multiplying lse with log2
            return False
        return True

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        # We want kBlockKGmem to be a power of 2 so that when we do the summing,
        # it's just between threads in the same warp
        gmem_k_block_size = (
            128
            if self.head_dim_padded % 128 == 0
            else (
                64
                if self.head_dim_padded % 64 == 0
                else (32 if self.head_dim_padded % 32 == 0 else 16)
            )
        )
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        # atom_universal_copy: universal copy atom for O & dO load
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tOdO_layout: thread layout for O & dO load
        self.gmem_threads_per_row = gmem_k_block_size // async_copy_elems
        assert self.num_threads % self.gmem_threads_per_row == 0
        tOdO_layout = cute.make_ordered_layout(
            (self.num_threads // self.gmem_threads_per_row, self.gmem_threads_per_row),
            order=(1, 0),
        )
        # Value layouts for copies
        vOdO_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tOdO_layout, vOdO_layout
        )
        self.gmem_tiled_copy_dO = cute.make_tiled_copy_tv(
            atom_universal_copy, tOdO_layout, vOdO_layout
        )

        async_copy_elems_accum = universal_copy_bits // cutlass.Float32.width
        atom_universal_copy_accum = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        assert (
            self.m_block_size * self.head_dim_padded // async_copy_elems_accum
        ) % self.num_threads == 0
        self.gmem_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_universal_copy_accum,
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not mO.element_type in [cutlass.Float16, cutlass.BFloat16]
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(not mdPsum.element_type in [cutlass.Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if cutlass.const_expr(mdQaccum is not None):
            if cutlass.const_expr(not mdQaccum.element_type in [cutlass.Float32]):
                raise TypeError("dQaccum tensor must be Float32")
        if cutlass.const_expr(mLSE is not None):
            assert (
                mLSElog2 is not None
            ), "If mLSE is provided, mLSElog2 must also be provided"
            if cutlass.const_expr(not mLSE.element_type in [cutlass.Float32]):
                raise TypeError("LSE tensor must be Float32")
            if cutlass.const_expr(not mLSElog2.element_type in [cutlass.Float32]):
                raise TypeError("LSElog2 tensor must be Float32")

        self._setup_attributes()

        # grid_dim: (m_block, num_head, batch_size)
        grid_dim = (
            cute.ceil_div(mO.shape[1], self.m_block_size),
            cute.size(mO.shape[2]),
            cute.size(mO.shape[0]),
        )
        self.kernel(
            mO,
            mdO,
            mdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dO,
            self.gmem_tiled_copy_dQaccum,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dO: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, num_head, batch_size = cute.arch.block_idx()

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkOdO_shape = (self.m_block_size, self.head_dim_padded)
        # (m_block_size, head_dim)
        gO = cute.local_tile(
            mO[batch_size, None, num_head, None], blkOdO_shape, (m_block, 0)
        )
        gdO = cute.local_tile(
            mdO[batch_size, None, num_head, None], blkOdO_shape, (m_block, 0)
        )

        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        gmem_thr_copy_dO = gmem_tiled_copy_dO.get_slice(tidx)
        # (CPY_Atom, CPY_M, CPY_K)
        tOgO = gmem_thr_copy_O.partition_S(gO)
        tOgdO = gmem_thr_copy_dO.partition_S(gdO)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for KV
        cOdO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tOcO = gmem_thr_copy_O.partition_S(cOdO)
        t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cOdO)
        tOpO = utils.predicate_k(tOcO, limit=mO.shape[3])
        tOcdO = gmem_thr_copy_dO.partition_S(cOdO)
        t0OcdO = gmem_thr_copy_dO.get_slice(0).partition_S(cOdO)
        tOpdO = utils.predicate_k(tOcdO, limit=mdO.shape[3])

        seqlen_q = mO.shape[1]
        seqlen_q_rounded = cute.round_up(seqlen_q, self.m_block_size)

        if cutlass.const_expr(mLSE is not None):
            gLSE = cute.local_tile(
                mLSE[batch_size, num_head, None], (self.m_block_size,), (m_block,)
            )
            lse = cutlass.Float32.inf
            if tidx < seqlen_q - m_block * self.m_block_size:
                lse = gLSE[tidx]

        tOrO = cute.make_fragment_like(tOgO)
        tOrdO = cute.make_fragment_like(tOgdO)
        assert cute.size(tOgO, mode=[0]) == cute.size(tOgdO, mode=[0])
        assert cute.size(tOgO, mode=[1]) == cute.size(tOgdO, mode=[1])
        assert cute.size(tOgO, mode=[2]) == cute.size(tOgdO, mode=[2])
        for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
            # Instead of using tOcO, we using t0OcO and subtract the offset from the limit
            # (seqlen_q - m_block * kBlockM). This is because the entries of t0OcO are known at compile time.
            if t0OcO[0, m, 0][0] < seqlen_q - m_block * self.m_block_size - tOcO[0][0]:
                cute.copy(
                    gmem_thr_copy_O,
                    tOgO[None, m, None],
                    tOrO[None, m, None],
                    pred=(
                        tOpO[None, m, None]
                        if cutlass.const_expr(self.check_hdim_oob)
                        else None
                    ),
                )
                cute.copy(
                    gmem_thr_copy_dO,
                    tOgdO[None, m, None],
                    tOrdO[None, m, None],
                    pred=(
                        tOpdO[None, m, None]
                        if cutlass.const_expr(self.check_hdim_oob)
                        else None
                    ),
                )
        # Sum across the "k" dimension
        dpsum = (
            tOrO.load().to(cutlass.Float32) * tOrdO.load().to(cutlass.Float32)
        ).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1))
        dpsum = utils.warp_reduce(dpsum, operator.add, width=self.gmem_threads_per_row)
        dP_sum = cute.make_fragment(cute.size(tOrO, mode=[1]), cutlass.Float32)
        dP_sum.store(dpsum)

        # Write dPsum from rmem -> gmem
        gdPsum = cute.local_tile(
            mdPsum[batch_size, num_head, None], (self.m_block_size,), (m_block,)
        )
        # Only the thread corresponding to column 0 writes out the lse to gmem
        if tOcO[0, 0, 0][1] == 0:
            for m in cutlass.range(cute.size(dP_sum), unroll_full=True):
                row = tOcO[0, m, 0][0]
                gdPsum[row] = (
                    dP_sum[m]
                    if row < mO.shape[1] - m_block * self.m_block_size
                    else 0.0
                )

        # Clear dQaccum
        if cutlass.const_expr(mdQaccum is not None):
            blkdQaccum_shape = (self.m_block_size * self.head_dim_padded,)
            gdQaccum = cute.local_tile(
                mdQaccum[batch_size, num_head, None], blkdQaccum_shape, (m_block,)
            )
            gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
            tQgQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
            zero = cute.make_fragment_like(tQgQaccum)
            zero.fill(0.0)
            cute.copy(gmem_tiled_copy_dQaccum, zero, tQgQaccum)

        if cutlass.const_expr(mLSE is not None):
            gLSElog2 = cute.local_tile(
                mLSElog2[batch_size, num_head, None], (self.m_block_size,), (m_block,)
            )
            LOG2_E = math.log2(math.e)
            if tidx < seqlen_q_rounded - m_block * self.m_block_size:
                gLSElog2[tidx] = lse * LOG2_E if lse != -cutlass.Float32.inf else 0.0
