# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_bwd_postprocess_kernel.h
# from Cutlass C++ to Cute-DSL.
import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp

from sglang.srt.sparse_attention.kernels.attention import ampere_helpers as sm80_utils
from sglang.srt.sparse_attention.kernels.attention import utils


class FlashAttentionBackwardPostprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        # tiled_mma: cute.TiledMma,
        head_dim: int,
        m_block_size: int = 128,
        num_threads: int = 256,
        AtomLayoutMdQ: int = 1,
        dQ_swapAB: bool = False,
    ):
        """Initializes the configuration for a flash attention v2 kernel.

        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        """
        self.dtype = dtype
        self.m_block_size = m_block_size
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        self.check_hdim_oob = head_dim != self.head_dim_padded
        # self.tiled_mma = tiled_mma
        self.num_threads = num_threads
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.dQ_swapAB = dQ_swapAB

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        return True

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems_accum = universal_copy_bits // cutlass.Float32.width
        atom_async_copy_accum = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        # We don't do bound checking for the gmem -> smem load so we just assert here.
        assert (
            self.m_block_size * self.head_dim_padded // async_copy_elems_accum
        ) % self.tiled_mma.size == 0
        self.g2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.tiled_mma.size),
            cute.make_layout(async_copy_elems_accum),
        )
        atom_universal_copy_accum = cute.make_copy_atom(
            # multiply by 4 for Sm90
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=cutlass.Float32.width,
        )
        self.s2r_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_universal_copy_accum,
            cute.make_layout(self.tiled_mma.size),
            cute.make_layout(1),  # 4 for Sm90
        )

        async_copy_elems = universal_copy_bits // self.dtype.width
        # atom_universal_copy: universal copy atom for dQ store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tdQ_layout: thread layout for dQ store
        assert self.head_dim_padded % async_copy_elems == 0
        gmem_threads_per_row = math.gcd(
            self.head_dim_padded // async_copy_elems, self.tiled_mma.size
        )
        assert self.tiled_mma.size % gmem_threads_per_row == 0
        tdQ_layout = cute.make_ordered_layout(
            (self.tiled_mma.size // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        # Value layouts for copies
        vdQ_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_dQ = cute.make_tiled_copy_tv(
            atom_universal_copy, tdQ_layout, vdQ_layout
        )
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: dQaccum / dQ
        # ///////////////////////////////////////////////////////////////////////////////
        self.sdQaccum_layout = cute.make_layout(
            self.m_block_size * self.head_dim_padded
        )
        # We can't just use kHeadDim here. E.g. if MMA shape is 64 x 96 but split across 2 WGs,
        # then setting kBlockKSmem to 32 will cause "Static shape_div failure".
        # We want to treat it as 64 x 48, so kBlockKSmem should be 16.
        mma_shape_n = self.tiled_mma.get_tile_size(1)
        sdQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, mma_shape_n)
        self.sdQ_layout = cute.tile_to_shape(
            sdQ_layout_atom, (self.m_block_size, self.head_dim_padded), (0, 1)
        )

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(
            not mdQ.element_type in [cutlass.Float16, cutlass.BFloat16]
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(mdQaccum is not None):
            if cutlass.const_expr(not mdQaccum.element_type in [cutlass.Float32]):
                raise TypeError("dQaccum tensor must be Float32")

        num_mma_warps = self.num_threads // 32
        AtomLayoutdQ = (
            (self.AtomLayoutMdQ, num_mma_warps // self.AtomLayoutMdQ, 1)
            if cutlass.const_expr(not self.dQ_swapAB)
            else (num_mma_warps // self.AtomLayoutMdQ, self.AtomLayoutMdQ, 1)
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutdQ,
            permutation_mnk=(AtomLayoutdQ[0] * 16, AtomLayoutdQ[1] * 16, 16),
        )
        self.tiled_mma = tiled_mma

        self._setup_attributes()

        smem_size = max(
            cute.size_in_bytes(cutlass.Float32, self.sdQaccum_layout),
            cute.size_in_bytes(self.dtype, self.sdQ_layout),
        )

        # grid_dim: (m_block, num_head, batch_size)
        grid_dim = (
            cute.ceil_div(mdQ.shape[1], self.m_block_size),
            cute.size(mdQ.shape[2]),
            cute.size(mdQ.shape[0]),
        )
        self.kernel(
            mdQaccum,
            mdQ,
            scale,
            tiled_mma,
            self.dQ_swapAB,
            self.sdQaccum_layout,
            self.sdQ_layout,
            self.g2s_tiled_copy_dQaccum,
            self.s2r_tiled_copy_dQaccum,
            self.gmem_tiled_copy_dQ,
        ).launch(
            grid=grid_dim,
            block=[tiled_mma.size, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        dQ_swapAB: cutlass.Constexpr,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        g2s_tiled_copy_dQaccum: cute.TiledCopy,
        s2r_tiled_copy_dQaccum: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, num_head, batch_size = cute.arch.block_idx()

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkdQaccum_shape = (self.m_block_size * self.head_dim_padded,)
        gdQaccum = cute.local_tile(
            mdQaccum[batch_size, num_head, None], blkdQaccum_shape, (m_block,)
        )
        blkdQ_shape = (self.m_block_size, self.head_dim_padded)
        gdQ = cute.local_tile(
            mdQ[batch_size, None, num_head, None], blkdQ_shape, (m_block, 0)
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        sdQaccum = smem.allocate_tensor(
            cutlass.Float32, sdQaccum_layout, byte_alignment=1024
        )
        sdQ = cute.make_tensor(
            cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype), sdQ_layout
        )

        seqlen_q = mdQ.shape[1]
        seqlen_q_rounded = cute.round_up(seqlen_q, self.m_block_size)

        # Step 1: load dQaccum from gmem to smem
        g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_slice(tidx)
        tdQgdQaccum = g2s_thr_copy_dQaccum.partition_S(gdQaccum)
        tdQsdQaccumg2s = g2s_thr_copy_dQaccum.partition_D(sdQaccum)
        # print(tdQgdQaccum)
        # print(tdQsdQaccum)
        cute.copy(g2s_tiled_copy_dQaccum, tdQgdQaccum, tdQsdQaccumg2s)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Step 2: load dQ from smem to rmem
        s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_slice(tidx)
        tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum)
        # print(s2r_tiled_copy_dQaccum)
        # print(sdQaccum)
        # thr_mma = tiled_mma.get_slice(tidx)
        # print(tiled_mma)
        acc_shape = tiled_mma.partition_shape_C(
            (self.m_block_size, self.head_dim_padded)
            if cutlass.const_expr(not dQ_swapAB)
            else (self.head_dim_padded, self.m_block_size)
        )
        acc = cute.make_fragment(acc_shape, cutlass.Float32)
        assert cute.size(acc) == cute.size(tdQsdQaccum)
        tdQrdQaccum = s2r_thr_copy_dQaccum.retile(acc)
        # Somehow even after retiling the layouts of tdQsdQaccum and tdQrdQaccum are different.
        # So we have to do a for loop to copy
        # cute.copy(s2r_tiled_copy_dQaccum, tdQsdQaccum, tdQrdQaccum)
        # print(acc)
        # print(tdQsdQaccum)  # ((1, 1), 64)
        # print(tdQrdQaccum)  # ((1, 4), 4, 4)
        for i in cutlass.range(cute.size(tdQsdQaccum), unroll_full=True):
            tdQrdQaccum[i] = tdQsdQaccum[i]
        # Convert tdQrdQaccum from fp32 to fp16/bf16
        rdQ = cute.make_fragment_like(acc, self.dtype)
        rdQ.store((acc.load() * scale).to(self.dtype))

        # Step 3: Copy dQ from register to smem
        cute.arch.barrier()  # make sure all threads have finished loading dQaccum
        smem_copy_atom_dQ = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=cutlass.Float32.width,
        )
        smem_thr_copy_dQ = cute.make_tiled_copy_C(
            smem_copy_atom_dQ, tiled_mma
        ).get_slice(tidx)
        taccdQrdQ = smem_thr_copy_dQ.retile(rdQ)
        taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ)
        cute.copy(smem_copy_atom_dQ, taccdQrdQ, taccdQsdQ)
        # print(taccdQrdQ)
        # print(taccdQsdQ)

        # Step 4: Copy dQ from smem to register to prepare for coalesced write to gmem
        gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
        tdQgdQ = gmem_thr_copy_dQ.partition_S(gdQ)
        tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
        tdQrdQ = cute.make_fragment_like(tdQsdQ, self.dtype)
        cute.arch.barrier()  # make sure all smem stores are done
        # TODO: check OOB when reading from smem if kBlockM isn't evenly tiled
        cute.autovec_copy(tdQsdQ, tdQrdQ)

        # Step 5: Copy dQ from register to gmem
        cdQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
        tdQpdQ = utils.predicate_k(tdQcdQ, limit=mdQ.shape[3])
        for rest_m in cutlass.range(cute.size(tdQrdQ.shape[1]), unroll_full=True):
            if tdQcdQ[0, rest_m, 0][0] < mdQ.shape[1] - m_block * self.m_block_size:
                cute.copy(
                    gmem_tiled_copy_dQ,
                    tdQrdQ[None, rest_m, None],
                    tdQgdQ[None, rest_m, None],
                    pred=tdQpdQ[None, rest_m, None],
                )
