from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr


class CuSeqlensToBlocksKernel:
    """Single-CTA prep for block-packed shear scheduling: computes the cumulative
    per-batch group-block counts and the block -> batch index map in one launch."""

    def __init__(
        self,
        tile: int = 128,
        num_threads: int = 1024,
        seqlen_multiple: int = 1,
        use_pdl: bool = False,
    ):
        self.tile = tile
        self.num_threads = num_threads
        assert num_threads % 32 == 0
        self.num_warps = num_threads // cute.arch.WARP_SIZE
        self.seqlen_multiple = seqlen_multiple
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        mCuBlocks: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        mBlocksToBatchIdx: cute.Tensor,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        @cute.struct
        class SharedStorage:
            warp_block_count: cute.struct.MemRange[Int32, self.num_warps]
            cu_blocks: cute.struct.MemRange[Int32, self.num_threads + 1]

        self.kernel(
            mCuBlocks,
            mCuSeqlens,
            mBlocksToBatchIdx,
            SharedStorage,
        ).launch(
            grid=[1, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mCuBlocks: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        mBlocksToBatchIdx: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        batch_size = mCuBlocks.shape[0] - 1
        batch_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_block_count = storage.warp_block_count.get_tensor(
            cute.make_layout(self.num_warps)
        )
        sCuBlocks = storage.cu_blocks.get_tensor(cute.make_layout(self.num_threads + 1))

        if batch_idx == 0:
            mCuBlocks[0] = 0
            sCuBlocks[0] = 0

        seqlen = Int32(0)
        if batch_idx < batch_size:
            seqlen = mCuSeqlens[batch_idx + 1] - mCuSeqlens[batch_idx]
        seqlen *= self.seqlen_multiple
        num_blocks = (seqlen + self.tile - 1) // self.tile

        total_blocks_for_batch = num_blocks
        for delta in (1, 2, 4, 8, 16):
            other = cute.arch.shuffle_sync_up(
                total_blocks_for_batch, delta, mask_and_clamp=0
            )
            if lane_idx >= delta:
                total_blocks_for_batch += other

        if lane_idx == 31:
            warp_block_count[warp_idx] = total_blocks_for_batch

        cute.arch.sync_threads()

        if warp_idx * 32 < batch_size:
            for idx in cutlass.range(warp_idx):
                total_blocks_for_batch += warp_block_count[idx]

            if batch_idx < batch_size:
                mCuBlocks[batch_idx + 1] = total_blocks_for_batch
                sCuBlocks[batch_idx + 1] = total_blocks_for_batch

        cute.arch.sync_threads()

        total_blocks = sCuBlocks[batch_size]
        num_iters = (total_blocks + self.num_threads - 1) // self.num_threads
        for it in cutlass.range(num_iters, unroll=1):
            block = it * self.num_threads + batch_idx
            if block < total_blocks:
                lo = Int32(0)
                hi = Int32(batch_size)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if sCuBlocks[mid + 1] <= block:
                        lo = mid + 1
                    else:
                        hi = mid
                mBlocksToBatchIdx[block] = lo
