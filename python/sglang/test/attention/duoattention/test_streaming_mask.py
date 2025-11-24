import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
import numpy as np
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.layout import LayoutEnum

from sglang.srt.sparse_attention.kernels.attention.mask import AttentionMask
from sglang.test.attention.duoattention.streaming_attention_ref import (
    construct_streaming_mask,
)


class StreamingMaskTester:
    """Test class for streaming mask functionality following CuteDSL pattern."""

    def __init__(
        self,
        m_block_size: int,
        n_block_size: int,
        num_threads: int = 128,
    ):
        # These are now treated as compile-time constants for the kernel
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mOutput: cute.Tensor,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        window_size_left: cutlass.Int32,
        sink_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Launch the streaming mask test kernel."""
        # Do NOT pass m_block_size and n_block_size here.
        # The kernel will access them via `self`.
        self.kernel(
            mOutput,
            seqlen_q,
            seqlen_k,
            window_size_left,
            sink_size,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOutput: cute.Tensor,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        window_size_left: cutlass.Int32,
        sink_size: cutlass.Int32,
    ):
        tidx = cute.arch.thread_idx()[0]

        m_block_size = self.m_block_size
        n_block_size = self.n_block_size

        # Follow FlashAttention Sm90 configuration exactly
        # The tiled_mma is configured for 64x64 processing per warpgroup
        # When block_size > 64, we need to loop over multiple tiles in both M and N directions
        atom_layout_mnk = (1, 1, 1)

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            cutlass.Float16,
            cutlass.Float16,
            LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            cutlass.Float32,
            atom_layout_mnk,
            tiler_mn=(64, 64),  # Each warpgroup processes 64x64
        )

        thr_mma = tiled_mma.get_slice(tidx)

        # Calculate number of tiles needed (each tile is 64x64)
        num_m_tiles = cutlass.const_expr(m_block_size // 64)
        num_n_tiles = cutlass.const_expr(n_block_size // 64)

        # AttentionMask should use the tile size (64x64), not the full block size
        # When we pass m_block and n_block indices, it will calculate:
        # global_row = m_block * 64 + local_row
        # global_col = n_block * 64 + local_col
        mask = AttentionMask(
            m_block_size=64,
            n_block_size=64,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            window_size_left=window_size_left,
            sink_size=sink_size,
            enable_streaming=True,
        )

        # Setup copy operation outside the loop
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
        tiled_copy_C = cute.make_tiled_copy_C(copy_atom, tiled_mma)
        thrd_copy_C = tiled_copy_C.get_slice(tidx)

        # Loop over M and N tiles (each warpgroup processes 64x64 at a time)
        for m_tile_idx in cutlass.range_constexpr(num_m_tiles):
            for n_tile_idx in cutlass.range_constexpr(num_n_tiles):
                m_block = cutlass.Int32(m_tile_idx)
                n_block = cutlass.Int32(n_tile_idx)

                # Create accumulator fragment for this 64x64 tile
                acc_shape = tiled_mma.partition_shape_C((64, 64))
                acc_S = cute.make_fragment(acc_shape, cutlass.Float32)

                self.clear_acc(acc_S)

                # Apply mask for this tile
                mask.apply_streaming_mask(
                    acc_S, m_block, n_block, thr_mma, mask_seqlen=True
                )

                # Copy this tile to global memory
                acc_S_view = thrd_copy_C.retile(acc_S)

                # Use cute.local_tile to select the correct 64x64 tile from output
                # This preserves the layout unlike domain_offset
                gOutput_tile = cute.local_tile(
                    mOutput, (64, 64), (m_tile_idx, n_tile_idx)
                )
                gmem_thr_out = thrd_copy_C.partition_D(gOutput_tile)

                cute.copy(thrd_copy_C, acc_S_view, gmem_thr_out)

    @cute.jit
    def clear_acc(
        self,
        acc_S: cute.Tensor,
    ):
        """
        Clear the accumulator tensor
        """
        for i in cutlass.range(cute.size(acc_S)):
            acc_S[i] = 0.0


def run_test(
    m_block_size: int,
    n_block_size: int,
    seqlen_q: int,
    seqlen_k: int,
    window_size_left: int,
    sink_size: int,
):
    cache_key = (m_block_size, n_block_size)
    if cache_key not in run_test.tester_cache:
        print(
            f"Compiling new kernel for block size ({m_block_size}, {n_block_size})..."
        )
        run_test.tester_cache[cache_key] = StreamingMaskTester(
            m_block_size=m_block_size,
            n_block_size=n_block_size,
        )

    tester = run_test.tester_cache[cache_key]

    output = torch.empty(
        (m_block_size, n_block_size), dtype=torch.float32, device="cuda"
    )
    mOutput = from_dlpack(output.detach(), assumed_align=16)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    tester(
        mOutput=mOutput,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        window_size_left=window_size_left,
        sink_size=sink_size,
        stream=current_stream,
    )

    torch.cuda.synchronize()

    result_cpu = output.cpu().numpy()

    expected_mask_bool = construct_streaming_mask(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        sink_size=sink_size,
        local_size=window_size_left,
        is_causal=True,
        device=torch.device("cpu"),
    )

    expected_mask = torch.zeros_like(expected_mask_bool, dtype=torch.float32)
    expected_mask[expected_mask_bool] = -float("inf")
    expected_mask = expected_mask.numpy()

    np.testing.assert_allclose(result_cpu, expected_mask, atol=1e-6)

    print(
        f"Test passed for m_block={m_block_size}, n_block={n_block_size}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}!"
    )


# Initialize tester cache
run_test.tester_cache = {}


if __name__ == "__main__":
    print("\nRunning original test case (64x64 block)...")
    run_test(
        m_block_size=64,
        n_block_size=64,
        seqlen_q=64,
        seqlen_k=64,
        window_size_left=128,  # window > seqlen, so it's fully causal + sink
        sink_size=4,
    )

    # Note: For 128x128, we need special handling as single warpgroup
    # may not cover full block in one pass
    print("\nTesting both dimensions expansion (128x128 block)...")
    run_test(
        m_block_size=128,
        n_block_size=128,
        seqlen_q=128,
        seqlen_k=128,
        window_size_left=128,
        sink_size=4,
    )
