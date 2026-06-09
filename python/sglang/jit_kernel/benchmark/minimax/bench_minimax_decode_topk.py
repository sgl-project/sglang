"""Benchmark: MiniMax-M3 single-stage radix-select decode topk (JIT CUDA) vs the
2-stage split-K Triton baseline (_topk_index_partial_kernel + _topk_index_merge_kernel).

Both consume the decode score tensor [num_heads, batch, max_seqblock] and produce
topk_idx [num_heads, batch, topk]. The JIT kernel is one launch with no
intermediate buffers; the baseline is two launches with split-K partials.
"""

import torch
import triton

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.minimax_decode_topk import minimax_decode_topk
from sglang.srt.layers.attention.minimax_sparse_ops.decode.flash_with_topk_idx import (
    _topk_index_merge_kernel,
    _topk_index_partial_kernel,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-benchmark-1-gpu-large")

BLOCK_SIZE = 128
TOPK = 16
NUM_HEADS = 1  # per-rank index heads at TP>=4


def _triton_2stage(score, seq_lens):
    num_q_heads, batch_size, max_seqblock = score.shape
    TOPK_TARGET_GRID = 64
    MAX_NUM_TOPK_CHUNKS = 16
    t = max(
        1,
        min(MAX_NUM_TOPK_CHUNKS, TOPK_TARGET_GRID // max(1, batch_size * num_q_heads)),
    )
    nchunks = 1 << (t.bit_length() - 1)
    bt = triton.next_power_of_2(TOPK)
    chunk_blocks = (max_seqblock + nchunks - 1) // nchunks
    out = torch.empty(
        (num_q_heads, batch_size, TOPK), device=score.device, dtype=torch.int32
    )
    tsp = torch.empty(
        nchunks, num_q_heads, batch_size, bt, dtype=torch.float32, device=score.device
    )
    tip = torch.empty(
        nchunks, num_q_heads, batch_size, bt, dtype=torch.int32, device=score.device
    )
    _topk_index_partial_kernel[(batch_size, num_q_heads, nchunks)](
        score,
        tsp,
        tip,
        seq_lens,
        BLOCK_SIZE,
        TOPK,
        chunk_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        tsp.stride(0),
        tsp.stride(1),
        tsp.stride(2),
        tsp.stride(3),
        tip.stride(0),
        tip.stride(1),
        tip.stride(2),
        tip.stride(3),
    )
    _topk_index_merge_kernel[(batch_size, num_q_heads)](
        tsp,
        tip,
        out,
        seq_lens,
        BLOCK_SIZE,
        TOPK,
        tsp.stride(0),
        tsp.stride(1),
        tsp.stride(2),
        tsp.stride(3),
        tip.stride(0),
        tip.stride(1),
        tip.stride(2),
        tip.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_TOPK_CHUNKS=nchunks,
    )
    return out


def _jit(score, seq_lens):
    return minimax_decode_topk(score, seq_lens, BLOCK_SIZE, TOPK)


FN_MAP = {"jit": _jit, "triton_2stage": _triton_2stage}


@marker.parametrize("ctx", [4096, 32768, 131072, 524288], [4096, 524288])
@marker.parametrize("batch", [1, 4, 16, 64, 256], [1, 64])
@marker.benchmark("impl", ["jit", "triton_2stage"])
def benchmark(ctx: int, batch: int, impl: str):
    max_seqblock = (524288 + BLOCK_SIZE - 1) // BLOCK_SIZE
    nb = min((ctx + BLOCK_SIZE - 1) // BLOCK_SIZE, max_seqblock)
    score = torch.full(
        (NUM_HEADS, batch, max_seqblock),
        float("-inf"),
        dtype=torch.float32,
        device="cuda",
    )
    score[:, :, :nb] = torch.randn(NUM_HEADS, batch, nb, device="cuda") * 5.0
    score[:, :, nb - 1] = 1e29  # forced local block
    seq_lens = torch.full((batch,), ctx, device="cuda", dtype=torch.int32)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(score, seq_lens),
        graph_clone_args=(0, 1),  # both read-only inputs
        memory_args=(score,),
    )


if __name__ == "__main__":
    benchmark.run()
