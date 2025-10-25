import argparse
import random

import numpy as np
import torch
import triton
from triton.testing import do_bench
from vsa import BLOCK_M, BLOCK_N, block_sparse_bwd, block_sparse_fwd


def set_seed(seed: int = 42):
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark Block Sparse Attention")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Number of kv blocks each q block attends to",
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[49152],
        help="Sequence lengths to benchmark",
    )
    return parser.parse_args()


def create_input_tensors(batch, head, seq_len, headdim):
    """Create random input tensors for attention."""
    q = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(batch, head, seq_len, headdim, dtype=torch.bfloat16, device="cuda")
    return q, k, v


def generate_block_sparse_pattern(bs, h, num_q_blocks, num_kv_blocks, k, device="cuda"):
    """
    Generate a block sparse pattern where each q block attends to exactly k kv blocks.

    Args:
        bs: batch size
        h: number of heads
        num_q_blocks: number of query blocks
        num_kv_blocks: number of key-value blocks
        k: number of kv blocks each q block attends to
        device: device to create tensors on

    Returns:
        q2k_block_sparse_index: [bs, h, num_q_blocks, k]
            Contains the indices of kv blocks that each q block attends to.
        q2k_block_sparse_num: [bs, h, num_q_blocks]
            Contains the number of kv blocks that each q block attends to (all equal to k).
        k2q_block_sparse_index: [bs, h, num_kv_blocks, num_q_blocks]
            Contains the indices of q blocks that attend to each kv block.
        k2q_block_sparse_num: [bs, h, num_kv_blocks]
            Contains the number of q blocks that attend to each kv block.
        block_sparse_mask: [bs, h, num_q_blocks, num_kv_blocks]
            Binary mask where 1 indicates attention connection.
    """
    # Ensure k is not larger than num_kv_blocks
    k = min(k, num_kv_blocks)

    # Create random scores for sampling
    scores = torch.rand(bs, h, num_q_blocks, num_kv_blocks, device=device)

    # Get top-k indices for each q block
    _, q2k_block_sparse_index = torch.topk(scores, k, dim=-1)
    q2k_block_sparse_index = q2k_block_sparse_index.to(torch.int32)

    # sort q2k_block_sparse_index
    q2k_block_sparse_index, _ = torch.sort(q2k_block_sparse_index, dim=-1)

    # All q blocks attend to exactly k kv blocks
    q2k_block_sparse_num = torch.full(
        (bs, h, num_q_blocks), k, dtype=torch.int32, device=device
    )

    # Create the corresponding mask
    block_sparse_mask = torch.zeros(
        bs, h, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )

    # Fill in the mask based on the indices
    for b in range(bs):
        for head in range(h):
            for q_idx in range(num_q_blocks):
                kv_indices = q2k_block_sparse_index[b, head, q_idx]
                block_sparse_mask[b, head, q_idx, kv_indices] = True

    # Create the reverse mapping (k2q)
    # First, initialize lists to collect q indices for each kv block
    k2q_indices_list = [[[] for _ in range(num_kv_blocks)] for _ in range(bs * h)]

    # Populate the lists based on q2k mapping
    for b in range(bs):
        for head in range(h):
            flat_idx = b * h + head
            for q_idx in range(num_q_blocks):
                kv_indices = q2k_block_sparse_index[b, head, q_idx].tolist()
                for kv_idx in kv_indices:
                    k2q_indices_list[flat_idx][kv_idx].append(q_idx)

    # Find the maximum number of q blocks that attend to any kv block
    max_q_per_kv = 0
    for flat_idx in range(bs * h):
        for kv_idx in range(num_kv_blocks):
            max_q_per_kv = max(max_q_per_kv, len(k2q_indices_list[flat_idx][kv_idx]))

    # Create tensors for k2q mapping
    k2q_block_sparse_index = torch.full(
        (bs, h, num_kv_blocks, max_q_per_kv), -1, dtype=torch.int32, device=device
    )
    k2q_block_sparse_num = torch.zeros(
        (bs, h, num_kv_blocks), dtype=torch.int32, device=device
    )

    # Fill the tensors
    for b in range(bs):
        for head in range(h):
            flat_idx = b * h + head
            for kv_idx in range(num_kv_blocks):
                q_indices = k2q_indices_list[flat_idx][kv_idx]
                num_q = len(q_indices)
                k2q_block_sparse_num[b, head, kv_idx] = num_q
                if num_q > 0:
                    k2q_block_sparse_index[b, head, kv_idx, :num_q] = torch.tensor(
                        q_indices, dtype=torch.int32, device=device
                    )

    return (
        q2k_block_sparse_index,
        q2k_block_sparse_num,
        k2q_block_sparse_index,
        k2q_block_sparse_num,
        block_sparse_mask,
    )


def benchmark_block_sparse_attention(
    q,
    k,
    v,
    q2k_block_sparse_index,
    q2k_block_sparse_num,
    k2q_block_sparse_index,
    k2q_block_sparse_num,
    flops,
):
    """Benchmark block sparse attention forward and backward passes."""
    print("\n=== BLOCK SPARSE ATTENTION BENCHMARK ===")

    # Forward pass
    # Warm-up run
    variable_block_sizes = (
        torch.ones(q2k_block_sparse_index.shape[2], device=q.device).int() * BLOCK_M
    )
    o, l_vec = block_sparse_fwd(
        q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, variable_block_sizes
    )
    torch.cuda.synchronize()

    # Benchmark forward
    fwd_time = do_bench(
        lambda: block_sparse_fwd(
            q, k, v, q2k_block_sparse_index, q2k_block_sparse_num, variable_block_sizes
        ),
        warmup=5,
        rep=20,
        quantiles=None,
    )

    sparse_tflops = flops / fwd_time * 1e-12 * 1e3
    print(f"Block Sparse Forward - TFLOPS: {sparse_tflops:.2f}")

    # Backward pass
    grad_output = torch.randn_like(o)

    # Warm-up runs
    for _ in range(5):
        block_sparse_bwd(
            q,
            k,
            v,
            o,
            l_vec,
            grad_output,
            k2q_block_sparse_index,
            k2q_block_sparse_num,
            variable_block_sizes,
        )
    torch.cuda.synchronize()

    # Benchmark backward
    bwd_time = do_bench(
        lambda: block_sparse_bwd(
            q,
            k,
            v,
            o,
            l_vec,
            grad_output,
            k2q_block_sparse_index,
            k2q_block_sparse_num,
            variable_block_sizes,
        ),
        warmup=5,
        rep=20,
        quantiles=None,
    )
    bwd_flops = 2.5 * flops  # Approximation

    sparse_bwd_tflops = bwd_flops / bwd_time * 1e-12 * 1e3
    print(f"Block Sparse Backward - TFLOPS: {sparse_bwd_tflops:.2f}")

    return sparse_tflops, sparse_bwd_tflops


def main():
    args = parse_arguments()

    set_seed(42)

    # Extract parameters
    batch = args.batch_size
    head = args.num_heads
    headdim = args.head_dim

    print(f"Block Sparse Attention Benchmark")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}")

    # Test with different sequence lengths
    for seq_len in args.seq_lengths:
        # Skip very long sequences if they might cause OOM
        if seq_len > 16384 and batch > 1:
            continue

        print("=" * 100)
        print(f"\nSequence length: {seq_len}")

        # Calculate theoretical FLOPs for attention
        flops = 4 * batch * head * headdim * seq_len * seq_len

        # Create input tensors
        q, k, v = create_input_tensors(batch, head, seq_len, headdim)

        # Setup block sparse parameters
        num_q_blocks = seq_len // BLOCK_M
        num_kv_blocks = seq_len // BLOCK_N

        # Determine k value (number of kv blocks per q block)
        topk = args.topk
        if topk is None:
            topk = num_kv_blocks // 10  # Default to ~90% sparsity if k is not specified
        topk = max(1, topk)
        print(
            f"Using topk={topk} kv blocks per q block (out of {num_kv_blocks} total kv blocks)"
        )

        # Generate block sparse pattern
        (
            q2k_block_sparse_index,
            q2k_block_sparse_num,
            k2q_block_sparse_index,
            k2q_block_sparse_num,
            _,
        ) = generate_block_sparse_pattern(
            batch, head, num_q_blocks, num_kv_blocks, topk, device="cuda"
        )

        # Benchmark block sparse attention
        sparse_fwd, sparse_bwd = benchmark_block_sparse_attention(
            q,
            k,
            v,
            q2k_block_sparse_index,
            q2k_block_sparse_num,
            k2q_block_sparse_index,
            k2q_block_sparse_num,
            flops,
        )

        # Print results
        print("\n=== PERFORMANCE RESULTS ===")
        print(f"Block Sparse Forward - TFLOPS: {sparse_fwd:.2f}")
        print(f"Block Sparse Backward - TFLOPS: {sparse_bwd:.2f}")


if __name__ == "__main__":
    main()
