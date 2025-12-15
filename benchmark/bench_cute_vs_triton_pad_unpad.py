"""
Standalone benchmark comparing CuTe vs Triton implementations of pad/unpad kernels.

Usage:
    python bench_cute_vs_triton_pad_unpad.py [--max-seq-len 1024] [--num-heads 16] [--head-dim 128] [--dtype bfloat16] [--repeats 10]
"""

from __future__ import annotations

import argparse
import time

import torch
import triton

from sglang.srt.layers.attention.cute_ops import (
    CutePadDraftExtendQueryKernel,
    CuteUnpadDraftExtendOutputKernel,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    pad_draft_extend_query_kernel,
    unpad_draft_extend_output_kernel,
)


def _cuda_sync():
    """Synchronize CUDA operations."""
    torch.cuda.synchronize()


def _time(fn, repeats: int = 10, warmup: int = 3) -> float:
    """Time a function execution, averaging over repeats."""
    # Warmup
    for _ in range(warmup):
        fn()
    _cuda_sync()

    # Actual timing
    times = []
    for _ in range(repeats):
        _cuda_sync()
        t0 = time.time()
        fn()
        _cuda_sync()
        times.append(time.time() - t0)
    return sum(times) / len(times)


def _make_pad_inputs(
    batch_size: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str = "cuda",
):
    """Create inputs for pad kernel benchmark."""
    seq_lens = torch.randint(
        1, max_seq_len + 1, (batch_size,), device=device, dtype=torch.int32
    )
    cumsum = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cumsum[1:] = torch.cumsum(seq_lens, dim=0)
    q = torch.randn(int(cumsum[-1]), num_heads, head_dim, device=device, dtype=dtype)
    padded_q = torch.zeros(
        batch_size, max_seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    return q, padded_q, seq_lens, cumsum


def _make_unpad_inputs(
    batch_size: int,
    token_per_batch: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str = "cuda",
):
    """Create inputs for unpad kernel benchmark."""
    accept_lengths = torch.randint(
        1, token_per_batch + 1, (batch_size,), device=device, dtype=torch.int32
    )
    cumsum = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cumsum[1:] = torch.cumsum(accept_lengths, dim=0)
    raw_out = torch.randn(
        batch_size, token_per_batch, num_heads, head_dim, device=device, dtype=dtype
    )
    output = torch.empty(
        int(cumsum[-1]), num_heads, head_dim, device=device, dtype=dtype
    )
    return raw_out, output, accept_lengths, cumsum


def benchmark_pad_triton(
    batch_size: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    repeats: int,
):
    """Benchmark Triton pad kernel."""
    q, padded_q, seq_lens, cumsum = _make_pad_inputs(
        batch_size, max_seq_len, num_heads, head_dim, dtype
    )
    BLOCK_SIZE = 64
    num_head_blocks = triton.cdiv(num_heads, BLOCK_SIZE)
    num_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)
    grid = (batch_size * max_seq_len, num_head_blocks, num_dim_blocks)

    def run():
        pad_draft_extend_query_kernel[grid](
            q_ptr=q,
            padded_q_ptr=padded_q,
            seq_lens_q_ptr=seq_lens,
            cumsum_ptr=cumsum,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return _time(run, repeats)


def benchmark_pad_cute(
    batch_size: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    repeats: int,
):
    """Benchmark CuTe pad kernel."""
    pad_kernel = CutePadDraftExtendQueryKernel()
    q, padded_q, seq_lens, cumsum = _make_pad_inputs(
        batch_size, max_seq_len, num_heads, head_dim, dtype
    )

    def run():
        pad_kernel(q=q, padded_q=padded_q, seq_lens_q=seq_lens, cumsum=cumsum)

    return _time(run, repeats)


def benchmark_unpad_triton(
    batch_size: int,
    token_per_batch: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    repeats: int,
):
    """Benchmark Triton unpad kernel."""
    raw_out, output, accept_lengths, cumsum = _make_unpad_inputs(
        batch_size, token_per_batch, num_heads, head_dim, dtype
    )
    BLOCK_SIZE = 64
    num_head_blocks = triton.cdiv(num_heads, BLOCK_SIZE)
    num_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)
    grid = (batch_size * token_per_batch, num_head_blocks, num_dim_blocks)

    def run():
        unpad_draft_extend_output_kernel[grid](
            raw_out_ptr=raw_out,
            output_ptr=output,
            accept_length_ptr=accept_lengths,
            cumsum_ptr=cumsum,
            batch_size=batch_size,
            token_per_batch=token_per_batch,
            tp_q_head_num=num_heads,
            v_head_dim=head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return _time(run, repeats)


def benchmark_unpad_cute(
    batch_size: int,
    token_per_batch: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    repeats: int,
):
    """Benchmark CuTe unpad kernel."""
    unpad_kernel = CuteUnpadDraftExtendOutputKernel()
    raw_out, output, accept_lengths, cumsum = _make_unpad_inputs(
        batch_size, token_per_batch, num_heads, head_dim, dtype
    )

    def run():
        unpad_kernel(
            raw_out=raw_out,
            output=output,
            accept_lengths=accept_lengths,
            cumsum=cumsum,
        )

    return _time(run, repeats)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pad/unpad kernels (Triton vs CuTe)."
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes, or 'auto' for auto-generated",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(
        f"Config: max_seq_len={args.max_seq_len}, "
        f"heads={args.num_heads}, head_dim={args.head_dim}, "
        f"dtype={args.dtype}, repeats={args.repeats}"
    )

    # Generate batch sizes
    if args.batch_sizes == "auto" or args.batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        batch_sizes.extend([i for i in range(3, 256, 8) if i not in batch_sizes])
        batch_sizes = sorted(set(batch_sizes))
    else:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print(
        f"Testing {len(batch_sizes)} batch sizes: {batch_sizes[:10]}...{batch_sizes[-5:]}"
    )
    print()

    # Benchmark pad kernel
    print("=" * 60)
    print("PAD KERNEL BENCHMARK")
    print("=" * 60)
    print(
        f"{'batch_size':>12} | {'Triton (ms)':>12} | {'CuTe (ms)':>12} | {'Speedup':>10}"
    )
    print("-" * 60)
    for batch_size in batch_sizes:
        try:
            t_triton = benchmark_pad_triton(
                batch_size,
                args.max_seq_len,
                args.num_heads,
                args.head_dim,
                dtype,
                args.repeats,
            )
            t_cute = benchmark_pad_cute(
                batch_size,
                args.max_seq_len,
                args.num_heads,
                args.head_dim,
                dtype,
                args.repeats,
            )
            speedup = t_triton / t_cute if t_cute > 0 else 0.0
            print(
                f"{batch_size:12d} | {t_triton*1e3:12.3f} | {t_cute*1e3:12.3f} | {speedup:10.2f}x"
            )
        except Exception as e:
            print(f"{batch_size:12d} | ERROR: {e}")

    print()
    print("=" * 60)
    print("UNPAD KERNEL BENCHMARK")
    print("=" * 60)
    print(
        f"{'batch_size':>12} | {'Triton (ms)':>12} | {'CuTe (ms)':>12} | {'Speedup':>10}"
    )
    print("-" * 60)
    for batch_size in batch_sizes:
        try:
            t_triton = benchmark_unpad_triton(
                batch_size,
                args.max_seq_len,
                args.num_heads,
                args.head_dim,
                dtype,
                args.repeats,
            )
            t_cute = benchmark_unpad_cute(
                batch_size,
                args.max_seq_len,
                args.num_heads,
                args.head_dim,
                dtype,
                args.repeats,
            )
            speedup = t_triton / t_cute if t_cute > 0 else 0.0
            print(
                f"{batch_size:12d} | {t_triton*1e3:12.3f} | {t_cute*1e3:12.3f} | {speedup:10.2f}x"
            )
        except Exception as e:
            print(f"{batch_size:12d} | ERROR: {e}")


if __name__ == "__main__":
    main()
