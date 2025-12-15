"""
Standalone benchmark comparing CuTe vs Triton implementations of pad/unpad kernels.

Usage:
    python benchmark/bench_cute_vs_triton_pad_unpad.py --max-seq-len 1024 --num-heads 16 --head-dim 128
"""

from __future__ import annotations

import argparse

import torch
import triton
from triton.testing import do_bench

from sglang.srt.layers.attention.cute_ops import (
    CutePadDraftExtendQueryKernel,
    CuteUnpadDraftExtendOutputKernel,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    pad_draft_extend_query_kernel,
    unpad_draft_extend_output_kernel,
)

try:
    import cutlass.cute.testing as cute_testing  # type: ignore
except Exception:
    cute_testing = None


def _bench_ms(
    *,
    timer: str,
    fn,
    warmup: int,
    iters: int,
    workspace_generator=None,
    workspace_count: int = 1,
    use_cuda_graphs: bool = False,
) -> float:
    """
    Returns time in milliseconds.

    timer="triton": uses triton.testing.do_bench (CUDA events)
    timer="cute": uses cutlass.cute.testing.benchmark (CUDA events + workspace cycling + optional CUDA graphs)
    """
    if timer == "cute":
        if cute_testing is None:
            raise RuntimeError(
                "cutlass.cute.testing is not available; use --timer triton"
            )
        if workspace_generator is None:
            us = cute_testing.benchmark(
                fn,
                warmup_iterations=warmup,
                iterations=iters,
                use_cuda_graphs=use_cuda_graphs,
            )
        else:
            us = cute_testing.benchmark(
                fn,
                warmup_iterations=warmup,
                iterations=iters,
                workspace_generator=workspace_generator,
                workspace_count=workspace_count,
                use_cuda_graphs=use_cuda_graphs,
            )
        return float(us) / 1000.0

    # timer == "triton": do_bench expects a zero-arg callable.
    if workspace_generator is None or workspace_count <= 1:
        # First call triggers compilation (Triton/CuTe). Exclude it from timing.
        fn()
        torch.cuda.synchronize()
        return float(do_bench(fn, warmup=warmup, rep=iters))

    # Workspace cycling to reduce L2 effects.
    workspaces = [workspace_generator() for _ in range(workspace_count)]
    idx = {"i": 0}

    def run():
        ws = workspaces[idx["i"]]
        idx["i"] = (idx["i"] + 1) % workspace_count
        fn(*ws)

    run()
    torch.cuda.synchronize()
    return float(do_bench(run, warmup=warmup, rep=iters))


def _make_pad_inputs(
    batch_size: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    fill_ratio: float,
    device: str = "cuda",
):
    """Create inputs for pad kernel benchmark."""
    if fill_ratio >= 1.0:
        seq_lens = torch.full(
            (batch_size,), max_seq_len, device=device, dtype=torch.int32
        )
    else:
        fixed_len = max(1, int(max_seq_len * fill_ratio))
        seq_lens = torch.full(
            (batch_size,), fixed_len, device=device, dtype=torch.int32
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
    fill_ratio: float,
    device: str = "cuda",
):
    """Create inputs for unpad kernel benchmark."""
    if fill_ratio >= 1.0:
        accept_lengths = torch.full(
            (batch_size,), token_per_batch, device=device, dtype=torch.int32
        )
    else:
        fixed_len = max(1, int(token_per_batch * fill_ratio))
        accept_lengths = torch.full(
            (batch_size,), fixed_len, device=device, dtype=torch.int32
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
    fill_ratio: float,
    warmup: int,
    iters: int,
    timer: str = "triton",
    workspace_count: int = 1,
    use_cuda_graphs: bool = False,
):
    """Benchmark Triton pad kernel."""
    BLOCK_SIZE = 64
    num_head_blocks = triton.cdiv(num_heads, BLOCK_SIZE)
    num_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)
    grid = (batch_size * max_seq_len, num_head_blocks, num_dim_blocks)

    def kernel(q, padded_q, seq_lens, cumsum):
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

    def workspace():
        return _make_pad_inputs(
            batch_size, max_seq_len, num_heads, head_dim, dtype, fill_ratio
        )

    q0, padded0, seq0, cumsum0 = workspace()
    active_tokens = int(cumsum0[-1].item())
    elems = active_tokens * num_heads * head_dim
    gb = (2 * elems * torch.tensor([], dtype=dtype).element_size()) / 1e9

    if timer == "cute":

        def ws_gen():
            q, p, s, c = workspace()
            return cute_testing.JitArguments(q, p, s, c)

        ms = _bench_ms(
            timer=timer,
            fn=kernel,
            warmup=warmup,
            iters=iters,
            workspace_generator=ws_gen,
            workspace_count=workspace_count,
            use_cuda_graphs=use_cuda_graphs,
        )
    else:
        ms = _bench_ms(
            timer=timer,
            fn=lambda: kernel(q0, padded0, seq0, cumsum0),
            warmup=warmup,
            iters=iters,
            workspace_generator=workspace,
            workspace_count=workspace_count,
            use_cuda_graphs=use_cuda_graphs,
        )
    return ms, gb


def benchmark_pad_cute(
    pad_kernel: CutePadDraftExtendQueryKernel,
    batch_size: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    fill_ratio: float,
    warmup: int,
    iters: int,
    timer: str = "triton",
    workspace_count: int = 1,
    use_cuda_graphs: bool = False,
):
    """Benchmark CuTe pad kernel."""

    def kernel(q, padded_q, seq_lens, cumsum):
        pad_kernel(q=q, padded_q=padded_q, seq_lens_q=seq_lens, cumsum=cumsum)

    def workspace():
        return _make_pad_inputs(
            batch_size, max_seq_len, num_heads, head_dim, dtype, fill_ratio
        )

    q0, padded0, seq0, cumsum0 = workspace()
    active_tokens = int(cumsum0[-1].item())
    elems = active_tokens * num_heads * head_dim
    gb = (2 * elems * torch.tensor([], dtype=dtype).element_size()) / 1e9

    if timer == "cute":

        def ws_gen():
            q, p, s, c = workspace()
            return cute_testing.JitArguments(q, p, s, c)

        ms = _bench_ms(
            timer=timer,
            fn=kernel,
            warmup=warmup,
            iters=iters,
            workspace_generator=ws_gen,
            workspace_count=workspace_count,
            use_cuda_graphs=use_cuda_graphs,
        )
    else:
        ms = _bench_ms(
            timer=timer,
            fn=lambda: kernel(q0, padded0, seq0, cumsum0),
            warmup=warmup,
            iters=iters,
            workspace_generator=workspace,
            workspace_count=workspace_count,
            use_cuda_graphs=use_cuda_graphs,
        )
    return ms, gb


def benchmark_unpad_triton(
    batch_size: int,
    token_per_batch: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    fill_ratio: float,
    warmup: int,
    iters: int,
    timer: str = "triton",
    workspace_count: int = 1,
    use_cuda_graphs: bool = False,
):
    """Benchmark Triton unpad kernel."""
    BLOCK_SIZE = 64
    num_head_blocks = triton.cdiv(num_heads, BLOCK_SIZE)
    num_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)
    grid = (batch_size * token_per_batch, num_head_blocks, num_dim_blocks)

    def kernel(raw_out, output, accept_lengths, cumsum):
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

    def workspace():
        return _make_unpad_inputs(
            batch_size, token_per_batch, num_heads, head_dim, dtype, fill_ratio
        )

    raw0, out0, acc0, cumsum0 = workspace()
    active_tokens = int(cumsum0[-1].item())
    elems = active_tokens * num_heads * head_dim
    gb = (2 * elems * torch.tensor([], dtype=dtype).element_size()) / 1e9

    if timer == "cute":

        def ws_gen():
            r, o, a, c = workspace()
            return cute_testing.JitArguments(r, o, a, c)

        ms = _bench_ms(
            timer=timer,
            fn=kernel,
            warmup=warmup,
            iters=iters,
            workspace_generator=ws_gen,
            workspace_count=workspace_count,
            use_cuda_graphs=use_cuda_graphs,
        )
    else:
        ms = _bench_ms(
            timer=timer,
            fn=lambda: kernel(raw0, out0, acc0, cumsum0),
            warmup=warmup,
            iters=iters,
            workspace_generator=workspace,
            workspace_count=workspace_count,
            use_cuda_graphs=use_cuda_graphs,
        )
    return ms, gb


def benchmark_unpad_cute(
    unpad_kernel: CuteUnpadDraftExtendOutputKernel,
    batch_size: int,
    token_per_batch: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    fill_ratio: float,
    warmup: int,
    iters: int,
    timer: str = "triton",
    workspace_count: int = 1,
    use_cuda_graphs: bool = False,
):
    """Benchmark CuTe unpad kernel."""

    def kernel(raw_out, output, accept_lengths, cumsum):
        unpad_kernel(
            raw_out=raw_out,
            output=output,
            accept_lengths=accept_lengths,
            cumsum=cumsum,
        )

    def workspace():
        return _make_unpad_inputs(
            batch_size, token_per_batch, num_heads, head_dim, dtype, fill_ratio
        )

    raw0, out0, acc0, cumsum0 = workspace()
    active_tokens = int(cumsum0[-1].item())
    elems = active_tokens * num_heads * head_dim
    gb = (2 * elems * torch.tensor([], dtype=dtype).element_size()) / 1e9

    if timer == "cute":

        def ws_gen():
            r, o, a, c = workspace()
            return cute_testing.JitArguments(r, o, a, c)

        ms = _bench_ms(
            timer=timer,
            fn=kernel,
            warmup=warmup,
            iters=iters,
            workspace_generator=ws_gen,
            workspace_count=workspace_count,
            use_cuda_graphs=use_cuda_graphs,
        )
    else:
        ms = _bench_ms(
            timer=timer,
            fn=lambda: kernel(raw0, out0, acc0, cumsum0),
            warmup=warmup,
            iters=iters,
            workspace_generator=workspace,
            workspace_count=workspace_count,
            use_cuda_graphs=use_cuda_graphs,
        )
    return ms, gb


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
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--timer",
        type=str,
        default="triton",
        choices=["triton", "cute"],
        help="Timing backend: triton.do_bench or cutlass.cute.testing.benchmark",
    )
    parser.add_argument(
        "--workspace-count",
        type=int,
        default=1,
        help="Cycle through N workspaces to reduce L2 effects.",
    )
    parser.add_argument(
        "--use-cuda-graphs",
        action="store_true",
        help="Use CUDA graphs in cutlass.cute.testing.benchmark (timer=cute).",
    )
    parser.add_argument(
        "--fill-ratio",
        type=float,
        default=1.0,
        help="0<r<=1.0. r=1 uses full-length (no early-exit).",
    )
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
        f"dtype={args.dtype}, warmup={args.warmup}, iters={args.iters}, "
        f"fill_ratio={args.fill_ratio}, timer={args.timer}, "
        f"workspace_count={args.workspace_count}, cuda_graphs={args.use_cuda_graphs}"
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

    # Reuse CuTe kernel objects so we don't recompile every batch size.
    pad_kernel = CutePadDraftExtendQueryKernel()
    unpad_kernel = CuteUnpadDraftExtendOutputKernel()

    # Benchmark pad kernel
    print("=" * 60)
    print("PAD KERNEL BENCHMARK")
    print("=" * 60)
    print(
        f"{'batch_size':>12} | {'Triton (ms)':>12} | {'CuTe (ms)':>12} | {'Speedup':>10} | {'T GB/s':>8} | {'C GB/s':>8}"
    )
    print("-" * 60)
    with torch.inference_mode():
        for batch_size in batch_sizes:
            try:
                t_ms, t_gb = benchmark_pad_triton(
                    batch_size,
                    args.max_seq_len,
                    args.num_heads,
                    args.head_dim,
                    dtype,
                    args.fill_ratio,
                    args.warmup,
                    args.iters,
                    timer=args.timer,
                    workspace_count=args.workspace_count,
                    use_cuda_graphs=args.use_cuda_graphs,
                )
                c_ms, c_gb = benchmark_pad_cute(
                    pad_kernel,
                    batch_size,
                    args.max_seq_len,
                    args.num_heads,
                    args.head_dim,
                    dtype,
                    args.fill_ratio,
                    args.warmup,
                    args.iters,
                    timer=args.timer,
                    workspace_count=args.workspace_count,
                    use_cuda_graphs=args.use_cuda_graphs,
                )
                speedup = t_ms / c_ms if c_ms > 0 else 0.0
                t_bw = t_gb / (t_ms / 1e3) if t_ms > 0 else 0.0
                c_bw = c_gb / (c_ms / 1e3) if c_ms > 0 else 0.0
                print(
                    f"{batch_size:12d} | {t_ms:12.3f} | {c_ms:12.3f} | {speedup:10.2f}x | {t_bw:8.1f} | {c_bw:8.1f}"
                )
            except Exception as e:
                print(f"{batch_size:12d} | ERROR: {e}")

    print()
    print("=" * 60)
    print("UNPAD KERNEL BENCHMARK")
    print("=" * 60)
    print(
        f"{'batch_size':>12} | {'Triton (ms)':>12} | {'CuTe (ms)':>12} | {'Speedup':>10} | {'T GB/s':>8} | {'C GB/s':>8}"
    )
    print("-" * 60)
    with torch.inference_mode():
        for batch_size in batch_sizes:
            try:
                t_ms, t_gb = benchmark_unpad_triton(
                    batch_size,
                    args.max_seq_len,
                    args.num_heads,
                    args.head_dim,
                    dtype,
                    args.fill_ratio,
                    args.warmup,
                    args.iters,
                    timer=args.timer,
                    workspace_count=args.workspace_count,
                    use_cuda_graphs=args.use_cuda_graphs,
                )
                c_ms, c_gb = benchmark_unpad_cute(
                    unpad_kernel,
                    batch_size,
                    args.max_seq_len,
                    args.num_heads,
                    args.head_dim,
                    dtype,
                    args.fill_ratio,
                    args.warmup,
                    args.iters,
                    timer=args.timer,
                    workspace_count=args.workspace_count,
                    use_cuda_graphs=args.use_cuda_graphs,
                )
                speedup = t_ms / c_ms if c_ms > 0 else 0.0
                t_bw = t_gb / (t_ms / 1e3) if t_ms > 0 else 0.0
                c_bw = c_gb / (c_ms / 1e3) if c_ms > 0 else 0.0
                print(
                    f"{batch_size:12d} | {t_ms:12.3f} | {c_ms:12.3f} | {speedup:10.2f}x | {t_bw:8.1f} | {c_bw:8.1f}"
                )
            except Exception as e:
                print(f"{batch_size:12d} | ERROR: {e}")


if __name__ == "__main__":
    main()
