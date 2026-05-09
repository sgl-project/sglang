"""
Benchmark for triton-backend sgemm LoRA kernels (sgemm_lora_a_fwd / sgemm_lora_b_fwd).

Uses triton.testing.perf_report — zero arguments, just run:

    PYTHONPATH=python python python/sglang/jit_kernel/benchmark/bench_sgemm_lora.py

Style follows python/sglang/jit_kernel/benchmark/bench_norm.py.

Three benchmarks:
  1. LoRA-A (shrink)         — single adapter, Triton kernel
  2. LoRA-B single adapter   — bs=1, one adapter, Triton kernel
  3. LoRA-B multi adapter    — bs=2, two adapters, Triton kernel
"""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import run_benchmark_no_cudagraph
from sglang.srt.lora.triton_ops.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.srt.lora.triton_ops.sgemm_lora_b import sgemm_lora_b_fwd
from sglang.srt.lora.utils import LoRABatchInfo

DTYPE = torch.float16
DEVICE = "cuda"

# Sweep axes — realistic LoRA shapes
RANK_LIST = [16, 32, 64]
HIDDEN_DIM_LIST = [4096, 8192]
NUM_TOKENS_LIST = [1, 8, 64, 256, 1024]

CONFIGS = list(itertools.product(HIDDEN_DIM_LIST, RANK_LIST, NUM_TOKENS_LIST))


# ---------------------------------------------------------------------------
# Helpers: build LoRABatchInfo for different scenarios
# ---------------------------------------------------------------------------
def _build_single_adapter_batch(
    num_tokens: int, rank: int, device: str = DEVICE
) -> LoRABatchInfo:
    """bs=1, single adapter — Triton kernel with HAS_BASE_OUTPUT=False."""
    seg_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    seg_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    weight_indices = torch.zeros(1, dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([rank], dtype=torch.int32, device=device)
    scalings = torch.tensor([1.0], dtype=torch.float32, device=device)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        max_len=num_tokens,
        seg_lens=seg_lens,
        permutation=None,
    )


def _build_multi_adapter_batch(
    num_tokens: int, rank: int, device: str = DEVICE
) -> LoRABatchInfo:
    """bs=2 with 2 different adapters — forces the Triton kernel path."""
    half = num_tokens // 2
    remainder = num_tokens - half
    seg_lens = torch.tensor([half, remainder], dtype=torch.int32, device=device)
    seg_indptr = torch.tensor([0, half, num_tokens], dtype=torch.int32, device=device)
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([rank, rank], dtype=torch.int32, device=device)
    scalings = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=2,
        num_segments=2,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        max_len=max(half, remainder),
        seg_lens=seg_lens,
        permutation=None,
    )


# ---------------------------------------------------------------------------
# LoRA-A benchmark (shrink: x @ A^T)
# ---------------------------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_dim", "rank", "num_tokens"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=["triton_sgemm", "torch_ref"],
        line_names=["sgemm_lora_a (Triton)", "torch.matmul reference"],
        styles=[("green", "-"), ("blue", "--")],
        ylabel="us",
        plot_name="sgemm-lora-a-performance",
        args={},
    )
)
def bench_lora_a(hidden_dim, rank, num_tokens, provider):
    batch_info = _build_single_adapter_batch(num_tokens, rank)
    x = torch.randn(num_tokens, hidden_dim, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(1, rank, hidden_dim, dtype=DTYPE, device=DEVICE) * 0.02

    if provider == "triton_sgemm":
        fn = lambda: sgemm_lora_a_fwd(x, weights, batch_info, stack_num=1)
    else:
        w = weights[0]
        fn = lambda: x @ w.T

    return run_benchmark_no_cudagraph(fn)


# ---------------------------------------------------------------------------
# LoRA-B single adapter (bs=1, Triton kernel with HAS_BASE_OUTPUT=False)
# ---------------------------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_dim", "rank", "num_tokens"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=["sgemm_fwd", "torch_ref"],
        line_names=["sgemm_lora_b single (Triton)", "torch.matmul reference"],
        styles=[("green", "-"), ("blue", "--")],
        ylabel="us",
        plot_name="sgemm-lora-b-single-adapter",
        args={},
    )
)
def bench_lora_b_single(hidden_dim, rank, num_tokens, provider):
    batch_info = _build_single_adapter_batch(num_tokens, rank)
    x = torch.randn(num_tokens, rank, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(1, hidden_dim, rank, dtype=DTYPE, device=DEVICE) * 0.02

    if provider == "sgemm_fwd":
        fn = lambda: sgemm_lora_b_fwd(x, weights, batch_info)
    else:
        w = weights[0]
        fn = lambda: x @ w.T

    return run_benchmark_no_cudagraph(fn)


# ---------------------------------------------------------------------------
# LoRA-B multi adapter (forces Triton kernel, tests P1/P2/P3)
# ---------------------------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_dim", "rank", "num_tokens"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=["triton_sgemm", "torch_ref"],
        line_names=["sgemm_lora_b multi (Triton)", "torch.matmul reference"],
        styles=[("green", "-"), ("blue", "--")],
        ylabel="us",
        plot_name="sgemm-lora-b-multi-adapter",
        args={},
    )
)
def bench_lora_b_multi(hidden_dim, rank, num_tokens, provider):
    batch_info = _build_multi_adapter_batch(num_tokens, rank)
    total_tokens = num_tokens
    x = torch.randn(total_tokens, rank, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(2, hidden_dim, rank, dtype=DTYPE, device=DEVICE) * 0.02

    if provider == "triton_sgemm":
        fn = lambda: sgemm_lora_b_fwd(x, weights, batch_info)
    else:
        # Reference: concatenated matmul for both halves
        half = num_tokens // 2
        w0, w1 = weights[0], weights[1]
        fn = lambda: torch.cat([x[:half] @ w0.T, x[half:] @ w1.T], dim=0)

    return run_benchmark_no_cudagraph(fn)


if __name__ == "__main__":
    print("Benchmarking sgemm_lora_a (shrink)...")
    bench_lora_a.run(print_data=True)

    print("\nBenchmarking sgemm_lora_b single adapter (cuBLAS fast-path)...")
    bench_lora_b_single.run(print_data=True)

    print("\nBenchmarking sgemm_lora_b multi adapter (Triton kernel)...")
    bench_lora_b_multi.run(print_data=True)