"""Benchmark for embedding_lora_a kernel.

Measures kernel latency across different batch sizes, sequence lengths, and
LoRA ranks. Uses ``triton.testing.perf_report`` for consistent output that
matches the style of jit_kernel benchmarks.

Usage:
    python benchmark/lora/bench_embedding_lora_a.py
"""

import itertools

import torch
import triton
import triton.testing

from sglang.srt.lora.triton_ops.embedding_lora_a import embedding_lora_a_fwd
from sglang.srt.lora.utils import LoRABatchInfo

# ==============================================================================
# Constants
# ==============================================================================
DEVICE = "cuda"
DTYPE = torch.float16
VOCAB_SIZE = 32000
NUM_LORAS = 4
DEFAULT_QUANTILES = [0.5, 0.2, 0.8]

# ==============================================================================
# Parameter ranges
# ==============================================================================
BS_LIST = [1, 8, 32]
SEQ_LEN_LIST = [1, 128, 512]
RANK_LIST = [16, 64, 128, 256]
CONFIGS = list(itertools.product(BS_LIST, SEQ_LEN_LIST, RANK_LIST))


# ==============================================================================
# Helpers
# ==============================================================================
def create_batch_info(
    batch_size: int, seq_len: int, num_loras: int, rank: int
) -> LoRABatchInfo:
    """Build a LoRABatchInfo for the triton (non-chunked) backend."""
    total_tokens = batch_size * seq_len
    seg_lens = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=DEVICE
    )
    seg_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=batch_size,
        num_segments=batch_size,
        seg_indptr=seg_indptr,
        weight_indices=torch.randint(
            0, num_loras, (batch_size,), dtype=torch.int32, device=DEVICE
        ),
        lora_ranks=torch.full(
            (num_loras,), rank, dtype=torch.int32, device=DEVICE
        ),
        scalings=torch.ones(num_loras, dtype=torch.float32, device=DEVICE),
        max_len=seq_len,
        seg_lens=seg_lens,
        permutation=torch.arange(
            total_tokens, dtype=torch.int32, device=DEVICE
        ),
    )


def run_benchmark(fn, scale=1.0):
    """Execute benchmark via ``triton.testing.do_bench`` and return µs."""
    ms, min_ms, max_ms = triton.testing.do_bench(
        fn, quantiles=DEFAULT_QUANTILES
    )
    return 1000 * ms / scale, 1000 * max_ms / scale, 1000 * min_ms / scale


# ==============================================================================
# Benchmark — embedding_lora_a kernel
# ==============================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "rank"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["embedding_lora_a (triton)"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="embedding-lora-a",
        args={},
    )
)
def benchmark(batch_size: int, seq_len: int, rank: int, provider: str):
    total_tokens = batch_size * seq_len
    input_ids = torch.randint(
        0, VOCAB_SIZE, (total_tokens,), dtype=torch.int64, device=DEVICE
    )
    weights = torch.randn(
        NUM_LORAS, rank, VOCAB_SIZE, dtype=DTYPE, device=DEVICE
    )
    batch_info = create_batch_info(batch_size, seq_len, NUM_LORAS, rank)

    def fn():
        embedding_lora_a_fwd(input_ids, weights, batch_info, VOCAB_SIZE)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True) 