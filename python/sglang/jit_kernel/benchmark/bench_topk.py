"""Benchmark for the DeepSeek-V4 (DSA indexer) top-k transform kernels.

Providers:
  - jit_v1      : the JIT radix kernel (`topk_transform_512`, fixed K=512/1024)
  - jit_v2      : the JIT register/streaming/cluster kernel (`topk_transform_512_v2`)
  - flashinfer  : `flashinfer.top_k` (plain top-k, no page-table transform)
  - torch       : `torch.topk` (plain top-k, no page-table transform)

The JIT kernels perform top-k AND a page-table transform over an arbitrary page
size + page table; the flashinfer / torch baselines only do a *naive* top-k (they
do not support arbitrary page size + page table), so they are included purely as a
memory-bandwidth reference, as requested.

Run:
    CUDA_VISIBLE_DEVICES=7 python -m sglang.jit_kernel.benchmark.bench_topk
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.dsv4.topk import (
    plan_topk_v2,
    topk_transform_512,
    topk_transform_512_v2,
)

# Compressed page size used by the DSA indexer (real value is 256 // 4 = 64).
PAGE_SIZE = 64


def _make_inputs(batch_size: int, seq_len: int, k: int):
    """Build the (scores, seq_lens, page_table, out) tuple shared by both JIT paths."""
    torch.random.manual_seed(42)
    scores = torch.randn(batch_size, seq_len, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    # Identity-ish page table: page p -> p, so page_to_indices(i) == i. Keeps the
    # access pattern realistic (one int32 / page) without changing the top-k set.
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    out = torch.empty(batch_size, k, dtype=torch.int32, device="cuda")
    return scores, seq_lens, page_table, out


def _build_fn(provider: str, batch_size: int, seq_len: int, k: int):
    scores, seq_lens, page_table, out = _make_inputs(batch_size, seq_len, k)

    if provider == "jit_v1":

        def fn(scores):
            topk_transform_512(scores, seq_lens, page_table, out, PAGE_SIZE)
            return out

        return fn, scores, (out,)

    if provider == "jit_v2":
        metadata = plan_topk_v2(seq_lens)  # amortized once per metadata init

        def fn(scores):
            topk_transform_512_v2(
                scores, seq_lens, page_table, out, PAGE_SIZE, metadata
            )
            return out

        return fn, scores, (out,)

    if provider == "flashinfer":
        import flashinfer

        def fn(scores):
            return flashinfer.top_k(scores, k)[1]

        return fn, scores, "out"

    if provider == "torch":

        def fn(scores):
            return scores.topk(k, dim=-1).indices

        return fn, scores, "out"

    raise ValueError(f"unknown provider: {provider}")


@marker.parametrize(
    "seq_len",
    [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144],
    [4096, 65536],
)
@marker.parametrize("batch_size", [2**x for x in range(14)], [1, 128, 1024])
@marker.parametrize("k", [512, 1024], [512])
@marker.benchmark("provider", ["jit_v1", "jit_v2", "flashinfer", "torch"])
def benchmark(seq_len: int, batch_size: int, k: int, provider: str):
    if k > seq_len:
        marker.skip("k cannot be larger than seq_len")
    # Bound the scores tensor so the cuda-graph buffer rotation (~4x clone) stays
    # well within HBM. 4 GiB still covers the target regime (batch=1024, ctx=256K).
    if batch_size * seq_len * 4 > (4 << 30):
        marker.skip("input too large (>4GB)")

    fn, scores, mem_out = _build_fn(provider, batch_size, seq_len, k)
    return marker.do_bench(fn, input_args=(scores,), memory_output=mem_out)


if __name__ == "__main__":
    benchmark.run()
