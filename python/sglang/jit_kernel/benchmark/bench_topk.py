"""Benchmark for the DeepSeek-V4 (DSA indexer) top-k transform kernels.

All four providers do top-k selection AND a page-table transform of the selected
indices, so the comparison is apples-to-apples (top-k + transform):
  - jit_v1      : JIT radix kernel `topk_transform_512` (page_size=64)
  - jit_v2      : JIT register/streaming/cluster `topk_transform_512_v2` (page_size=64)
  - flashinfer  : `flashinfer.top_k_page_table_transform` -- a fused top-k + transform.
                  Its API is page_size=1, so it gathers through a per-token table.
  - torch       : `torch.topk` followed by a `gather` (page_size=1).

The JIT kernels support an arbitrary page size and run at the production page_size=64;
flashinfer only supports page_size=1 (a per-token table), so its transform is over a
64x larger table. The page-table tensor is excluded from the reported footprint (only
~top-k of its entries are read per row); bandwidth counts the fully-read scores plus
the written output.

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


def _make_p1_table(batch_size: int, seq_len: int):
    """Per-token (page_size=1) page table + lengths for the flashinfer/torch
    baselines. flashinfer's fused transform only supports page_size=1, so its
    src_page_table is (batch, seq); a 1:1 row->batch mapping (row_to_batch=None)
    keeps it on the same cluster path the JIT kernels are compared against."""
    src_page_table = (
        torch.arange(seq_len, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    return src_page_table, lengths


def _build_fn(provider: str, batch_size: int, seq_len: int, k: int):
    """Return (fn, input_args, memory_args, memory_output).

    All *read* tensors a provider touches go in `input_args` so do_bench rotates
    them across cuda-graph iterations (cold L2) -- including page_table, whose
    same ~topk entries are re-read every iteration and would otherwise stay
    L2-resident. But page_table is deliberately *excluded* from `memory_args`:
    only ~topk of its entries are read per row (not the whole table), so counting
    its full size would overstate the footprint. The reported bandwidth therefore
    counts the fully-read inputs (scores + seq_lens) plus the written output.
    """
    scores, seq_lens, page_table, out = _make_inputs(batch_size, seq_len, k)

    if provider == "jit_v1":

        def fn(scores, seq_lens, page_table):
            topk_transform_512(scores, seq_lens, page_table, out, PAGE_SIZE)
            return out

        return fn, (scores, seq_lens, page_table), (scores, seq_lens), (out,)

    if provider == "jit_v2":
        metadata = plan_topk_v2(seq_lens)  # amortized once per metadata init

        def fn(scores, seq_lens, page_table):
            topk_transform_512_v2(
                scores, seq_lens, page_table, out, PAGE_SIZE, metadata
            )
            return out

        return fn, (scores, seq_lens, page_table), (scores, seq_lens), (out,)

    if provider == "flashinfer":
        import flashinfer

        # Fused top-k + page-table transform (apples-to-apples with the JIT kernels).
        # page_size=1 per-token table; row_to_batch=None keeps the B200 cluster path.
        src_page_table, lengths = _make_p1_table(batch_size, seq_len)

        def fn(scores):
            return flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)

        return fn, (scores,), (scores,), "out"

    if provider == "torch":
        # torch.topk followed by a page_size=1 gather, so it also does the transform.
        src_page_table, _ = _make_p1_table(batch_size, seq_len)

        def fn(scores):
            idx = scores.topk(k, dim=-1).indices  # (batch, k) int64
            return torch.gather(src_page_table, 1, idx)

        return fn, (scores,), (scores,), "out"

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

    fn, input_args, mem_args, mem_out = _build_fn(provider, batch_size, seq_len, k)
    return marker.do_bench(
        fn, input_args=input_args, memory_args=mem_args, memory_output=mem_out
    )


if __name__ == "__main__":
    benchmark.run()
