"""Benchmark the QSA indexer decode scoring: TileLang kernel vs CUDA kernel.

Both write the same logits bit for bit (test_qsa_mqa_decode_cuda.py). Shape axes:
  - ``requests``: requests per batch.
  - ``rows_per_request``: 1 is plain decode; 4 is a target-verify block of four
    rows that share the request's pages.
  - ``context``: compressed keys per request (4 tokens per key on Qwen3.8).

Constants follow the Qwen3.8 indexer: 4 query heads of dim 128, 16 compressed keys
per page, and the fast top-k's 512 below which rows are not scored.

The CUDA kernel covers SM100, SM103 and SM120 only. On other devices, including
the H100 runner this benchmark is registered on, the cuda provider is skipped
and only the TileLang kernel is timed.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import DEFAULT_DEVICE
from sglang.srt.layers.attention.qsa.mqa import (
    cuda_qsa_mqa_decode,
    tilelang_qsa_mqa_decode,
    use_cuda_qsa_mqa_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HEADS = 4
HEAD_DIM = 128
PAGE = 16
MIN_SCORED_LEN = 512


@marker.parametrize("context", [2048, 8192, 32768], [2048])
@marker.parametrize("rows_per_request", [1, 4], [1, 4])
@marker.parametrize("requests", [16, 64, 256], [16])
@marker.benchmark("provider", ["tilelang", "cuda"])
def benchmark(requests: int, rows_per_request: int, context: int, provider: str):
    torch.manual_seed(0)
    rows = requests * rows_per_request
    max_pages = context // PAGE + 1
    k_cache = torch.randn(
        requests * max_pages, PAGE, 1, HEAD_DIM, device=DEFAULT_DEVICE
    ).to(torch.bfloat16)
    q = torch.randn(rows, HEADS, HEAD_DIM, device=DEFAULT_DEVICE).to(torch.bfloat16)
    if provider == "cuda" and not use_cuda_qsa_mqa_decode(q, k_cache):
        marker.skip("the CUDA QSA decode kernel does not cover this device")
    table = torch.randperm(requests * max_pages, device=DEFAULT_DEVICE)
    page_table = (
        table.reshape(requests, max_pages)
        .repeat_interleave(rows_per_request, dim=0)
        .to(torch.int32)
    )
    lengths = (
        torch.full((requests, 1), context, device=DEFAULT_DEVICE)
        + torch.arange(rows_per_request, device=DEFAULT_DEVICE)
    ).flatten()
    lengths = lengths.to(torch.int32)
    max_model_len = context + rows_per_request
    fn = cuda_qsa_mqa_decode if provider == "cuda" else tilelang_qsa_mqa_decode
    kwargs = dict(min_scored_len=MIN_SCORED_LEN)
    if provider == "cuda":
        kwargs["rows_per_request"] = rows_per_request
    key_bytes = requests * context * HEAD_DIM * k_cache.element_size()
    logit_bytes = rows * context * 4
    return marker.do_bench(
        fn,
        input_args=(q, k_cache, page_table, lengths, max_model_len),
        input_kwargs=kwargs,
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=key_bytes + logit_bytes,
    )


if __name__ == "__main__":
    benchmark.run()
