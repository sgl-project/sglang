import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.dsv4.topk import (
    plan_topk_v2,
    topk_transform_512,
    topk_transform_512_v2,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=120, stage="jit-kernel-benchmark", runner_config="amd")

# Compressed page size used by the DSA indexer (real value is 256 // 4 = 64).
PAGE_SIZE = 64


def _make_inputs(batch_size: int, seq_len: int, k: int):
    torch.random.manual_seed(42)
    scores = torch.randn(batch_size, seq_len, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    out = torch.empty(batch_size, k, dtype=torch.int32, device="cuda")
    return scores, seq_lens, page_table, out


def _make_p1_table(batch_size: int, seq_len: int):
    # flashinfer / torch do a per-token (page_size=1) gather, so they need a
    # (batch, seq) table (one entry per position) rather than the page-size-64 one.
    src_page_table = (
        torch.arange(seq_len, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    return src_page_table, lengths


def _build_fn(provider: str, batch_size: int, seq_len: int, k: int):
    scores, seq_lens, page_table, out = _make_inputs(batch_size, seq_len, k)
    N = PAGE_SIZE

    def fn(scores, seq_lens, page_table):
        if provider == "jit_v1":
            topk_transform_512(scores, seq_lens, page_table, out, N)
            return out
        elif provider == "jit_v2":
            topk_transform_512_v2(scores, seq_lens, page_table, out, N, metadata)
            return out
        elif provider == "flashinfer":
            from flashinfer import top_k_page_table_transform

            return top_k_page_table_transform(scores, page_table, seq_lens, k)
        elif provider == "torch":
            idx = scores.topk(k, dim=-1).indices  # (batch, k) int64
            return torch.gather(page_table, 1, idx)
        else:
            raise ValueError(f"unknown provider {provider}")

    if provider in ("flashinfer", "torch"):
        page_table, seq_lens = _make_p1_table(batch_size, seq_len)
    if provider == "jit_v2":
        metadata = plan_topk_v2(seq_lens)
    return fn, (scores, seq_lens, page_table)


@marker.parametrize("k", [512, 1024, 2048], [512])
@marker.parametrize("seq_len", [2**x for x in range(10, 19)], [4096, 65536])
@marker.parametrize("batch_size", [2**x for x in range(13)], [1, 128, 1024])
@marker.benchmark("provider", ["jit_v1", "jit_v2", "flashinfer", "torch"])
def benchmark(seq_len: int, batch_size: int, k: int, provider: str):
    if k > seq_len:
        marker.skip("k cannot be larger than seq_len")
    if k == 2048 and provider == "jit_v1":
        marker.skip("jit_v1 does not support k=2048")

    fn, input_args = _build_fn(provider, batch_size, seq_len, k)
    return marker.do_bench(fn, input_args=input_args, memory_args=input_args[:2])


if __name__ == "__main__":
    benchmark.run()
