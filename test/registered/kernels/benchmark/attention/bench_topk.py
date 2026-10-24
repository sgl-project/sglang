import itertools

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_packed_v2,
    topk_transform_paged,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# Compressed page size used by the DSA indexer (real value is 256 // 4 = 64).
PAGE_SIZE = 64
# NOTE: currently torch baseline is disabled, since it's too slow
DISABLE_TORCH = True


def _make_inputs(batch_size: int, seq_len: int, k: int, page_size: int = PAGE_SIZE):
    torch.random.manual_seed(42)
    scores = torch.randn(batch_size, seq_len, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    num_pages = (seq_len + page_size - 1) // page_size
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    out = torch.empty(batch_size, k, dtype=torch.int32, device="cuda")
    return scores, seq_lens, page_table, out


def _build_paged_fn(
    provider: str, batch_size: int, seq_len: int, k: int, page_size: int
):
    scores, seq_lens, page_table, out = _make_inputs(batch_size, seq_len, k, page_size)
    N = page_size

    def fn(scores, seq_lens, page_table):
        if provider == "jit_v1":
            topk_transform_paged(scores, seq_lens, page_table, out, N)
            return out
        elif provider == "jit_v2":
            topk_transform_paged_v2(scores, seq_lens, page_table, out, N, metadata)
            return out
        elif provider == "flashinfer":
            from flashinfer import top_k_page_table_transform

            return top_k_page_table_transform(
                scores,
                page_table,
                seq_lens,
                k,
                dsa_graph_safe=True,
                page_size=N,
                out=out,
            )
        elif provider == "torch":
            raw_indices = scores.topk(k, dim=-1).indices
            physical_pages = torch.gather(page_table, 1, raw_indices // N)
            out.copy_(physical_pages * N + raw_indices % N)
            return out
        else:
            raise ValueError(f"unknown provider {provider}")

    if provider == "jit_v2":
        metadata = plan_topk_v2(seq_lens)
    return fn, (scores, seq_lens, page_table)


def _build_ragged_fn(provider: str, batch_size: int, seq_len: int, k: int):
    scores, seq_lens, _, out = _make_inputs(batch_size, seq_len, k)
    offsets = torch.arange(batch_size, dtype=torch.int32, device="cuda") * seq_len

    def fn(scores, seq_lens, offsets):
        if provider == "jit_v1":
            from sgl_kernel import fast_topk_transform_ragged_fused

            return fast_topk_transform_ragged_fused(scores, seq_lens, offsets, k)
        elif provider == "jit_v2":
            topk_transform_ragged_v2(
                scores, seq_lens, out_offsets=offsets, out_indices=out
            )
            return out
        elif provider == "flashinfer":
            from flashinfer import top_k_ragged_transform

            return top_k_ragged_transform(scores, offsets, seq_lens, k)
        elif provider == "torch":
            idx = scores.topk(k, dim=-1).indices.to(torch.int32)  # (batch, k)
            return idx + offsets.unsqueeze(1)
        else:
            raise ValueError(f"unknown provider {provider}")

    return fn, (scores, seq_lens, offsets)


PRROVIDERS = ["jit_v1", "jit_v2", "flashinfer"]
if not DISABLE_TORCH:
    PRROVIDERS.append("torch")


@marker.parametrize("page_size", [1, 64], [1, 64])
@marker.parametrize("k", [512, 1024, 2048], [512])
@marker.parametrize("seq_len", [2**x for x in range(10, 19)], [4096, 65536])
@marker.parametrize("batch_size", [2**x for x in range(13)], [1, 128, 1024])
@marker.benchmark("provider", PRROVIDERS)
def benchmark_paged(
    seq_len: int, batch_size: int, k: int, page_size: int, provider: str
):
    seed = seq_len ^ (batch_size << 16) ^ (k << 32) ^ (page_size << 48)
    torch.random.manual_seed(seed)
    if k > seq_len:
        marker.skip("k cannot be larger than seq_len")
    if k == 2048 and provider == "jit_v1":
        marker.skip("jit_v1 does not support k=2048")

    fn, input_args = _build_paged_fn(provider, batch_size, seq_len, k, page_size)
    return marker.do_bench(fn, input_args=input_args, memory_args=input_args[:2])


@marker.parametrize("k", [512, 1024, 2048], [2048])
@marker.parametrize("seq_len", [2**x for x in range(10, 19)], [4096, 65536])
# NOTE: prefill workload should be heavier than decode; not common for short extend
@marker.parametrize("batch_size", [2**x for x in range(7, 14)], [128, 1024])
@marker.benchmark("provider", PRROVIDERS)
def benchmark_ragged(seq_len: int, batch_size: int, k: int, provider: str):
    seed = seq_len ^ (batch_size << 16) ^ (k << 32)
    torch.random.manual_seed(seed)
    if k > seq_len:
        marker.skip("k cannot be larger than seq_len")
    if k != 2048 and provider == "jit_v1":
        marker.skip("jit_v1 (here, sgl-AOT) only support k=2048")

    fn, input_args = _build_ragged_fn(provider, batch_size, seq_len, k)
    return marker.do_bench(fn, input_args=input_args, memory_args=input_args[:2])


# Packed PAGED extend top-k, the DSA prefill layout on a paged-KV prefill backend.
# A chunk's requests share one key axis; each new token is a row ending at its position.
PACKED_PROVIDERS = ["sgl_kernel", "jit_v2"]  # sgl_kernel is the legacy kernel
PACKED_TOPK = 2048


def _make_packed_inputs(num_rows: int, mean_window: int, shape: str):
    # Requests are (key-axis base, keys, new rows); "single" is one request.
    # "mix3": three at unaligned bases, prefixes near 0.5x, 1x and 1.5x the window.
    if shape == "single":
        reqs = [(0, max(0, mean_window - num_rows // 2) + num_rows, num_rows)]
    else:
        reqs, base = [], 0
        news = [num_rows // 3, num_rows // 3, num_rows - 2 * (num_rows // 3)]
        for q, (new, frac) in enumerate(zip(news, (0.5, 1.0, 1.5))):
            base += q + 1
            keys = (max(0, int(mean_window * frac) - new // 2) | 1) + new
            reqs.append((base, keys, new))
            base += keys
    num_cols = reqs[-1][0] + reqs[-1][1]
    stride = -(-num_cols // 256) * 256  # DeepGEMM pads the logits row stride
    if num_rows * stride * 4 > 24 << 30:
        marker.skip("logits over 24 GiB: the DSA indexer splits this call into chunks")
    i32 = dict(dtype=torch.int32, device="cuda")
    rows = [(q, b, k - n + 1 + t) for q, (b, k, n) in enumerate(reqs) for t in range(n)]
    row_to_req, row_starts, lengths = torch.tensor(rows, **i32).T.contiguous()
    cu_q = torch.tensor([0, *itertools.accumulate(n for _, _, n in reqs)], **i32)
    # Requests own distinct pages of a permuted pool (the indexer's real_page_table);
    # page_table_1 is the page-size-1 table the legacy kernel reads.
    num_pages = [-(-k // PAGE_SIZE) for _, k, _ in reqs]
    pool = torch.randperm(sum(num_pages), **i32)
    page_table = torch.full((len(reqs), max(num_pages)), -1, **i32)
    for q, first in enumerate(itertools.accumulate([0, *num_pages[:-1]])):
        page_table[q, : num_pages[q]] = pool[first : first + num_pages[q]]
    pos = torch.arange(max(k for _, k, _ in reqs), **i32)
    page_table_1 = page_table[:, pos // PAGE_SIZE] * PAGE_SIZE + pos % PAGE_SIZE
    scores = create_random(num_rows, stride, dtype=torch.float32)

    def check(out: torch.Tensor) -> None:
        # Exact against torch.topk: each slot maps back to a key of the row's request,
        # and each row holds its window's top-k value multiset, bitwise.
        owned = pos < torch.tensor([k for _, k, _ in reqs], **i32)[:, None]
        bases = torch.tensor([b for b, _, _ in reqs], **i32)[:, None]
        col_of_slot = torch.full((int(page_table_1.max()) + 1,), -1, device="cuda")
        col_of_slot[page_table_1[owned].long()] = (bases + pos)[owned].long()
        cols = col_of_slot[out.clamp(min=0).long()]
        assert torch.all((out == -1) | ((out >= 0) & (cols >= 0))), "bad slot"
        view, col = scores[:, :num_cols], torch.arange(num_cols, device="cuda")
        got = torch.where(out >= 0, view.gather(1, cols.clamp(min=0)), -torch.inf)
        for r in range(0, num_rows, 512):
            s = row_starts[r : r + 512, None]
            in_window = (col >= s) & (col < s + lengths[r : r + 512, None])
            ref = torch.where(in_window, view[r : r + 512], -torch.inf)
            ref = ref.topk(PACKED_TOPK).values.sort(dim=1).values
            assert torch.equal(got[r : r + 512].sort(dim=1).values, ref), "top-k"

    input_args = (
        scores,
        lengths,
        row_starts,
        row_to_req,
        cu_q,
        page_table,
        page_table_1,
    )
    return num_cols, input_args, check


@marker.parametrize("num_rows", [1024, 4096, 8192], [1024])
@marker.parametrize("mean_window", [4096, 32768, 131072, 409600], [32768])
@marker.parametrize("shape", ["mix3", "single"], ["mix3"])
@marker.benchmark("provider", PACKED_PROVIDERS)
def benchmark_packed(num_rows: int, mean_window: int, shape: str, provider: str):
    import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel

    torch.random.manual_seed(num_rows ^ (mean_window << 16) ^ (shape == "mix3"))
    num_cols, input_args, check = _make_packed_inputs(
        num_rows=num_rows, mean_window=mean_window, shape=shape
    )
    out = torch.empty(num_rows, PACKED_TOPK, dtype=torch.int32, device="cuda")

    def fn(scores, lengths, row_starts, row_to_req, cu_q, page_table, page_table_1):
        scores = scores[:, :num_cols]  # a view of the padded buffer, as in the indexer
        if provider == "sgl_kernel":
            torch.ops.sgl_kernel.fast_topk_transform_fused(
                score=scores,
                lengths=lengths,
                dst_page_table=out,
                src_page_table=page_table_1,
                cu_seqlens_q=cu_q,
                row_starts=row_starts,
            )
        elif provider == "jit_v2":
            topk_transform_packed_v2(
                scores=scores,
                seq_lens=lengths,
                page_tables=page_table,
                out_page_indices=out,
                page_size=PAGE_SIZE,
                row_starts=row_starts,
                row_to_batch=row_to_req,
            )
        else:
            raise ValueError(f"unknown provider {provider}")
        return out

    out.fill_(-7)  # poison, so a slot either provider leaves unwritten fails the check
    check(fn(*input_args))
    # Rows read only their windows, so input bytes over time would overstate bandwidth.
    # do_bench flushes L2, so reads are colder than the indexer's fresh logits.
    return marker.do_bench(fn, input_args=input_args, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark_paged.run()
    benchmark_ragged.run()
    benchmark_packed.run()
