import itertools
import os
from typing import Any, Optional

import sgl_kernel
import torch
import triton
import triton.testing
from sgl_kernel import (
    fast_topk_v2,
)

from sglang.jit_kernel.topk_indexer import fast_topk_v3

SEED = 42
MAX_SEQ_LEN = 131072

USE_TORCH_ORI = True

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def _ref_torch_impl_ori(
    score: torch.Tensor,
    seq_len: int,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert score.dim() == 2
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    else:
        ks = row_starts.cpu().tolist()
        ke = (row_starts + seq_len).tolist()
        scores = []
        for i, (start, end) in enumerate(zip(ks, ke)):
            scores.append(score[i, start:end].unsqueeze(0))

        score = torch.cat(scores, dim=0)
        return torch.topk(score, topk, dim=-1, sorted=False).indices


def _ref_torch_impl(score, seq_len, topk, row_starts=None):
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    else:
        idx = torch.arange(seq_len, device=score.device)
        idx = idx.unsqueeze(0) + row_starts.unsqueeze(1)
        sliced = torch.gather(score, 1, idx)
        return torch.topk(sliced, topk, dim=-1, sorted=False).indices


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
    topk_indices_offset: Optional[torch.Tensor] = None,
    max_permit_error: int = 0,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()

    wrong_values = 0
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i

        offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0

        if len(more) > 0 or len(less) > 0:
            # check whether more values are the same with less values
            # if so, either one is acceptable, since their values are the same
            more_values = sorted(score[i, idx - offset].item() for idx in more)
            less_values = sorted(score[i, idx - offset].item() for idx in less)
            if more_values != less_values:
                wrong_values += len(more)
                print(
                    f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"
                )
        assert wrong_values <= max_permit_error, f"{wrong_values=}, {max_permit_error=}"


def calculate_diff(bs, k, seq_len, has_row_starts):
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    if has_row_starts:
        score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    else:
        score = torch.randn(bs, seq_len, dtype=torch.float32, device="cuda")

    score_max = score.max()
    score_min = score.min()

    score = (score - score_min) / (score_max - score_min + 1e-6) * 255

    # score = torch.arange(MAX_SEQ_LEN, dtype=torch.float32, device="cuda").view(1, -1).expand(bs, -1)

    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None

    if USE_TORCH_ORI:
        indices_ref = _ref_torch_impl_ori(score, seq_len, k, row_starts=row_starts)
    else:
        indices_ref = _ref_torch_impl(score, seq_len, k, row_starts=row_starts)

    indices_old = fast_topk_v2(score, lengths, k, row_starts=row_starts)

    indices_our = fast_topk_v3(score, lengths, k, row_starts=row_starts)

    # sort and compare
    indices_ref = torch.sort(indices_ref, dim=-1).values
    indices_old = torch.sort(indices_old, dim=-1).values
    indices_our = torch.sort(indices_our, dim=-1).values

    # Tests can pass with max_permit_error=3, set to 5 for safety
    # assert_equal(score, indices_ref, indices_old, bs, k, seq_len, max_permit_error=5)

    assert_equal(score, indices_ref, indices_our, bs, k, seq_len, max_permit_error=5)


bs = [1, 2, 4, 8]
k = [2048]  # we only support 2048 now
# 32k smem
seq_len = [16384, 65536, 98304, 120000]
has_row_starts = [True, False]

configs = list(itertools.product(bs, k, seq_len, has_row_starts))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "k", "seq_len", "has_row_starts"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "radix_2602", "radix"],
        line_names=["torch", "radix_2602", "radix"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="Latency",
        plot_name="top2048-performance",
        args={},
    )
)
def benchmark(bs: int, k: int, seq_len: int, has_row_starts: bool, provider) -> None:
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    if has_row_starts:
        score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    else:
        score = torch.randn(bs, seq_len, dtype=torch.float32, device="cuda")

    score_max = score.max()
    score_min = score.min()

    score = (score - score_min) / (score_max - score_min + 1e-6) * 255

    # score = torch.arange(MAX_SEQ_LEN, dtype=torch.float32, device="cuda").view(1, -1).expand(bs, -1)

    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        if USE_TORCH_ORI:
            # torch_impl_ori does not satisfy cudagraph capture
            fn = lambda: _ref_torch_impl_ori(score, seq_len, k, row_starts=row_starts)
            ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        else:
            fn = lambda: _ref_torch_impl(score, seq_len, k, row_starts=row_starts)
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                fn, quantiles=quantiles
            )
    else:
        if provider == "radix_2602":
            fn = lambda: fast_topk_v3(score, lengths, k, row_starts=row_starts)
        else:
            fn = lambda: fast_topk_v2(score, lengths, k, row_starts=row_starts)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(  # do_bench(  #
            fn, quantiles=quantiles
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    # Correctness check - simplified for CI
    if IS_CI:
        # Only test one configuration in CI
        test_configs = [configs[0]]
    else:
        test_configs = configs

    for cfg in test_configs:
        print(f"cfg : {cfg}")
        calculate_diff(*cfg)

    print("\n" + "=" * 60)
    print("Starting performance benchmark...")
    benchmark.run(print_data=True)
