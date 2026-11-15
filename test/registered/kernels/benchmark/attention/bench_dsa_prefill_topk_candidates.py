"""Benchmark of the prefill DSA candidate top-k: fp32 vs bf16 slabs, on GLM-like
synthetic slabs (count ~ 4700 of 12288, k = 2048) and the DeepSeek shapes (k = 1024 / 512)."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsa.litetopk_prefill.topk_candidates import (
    TOPK_CANDIDATES_CAP,
    topk_candidates,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="nightly", runner_config="4-gpu-b200")


def make_inputs(rows: int, k: int, mean_count: int):
    gen = torch.Generator(device="cuda").manual_seed(rows + k)
    cap = TOPK_CANDIDATES_CAP
    count = (
        (
            torch.randn(rows, device="cuda", generator=gen) * 0.15 * mean_count
            + mean_count
        )
        .clamp(k + 1, cap)
        .int()
    )
    scores = torch.randn(rows, cap, device="cuda", generator=gen) * 5 - 60
    live = torch.arange(cap, device="cuda")[None, :] < count[:, None]
    scores = torch.where(live, scores, torch.full_like(scores, -float("inf")))
    thr = (
        torch.where(live, scores, torch.full_like(scores, float("inf")))
        .amin(dim=1)
        .half()
        .float()
    )
    indices = (
        torch.arange(cap, dtype=torch.int32, device="cuda")[None, :].repeat(rows, 1)
        * 64
        + 3
    )
    out = torch.empty(rows, k, dtype=torch.int32, device="cuda")
    return scores, indices, count, thr, out


@marker.parametrize("rows", [1024, 2048, 4096, 8192], [1024, 4096])
@marker.parametrize(
    "k,mean_count", [(2048, 4700), (1024, 2850), (512, 1840)], [(2048, 4700)]
)
@marker.benchmark("dtype", ["fp32", "bf16"])
def benchmark(rows: int, k: int, mean_count: int, dtype: str):
    scores, indices, count, thr, out = make_inputs(rows, k, mean_count)
    if dtype == "bf16":
        scores_bf16 = scores.to(torch.bfloat16)
        # the bf16 slab's T: a bf16 value <= every rounded candidate (the slab contract)
        live = torch.arange(scores.shape[1], device="cuda")[None, :] < count[:, None]
        low = torch.where(
            live, scores_bf16.float(), torch.full_like(scores, float("inf"))
        ).amin(dim=1)
        thr = (low - 1.0).to(torch.bfloat16).float()
        fn = lambda scores_bf16, indices, count, thr, out: topk_candidates(
            scores_bf16, indices, count, thr, out
        )
        args = (scores_bf16, indices, count, thr, out)
        mem = (scores_bf16, indices)
    else:
        fn = lambda scores, indices, count, thr, out: topk_candidates(
            scores, indices, count, thr, out
        )
        args = (scores, indices, count, thr, out)
        mem = (scores, indices)
    return marker.do_bench(fn, input_args=args, memory_args=mem, memory_output=(out,))


if __name__ == "__main__":
    benchmark.run()
