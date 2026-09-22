"""Benchmark SM90 DeepSelect FP32 Top-K against torch.topk."""

import itertools

import torch
import triton.testing

from sgl_kernel import deepselect_topk_fp32


configs = list(
    itertools.product([1, 6, 32], [16384, 32768, 65536, 131072], [512, 2048])
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["rows", "width", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["deepselect", "torch"],
        line_names=["DeepSelect AOT", "torch.topk"],
        styles=[("green", "-"), ("red", "--")],
        ylabel="µs (median)",
        plot_name="deepselect-sm90-fp32-topk",
        args={},
    )
)
def benchmark(rows, width, topk, provider):
    torch.manual_seed(913)
    scores = torch.randn((rows, width), dtype=torch.float32, device="cuda")
    if provider == "deepselect":

        def fn():
            return deepselect_topk_fp32(scores, topk)

    else:

        def fn():
            return torch.topk(scores, topk, dim=-1, sorted=False)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This benchmark requires an SM90 GPU")
    benchmark.run(print_data=True)
