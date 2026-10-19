"""Compare router GEMM latency and mean absolute error against FP64.

Run this file on Hopper or Blackwell. Weight preparation is outside the timed region;
split-K reduction is included. Latencies use rotating inputs and CUDA graphs.
All providers use the same random inputs (seed 0, X std 1, W std 0.02).
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random, get_benchmark_range
from sglang.kernels.ops.gemm.router_gemm_cutile import router_gemm_bf16x3_cutile
from sglang.kernels.ops.gemm.router_gemv import split_fp32_to_bf16x3
from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

PROVIDERS = ["SGLang tiny FP32", "Cast + TF32", "cuBLAS BF16x3", "cuTile BF16x3"]


def _cast_tf32(x, w):
    return x.float() @ w.T


def _cublas(x, parts):
    n = parts.shape[0] // 3
    return torch.mm(x, parts.T, out_dtype=torch.float32).view(-1, 3, n).sum(1)


def _print_table(title, rows, number_format):
    print(f"\n### {title}\n")
    print("| Tokens | " + " | ".join(PROVIDERS) + " |")
    print("| ---: " * (len(PROVIDERS) + 1) + "|")
    for m, values in rows:
        cells = ["—" if v is None else format(v, number_format) for v in values]
        print(f"| {m} | " + " | ".join(cells) + " |")


def benchmark(batch_sizes, n, k):
    torch.manual_seed(0)
    w = create_random(n, k, dtype=torch.float32) * 0.02
    parts = split_fp32_to_bf16x3(w)
    latencies, errors = [], []
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for m in batch_sizes:
            x = create_random(m, k, dtype=torch.bfloat16)
            ref = x.double() @ w.double().T
            cases = [
                (
                    tiny_gemm_bf16,
                    w,
                    {"out_dtype": torch.float32, "max_m": 16 if m <= 16 else 32},
                ),
                (_cast_tf32, w, {}),
                (_cublas, parts, {}),
                (router_gemm_bf16x3_cutile, parts, {}),
            ]
            times, maes = [], []
            for fn, weight, kwargs in cases:
                if fn is tiny_gemm_bf16 and m > 32:
                    times.append(None)
                    maes.append(None)
                    continue
                result = marker.do_bench(
                    fn, input_args=(x, weight), input_kwargs=kwargs
                )
                out = fn(x, weight, **kwargs)
                times.append(result.times[0] * 1e6)
                maes.append((out.double() - ref).abs().mean().item())
            latencies.append((m, times))
            errors.append((m, maes))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    print(f"\n## K{k}/N{n}")
    _print_table("Latency (µs)", latencies, ".2f")
    _print_table("Mean absolute error vs FP64", errors, ".2e")


if __name__ == "__main__":
    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    if major not in (9, 10):
        print("Hopper or Blackwell required.")
    else:
        batch_sizes = get_benchmark_range(
            [
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
                16896,
            ],
            [1, 32, 1024, 16384],
        )
        for n, k in get_benchmark_range([(256, 5120), (128, 6144)], [(256, 5120)]):
            benchmark(batch_sizes, n, k)
