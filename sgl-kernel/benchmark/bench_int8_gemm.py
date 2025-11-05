import argparse
import copy
import itertools
import os

import torch
import triton
from sgl_kernel import int8_scaled_mm

# Optional vLLM import
try:
    from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm

    VLLM_AVAILABLE = True
except ImportError:
    vllm_scaled_mm = None
    VLLM_AVAILABLE = False

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


WEIGHT_SHAPES = {
    "meta-llama/Llama-3.1-8B-Instruct": [
        ([4096, 6144], 1),
        ([4096, 4096], 0),
        ([4096, 28672], 1),
        ([14336, 4096], 0),
    ],
    "meta-llama/Llama-3.3-70B-Instruct": [
        ([8192, 10240], 1),
        ([8192, 8192], 0),
        ([8192, 57344], 1),
        ([28672, 8192], 0),
    ],
    "mistralai/Mistral-Large-Instruct-2407": [
        ([12288, 14336], 1),
        ([12288, 12288], 0),
        ([12288, 57344], 1),
        ([28672, 12288], 0),
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        ([3584, 4608], 1),
        ([3584, 3584], 0),
        ([3584, 37888], 1),
        ([18944, 3584], 0),
    ],
    "Qwen/Qwen2.5-32B-Instruct": [
        ([5120, 7168], 1),
        ([5120, 5120], 0),
        ([5120, 55296], 1),
        ([27648, 5120], 0),
    ],
    "Qwen/Qwen2.5-72B-Instruct": [
        ([8192, 10240], 1),
        ([8192, 8192], 0),
        ([8192, 59136], 1),
        ([29568, 8192], 0),
    ],
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": [
        ([2048, 3072], 1),
        ([2048, 4096], 1),
        ([2048, 2048], 0),
        ([2048, 576], 0),
        ([2048, 21888], 1),
        ([10944, 2048], 0),
        ([2048, 2816], 1),
        ([1408, 2048], 0),
    ],
}


# CI environment uses simplified parameters
if IS_CI:
    batch_sizes = [1]  # Single batch size for CI
else:
    batch_sizes = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Filter providers based on vLLM availability
if VLLM_AVAILABLE:
    line_vals = ["vllm", "sgl-kernel"]
    line_names = ["vllm int8 gemm", "sgl-kernel int8 gemm"]
    styles = [("blue", "-"), ("orange", "-")]
else:
    line_vals = ["sgl-kernel"]
    line_names = ["sgl-kernel int8 gemm"]
    styles = [("orange", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=batch_sizes,
        x_log=False,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="GB/s",
        plot_name="int8 scaled matmul",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    M = batch_size
    a = to_int8(torch.randn((M, K), device="cuda") * 5)
    b = to_int8(torch.randn((N, K), device="cuda").t() * 5)
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    bias = torch.randn((N,), device="cuda", dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "sgl-kernel":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: int8_scaled_mm(a, b, scale_a, scale_b, torch.float16, bias),
            quantiles=quantiles,
        )
    elif provider == "vllm":
        if not VLLM_AVAILABLE:
            return (0, 0, 0)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: vllm_scaled_mm(a, b, scale_a, scale_b, torch.float16, bias),
            quantiles=quantiles,
        )
    gbps = (
        lambda ms: (
            (2 * M * N * K - M * N) * a.element_size()
            + (3 * M * N) * scale_a.element_size()
        )
        * 1e-9
        / (ms * 1e-3)
    )
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def prepare_shapes(args):
    KN_model_names = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        assert model in WEIGHT_SHAPES
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KN.append(model)
            KN_model_names.append(KN)
    return KN_model_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["meta-llama/Llama-3.1-8B-Instruct"],
        help="List of models to benchmark",
    )
    parser.add_argument(
        "--tp-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="List of tensor parallel sizes",
    )
    args = parser.parse_args()

    # Skip in CI environment due to architecture compatibility issues
    if IS_CI:
        print(
            "Skipping INT8 GEMM benchmark in CI environment due to architecture compatibility issues"
        )
        print("INT8 operations may not be supported on all GPU architectures")
    else:
        KN_model_names = prepare_shapes(args)
        for K, N, model_name in KN_model_names:
            print(f"{model_name} N={N} K={K}: ")
            benchmark.run(print_data=True, N=N, K=K)

        print("Benchmark finished!")
