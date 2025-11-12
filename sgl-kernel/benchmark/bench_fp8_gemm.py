import argparse
import copy
import itertools
import os
from typing import Optional, Tuple

import torch
import triton
from sgl_kernel import fp8_scaled_mm as sgl_scaled_mm
from sgl_kernel import sgl_per_tensor_quant_fp8

# Optional vLLM import
try:
    from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm
    from vllm._custom_ops import scaled_fp8_quant as vllm_scaled_fp8_quant

    VLLM_AVAILABLE = True
except ImportError:
    vllm_scaled_mm = None
    vllm_scaled_fp8_quant = None
    VLLM_AVAILABLE = False

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

# Weight Shapes are in the format
# ([K, N], TP_SPLIT_DIM)
# Example:
#  A shape of ([14336, 4096], 0) indicates the following GEMM shape,
#   - TP1 : K = 14336, N = 4096
#   - TP2 : K = 7168, N = 4096
#  A shape of ([4096, 6144], 1) indicates the following GEMM shape,
#   - TP1 : K = 4096, N = 6144
#   - TP4 : K = 4096, N = 1536

# TP1 shapes
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


def sglang_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    is_static = True
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        is_static = False
    sgl_per_tensor_quant_fp8(input, output, scale, is_static)

    return output, scale


# CI environment uses simplified parameters
if IS_CI:
    batch_sizes = [1]  # Single batch size for CI
else:
    batch_sizes = [1, 16, 64, 128, 256, 512, 1024, 2048]

# Filter line_vals based on vLLM availability
if VLLM_AVAILABLE:
    line_vals = [
        "vllm-fp8-fp16",
        "vllm-fp8-bf16",
        "sglang-fp8-fp16",
        "sglang-fp8-bf16",
    ]
    line_names = [
        "vllm-fp8-fp16",
        "vllm-fp8-bf16",
        "sglang-fp8-fp16",
        "sglang-fp8-bf16",
    ]
    styles = [("green", "-"), ("green", "--"), ("blue", "-"), ("blue", "--")]
else:
    line_vals = [
        "sglang-fp8-fp16",
        "sglang-fp8-bf16",
    ]
    line_names = [
        "sglang-fp8-fp16",
        "sglang-fp8-bf16",
    ]
    styles = [("blue", "-"), ("blue", "--")]


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
        plot_name="fp8 scaled matmul",
        args={},
    )
)
def benchmark(batch_size, provider, N, K):
    # M, N, K = batch_size, 4096, 8192
    M = batch_size
    a = torch.ones((M, K), device="cuda") * 5.0
    b = torch.ones((N, K), device="cuda") * 5.0
    # vLLM expects scalar scales, while sglang can handle per-token scales
    scale_a_scalar = torch.randn(1, device="cuda", dtype=torch.float32)
    scale_b_scalar = torch.randn(1, device="cuda", dtype=torch.float32)
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    dtype = torch.float16 if "fp16" in provider else torch.bfloat16

    if "vllm-fp8" in provider:
        if not VLLM_AVAILABLE:
            # Return zero if vLLM is not available
            return (0, 0, 0)
        a_fp8, scale_a_fp8 = vllm_scaled_fp8_quant(a, scale_a_scalar)
        b_fp8, scale_b_fp8 = vllm_scaled_fp8_quant(b, scale_b_scalar)
        b_fp8 = b_fp8.t()
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: vllm_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype),
            quantiles=quantiles,
        )
    elif "sglang-fp8" in provider:
        a_fp8, scale_a_fp8 = sglang_scaled_fp8_quant(a, scale_a)
        b_fp8, scale_b_fp8 = sglang_scaled_fp8_quant(b, scale_b)
        b_fp8 = b_fp8.t()
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: sgl_scaled_mm(
                a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, dtype, bias=None
            ),
            quantiles=quantiles,
        )

    gbps = lambda ms: (2 * M * N * K + M * N) * a.element_size() * 1e-9 / (ms * 1e-3)
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

    # Simplify for CI environment
    if IS_CI:
        args.models = [args.models[0]]  # Use only first model
        args.tp_sizes = [args.tp_sizes[0]]  # Use only first TP size

    KN_model_names = prepare_shapes(args)
    for K, N, model_name in KN_model_names:
        print(f"{model_name} N={N} K={K}: ")
        benchmark.run(print_data=True, N=N, K=K)

    print("Benchmark finished!")
