"""Cross-model benchmark for the KDA SM120 FP8 skinny GEMM."""

from __future__ import annotations

import sys

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.kda_kernels.sm120_fp8_skinny_gemm_sm120 import (
    _run_sm120_fp8_skinny_gemm_quantized,
)
from sglang.kernels.ops.gemm.sm120_fp8_gemv import sm120_fp8_gemv
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    apply_fp8_linear_bmm_flashinfer,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


def _make_inputs(m: int, n: int, k: int):
    input = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = (
        torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
        .t()
    )
    weight_scale = torch.tensor(0.02, dtype=torch.float32, device="cuda")
    input_scale = torch.tensor(0.025, dtype=torch.float32, device="cuda")
    output_scale = input_scale * weight_scale
    return input, weight, weight_scale, input_scale, output_scale


def _torch_impl(input, weight, weight_scale, input_scale, output_scale):
    del output_scale
    return apply_fp8_linear(
        input,
        weight,
        weight_scale,
        input_scale,
        cutlass_fp8_supported=False,
        pad_output=False,
    )


def _kda_impl(input, weight, weight_scale, input_scale, output_scale):
    del weight_scale
    qinput, _ = static_quant_fp8(input, input_scale, repeat_scale=False)
    return _run_sm120_fp8_skinny_gemm_quantized(qinput, weight, output_scale)


def _flashinfer_impl(input, weight, weight_scale, input_scale, output_scale):
    del output_scale
    return apply_fp8_linear_bmm_flashinfer(input, weight, weight_scale, input_scale)


def _native_gemv_impl(input, weight, weight_scale, input_scale, output_scale):
    del weight_scale
    qinput, _ = static_quant_fp8(input, input_scale, repeat_scale=False)
    return sm120_fp8_gemv(qinput, weight.t(), output_scale.reshape(1))


FN_MAP = {
    "kda": _kda_impl,
    "torch": _torch_impl,
    "flashinfer": _flashinfer_impl,
    "native": _native_gemv_impl,
}


PROJECTIONS = [
    ("Qwen3.8-27B/gdn-in", 16384, 5120),
    ("Qwen3.8-27B/attn-qkv", 8192, 5120),
    ("Qwen3.8-27B/out", 5120, 6144),
    ("Qwen3-8B+Llama-3.1-8B/qkv", 6144, 4096),
    ("Qwen3-8B+Llama-3.1-8B/o", 4096, 4096),
    ("Qwen3-8B/gate-up", 24576, 4096),
    ("Qwen3-8B/down", 4096, 12288),
    ("Qwen3-14B/qkv", 7168, 5120),
    ("Qwen3-14B/o", 5120, 5120),
    ("Qwen3-14B/gate-up", 34816, 5120),
    ("Qwen3-14B/down", 5120, 17408),
    ("Llama-3.1-8B/gate-up", 28672, 4096),
    ("Llama-3.1-8B/down", 4096, 14336),
    ("Nemotron-3-Super/mamba-in", 18560, 4096),
    ("Nemotron-3-Super/mamba-out", 4096, 8192),
    ("Nemotron-3-Super/shared-up", 5376, 4096),
    ("Nemotron-3-Super/shared-down", 4096, 5376),
]
M_VALUES = (1, 2, 4, 8, 9)
MODEL_FP8_CASES = [(model, m, n, k) for m in M_VALUES for model, n, k in PROJECTIONS]
NATIVE_M1_CASES = [
    ("Qwen3.8-27B/attn-qkv", 8192, 5120),
    ("Qwen3.8-27B/out", 5120, 6144),
    ("Qwen3-8B+Llama-3.1-8B/o", 4096, 4096),
    ("Qwen3-14B/qkv", 7168, 5120),
    ("Nemotron-3-Super/shared-up", 5376, 4096),
]


# The full sweep covers decode/verify M values and representative per-tensor
# FP8 projections from several model families.
@marker.parametrize(
    "model,m,n,k",
    MODEL_FP8_CASES,
    [("Qwen3.8-27B/attn-qkv", 9, 8192, 5120)],
)
@marker.benchmark("provider", ["kda", "torch", "flashinfer"])
def benchmark(model: str, m: int, n: int, k: int, provider: str):
    del model
    args = _make_inputs(m, n, k)
    return marker.do_bench(
        FN_MAP[provider],
        input_args=args,
        graph_clone_args=(0, 1, 2, 3),
        disable_log_bandwidth=True,
    )


@marker.parametrize(
    "model,n,k",
    NATIVE_M1_CASES,
    [("Qwen3.8-27B/attn-qkv", 8192, 5120)],
)
@marker.benchmark("provider", ["kda", "native"])
def benchmark_m1_native(model: str, n: int, k: int, provider: str):
    """Compare M=1 with SGLang's existing SM120 GEMV before dispatching."""
    del model
    args = _make_inputs(1, n, k)
    return marker.do_bench(
        FN_MAP[provider],
        input_args=args,
        graph_clone_args=(0, 1, 2, 3),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    if not (
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
    ):
        print("[skip] KDA FP8 skinny GEMM benchmark requires CUDA SM120")
        sys.exit(0)
    benchmark.run()
    benchmark_m1_native.run()
