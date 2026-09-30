"""ConvRot INT8 linear (rotate + quant + GEMM, and GEMM alone) vs torch BF16 addmm."""

import itertools

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.gemm.convrot_int8 import (
    SUPPORTED_CAPABILITIES,
    convrot_int8_fused_linear,
    convrot_int8_linear_prequant,
    convrot_rotate_quantize_activation,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

GROUP_SIZE = 256

# Qwen-Image DiT linears at 1024x1024; same shapes as ops/gemm/test_convrot_int8.py.
M_VALUES = [3, 20, 2048, 4096]
KN_VALUES = [(3072, 3072), (3072, 12288), (12288, 3072)]
SHAPES = [(M, K, N) for M, (K, N) in itertools.product(M_VALUES, KN_VALUES)]


def _fused(x, weight_q, weight_scale, bias):
    return convrot_int8_fused_linear(
        x, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
    )


def _prequant(x_q, x_scale, weight_q, weight_scale, bias):
    return convrot_int8_linear_prequant(
        x_q, x_scale, weight_q, weight_scale, bias=bias, group_size=GROUP_SIZE
    )


def _torch_bf16(x, weight, bias):
    return torch.addmm(bias, x, weight.t())


@marker.parametrize("shape", SHAPES, [(2048, 3072, 3072), (2048, 12288, 3072)])
@marker.benchmark("provider", ["convrot_int8", "convrot_int8_prequant", "torch_bf16"])
def benchmark(shape, provider):
    if torch.cuda.get_device_capability() not in SUPPORTED_CAPABILITIES:
        marker.skip("convrot_int8 kernels carry no code for this GPU")
    M, K, N = shape
    x = create_random(M, K)
    weight = create_random(N, K) * 0.02
    bias = create_random(N)
    weight_q, weight_scale = convrot_rotate_quantize_activation(
        weight, group_size=GROUP_SIZE
    )
    x_q, x_scale = convrot_rotate_quantize_activation(x, group_size=GROUP_SIZE)
    # Each provider gets only the tensors it reads, so the bandwidth column and
    # the per-iteration L2 rotation count what that provider actually touches.
    fn, args = {
        "convrot_int8": (_fused, (x, weight_q, weight_scale, bias)),
        "convrot_int8_prequant": (
            _prequant,
            (x_q, x_scale, weight_q, weight_scale, bias),
        ),
        "torch_bf16": (_torch_bf16, (x, weight, bias)),
    }[provider]
    return marker.do_bench(fn, input_args=args, flops=2.0 * M * N * K)


if __name__ == "__main__":
    benchmark.run()
