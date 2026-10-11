"""Benchmark CuTe DSL dual GEMM against SGLang's production operator chains."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.gemm import (
    dual_gemm_swiglu,
    dual_gemm_swiglu_fp8,
    fp8_scaled_mm,
)
from sglang.kernels.ops.gemm.cutedsl_dual_gemm import DualGemmQuantMode
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="nightly", runner_config="4-gpu-b200")

_DECODE_SHAPES = [
    (1, 128, 128),
    (1, 4096, 14336),  # Llama 3 8B
    (2, 4096, 14336),
    (4, 4096, 14336),
    (8, 4096, 14336),
    (16, 4096, 14336),
    (1, 3584, 18944),  # Qwen 2 7B
    (2, 3584, 18944),
    (4, 3584, 18944),
    (8, 3584, 18944),
    (16, 3584, 18944),
]

_QUANT_MODES = {
    mode.name.lower(): mode
    for mode in (
        DualGemmQuantMode.STATIC_PER_TENSOR,
        DualGemmQuantMode.STATIC_PER_TOKEN,
        DualGemmQuantMode.DYNAMIC_PER_TENSOR,
        DualGemmQuantMode.DYNAMIC_PER_TOKEN,
    )
}


def _sglang_dual_gemm(
    x,
    gate_up_weight,
    x_scale,
    gate_up_weight_scale,
    output_scale,
    quant_mode,
):
    gate_up = fp8_scaled_mm(
        x,
        gate_up_weight.T,
        x_scale,
        gate_up_weight_scale,
        torch.bfloat16,
    )
    activation = silu_and_mul(gate_up)
    if quant_mode.is_dynamic:
        return scaled_fp8_quant(
            activation,
            use_per_token_if_dynamic=quant_mode.is_per_token,
        )
    quantized = (
        (activation.float() / output_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return quantized, output_scale


@marker.parametrize(
    "num_tokens,hidden_size,intermediate_size",
    _DECODE_SHAPES,
)
@marker.parametrize("quantization", list(_QUANT_MODES))
@marker.benchmark("impl", ["cutedsl", "sglang"], unit="us")
def benchmark_fp8(
    num_tokens,
    hidden_size,
    intermediate_size,
    quantization,
    impl,
):
    quant_mode = _QUANT_MODES[quantization]
    generator = torch.Generator(device="cuda").manual_seed(20261010)
    x, x_scale = scaled_fp8_quant(
        torch.randn(
            (num_tokens, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25,
        use_per_token_if_dynamic=quant_mode.is_per_token,
    )
    gate_up_weight, gate_up_weight_scale = scaled_fp8_quant(
        torch.randn(
            (2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25,
        use_per_token_if_dynamic=True,
    )
    gate_up_weight_scale = gate_up_weight_scale.reshape(-1)
    output_scale = None
    if not quant_mode.is_dynamic:
        _, output_scale = _sglang_dual_gemm(
            x,
            gate_up_weight,
            x_scale,
            gate_up_weight_scale,
            None,
            (
                DualGemmQuantMode.DYNAMIC_PER_TOKEN
                if quant_mode.is_per_token
                else DualGemmQuantMode.DYNAMIC_PER_TENSOR
            ),
        )

    fn = dual_gemm_swiglu_fp8 if impl == "cutedsl" else _sglang_dual_gemm
    return marker.do_bench(
        fn,
        input_args=(
            x,
            gate_up_weight,
            x_scale,
            gate_up_weight_scale,
            output_scale,
            quant_mode,
        ),
    )


def _sglang_float_dual_gemm(x, gate_up_weight):
    return silu_and_mul(torch.nn.functional.linear(x, gate_up_weight))


@marker.parametrize(
    "num_tokens,hidden_size,intermediate_size",
    _DECODE_SHAPES,
)
@marker.parametrize("dtype", [torch.bfloat16, torch.float16])
@marker.benchmark("impl", ["cutedsl", "sglang"], unit="us")
def benchmark_fp16(
    num_tokens,
    hidden_size,
    intermediate_size,
    dtype,
    impl,
):
    generator = torch.Generator(device="cuda").manual_seed(20261010)
    x = (
        torch.randn(
            (num_tokens, hidden_size),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.25
    )
    gate_up_weight = (
        torch.randn(
            (2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * 0.25
    )
    fn = dual_gemm_swiglu if impl == "cutedsl" else _sglang_float_dual_gemm
    return marker.do_bench(fn, input_args=(x, gate_up_weight))


if __name__ == "__main__":
    benchmark_fp8.run()
    benchmark_fp16.run()
