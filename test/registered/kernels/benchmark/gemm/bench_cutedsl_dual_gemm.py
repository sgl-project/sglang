"""Benchmark fused FP8 dual GEMM against SGLang's production operator chain."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.gemm import dual_gemm_swiglu_fp8, fp8_scaled_mm
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="nightly", runner_config="4-gpu-b200")


def _sglang_dual_gemm(
    x,
    gate_up_weight,
    x_scale,
    gate_up_weight_scale,
    output_scale,
):
    gate_up = fp8_scaled_mm(
        x,
        gate_up_weight.T,
        x_scale,
        gate_up_weight_scale,
        torch.bfloat16,
    )
    activation = silu_and_mul(gate_up)
    if output_scale is None:
        return scaled_fp8_quant(activation, use_per_token_if_dynamic=True)
    return scaled_fp8_quant(activation, scale=output_scale)


@marker.parametrize(
    "num_tokens,hidden_size,intermediate_size",
    [
        (1, 128, 128),
        (1, 2048, 11008),
        (1, 4096, 14336),  # Llama 3 8B
        (1, 3584, 18944),  # Qwen 2 7B
    ],
)
@marker.parametrize("quantization", ["dynamic", "static"])
@marker.benchmark("impl", ["cutedsl", "sglang"], unit="us")
def benchmark(
    num_tokens,
    hidden_size,
    intermediate_size,
    quantization,
    impl,
):
    generator = torch.Generator(device="cuda").manual_seed(20261010)
    x, x_scale = scaled_fp8_quant(
        torch.randn(
            (num_tokens, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.25,
        use_per_token_if_dynamic=True,
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
    x_scale = x_scale.reshape(-1)
    gate_up_weight_scale = gate_up_weight_scale.reshape(-1)
    output_scale = None
    if quantization == "static":
        _, output_scale = _sglang_dual_gemm(
            x,
            gate_up_weight,
            x_scale,
            gate_up_weight_scale,
            None,
        )
        output_scale = output_scale.reshape(1)

    fn = dual_gemm_swiglu_fp8 if impl == "cutedsl" else _sglang_dual_gemm
    return marker.do_bench(
        fn,
        input_args=(
            x,
            gate_up_weight,
            x_scale,
            gate_up_weight_scale,
            output_scale,
        ),
    )


if __name__ == "__main__":
    benchmark.run()
