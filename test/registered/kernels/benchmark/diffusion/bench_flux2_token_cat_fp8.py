import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import try_flux2_token_cat_fp8
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("tokens", [512, 4096], [4096])
@marker.benchmark("impl", ["cat_then_quant", "fused"], unit="us")
def benchmark(tokens: int, impl: str):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260831 + tokens)
    attention = torch.randn(
        (1, tokens, 6144),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    mlp = torch.randn(
        (1, tokens, 18432),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    input_scale = torch.tensor([0.013], dtype=torch.float32, device="cuda")

    if impl == "cat_then_quant":

        def fn():
            return static_quant_fp8(torch.cat([attention, mlp], dim=-1), input_scale)[0]

    else:

        def fn():
            return try_flux2_token_cat_fp8(attention, mlp, input_scale)

    return marker.do_bench(fn, use_cuda_graph=False, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
