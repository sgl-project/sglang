import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    fused_layernorm_modulate_fp8_quant_raw,
    fused_layernorm_modulate_raw,
)
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 6144
EPS = 1e-6


@marker.parametrize("rows", [512, 4096, 4608], [512])
@marker.benchmark("impl", ["split", "fused"], unit="us")
def benchmark(rows: int, impl: str):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260831 + rows)
    x = torch.randn((1, rows, HIDDEN), dtype=DTYPE, device=DEVICE, generator=generator)
    scale = torch.randn((1, HIDDEN), dtype=DTYPE, device=DEVICE, generator=generator)
    shift = torch.randn((1, HIDDEN), dtype=DTYPE, device=DEVICE, generator=generator)
    input_scale = torch.tensor(0.03125, dtype=torch.float32, device=DEVICE)

    if impl == "split":

        def fn():
            normalized = fused_layernorm_modulate_raw(x, scale, shift, EPS)
            return static_quant_fp8(normalized, input_scale)[0]

    else:

        def fn():
            return fused_layernorm_modulate_fp8_quant_raw(
                x, scale, shift, input_scale, EPS
            )

    return marker.do_bench(fn, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
