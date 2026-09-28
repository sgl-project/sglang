import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    fused_norm_scale_shift_fp8,
    fused_scale_residual_norm_scale_shift_fp8,
)
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    ScaleResidualLayerNormScaleShift,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 3072
EPS = 1e-6


@marker.parametrize("rows", [128, 1024, 4096], [128])
@marker.parametrize("residual_path", [False, True], [False, True])
@marker.benchmark("impl", ["split", "fused"], unit="us")
def benchmark(rows: int, residual_path: bool, impl: str):
    if impl == "fused" and torch.cuda.get_device_capability()[0] < 10:
        marker.skip("Fused Qwen-Image norm+FP8 quant requires NVIDIA Blackwell")

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260831 + rows + int(residual_path))
    x = torch.randn((1, rows, HIDDEN), dtype=DTYPE, device=DEVICE, generator=generator)
    residual = torch.randn_like(x)
    gate = torch.randn((HIDDEN,), dtype=DTYPE, device=DEVICE, generator=generator)
    scale = torch.randn((HIDDEN,), dtype=DTYPE, device=DEVICE, generator=generator)
    shift = torch.randn((HIDDEN,), dtype=DTYPE, device=DEVICE, generator=generator)
    input_scale = torch.tensor(0.03125, dtype=torch.float32, device=DEVICE)

    if residual_path:
        layer = ScaleResidualLayerNormScaleShift(
            HIDDEN, eps=EPS, elementwise_affine=False, dtype=DTYPE
        ).to(DEVICE)

        if impl == "split":

            def fn():
                normalized, residual_out = layer.forward_cuda(
                    residual, x, gate, shift, scale
                )
                quantized, _ = static_quant_fp8(normalized, input_scale)
                return quantized, residual_out

        else:

            def fn():
                return fused_scale_residual_norm_scale_shift_fp8(
                    residual, x, gate, scale, shift, input_scale, EPS
                )

    else:
        layer = LayerNormScaleShift(
            HIDDEN, eps=EPS, elementwise_affine=False, dtype=DTYPE
        ).to(DEVICE)

        if impl == "split":

            def fn():
                normalized = layer.forward_cuda(x, shift, scale)
                return static_quant_fp8(normalized, input_scale)[0]

        else:

            def fn():
                return fused_norm_scale_shift_fp8(x, scale, shift, input_scale, EPS)

    return marker.do_bench(fn, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
