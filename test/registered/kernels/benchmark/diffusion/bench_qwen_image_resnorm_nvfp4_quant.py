import flashinfer
import torch

from sglang.kernels.ops.diffusion import (
    fused_scale_residual_norm_scale_shift,
    try_fused_scale_residual_norm_scale_shift_nvfp4,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="standalone Qwen-Image NVFP4 residual-norm benchmark",
)


def _benchmark(fn, iterations: int = 100) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def _run_case(token_count: int) -> None:
    hidden_size = 3072
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260830 + token_count)

    def randn(shape):
        return torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        ).contiguous()

    x = randn((1, token_count, hidden_size))
    residual = randn(x.shape)
    input_bias = randn((hidden_size,))
    gate = randn((1, 1, hidden_size))
    scale = randn((1, 1, hidden_size))
    shift = randn((1, 1, hidden_size))
    global_scale = torch.tensor(0.625, device="cuda", dtype=torch.float32)

    def baseline():
        modulated, residual_out = fused_scale_residual_norm_scale_shift(
            residual, x + input_bias, gate, None, None, scale, shift, "layer", 1e-6
        )
        quantized, quant_scales = flashinfer.fp4_quantize(
            modulated.view(-1, hidden_size), global_scale
        )
        return quantized, quant_scales, residual_out

    def fused():
        result = try_fused_scale_residual_norm_scale_shift_nvfp4(
            residual,
            x,
            input_bias,
            gate,
            None,
            None,
            scale,
            shift,
            global_scale,
            "layer",
            1e-6,
        )
        assert result is not None
        (quantized, quant_scales), residual_out = result
        return quantized, quant_scales, residual_out

    expected = baseline()
    actual = fused()
    exact = [torch.equal(lhs, rhs) for lhs, rhs in zip(actual, expected)]
    baseline_us = _benchmark(baseline)
    fused_us = _benchmark(fused)
    print(
        {
            "tokens": token_count,
            "baseline_us": baseline_us,
            "fused_us": fused_us,
            "speedup": baseline_us / fused_us,
            "exact": exact,
        }
    )


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("This benchmark requires an NVIDIA Blackwell SM10x GPU")
    for tokens in (17, 1024, 4096, 4608):
        _run_case(tokens)
