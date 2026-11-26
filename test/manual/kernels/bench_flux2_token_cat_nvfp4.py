import time

import flashinfer
import torch

from sglang.kernels.ops.diffusion import try_flux2_token_cat_nvfp4


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


def _benchmark_wall(fn, iterations: int = 100) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / iterations / 1000


def _run_case(token_count: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260830 + token_count)
    attention = torch.randn(
        1,
        token_count,
        6144,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    mlp = torch.randn(
        1,
        token_count,
        18432,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    global_scale = torch.tensor(0.625, device="cuda", dtype=torch.float32)

    def baseline():
        return flashinfer.fp4_quantize(
            torch.cat([attention, mlp], dim=-1).view(-1, 24576), global_scale
        )

    def fused():
        result = try_flux2_token_cat_nvfp4(attention, mlp, global_scale)
        assert result is not None
        return result

    expected = baseline()
    actual = fused()
    exact = [torch.equal(lhs, rhs) for lhs, rhs in zip(actual, expected)]
    baseline_us = _benchmark(baseline)
    fused_us = _benchmark(fused)
    baseline_wall_us = _benchmark_wall(baseline)
    fused_wall_us = _benchmark_wall(fused)
    print(
        {
            "tokens": token_count,
            "baseline_us": baseline_us,
            "fused_us": fused_us,
            "speedup": baseline_us / fused_us,
            "baseline_wall_us": baseline_wall_us,
            "fused_wall_us": fused_wall_us,
            "wall_speedup": baseline_wall_us / fused_wall_us,
            "exact": exact,
        }
    )


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("This benchmark requires an NVIDIA Blackwell SM103 GPU")
    for tokens in (17, 512, 4096, 4608):
        _run_case(tokens)
