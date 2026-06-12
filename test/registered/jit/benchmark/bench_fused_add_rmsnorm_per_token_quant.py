import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.fused_add_rmsnorm_per_token_quant import (
    fused_add_rmsnorm_per_token_quant,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-benchmark-1-gpu-large")


def _unfused(x, residual, weight, eps):
    # Two-kernel reference: flashinfer fused_add_rmsnorm + sgl per-token quant.
    from sgl_kernel import fused_add_rmsnorm

    fused_add_rmsnorm(x, residual, weight, eps)
    return sglang_per_token_quant_fp8(x)


def _fused(x, residual, weight, eps):
    return fused_add_rmsnorm_per_token_quant(x, residual, weight, eps=eps)


@marker.parametrize("dim", [4096, 8192], [8192])
@marker.parametrize("batch_size", [2**x for x in range(0, 13)], [16, 64])
@marker.benchmark("impl", ["unfused", "fused"])
def benchmark(dim: int, batch_size: int, impl: str):
    torch.random.manual_seed(42)
    x = torch.randn(batch_size, dim, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(batch_size, dim, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(dim, dtype=torch.bfloat16, device="cuda")
    fn = _fused if impl == "fused" else _unfused
    # x/residual are mutated in place by both impls; numerics are irrelevant
    # for timing, so reuse the same buffers across iterations.
    return marker.do_bench(fn, input_args=(x, residual, weight, 1e-6))


if __name__ == "__main__":
    # standalone runner (CI uses the marker-decorated entry above)
    import triton.testing

    print(f"{'M':>6} {'D':>6} {'unfused(us)':>12} {'fused(us)':>10} {'speedup':>8}")
    for d in (4096, 8192):
        for m in (1, 4, 16, 64, 256, 1024, 4096):
            x = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
            r = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(d, dtype=torch.bfloat16, device="cuda")
            t_unfused = triton.testing.do_bench(
                lambda: _unfused(x, r, w, 1e-6), warmup=25, rep=100
            )
            t_fused = triton.testing.do_bench(
                lambda: _fused(x, r, w, 1e-6), warmup=25, rep=100
            )
            print(
                f"{m:>6} {d:>6} {t_unfused*1e3:>12.2f} {t_fused*1e3:>10.2f}"
                f" {t_unfused/t_fused:>7.2f}x"
            )
