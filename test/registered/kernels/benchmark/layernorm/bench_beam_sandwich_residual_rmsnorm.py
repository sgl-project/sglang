"""Beam norm/residual latency at decode and prefill sizes."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.layernorm import _BEAM_SANDWICH_RESIDUAL_RMSNORM
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=13, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("rows", [1, 16, 64, 128, 1024, 8192], [1, 128])
@marker.benchmark("impl", ["triton", "torch"])
def benchmark(rows: int, impl: str):
    x = torch.randn(rows, 5120, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.ones(5120, dtype=x.dtype, device="cuda")
    return marker.do_bench(
        lambda *args: _BEAM_SANDWICH_RESIDUAL_RMSNORM(
            *args, backend=KernelBackend(impl)
        ),
        input_args=(x, residual, weight, weight, 1e-6, (0.25, 0.968)),
        memory_args=(x, residual, weight),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
