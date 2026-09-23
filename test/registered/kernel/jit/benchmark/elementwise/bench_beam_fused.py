"""Beam norm/residual and gate latency at decode and prefill sizes."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.elementwise import _BEAM_ATTENTION_GATE
from sglang.kernels.ops.layernorm import _BEAM_SANDWICH_RESIDUAL_RMSNORM
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("rows", [1, 16, 64, 128, 1024, 8192], [1, 128])
@marker.parametrize("op_name", ["norm", "norm_fp32", "gate"])
@marker.benchmark("impl", ["triton", "torch"])
def benchmark(rows: int, op_name: str, impl: str):
    hidden = 6 * 128 if op_name == "gate" else 5120
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.ones(hidden, dtype=x.dtype, device="cuda")
    if op_name == "gate":
        op = _BEAM_ATTENTION_GATE
        bias = torch.randn(6, dtype=x.dtype, device="cuda")
        args = (x, residual, bias, 128)
        memory_args = (x, residual, bias)
    else:
        op = _BEAM_SANDWICH_RESIDUAL_RMSNORM
        args = (
            x,
            residual,
            weight,
            weight,
            1e-6,
            (0.25, 0.968),
            op_name == "norm_fp32",
        )
        memory_args = (x, residual, weight)
    return marker.do_bench(
        lambda *args: op(*args, backend=KernelBackend(impl)),
        input_args=args,
        memory_args=memory_args,
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
