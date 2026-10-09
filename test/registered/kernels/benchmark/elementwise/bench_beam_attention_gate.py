"""Beam attention-gate latency at decode and prefill sizes."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.elementwise import _BEAM_ATTENTION_GATE
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=7, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("rows", [1, 16, 64, 128, 1024, 8192], [1, 128])
@marker.parametrize("packed_gate", [False, True])
@marker.benchmark("impl", ["triton", "torch"])
def benchmark(rows: int, packed_gate: bool, impl: str):
    x = torch.randn(rows, 6 * 128, dtype=torch.bfloat16, device="cuda")
    if packed_gate:
        # TP8 QKV+gate widths: Q=768, K=128, V=128, gate=768.
        gate_source = torch.randn(rows, 1792, dtype=x.dtype, device=x.device)
    else:
        gate_source = torch.randn_like(x)
    gate_start = gate_source.shape[1] - x.shape[1]
    bias = torch.randn(6, dtype=x.dtype, device="cuda")
    # The marker rotates clones, and a clone of a strided slice is contiguous;
    # rotate the backing tensor and slice the gate inside the timed call.
    return marker.do_bench(
        lambda x, gate_source, bias: _BEAM_ATTENTION_GATE(
            x, gate_source[:, gate_start:], bias, 128, backend=KernelBackend(impl)
        ),
        input_args=(x, gate_source, bias),
        memory_args=(x, gate_source[:, gate_start:], bias),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
