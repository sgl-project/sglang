"""Bandwidth benchmark for the Inkling MoE silu_and_mul over both gate/up layouts.

Re-run after touching the tuned configs in
``sglang.kernels.ops.moe.inkling_silu_config``. The op is bandwidth-bound, so the
number to watch is achieved bandwidth, not latency.
"""

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.moe.inkling_moe import silu_and_mul_triton
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("layout", ["interleaved", "contiguous"])
@marker.parametrize("n", [512, 2048, 4608, 6144, 8192], [2048])
@marker.parametrize("m", [1, 16, 128, 1024, 4096, 16384], [128, 4096])
@marker.benchmark("weights", ["no_weights", "topk_weights"])
def benchmark(layout: str, n: int, m: int, weights: str):
    gateup = create_random(m, 2 * n)
    topk_weights = create_random(m) if weights == "topk_weights" else None
    return marker.do_bench(
        silu_and_mul_triton,
        input_args=(gateup, topk_weights),
        input_kwargs={"use_interleaved": layout == "interleaved"},
    )


if __name__ == "__main__":
    benchmark.run()
