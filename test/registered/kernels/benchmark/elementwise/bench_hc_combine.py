import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.elementwise.hc_combine import hc_combine as jit_hc_combine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")

HC_COUNT = 4
HIDDEN_SIZE = 2560


def torch_impl_hc_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    """Eager baseline (the production unfused chain, sans torch.compile)."""
    R = residual.unflatten(-1, (hc, hs))
    gates = 2 * torch.sigmoid(F.linear(normed_residual, inject_weight) / hc)
    injection = block_output.unsqueeze(-2) * gates.unsqueeze(-1)
    return (R + injection).flatten(-2)


FN_MAP = {
    "jit": jit_hc_combine,
    "torch_eager": torch_impl_hc_combine,
}


@marker.parametrize("num_tokens", [1, 8, 64, 512, 2048, 8192], [1, 512, 8192])
@marker.benchmark("impl", ["jit", "torch_eager"])
def benchmark(num_tokens: int, impl: str):
    block_output = create_random(num_tokens, HIDDEN_SIZE)
    residual = create_random(num_tokens, HC_COUNT * HIDDEN_SIZE)
    normed_residual = create_random(num_tokens, HC_COUNT * HIDDEN_SIZE)
    inject_weight = create_random(HC_COUNT, HC_COUNT * HIDDEN_SIZE) * 0.02
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(
            block_output,
            residual,
            normed_residual,
            inject_weight,
            HC_COUNT,
            HIDDEN_SIZE,
        ),
        # y / r / n are read -> clone them per iter to avoid L2 reuse; the
        # weight is tiny.
        graph_clone_args=(0, 1, 2),
    )


if __name__ == "__main__":
    benchmark.run()
