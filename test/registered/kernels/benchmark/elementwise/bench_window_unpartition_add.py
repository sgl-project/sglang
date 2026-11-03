import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.elementwise.window_unpartition_add import (
    window_unpartition_add,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.sam import make_windows, window_unpartition_add_reference

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("batch", [1, 2, 4, 8], [1, 8])
@marker.parametrize("height,width", [(40, 40), (64, 64)], [(40, 40)])
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(batch: int, height: int, width: int, impl: str):
    shortcut = torch.randn(
        batch, height, width, 768, device="cuda", dtype=torch.bfloat16
    )
    windows = make_windows(torch.randn_like(shortcut), 14)
    fn = window_unpartition_add if impl == "jit" else window_unpartition_add_reference
    return marker.do_bench(
        fn,
        input_args=(windows, shortcut, 14),
        graph_clone_args=(0, 1),
    )


if __name__ == "__main__":
    benchmark.run()
