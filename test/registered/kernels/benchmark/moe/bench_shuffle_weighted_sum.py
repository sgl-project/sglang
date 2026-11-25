"""Whole mapped reduction, with cold-L2 graph input/output rotation."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe.shuffle_weighted_sum import (
    _shuffle_weighted_sum_triton,
    shuffle_weighted_sum,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("tokens,width,topk", [(32, 256, 2), (512, 256, 2), (64, 4096, 8)])
@marker.benchmark("implementation", ["cuda", "triton"])
def benchmark(tokens: int, width: int, topk: int, implementation: str):
    values = torch.randn((tokens * topk, width), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((tokens, width), device="cuda", dtype=torch.bfloat16)
    mapping = torch.randperm(tokens * topk, device="cuda", dtype=torch.int32)
    weights = torch.rand((tokens, topk), device="cuda", dtype=torch.float32)
    function = (
        shuffle_weighted_sum
        if implementation == "cuda"
        else _shuffle_weighted_sum_triton
    )
    return marker.do_bench(
        function,
        input_args=(values, output, mapping, weights),
        memory_args=(values, mapping, weights),
        memory_output=(output,),
        graph_clone_args=(0, 1, 2, 3),
    )


if __name__ == "__main__":
    benchmark.run()
