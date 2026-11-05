import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.elementwise.sam_rel_pos_bias import sam_rel_pos_bias
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.sam import sam_rel_pos_bias_reference

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# SAM ViT-B global attention: 12 heads.
NUM_HEADS = 12


@marker.parametrize("batch", [1, 2, 4, 8], [1, 8])
@marker.parametrize("queries,key_size", [(1600, 40), (4096, 64)], [(1600, 40)])
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(batch: int, queries: int, key_size: int, impl: str):
    rel_h = torch.randn(
        batch, NUM_HEADS, queries, key_size, 1, device="cuda", dtype=torch.bfloat16
    )
    rel_w = torch.randn(
        batch, NUM_HEADS, queries, 1, key_size, device="cuda", dtype=torch.bfloat16
    )
    fn = sam_rel_pos_bias if impl == "jit" else sam_rel_pos_bias_reference
    return marker.do_bench(
        fn,
        input_args=(rel_h, rel_w),
        # Both inputs are read; clone them per iteration to defeat L2 reuse.
        graph_clone_args=(0, 1),
    )


if __name__ == "__main__":
    benchmark.run()
