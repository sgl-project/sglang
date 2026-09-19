import torch
from sgl_kernel import moe_sum as aot_moe_sum

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.moe.moe_sum import moe_sum as jit_moe_sum
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

FN_MAP = {"jit": jit_moe_sum, "aot": aot_moe_sum}


@marker.parametrize("num_tokens", [1, 8, 128, 1024, 4096], [1, 128])
@marker.parametrize("topk", [2, 3, 4], [2, 4])
@marker.parametrize("hidden_size", [128, 512, 1024, 4096, 8192], [1024, 4096])
@marker.parametrize("dtype", [torch.float16, torch.bfloat16])
@marker.benchmark("impl", ["jit", "aot"])
def benchmark(
    num_tokens: int, topk: int, hidden_size: int, dtype: torch.dtype, impl: str
):
    input = create_random(num_tokens, topk, hidden_size, dtype=dtype)
    output = torch.empty((num_tokens, hidden_size), dtype=dtype, device="cuda")
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(input, output),
        memory_args=(input,),
        memory_output=(output,),
        graph_clone_args=(0,),
    )


if __name__ == "__main__":
    benchmark.run()
