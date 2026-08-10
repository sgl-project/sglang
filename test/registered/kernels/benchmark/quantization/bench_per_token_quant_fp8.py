import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.quantization.per_token_quant_fp8 import per_token_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _sglang_quant(input, output, scale):
    per_token_quant_fp8(input, output, scale)


@marker.parametrize("num_tokens", [1, 39, 128, 512, 1392, 7807], [39, 1392])
@marker.parametrize("hidden_dim", [512, 1076, 1368, 1536, 2048, 4096], [1536])
@marker.benchmark("dtype", [torch.float16, torch.bfloat16])
def benchmark(num_tokens: int, hidden_dim: int, dtype: torch.dtype):
    input = create_random(num_tokens, hidden_dim, dtype=dtype)
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scale = torch.empty((num_tokens, 1), dtype=torch.float32, device="cuda")
    return marker.do_bench(
        _sglang_quant,
        input_args=(input, output, scale),
        memory_args=(input,),
        memory_output=(output, scale),
        graph_clone_args=(0, 1, 2),
    )


if __name__ == "__main__":
    benchmark.run()
