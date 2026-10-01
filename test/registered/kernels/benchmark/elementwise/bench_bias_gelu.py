import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.elementwise.bias_gelu import bias_gelu_tanh
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def torch_bias_gelu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.gelu(input + bias, approximate="tanh")


FN_MAP = {
    "jit": bias_gelu_tanh,
    "torch": torch_bias_gelu,
}


@marker.parametrize(
    "rows,hidden_dim",
    [(32760, 5120), (32760, 13824)],
    [(4096, 13824)],
)
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(rows: int, hidden_dim: int, impl: str):
    input = torch.randn(rows, hidden_dim, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(hidden_dim, dtype=torch.bfloat16, device="cuda")
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(input, bias),
        memory_args=(input, bias),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
