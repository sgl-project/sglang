import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.diffusion import fused_gelu_tanh_cat
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def eager(attn, mlp):
    return torch.cat((attn, F.gelu(mlp, approximate="tanh")), dim=-1)


@marker.parametrize("tokens", [859, 4608, 9233], [859, 9233])
@marker.benchmark("provider", ["eager", "fused"])
def benchmark(tokens, provider):
    attn = create_random(1, tokens, 3072)
    mlp = create_random(1, tokens, 12288)
    fn = eager if provider == "eager" else fused_gelu_tanh_cat
    return marker.do_bench(fn, input_args=(attn, mlp))


if __name__ == "__main__":
    benchmark.run()
