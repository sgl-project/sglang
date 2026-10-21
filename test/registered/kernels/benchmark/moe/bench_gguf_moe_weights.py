"""Compare GGUF in-place weighting without hiding memory traffic in L2."""

import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe.gguf_moe_weights_aot import resolve_gguf_moe_weights
from sglang.srt.utils import is_musa
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def torch_weight(outputs, weights, ids):
    inactive = ids < 0
    outputs.masked_fill_(inactive.reshape(-1, 1), 0)
    outputs.view(*ids.shape, -1).mul_(weights.masked_fill(inactive, 0).unsqueeze(-1))


@marker.parametrize("tokens", [1, 64, 1024, 8192], [1024])
@marker.parametrize("hidden", [4096, 7168], [4096])
@marker.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32], [torch.float16]
)
@marker.benchmark("provider", ["aot", "torch"])
def benchmark(tokens, hidden, dtype, provider):
    device = "musa" if is_musa() else "cuda"
    native = resolve_gguf_moe_weights(device)
    if provider == "aot" and native is None:
        marker.skip("Native GGUF weighting requires a rebuilt kernel provider")
    outputs = torch.ones(tokens * 8, hidden, device=device, dtype=dtype)
    # Unit weights keep repeated in-place calls numerically stable.
    weights = torch.ones(tokens, 8, device=device)
    ids = torch.zeros(tokens, 8, dtype=torch.int32, device=device)
    ids[::3, -1] = -1
    return marker.do_bench(
        native if provider == "aot" else torch_weight,
        input_args=(outputs, weights, ids),
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark.run()
