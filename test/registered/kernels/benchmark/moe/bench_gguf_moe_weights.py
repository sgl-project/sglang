"""Compare all-valid GGUF weighting with the original in-place product."""

import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe.gguf_moe_vec import gguf_moe_a8_vec
from sglang.kernels.ops.moe.gguf_moe_weights_aot import resolve_gguf_moe_weights
from sglang.srt.utils import is_musa
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def torch_weight(outputs, weights, ids):
    outputs.view(*ids.shape, -1).mul_(weights.unsqueeze(-1))


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
    return marker.do_bench(
        native if provider == "aot" else torch_weight,
        input_args=(outputs, weights, ids),
        memory_output=(outputs,),
    )


@marker.parametrize("tokens", [1, 32, 64, 128], [64])
@marker.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32], [torch.float16]
)
@marker.benchmark("provider", ["jit", "aot"])
def benchmark_vector(tokens, dtype, provider):
    """Compare the same all-valid Q4_0 vector operation, including private scratch."""
    from sgl_kernel.quantization import ggml_moe_a8_vec

    if is_musa():
        marker.skip("Local GGUF vector projection is CUDA-only")
    inputs = torch.full((tokens, 4096), 0.25, device="cuda", dtype=dtype)
    packed = torch.full(
        (4, 128, 4096 // 32 * 18), 0x31, device="cuda", dtype=torch.uint8
    )
    ids = torch.zeros(tokens, 2, dtype=torch.int32, device="cuda")
    return marker.do_bench(
        gguf_moe_a8_vec if provider == "jit" else ggml_moe_a8_vec,
        input_args=(inputs, packed, ids, 2, 2, 128, tokens),
    )


if __name__ == "__main__":
    benchmark.run()
    benchmark_vector.run()
