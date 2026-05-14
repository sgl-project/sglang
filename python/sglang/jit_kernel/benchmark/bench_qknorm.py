import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import (
    create_random,
    get_benchmark_range,
)
from sglang.jit_kernel.norm import fused_inplace_qknorm
from sglang.srt.utils import get_current_device_stream_fast
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="base-b-kernel-benchmark-1-gpu-large")

alt_stream = torch.cuda.Stream()

torch._dynamo.config.recompile_limit = 100


# NOTE: now aot fallback to flashinfer
def sglang_aot_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    from flashinfer import rmsnorm  # lazy import to avoid crash

    current_stream = get_current_device_stream_fast()
    alt_stream.wait_stream(current_stream)
    rmsnorm(q, q_weight, out=q)
    with torch.cuda.stream(alt_stream):
        rmsnorm(k, k_weight, out=k)
    current_stream.wait_stream(alt_stream)


@torch.compile()
def torch_impl_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    q_mean = q.float().pow(2).mean(dim=-1, keepdim=True)
    k_mean = k.float().pow(2).mean(dim=-1, keepdim=True)
    q_norm = (q_mean + eps).rsqrt()
    k_norm = (k_mean + eps).rsqrt()
    q.copy_(q.float() * q_norm * q_weight.float())
    k.copy_(k.float() * k_norm * k_weight.float())


BS_RANGE = get_benchmark_range(
    full_range=[2**n for n in range(0, 14)],
    ci_range=[16],
)
GQA_RANGE = get_benchmark_range(
    full_range=[4, 8],
    ci_range=[4],
)
KV_HEAD_RANGE = get_benchmark_range(
    full_range=[1, 2, 4, 8],
    ci_range=[1],
)
HEAD_DIM_RANGE = get_benchmark_range(
    full_range=[128, 256, 512, 1024],
    ci_range=[128],
)
FN_MAP = {
    "aot": sglang_aot_qknorm,
    "jit": fused_inplace_qknorm,
    "torch": torch_impl_qknorm,
}


@marker.mark_args("head_dim", HEAD_DIM_RANGE)
@marker.mark_args("GQA", GQA_RANGE)
@marker.mark_args("num_kv_heads", KV_HEAD_RANGE)
@marker.mark_args("batch_size", BS_RANGE)
@marker.mark_benchmark("impl", ["aot", "jit", "torch"])
def benchmark(head_dim: int, GQA: int, num_kv_heads: int, batch_size: int, impl: str):
    num_qo_heads = GQA * num_kv_heads
    q = create_random(batch_size, num_qo_heads, head_dim)
    k = create_random(batch_size, num_kv_heads, head_dim)
    q_weight = create_random(head_dim)
    k_weight = create_random(head_dim)
    return marker.bench_one_function(
        FN_MAP[impl],
        input_args=(q, k, q_weight, k_weight),
        memory_args="all",
    )


if __name__ == "__main__":
    benchmark.run()
