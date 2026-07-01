import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.norm import fused_inplace_qknorm_across_heads
from sglang.srt.utils import get_current_device_stream_fast
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

alt_stream = torch.cuda.Stream()


def sglang_jit_qknorm_across_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    fused_inplace_qknorm_across_heads(q, k, q_weight, k_weight)


def sglang_aot_qknorm_across_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    from sgl_kernel import rmsnorm

    current_stream = get_current_device_stream_fast()
    alt_stream.wait_stream(current_stream)
    rmsnorm(q, q_weight, out=q)
    with torch.cuda.stream(alt_stream):
        rmsnorm(k, k_weight, out=k)
    current_stream.wait_stream(alt_stream)


def flashinfer_qknorm_across_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    from flashinfer import rmsnorm

    rmsnorm(q, q_weight, out=q)
    rmsnorm(k, k_weight, out=k)


@torch.compile()
def torch_impl_qknorm_across_heads(
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


FN_MAP = {
    "jit": sglang_jit_qknorm_across_heads,
    "aot": sglang_aot_qknorm_across_heads,
    "flashinfer": flashinfer_qknorm_across_heads,
    "torch": torch_impl_qknorm_across_heads,
}


@marker.parametrize("hidden_dim", [512, 1024, 2048, 4096, 8192], [1024])
@marker.parametrize("batch_size", [2**n for n in range(0, 14)], [16])
@marker.benchmark("impl", ["jit", "aot", "flashinfer", "torch"])
def benchmark(batch_size: int, hidden_dim: int, impl: str):
    q = create_random(batch_size, hidden_dim)
    k = create_random(batch_size, hidden_dim)
    q_weight = create_random(hidden_dim)
    k_weight = create_random(hidden_dim)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(q, k, q_weight, k_weight),
        memory_output=(q, k),  # inplace write to q, k
    )


if __name__ == "__main__":
    benchmark.run()
