import itertools

import torch
import triton
import triton.testing
from sgl_kernel import rmsnorm

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.norm import fused_inplace_qknorm
from sglang.srt.utils import get_current_device_stream_fast

alt_stream = torch.cuda.Stream()


def sglang_aot_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:

    head_dim = q.shape[-1]
    q = q.view(-1, head_dim)
    k = k.view(-1, head_dim)

    current_stream = get_current_device_stream_fast()
    alt_stream.wait_stream(current_stream)
    rmsnorm(q, q_weight, out=q)
    with torch.cuda.stream(alt_stream):
        rmsnorm(k, k_weight, out=k)
    current_stream.wait_stream(alt_stream)


def sglang_jit_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:

    fused_inplace_qknorm(q, k, q_weight, k_weight)


def flashinfer_qknorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    from flashinfer import rmsnorm

    rmsnorm(q, q_weight, out=q)
    rmsnorm(k, k_weight, out=k)


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

LINE_VALS = ["aot", "jit", "fi", "torch"]
LINE_NAMES = ["SGL AOT Kernel", "SGL JIT Kernel", "FlashInfer", "PyTorch"]
STYLES = [("orange", "-"), ("blue", "--"), ("green", "-."), ("red", ":")]

configs = list(itertools.product(HEAD_DIM_RANGE, GQA_RANGE, KV_HEAD_RANGE, BS_RANGE))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["head_dim", "GQA", "num_kv_heads", "batch_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="qknorm-performance",
        args={},
    )
)
def benchmark(
    head_dim: int, GQA: int, num_kv_heads: int, batch_size: int, provider: str
):
    num_qo_heads = GQA * num_kv_heads
    q = torch.randn(
        (batch_size, num_qo_heads, head_dim), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    k = torch.randn(
        (batch_size, num_kv_heads, head_dim), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    q_weight = torch.randn(head_dim, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    k_weight = torch.randn(head_dim, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    FN_MAP = {
        "aot": sglang_aot_qknorm,
        "jit": sglang_jit_qknorm,
        "fi": flashinfer_qknorm,
        "torch": torch_impl_qknorm,
    }
    fn = lambda: FN_MAP[provider](q, k, q_weight, k_weight)
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
