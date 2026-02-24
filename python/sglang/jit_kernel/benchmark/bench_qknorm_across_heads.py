import itertools
from typing import Tuple

import torch
import triton
import triton.testing
from sgl_kernel import rmsnorm

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.jit_kernel.norm import fused_inplace_qknorm_across_heads
from sglang.srt.utils import get_current_device_stream_fast

IS_CI = is_in_ci()

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


DTYPE = torch.bfloat16
DEVICE = "cuda"

if IS_CI:
    BS_RANGE = [16]
    HIDDEN_DIM_RANGE = [1024]
else:
    BS_RANGE = [2**n for n in range(0, 14)]
    HIDDEN_DIM_RANGE = [512, 1024, 2048, 4096, 8192]

LINE_VALS = ["jit", "aot", "fi", "torch"]
LINE_NAMES = ["SGL JIT Kernel", "SGL AOT Kernel", "FlashInfer", "PyTorch"]
STYLES = [("blue", "-"), ("orange", "--"), ("green", "-."), ("red", ":")]

configs = list(itertools.product(BS_RANGE, HIDDEN_DIM_RANGE))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_dim"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="qknorm-across-heads-performance",
        args={},
    )
)
def benchmark(
    batch_size: int, hidden_dim: int, provider: str
) -> Tuple[float, float, float]:
    q = torch.randn((batch_size, hidden_dim), dtype=DTYPE, device=DEVICE)
    k = torch.randn((batch_size, hidden_dim), dtype=DTYPE, device=DEVICE)
    q_weight = torch.randn(hidden_dim, dtype=DTYPE, device=DEVICE)
    k_weight = torch.randn(hidden_dim, dtype=DTYPE, device=DEVICE)
    FN_MAP = {
        "jit": sglang_jit_qknorm_across_heads,
        "aot": sglang_aot_qknorm_across_heads,
        "fi": flashinfer_qknorm_across_heads,
        "torch": torch_impl_qknorm_across_heads,
    }
    fn = lambda: FN_MAP[provider](q, k, q_weight, k_weight)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)  # type: ignore
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
