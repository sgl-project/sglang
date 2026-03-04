import itertools

import torch
import triton
import triton.testing
from sgl_kernel import concat_mla_absorb_q as aot_absorb_q
from sgl_kernel import concat_mla_k as aot_k

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.jit_kernel.concat_mla import concat_mla_absorb_q as jit_absorb_q
from sglang.jit_kernel.concat_mla import concat_mla_k as jit_k

IS_CI = is_in_ci()

# Constants
NUM_LOCAL_HEADS = 128
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM

A_LAST_DIM = 512
B_LAST_DIM = 64

DTYPE = torch.bfloat16
DEVICE = "cuda"


def aot_concat_mla_k(k, k_nope, k_rope):
    aot_k(k, k_nope, k_rope)


def jit_concat_mla_k(k, k_nope, k_rope):
    jit_k(k, k_nope, k_rope)


def torch_concat_mla_k(k, k_nope, k_rope):
    nope_head_dim = k_nope.shape[-1]
    k[:, :, :nope_head_dim] = k_nope
    k[:, :, nope_head_dim:] = k_rope.expand(-1, k.shape[1], -1)


def aot_concat_mla_absorb_q(a, b):
    return aot_absorb_q(a, b)


def jit_concat_mla_absorb_q(a, b):
    return jit_absorb_q(a, b)


def torch_concat_mla_absorb_q(a, b, out):
    a_last_dim = a.shape[-1]
    out[:, :, :a_last_dim] = a
    out[:, :, a_last_dim:] = b


if IS_CI:
    NUM_TOKENS_VALS = [256, 1024]
else:
    NUM_TOKENS_VALS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

K_LINE_VALS = ["aot", "jit", "torch"]
K_LINE_NAMES = ["SGL AOT Kernel", "SGL JIT Kernel", "PyTorch"]
K_STYLES = [("orange", "-"), ("blue", "--"), ("green", "-.")]


def _create_concat_mla_k_data(num_tokens):
    """Allocate oversized containers and slice to produce non-contiguous tensors."""
    k_nope_container = torch.randn(
        (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + 128),
        dtype=DTYPE,
        device=DEVICE,
    )
    k_nope = k_nope_container[:, :, :QK_NOPE_HEAD_DIM]

    k_rope_container = torch.randn(
        (num_tokens, 1, 128 + QK_ROPE_HEAD_DIM),
        dtype=DTYPE,
        device=DEVICE,
    )
    k_rope = k_rope_container[:, :, -QK_ROPE_HEAD_DIM:]

    k = torch.empty(
        (num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM),
        dtype=DTYPE,
        device=DEVICE,
    )
    return k, k_nope, k_rope


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=NUM_TOKENS_VALS,
        line_arg="provider",
        line_vals=K_LINE_VALS,
        line_names=K_LINE_NAMES,
        styles=K_STYLES,
        ylabel="us",
        plot_name="concat-mla-k-performance",
        args={},
    )
)
def bench_concat_mla_k(num_tokens: int, provider: str):
    k, k_nope, k_rope = _create_concat_mla_k_data(num_tokens)

    FN_MAP = {
        "aot": aot_concat_mla_k,
        "jit": jit_concat_mla_k,
        "torch": torch_concat_mla_k,
    }
    fn = lambda: FN_MAP[provider](k, k_nope, k_rope)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if IS_CI:
    ABSORB_Q_VALS = list(itertools.product([4, 16], [16]))
else:
    ABSORB_Q_VALS = list(itertools.product([1, 4, 8, 16, 32], [1, 8, 32, 128]))

Q_LINE_VALS = ["aot", "jit", "torch"]
Q_LINE_NAMES = ["SGL AOT Kernel", "SGL JIT Kernel", "PyTorch"]
Q_STYLES = [("orange", "-"), ("blue", "--"), ("green", "-.")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dim_0", "dim_1"],
        x_vals=ABSORB_Q_VALS,
        line_arg="provider",
        line_vals=Q_LINE_VALS,
        line_names=Q_LINE_NAMES,
        styles=Q_STYLES,
        ylabel="us",
        plot_name="concat-mla-absorb-q-performance",
        args={},
    )
)
def bench_concat_mla_absorb_q(dim_0: int, dim_1: int, provider: str):
    a = torch.randn(dim_0, dim_1, A_LAST_DIM, dtype=DTYPE, device=DEVICE)
    b = torch.randn(dim_0, dim_1, B_LAST_DIM, dtype=DTYPE, device=DEVICE)

    if provider == "torch":
        out = torch.empty(
            dim_0, dim_1, A_LAST_DIM + B_LAST_DIM, dtype=DTYPE, device=DEVICE
        )
        fn = lambda: torch_concat_mla_absorb_q(a, b, out)
    else:
        FN_MAP = {
            "aot": aot_concat_mla_absorb_q,
            "jit": jit_concat_mla_absorb_q,
        }
        fn = lambda: FN_MAP[provider](a, b)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    bench_concat_mla_k.run(print_data=True)
    bench_concat_mla_absorb_q.run(print_data=True)
