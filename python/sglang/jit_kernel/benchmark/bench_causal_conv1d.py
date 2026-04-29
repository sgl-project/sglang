"""Benchmark for the JIT causal_conv1d forward / update kernels."""

import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.causal_conv1d import causal_conv1d_fwd, causal_conv1d_update
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn as triton_causal_conv1d_fn,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_update as triton_causal_conv1d_update,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-kernel-benchmark-1-gpu-large")

PAD_SLOT_ID = -1
WIDTH = 4

# Forward sweep — varlen format (dim, total_tokens) used by both kernels.
fwd_batch_sizes = get_benchmark_range(full_range=[1, 4, 16, 64], ci_range=[16])
fwd_seqlens = get_benchmark_range(
    full_range=[128, 512, 1024, 2048, 4096], ci_range=[1024]
)
fwd_dim_range = get_benchmark_range(full_range=[1024, 2048, 4096], ci_range=[2048])

# Update sweep — single-step decode (seqlen == 1).
upd_batch_sizes = get_benchmark_range(full_range=[1, 4, 16, 64, 256], ci_range=[16])
upd_dim_range = get_benchmark_range(full_range=[1024, 2048, 4096], ci_range=[2048])

fwd_configs = list(itertools.product(fwd_batch_sizes, fwd_dim_range, fwd_seqlens))
upd_configs = list(itertools.product(upd_batch_sizes, upd_dim_range))

PROVIDERS = ["jit", "triton"]
PROVIDER_NAMES = ["JIT CUDA", "Triton"]
PROVIDER_STYLES = [("red", "-"), ("blue", "-")]


def _make_fwd_inputs(batch, dim, seqlen, dtype):
    """Allocate forward-kernel inputs in **varlen** format (the format both kernels accept).

    Returns ``(x_varlen, weight, bias, conv_states, query_start_loc, seq_lens_cpu)``
    where ``x_varlen`` has shape ``(dim, batch*seqlen)`` and ``query_start_loc``
    splits it into ``batch`` equal-length sequences.
    """
    total_tokens = batch * seqlen
    x_varlen = torch.randn(dim, total_tokens, device=DEFAULT_DEVICE, dtype=dtype).contiguous()
    weight = torch.randn(dim, WIDTH, device=DEFAULT_DEVICE, dtype=dtype)
    bias = torch.randn(dim, device=DEFAULT_DEVICE, dtype=dtype)
    conv_states = torch.zeros(batch, dim, WIDTH - 1, device=DEFAULT_DEVICE, dtype=dtype)
    query_start_loc = torch.tensor(
        [i * seqlen for i in range(batch + 1)], dtype=torch.int32, device=DEFAULT_DEVICE
    )
    seq_lens_cpu = [seqlen] * batch
    return x_varlen, weight, bias, conv_states, query_start_loc, seq_lens_cpu


def _make_update_inputs(batch, dim, dtype):
    """Allocate update-kernel inputs (single-step decode)."""
    x = torch.randn(batch, dim, 1, device=DEFAULT_DEVICE, dtype=dtype)
    conv_state = torch.randn(batch, dim, WIDTH - 1, device=DEFAULT_DEVICE, dtype=dtype)
    weight = torch.randn(dim, WIDTH, device=DEFAULT_DEVICE, dtype=dtype)
    bias = torch.randn(dim, device=DEFAULT_DEVICE, dtype=dtype)
    return x, conv_state, weight, bias


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "dim", "seqlen"],
        x_vals=[list(c) for c in fwd_configs],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="us",
        plot_name="causal_conv1d-fwd",
        args={},
    )
)
def benchmark_fwd(batch_size, dim, seqlen, provider):
    """Benchmark a single mamba forward (prefill) call across kernel implementations."""
    dtype = DEFAULT_DTYPE
    x, weight, bias, conv_states, qsl, seq_lens_cpu = _make_fwd_inputs(
        batch_size, dim, seqlen, dtype
    )

    if provider == "jit":
        # JIT writes back into x in place; clone each iteration so the benchmark stays stable.
        def fn():
            xx = x.clone()
            cs = conv_states.clone()
            causal_conv1d_fwd(
                xx, weight, bias, cs, qsl, None, None, True, PAD_SLOT_ID
            )

    elif provider == "triton":

        def fn():
            xx = x.clone()
            cs = conv_states.clone()
            triton_causal_conv1d_fn(
                xx,
                weight,
                bias,
                cs,
                qsl,
                seq_lens_cpu,
                cache_indices=None,
                has_initial_state=None,
                activation="silu",
                pad_slot_id=PAD_SLOT_ID,
            )

    else:
        raise ValueError(f"unknown provider: {provider}")

    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "dim"],
        x_vals=[list(c) for c in upd_configs],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDER_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="us",
        plot_name="causal_conv1d-update",
        args={},
    )
)
def benchmark_update(batch_size, dim, provider):
    """Benchmark a single mamba decode (seqlen=1) update call across kernel implementations."""
    dtype = DEFAULT_DTYPE
    x, conv_state, weight, bias = _make_update_inputs(batch_size, dim, dtype)

    if provider == "jit":

        def fn():
            xx = x.clone()
            cs = conv_state.clone()
            causal_conv1d_update(
                xx, cs, weight, bias, True, None, None, PAD_SLOT_ID
            )

    elif provider == "triton":

        def fn():
            xx = x.clone()
            cs = conv_state.clone()
            triton_causal_conv1d_update(
                xx, cs, weight, bias=bias, activation="silu", pad_slot_id=PAD_SLOT_ID
            )

    else:
        raise ValueError(f"unknown provider: {provider}")

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark_fwd.run(print_data=True)
    benchmark_update.run(print_data=True)
