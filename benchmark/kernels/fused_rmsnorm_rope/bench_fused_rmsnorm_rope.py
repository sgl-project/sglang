"""Benchmark: fused RMSNorm+RoPE vs separate RMSNorm + rope_apply_head_dim.

Measures operator-level latency only (no full model forward).

Usage:
    PYTHONPATH=python:$PYTHONPATH python benchmark/kernels/fused_rmsnorm_rope/bench_fused_rmsnorm_rope.py
"""

import itertools
import os
import sys

import torch
import triton.testing as tt

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "test", "kernels")
)
from import_fused_norm import import_fused_norm_module

triton_ops = import_fused_norm_module()


# ---------------------------------------------------------------------------
# Separate (baseline) implementation: Triton rms_norm_fn + rope_apply_head_dim
# Matches the actual model fallback path in SelfAttention.forward()
# ---------------------------------------------------------------------------

from einops import rearrange

rms_norm_fn = triton_ops.rms_norm_fn


def rope_apply_head_dim(x, freqs, head_dim):
    x = rearrange(x, "b s (n d) -> b s n d", d=head_dim)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def separate_rmsnorm_rope(x, weight, freqs_complex, head_dim, eps):
    """Baseline: Triton RMSNorm (rms_norm_fn) + rope_apply_head_dim."""
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    x_normed = rms_norm_fn(x_2d, weight, bias=None, residual=None, eps=eps)
    x_normed = x_normed.view(shape)
    return rope_apply_head_dim(x_normed, freqs_complex, head_dim)


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------


def _build_inputs(B, S, D, head_dim, dtype=torch.bfloat16, device="cuda"):
    x = torch.randn(B, S, D, dtype=dtype, device=device)
    weight = torch.randn(D, dtype=dtype, device=device) * 0.5 + 1.0

    head_dim_half = head_dim // 2
    angles = torch.randn(S, head_dim_half, device=device, dtype=torch.float32) * 0.5
    cos = angles.cos().contiguous()
    sin = angles.sin().contiguous()

    freqs_complex = torch.polar(
        torch.ones_like(angles, dtype=torch.float64), angles.double()
    ).unsqueeze(
        1
    )  # [S, 1, head_dim//2]

    return dict(
        x=x,
        weight=weight,
        cos=cos,
        sin=sin,
        freqs_complex=freqs_complex,
    )


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _run_separate(inputs, head_dim, eps):
    return separate_rmsnorm_rope(
        inputs["x"], inputs["weight"], inputs["freqs_complex"], head_dim, eps
    )


def _run_fused(inputs, head_dim, eps):
    return triton_ops.fused_rmsnorm_rope(
        inputs["x"], inputs["weight"], inputs["cos"], inputs["sin"], head_dim, eps
    )


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

BATCH_SIZES = [1, 2, 4]
SEQ_LENS = [64, 256, 1024]
DIMS = [1536, 5120]

CONFIGS = list(itertools.product(BATCH_SIZES, SEQ_LENS, DIMS))

PROVIDERS = ["separate", "fused"]


@tt.perf_report(
    tt.Benchmark(
        x_names=["B", "S", "D"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDERS,
        ylabel="Runtime (ms)",
        plot_name="fused_rmsnorm_rope_vs_separate",
        args={
            "head_dim": 128,
            "eps": 1e-6,
            "dtype": "bf16",
            "device": "cuda",
            "warmup": 25,
            "rep": 100,
        },
    )
)
def bench(B, S, D, provider, head_dim, eps, dtype, device, warmup, rep):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dt = dtype_map[dtype]

    inputs = _build_inputs(B, S, D, head_dim, dtype=dt, device=device)

    if provider == "separate":
        ms = tt.do_bench(
            lambda: _run_separate(inputs, head_dim, eps), warmup=warmup, rep=rep
        )
    elif provider == "fused":
        ms = tt.do_bench(
            lambda: _run_fused(inputs, head_dim, eps), warmup=warmup, rep=rep
        )
    else:
        raise ValueError(provider)

    return ms


if __name__ == "__main__":
    bench.run(print_data=True, show_plots=False)
