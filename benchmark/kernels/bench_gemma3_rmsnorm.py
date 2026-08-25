"""Microbenchmark: Gemma3RMSNorm forward_xpu vs the forward_native reference,
across the same 2D/3D/4D (contiguous and non-contiguous "unflatten") shapes
covered by test/manual/layers/test_layernorm.py::TestGemma3RMSNorm.

Providers:
  native  Gemma3RMSNorm.forward_native  (pure PyTorch reference)
  xpu     Gemma3RMSNorm.forward_xpu     (sgl_kernel gemma_rmsnorm / gemma_fused_add_rmsnorm)

The "unflatten" shapes reproduce Gemma3's q_norm/k_norm input: a head slice cut
out of a wider qkv-style tensor via .split()+.unflatten(), so the leading dims
are not flattenable to 2D and the tensor is not contiguous.

Run:
    python benchmark/kernels/bench_gemma3_rmsnorm.py
"""

import torch
import triton

from sglang.srt.layers.layernorm import Gemma3RMSNorm

if not torch.xpu.is_available():
    raise RuntimeError("XPU is required for this benchmark")

DEVICE = "xpu"
DTYPE = torch.bfloat16
HEAD_DIMS = [1, 64, 128, 1024]
NUM_TOKENS = [64, 512, 4096]

# (case name, leading shape excluding head_dim, non_contiguous)
CASES = [
    ("2d", (83,), False),
    ("3d", (4, 19), False),
    ("4d", (1, 19, 8), False),
    ("3d_unflatten", (19, 4), True),
    ("4d_unflatten", (1, 19, 4), True),
]


def make_layer(head_dim):
    layer = Gemma3RMSNorm(head_dim).to(device=DEVICE, dtype=DTYPE)
    layer.weight.data.normal_(mean=0.0, std=0.1)
    return layer


def make_inputs(lead_shape, head_dim, add_residual, non_contiguous, num_tokens=None):
    # For the 2D case only, the leading shape is replaced by num_tokens so the
    # benchmark can sweep sequence length like the other norm benchmarks do.
    lead = (
        (num_tokens,) if lead_shape == (83,) and num_tokens is not None else lead_shape
    )
    scale = 1 / (2 * head_dim)

    if non_contiguous:
        *outer, num_heads = lead
        total_heads = num_heads + 3
        full = (
            torch.randn(*outer, total_heads * head_dim, device=DEVICE, dtype=DTYPE)
            * scale
        )
        x = full[..., : num_heads * head_dim].unflatten(-1, (num_heads, head_dim))
    else:
        x = torch.randn(*lead, head_dim, device=DEVICE, dtype=DTYPE) * scale

    residual = torch.randn_like(x) * scale if add_residual else None
    return x, residual


def run_native(layer, x, residual):
    return layer.forward_native(x, residual)


def run_xpu(layer, x, residual):
    return layer.forward_xpu(x, residual)


RUNNERS = {"native": run_native, "xpu": run_xpu}

_PROVIDERS = [
    ("native", "forward_native (pure PyTorch)", ("blue", "-")),
    ("xpu", "forward_xpu (sgl_kernel)", ("green", "-")),
]


def _check_correctness():
    """One-shot sanity check that forward_xpu agrees with forward_native."""
    add_residual = False
    for case, lead_shape, non_contiguous in CASES:
        for head_dim in HEAD_DIMS:
            layer = make_layer(head_dim)
            x, residual = make_inputs(
                lead_shape, head_dim, add_residual, non_contiguous
            )
            with torch.inference_mode():
                ref = layer.forward_native(x, residual)
                out = layer.forward_xpu(x.clone(), residual)
            assert torch.allclose(
                out, ref, atol=2e-2, rtol=2e-2
            ), f"{case} head_dim={head_dim} mismatch"
    print("correctness check passed (forward_xpu vs forward_native)")


configs = [
    triton.testing.Benchmark(
        x_names=["head_dim"],
        x_vals=HEAD_DIMS,
        line_arg="provider",
        line_vals=[p[0] for p in _PROVIDERS],
        line_names=[p[1] for p in _PROVIDERS],
        styles=[p[2] for p in _PROVIDERS],
        ylabel="latency (us)",
        plot_name=f"gemma3_rmsnorm_{case}",
        args={
            "case": case,
            "lead_shape": lead_shape,
            "non_contiguous": non_contiguous,
            "add_residual": False,
        },
    )
    for case, lead_shape, non_contiguous in CASES
]


@triton.testing.perf_report(configs)
def benchmark(case, lead_shape, non_contiguous, add_residual, head_dim, provider):
    layer = make_layer(head_dim)
    x, residual = make_inputs(lead_shape, head_dim, add_residual, non_contiguous)
    quantiles = [0.5, 0.2, 0.8]

    # add_residual is always False here, so forward_xpu never mutates x in
    # place (gemma_rmsnorm is out-of-place); no per-call cloning needed.
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: RUNNERS[provider](layer, x, residual), quantiles=quantiles
    )
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    torch.manual_seed(0)
    _check_correctness()
    dfs = benchmark.run(print_data=True, show_plots=False, return_df=True)
