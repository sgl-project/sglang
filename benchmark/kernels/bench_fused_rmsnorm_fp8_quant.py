"""Microbenchmark: fused RMSNorm + static per-tensor FP8 quant, comparing the
flashinfer default kernels against the CuTe-DSL kernels and the unfused
baseline (RMSNorm followed by a separate static FP8 quant).

Providers:
  unfused     RMSNorm.forward_cuda  +  static_quant_fp8
  fused       flashinfer rmsnorm_quant / fused_add_rmsnorm_quant (default)
  fused_cute  flashinfer rmsnorm_quant_cute / fused_add_rmsnorm_quant_cute

All fused providers produce an ``(fp8, scale)`` activation (and updated residual
when a residual is supplied), matching what a downstream FP8 static-per-tensor
linear consumes. Covers the no-residual and residual (fused-add) cases across a
few hidden sizes so you can pick the fastest kernel per shape.

Run:
    python benchmark/kernels/bench_fused_rmsnorm_fp8_quant.py
"""

import itertools

import numpy as np
import torch
import triton
from flashinfer.norm import fused_add_rmsnorm_quant, rmsnorm_quant
from flashinfer.testing import bench_gpu_time

from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.srt.layers.layernorm import RMSNorm, _flashinfer_rmsnorm_quant_available

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark")
if not _flashinfer_rmsnorm_quant_available:
    raise RuntimeError(
        "flashinfer rmsnorm_quant / fused_add_rmsnorm_quant is not available; "
        "install flashinfer to benchmark the fused path"
    )

try:
    from flashinfer.norm import fused_add_rmsnorm_quant_cute, rmsnorm_quant_cute

    _CUTE_AVAILABLE = True
except ImportError:
    _CUTE_AVAILABLE = False

DEVICE = "cuda"
DTYPE = torch.bfloat16
FP8_DTYPE = torch.float8_e4m3fn
HIDDEN_SIZES = [4096, 8192]
# Per-tensor reciprocal scale (q = normed / scale); 0.05 keeps normed/scale well
# within the e4m3 range for unit-scale activations.
SCALE_VALUE = 0.05


def make_layer(hidden_size):
    layer = RMSNorm(hidden_size).to(device=DEVICE, dtype=DTYPE)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    return layer


def make_inputs(num_tokens, hidden_size, add_residual):
    x = torch.randn(num_tokens, hidden_size, device=DEVICE, dtype=DTYPE)
    residual = torch.randn_like(x) if add_residual else None
    scale = torch.tensor([SCALE_VALUE], device=DEVICE, dtype=torch.float32)
    return x, residual, scale


def run_unfused(layer, x, residual, scale):
    out = layer(x, residual)
    if residual is not None:
        normed, residual_out = out
        q, q_scale = static_quant_fp8(normed, scale)
        return (q, q_scale), residual_out
    q, q_scale = static_quant_fp8(out, scale)
    return q, q_scale


def _run_fused(kernel, add_kernel, layer, x, residual, scale):
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    if residual is not None:
        # In-place: residual += x, then out = quant(rmsnorm(residual) * w).
        add_kernel(out, x, residual, layer.weight.data, scale, layer.variance_epsilon)
        return (out, scale), residual
    kernel(out, x, layer.weight.data, scale, layer.variance_epsilon)
    return out, scale


def run_fused_default(layer, x, residual, scale):
    return _run_fused(rmsnorm_quant, fused_add_rmsnorm_quant, layer, x, residual, scale)


def run_fused_cute(layer, x, residual, scale):
    return _run_fused(
        rmsnorm_quant_cute, fused_add_rmsnorm_quant_cute, layer, x, residual, scale
    )


RUNNERS = {
    "unfused": run_unfused,
    "fused": run_fused_default,
    "fused_cute": run_fused_cute,
}

# (provider key, plot label, style)
_PROVIDERS = [
    ("unfused", "rmsnorm + static_quant_fp8 (unfused)", ("blue", "-")),
    ("fused", "rmsnorm_quant (fused, default)", ("green", "-")),
]
if _CUTE_AVAILABLE:
    _PROVIDERS.append(
        ("fused_cute", "rmsnorm_quant_cute (fused, cute-dsl)", ("red", "-"))
    )


def _bench_ms(fn, args, quantiles=(0.5, 0.2, 0.8)):
    # Pass the GPU tensors as input_args so flashinfer's cold_l2_cache flush can
    # find them; a zero-arg callable trips its "no GPU tensors found" warning and
    # silently disables cold-L2 timing.
    times = bench_gpu_time(
        fn=fn,
        input_args=args,
        use_cuda_graph=True,
        dry_run_time_ms=25,
        repeat_time_ms=100,
    )
    return tuple(float(np.percentile(times, q * 100)) for q in quantiles)


def _check_correctness():
    """One-shot sanity check that every fused provider agrees with the unfused
    baseline within FP8 precision."""
    fused_providers = [p for p in RUNNERS if p != "unfused"]
    for hidden_size, add_residual in itertools.product(HIDDEN_SIZES, [False, True]):
        layer = make_layer(hidden_size)
        x, residual, scale = make_inputs(64, hidden_size, add_residual)
        with torch.inference_mode():
            ref = run_unfused(
                layer, x.clone(), residual.clone() if add_residual else None, scale
            )
        (uq, _), _ = ref if add_residual else (ref, None)
        ref_deq = uq.float() * scale
        for provider in fused_providers:
            if provider == "fused_cute" and not _CUTE_AVAILABLE:
                continue
            with torch.inference_mode():
                out = RUNNERS[provider](
                    layer, x.clone(), residual.clone() if add_residual else None, scale
                )
            (q, _), _ = out if add_residual else (out, None)
            cos = torch.nn.functional.cosine_similarity(
                (q.float() * scale).flatten(), ref_deq.flatten(), dim=0
            ).item()
            assert cos > 0.99, (
                f"{provider} h={hidden_size} residual={add_residual} cos={cos:.4f}"
            )
    print("correctness check passed (all fused providers vs unfused within FP8)")


configs = [
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[512, 1024, 2048, 4096, 8192, 16384],
        x_log=False,
        line_arg="provider",
        line_vals=[p[0] for p in _PROVIDERS],
        line_names=[p[1] for p in _PROVIDERS],
        styles=[p[2] for p in _PROVIDERS],
        ylabel="latency (ms)",
        plot_name=f"rmsnorm_fp8_quant_h{hidden_size}_residual{add_residual}",
        args={"hidden_size": hidden_size, "add_residual": add_residual},
    )
    for hidden_size, add_residual in itertools.product(HIDDEN_SIZES, [False, True])
]


@triton.testing.perf_report(configs)
def benchmark(num_tokens, hidden_size, add_residual, provider):
    layer = make_layer(hidden_size)
    x, residual, scale = make_inputs(num_tokens, hidden_size, add_residual)
    return _bench_ms(RUNNERS[provider], (layer, x, residual, scale))


if __name__ == "__main__":
    torch.manual_seed(0)
    _check_correctness()
    benchmark.run(print_data=True, show_plots=False)
