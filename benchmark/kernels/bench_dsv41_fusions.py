import json
import statistics
from functools import partial
from pathlib import Path

import flashinfer
import torch
import triton
from sgl_kernel import rmsnorm
from triton.testing import do_bench_cudagraph

from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace
from sglang.kernels.ops.attention.dsv4.q_rope_store import q_rope_store
from sglang.kernels.ops.layernorm.hc_combine_norm import (
    hc_combine_norm,
    hc_combine_norm_mxfp8,
)
from sglang.kernels.ops.layernorm.mhc import hc_combine
from sglang.kernels.ops.layernorm.mxfp8_epilogue import rmsnorm_mxfp8
from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize


def norm_quantize(x, weight):
    return flashinfer_mxfp8_quantize(rmsnorm(x, weight, 1e-6), True, 32, "cute-dsl")


def upstream_hc_norm_quantize(x, pre, weight):
    rows = x.shape[0]
    if rows <= 8:
        return hc_combine_norm_mxfp8(x, pre, weight, 1e-6)
    if rows <= 96:
        y = hc_combine_norm(x, pre, weight, 1e-6)
    else:
        y = rmsnorm(hc_combine(x, pre, 4, torch.bfloat16), weight, 1e-6)
    return flashinfer_mxfp8_quantize(y, True, 32, "cute-dsl")


def rope_then_store(q, output, freqs, positions):
    fused_rope_inplace(q[..., 448:], None, freqs, positions)
    output.copy_(q)


def profile_kernels(fn):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        fn()
        torch.cuda.synchronize()
    return [
        event.name for event in profile.events() if event.device_type.name == "CUDA"
    ]


def measure(name, rows, width, baseline, candidate):
    samples = {"baseline": [], "candidate": []}
    operators = {"baseline": baseline, "candidate": candidate}
    for fn in operators.values():
        for _ in range(5):
            fn()
    for repeat in range(5):
        order = (
            ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        )
        for label in order:
            samples[label].append(do_bench_cudagraph(operators[label], rep=100) * 1000)
    medians = {label: statistics.median(values) for label, values in samples.items()}
    record = {
        "operator": name,
        "rows": rows,
        "width": width,
        "us": samples,
        "median_us": medians,
        "latency_reduction_pct": 100 * (1 - medians["candidate"] / medians["baseline"]),
    }
    if rows == 128:
        record["profile"] = {
            label: profile_kernels(fn) for label, fn in operators.items()
        }
    print(json.dumps(record), flush=True)
    return record


def main():
    torch.manual_seed(20260928)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "flashinfer": flashinfer.__version__,
                "timing": "triton.testing.do_bench_cudagraph; five alternating-order rounds; rep=100ms",
                "control": "M=8 uses the same producer in both arms",
            }
        ),
        flush=True,
    )
    records = []
    for rows in (8, 9, 48, 96, 128, 129, 256, 384, 512):
        for width in (1280, 5120):
            x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            candidate = partial(rmsnorm_mxfp8, x, weight, 1e-6)
            baseline = candidate if rows <= 8 else partial(norm_quantize, x, weight)
            records.append(
                measure("rmsnorm_quantize", rows, width, baseline, candidate)
            )
        x = torch.randn(rows, 20480, device="cuda", dtype=torch.bfloat16)
        pre = torch.randn(rows, 4, device="cuda", dtype=torch.float32)
        weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
        records.append(
            measure(
                "hc_combine_norm_quantize",
                rows,
                5120,
                partial(upstream_hc_norm_quantize, x, pre, weight),
                partial(hc_combine_norm_mxfp8, x, pre, weight, 1e-6),
            )
        )
    freqs = torch.polar(
        torch.ones(8192, 32, device="cuda"), torch.randn(8192, 32, device="cuda")
    )
    for rows in (9, 128, 512, 1024, 4095):
        q = torch.randn(rows, 16, 512, device="cuda", dtype=torch.bfloat16)
        q_ref = q.clone()
        output = torch.empty(rows, 64, 512, device="cuda", dtype=q.dtype)[:, :16]
        positions = torch.arange(rows, device="cuda")

        records.append(
            measure(
                "q_rope_store",
                rows,
                8192,
                partial(rope_then_store, q_ref, output, freqs, positions),
                partial(q_rope_store, q, output, freqs, positions),
            )
        )
    Path("operator-results.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
