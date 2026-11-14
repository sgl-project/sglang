"""Compare the complete ConvRot projection/activation API with fused SwiGLU."""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding,
)
from sglang.kernels.ops.gemm.convrot_int8 import (
    convrot_int8_fused_linear,
    convrot_int8_swiglu,
    convrot_rotate_quantize_activation,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-small"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="129,1024,4096,16384")
    parser.add_argument("--shapes", default="5376:14336,3072:6144,4096:7168")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    rng = random.Random(123)
    flush = torch.empty(128 * 1024**2, device="cuda", dtype=torch.uint8)
    records = []
    for m in map(int, args.rows.split(",")):
        for shape in args.shapes.split(","):
            k, n = map(int, shape.split(":"))
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16) * 0.02
            wq, ws = convrot_rotate_quantize_activation(weight, 256)
            del weight

            def baseline():
                return silu_and_mul_with_activation_rounding(
                    convrot_int8_fused_linear(x, wq, ws, None, 256)
                )

            def fused():
                return convrot_int8_swiglu(x, wq, ws, None, 256)

            assert torch.equal(baseline(), fused())
            peak_bytes = {}
            for name, call in [("baseline", baseline), ("fused", fused)]:
                torch.cuda.synchronize()
                initial = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                result = call()
                torch.cuda.synchronize()
                peak_bytes[name] = torch.cuda.max_memory_allocated() - initial
                del result
            pairs = []
            for trial in range(args.repeats + 3):
                calls = [("baseline", baseline), ("fused", fused)]
                rng.shuffle(calls)
                pair = {}
                for name, call in calls:
                    flush.zero_()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    call()
                    end.record()
                    end.synchronize()
                    pair[name] = start.elapsed_time(end)
                if trial >= 3:
                    pairs.append(pair)
            ms = {name: statistics.median(p[name] for p in pairs) for name in pairs[0]}
            graph_ms = {
                name: triton.testing.do_bench_cudagraph(call, rep=100)
                for name, call in [("baseline", baseline), ("fused", fused)]
            }
            row = dict(
                m=m,
                n=n,
                k=k,
                ms=ms,
                graph_ms=graph_ms,
                peak_extra_allocated_bytes=peak_bytes,
                speedup=ms["baseline"] / ms["fused"],
                pairs=pairs,
            )
            records.append(row)
            print(
                f"M={m:5d} K={k:5d} N={n:5d}: {ms['baseline']:.4f} -> {ms['fused']:.4f} ms ({row['speedup']:.3f}x)"
            )
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        rows=records,
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
