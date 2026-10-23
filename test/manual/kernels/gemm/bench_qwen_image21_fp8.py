# SPDX-License-Identifier: Apache-2.0
"""SM89 sibling FP8 quantization sharing, retaining the original GEMMs.

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python .venv/bin/python test/manual/kernels/gemm/bench_qwen_image21_fp8.py
Uses cold-L2 marker measurements. Times are milliseconds.
"""

import argparse
import json
from pathlib import Path

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.gemm import fp8_scaled_mm
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8


def measure(fn, args):
    return (
        marker.do_bench(
            fn,
            input_args=args,
            warmup_iters=3,
            replay_iters=30,
            disable_log_bandwidth=True,
        ).times[0]
        * 1000
    )


def baseline(x, weights, scales):
    outputs = []
    for weight, scale in zip(weights, scales):
        quantized, input_scale = sglang_per_token_quant_fp8(x)
        outputs.append(
            fp8_scaled_mm(quantized, weight, input_scale, scale, torch.bfloat16)
        )
    return tuple(outputs)


def shared(x, weights, scales):
    quantized, input_scale = sglang_per_token_quant_fp8(x)
    return tuple(
        fp8_scaled_mm(quantized, weight, input_scale, scale, torch.bfloat16)
        for weight, scale in zip(weights, scales)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/qwen-image21-shared-fp8.json")
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (8, 9), "SM89 required"
    torch.manual_seed(42)
    report = {"gpu": torch.cuda.get_device_name(), "results": []}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for m, n, k, count in [
        (4096, 4096, 4096, 3),
        (4096, 12288, 4096, 2),
        (22, 4096, 4096, 3),
        (22, 12288, 4096, 2),
    ]:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weights = tuple(
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            .to(torch.float8_e4m3fn)
            .t()
            for _ in range(count)
        )
        scales = tuple(
            torch.rand(1, n, device="cuda", dtype=torch.float32) * 0.01
            for _ in range(count)
        )
        inputs = (x, weights, scales)
        assert all(
            torch.equal(a, b) for a, b in zip(baseline(*inputs), shared(*inputs))
        )
        row = {"m": m, "n": n, "k": k, "count": count, "repeats": []}
        for repeat in range(3):
            funcs = [("baseline", baseline), ("shared", shared)]
            if repeat % 2:
                funcs.reverse()
            row["repeats"].append({name: measure(fn, inputs) for name, fn in funcs})
        report["results"].append(row)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
