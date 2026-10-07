"""Validate and time the Qwen3-VL BF16 decode GEMM tactics on SM10x.

Example: CUDA_VISIBLE_DEVICES=0 python benchmark/qwen3_vl/bench_splitk.py

Each CUDA graph cycles through independent weights totaling at least 256 MiB,
which exceeds B300 L2. Reported times are per GEMM, including the split-K
reduction. The baseline uses the existing CuTe DSL/cuBLAS selector without the
new split-K allowlist. Inputs and outputs are reused, as in decode graphs.
"""

import argparse
import json

import torch
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
    SplitKTactic,
    run_splitk_dense,
)
from triton.testing import do_bench_cudagraph

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
    cutedsl_bf16_gemm_out,
    use_cutedsl_bf16_gemm,
)
from sglang.srt.layers.quantization.unquant import _BF16_SPLITK_TUNED_TACTICS


def benchmark(m, n, k, weight_mib, rep):
    count = max(2, (weight_mib * 1024**2) // (n * k * 2) + 1)
    inputs = [
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for _ in range(count)
    ]
    weights = [
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for _ in range(count)
    ]
    outputs = [
        torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(count)
    ]
    tactic = SplitKTactic(*_BF16_SPLITK_TUNED_TACTICS[(m, n, k)])

    def baseline():
        for x, weight, out in zip(inputs, weights, outputs):
            if use_cutedsl_bf16_gemm(m, n, k):
                cutedsl_bf16_gemm_out(x, weight, out)
            else:
                torch.mm(x, weight.T, out=out)

    def candidate():
        for x, weight, out in zip(inputs, weights, outputs):
            run_splitk_dense(x, weight.T, None, out, True, tactic)

    baseline()
    original = outputs[0].clone()
    candidate()
    reference = inputs[0].float() @ weights[0].float().T
    nrms = (
        ((outputs[0].float() - reference).square().mean() / reference.square().mean())
        .sqrt()
        .item()
    )
    cosine = torch.nn.functional.cosine_similarity(
        outputs[0].float().flatten(), original.float().flatten(), dim=0
    ).item()
    assert nrms < 0.0025, nrms
    assert cosine > 0.99998, cosine
    baseline_us = do_bench_cudagraph(baseline, rep=rep) * 1000 / count
    candidate_us = do_bench_cudagraph(candidate, rep=rep) * 1000 / count
    return {
        "m": m,
        "n": n,
        "k": k,
        "weight_count": count,
        "weight_mib": count * n * k * 2 / 1024**2,
        "tactic": list(_BF16_SPLITK_TUNED_TACTICS[(m, n, k)]),
        "baseline_us": baseline_us,
        "splitk_us": candidate_us,
        "speedup": baseline_us / candidate_us,
        "normalized_rmse": nrms,
        "cosine": cosine,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--weight-mib", type=int, default=256)
    parser.add_argument(
        "--rep", type=int, default=100, help="Timing duration in milliseconds"
    )
    args = parser.parse_args()
    if args.weight_mib < 256:
        parser.error("--weight-mib must be at least 256 to exceed B300 L2")
    torch.manual_seed(42)
    for m in args.batch_size:
        for n, k in [(6144, 2560), (2560, 4096), (2560, 9728)]:
            print(json.dumps(benchmark(m, n, k, args.weight_mib, args.rep)), flush=True)


if __name__ == "__main__":
    main()
