"""Compare PR17 Gluon MLA value/gate with the current SGLang native chain.

Uses the actual rocm_absorb_v_bmm and fused mla_output_gate entrypoints, with
native transposed BF16 weights. Both arms use CUDA graphs, alternating order.
The gate projection and output projection are outside this component benchmark.
"""

import argparse
import csv
import statistics
from types import SimpleNamespace

import torch
import triton

from sglang.kernels.ops.attention import mla_output_gate, mla_vc_gate_gluon_hip
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
    rocm_absorb_v_bmm,
)


def compare(rows, rounds):
    torch.manual_seed(100 + rows)
    device = "cuda"
    dtype = torch.bfloat16
    latent = torch.randn((rows, 12, 512), dtype=dtype, device=device)
    weight = (
        torch.randn((12, 128, 512), dtype=dtype, device=device) / 512**0.5
    ).transpose(1, 2)
    gate = torch.randn((rows, 1536), dtype=dtype, device=device)
    attn = SimpleNamespace(
        w_vc=weight,
        w_kc=torch.empty(0, dtype=dtype, device=device),
        w_scale=1.0,
        num_local_heads=12,
        o_proj=SimpleNamespace(weight=torch.empty(0, dtype=dtype, device=device)),
    )

    def native():
        value = rocm_absorb_v_bmm(attn, latent)
        assert mla_output_gate.covered(value, gate)
        return mla_output_gate.kimi_k3_mla_output_gate(value, gate)

    def gluon():
        return mla_vc_gate_gluon_hip.run(latent, weight, gate)

    expected, actual = native(), gluon()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    cosine = torch.nn.functional.cosine_similarity(
        expected.float().flatten(), actual.float().flatten(), dim=0
    ).item()
    assert cosine > 0.99999, cosine
    max_abs = (expected.float() - actual.float()).abs().max().item()
    for scale in (0.0, 8.0):
        gate.copy_(torch.randn_like(gate) * scale)
        torch.testing.assert_close(gluon(), native(), rtol=0.02, atol=0.02)
    gate.normal_()
    times = {"native": [], "gluon": []}
    for iteration in range(rounds):
        order = (("native", native), ("gluon", gluon))
        if iteration % 2:
            order = order[::-1]
        for name, fn in order:
            times[name].append(triton.testing.do_bench_cudagraph(fn, rep=100))
    native_ms = statistics.median(times["native"])
    gluon_ms = statistics.median(times["gluon"])
    return dict(
        rows=rows,
        native_ms=native_ms,
        gluon_ms=gluon_ms,
        speedup=native_ms / gluon_ms,
        cosine=cosine,
        max_abs=max_abs,
        rounds=rounds,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4018, 4096, 6144, 8192],
    )
    args = parser.parse_args()
    results = []
    for rows in args.rows:
        result = compare(rows, args.rounds)
        results.append(result)
        print(result, flush=True)
    if args.output:
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0])
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
