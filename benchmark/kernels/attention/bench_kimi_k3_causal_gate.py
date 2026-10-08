"""Compare fused Kimi-K3 causal attention/gate with current native chain."""

import argparse
import csv
import statistics

import torch
import triton

from sglang.kernels.ops.attention import (
    kimi_causal_gate_gluon_hip,
    mla_output_gate,
)
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd


def split_rows(rows, requests):
    lengths = [rows // requests] * requests
    lengths[-1] += rows - sum(lengths)
    return lengths


def compare(rows, requests, rounds, warmup, rep):
    torch.manual_seed(1701 + rows + requests)
    lengths = split_rows(rows, requests)
    q = torch.randn((rows, 12, 192), dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn((rows, 12, 128), dtype=torch.bfloat16, device="cuda")
    gate = torch.randn((rows, 1536), dtype=torch.bfloat16, device="cuda")
    qo_indptr = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int64,
        device="cuda",
    )
    kv_indptr = torch.zeros((requests + 1,), dtype=torch.int32, device="cuda")
    kv_indices = torch.empty((0,), dtype=torch.int64, device="cuda")
    native_attention = torch.empty((rows, 12, 128), dtype=torch.bfloat16, device="cuda")

    def native():
        extend_attention_fwd(
            q,
            k,
            v,
            native_attention,
            k[:0],
            v[:0],
            qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            True,
            None,
            max(lengths),
            1.0,
            1.0,
            sm_scale=192**-0.5,
            skip_prefix=True,
            page_size=1,
            extend_seq_lens_cpu=lengths,
        )
        return mla_output_gate.kimi_k3_mla_output_gate(
            native_attention.view(rows, 1536), gate
        )

    def gluon():
        return kimi_causal_gate_gluon_hip.run(
            q,
            k,
            v,
            gate,
            qo_indptr,
            scale=192**-0.5,
            max_query_len=max(lengths),
        )

    expected = native()
    actual = gluon()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)
    cosine = torch.nn.functional.cosine_similarity(
        expected.float().flatten(), actual.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999, cosine
    max_abs = (expected.float() - actual.float()).abs().max().item()

    times = {"native": [], "gluon": []}
    for iteration in range(rounds):
        order = (("native", native), ("gluon", gluon))
        if iteration % 2:
            order = order[::-1]
        for arm, fn in order:
            times[arm].append(triton.testing.do_bench(fn, warmup=warmup, rep=rep))
    native_ms = statistics.median(times["native"])
    gluon_ms = statistics.median(times["gluon"])
    return dict(
        rows=rows,
        requests=requests,
        native_ms=native_ms,
        gluon_ms=gluon_ms,
        speedup=native_ms / gluon_ms,
        cosine=cosine,
        max_abs=max_abs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 4, 8, 32])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()
    results = []
    for requests in args.requests:
        for rows in args.rows:
            result = compare(rows, requests, args.rounds, args.warmup, args.rep)
            results.append(result)
            print(result, flush=True)
    if args.output:
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0])
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
