"""Compare Kimi-K3 Gluon attention residual with current HIP fusion."""

import argparse
import csv
import statistics
from types import SimpleNamespace

import torch
import triton

from sglang.kernels.ops.attention import kimi_attn_residual_gluon_hip as adapter
from sglang.kernels.ops.attention.mla_gluon import attention_residual_norm
from sglang.srt.layers import attn_residual as native


CASES = (
    (16, 1, 4),
    (2, 8, 5),
    (2, 4, 5),
    (1, 4, 6),
    (1, 1, 4),
    (2, 4, 6),
    (2, 1, 4),
    (32, 4, 6),
    (32, 4, 5),
    (32, 1, 4),
    (32, 8, 5),
    (4, 4, 6),
    (16, 4, 5),
    (4, 8, 5),
    (64, 1, 4),
    (64, 8, 5),
    (8, 1, 4),
)


def compare(rows, valid_rows, mode, rounds, warmup, rep):
    torch.manual_seed(1701 + rows + valid_rows + mode)
    prefix = torch.randn((rows, 7168), dtype=torch.bfloat16, device="cuda")
    addend = torch.randn_like(prefix)
    native_bank = torch.randn((rows, 8, 7168), dtype=torch.bfloat16, device="cuda")
    gluon_bank = native_bank.clone()
    score_proj = SimpleNamespace(
        weight=torch.randn((1, 7168), dtype=torch.bfloat16, device="cuda")
    )
    score_norm = SimpleNamespace(
        weight=torch.randn((7168,), dtype=torch.bfloat16, device="cuda"),
        variance_epsilon=1e-5,
    )
    out_norm = SimpleNamespace(
        weight=torch.randn((7168,), dtype=torch.bfloat16, device="cuda"),
        variance_epsilon=1e-5,
    )
    cw = native.get_cw(score_proj, score_norm)
    entrypoint = adapter.entrypoint_name(rows, valid_rows, mode)
    assert entrypoint is not None
    selected = getattr(attention_residual_norm, entrypoint)
    has_addend = bool(mode & 1)
    write_bank = bool(mode & 2)
    addend_arg = addend if has_addend else None

    def native_call():
        return native._aggregate_hip(
            prefix,
            addend_arg,
            native_bank,
            valid_rows,
            score_proj,
            score_norm,
            out_norm,
            write_bank,
        )

    def gluon_call():
        return selected(
            prefix,
            prefix if addend_arg is None else addend_arg,
            gluon_bank,
            cw,
            out_norm.weight,
            valid_rows=valid_rows,
            has_addend=has_addend,
            write_bank=write_bank,
            apply_output_norm=True,
            score_eps=1e-5,
            output_eps=1e-5,
        )[:2]

    native_out, native_current = native_call()
    gluon_out, gluon_current = gluon_call()
    torch.testing.assert_close(gluon_current, native_current, rtol=0, atol=0)
    torch.testing.assert_close(gluon_out, native_out, rtol=0.02, atol=0.02)
    if write_bank:
        torch.testing.assert_close(
            gluon_bank[:, valid_rows], native_bank[:, valid_rows], rtol=0, atol=0
        )
    cosine = torch.nn.functional.cosine_similarity(
        native_out.float().flatten(), gluon_out.float().flatten(), dim=0
    ).item()
    max_abs = (native_out.float() - gluon_out.float()).abs().max().item()

    times = {"native": [], "gluon": []}
    for iteration in range(rounds):
        order = (("native", native_call), ("gluon", gluon_call))
        if iteration % 2:
            order = order[::-1]
        for arm, fn in order:
            times[arm].append(triton.testing.do_bench(fn, warmup=warmup, rep=rep))
    native_ms = statistics.median(times["native"])
    gluon_ms = statistics.median(times["gluon"])
    return dict(
        rows=rows,
        valid_rows=valid_rows,
        mode=mode,
        entrypoint=entrypoint,
        native_ms=native_ms,
        gluon_ms=gluon_ms,
        speedup=native_ms / gluon_ms,
        cosine=cosine,
        max_abs=max_abs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()
    results = []
    for rows, valid_rows, mode in CASES:
        result = compare(rows, valid_rows, mode, args.rounds, args.warmup, args.rep)
        results.append(result)
        print(result, flush=True)
    if args.output:
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0])
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
