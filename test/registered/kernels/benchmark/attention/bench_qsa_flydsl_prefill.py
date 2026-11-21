"""QSA extend attention on gfx950: the Triton kernels against AITER's FlyDSL sparse
prefill (SGLANG_ROCM_QSA_FLYDSL_PREFILL, ROCm/aiter#7556), with the operands
QwenSparseAttnBackend.forward_extend hands either one, at Qwen3.8-Flash-Next's
geometry (24 query heads, 2 KV heads, d=256, 512 blocks of 4 tokens per row).

A chunk is one call per QSA layer on one layout, new for each chunk over a prefix
(the op plans a layout on its first call; a prefix-free chunk reuses its layout);
"layer" is the median of the other calls. Both paths alternate on the
same operands. Selections: "shared" rows of a request rank its blocks by one score
plus noise, "random" rows rank them independently.

    python test/registered/kernels/benchmark/attention/bench_qsa_flydsl_prefill.py --sweep
"""

import argparse
import json
import os
import statistics

import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(
    est_time=120,
    stage="base-a-jit-kernel-benchmark",
    runner_config="amd",
    disabled="manual microbenchmark",
)

HEADS, KV_HEADS, DIM, RATIO, TOKEN_TOPK = 24, 2, 256, 4, 2048
SCALE = DIM**-0.5
SWEEP = [(16384, 0), (16384, 16384), (16384, 83968), (8192, 32768), (4096, 96256)]
SWEEP += [(rows, 98304) for rows in (2048, 1024, 512, 256, 64, 1)]


def selection(rows, prefix, pattern, device, chunk=2048):
    """indices [rows, 2051] of one request, from SGLang's block expander."""
    positions = prefix + torch.arange(rows, device=device)
    blocks_total = (prefix + rows) // RATIO
    base = torch.rand(blocks_total, device=device)
    take = min(TOKEN_TOPK // RATIO, blocks_total)
    out = []
    for start in range(0, rows, chunk):
        pos = positions[start : start + chunk]
        visible = ((pos + 1) // RATIO)[:, None]
        score = torch.rand(len(pos), blocks_total, device=device)
        if pattern == "shared":
            score = base + 0.05 * score
        cols = torch.arange(blocks_total, device=device)
        score = score.masked_fill(cols >= visible, -1.0)
        blocks = torch.full((len(pos), TOKEN_TOPK // RATIO), -1, device=device)
        blocks[:, :take] = score.topk(take, dim=1).indices
        blocks[torch.arange(blocks.shape[1], device=device) >= visible] = -1
        lengths = torch.full_like(pos, prefix + rows)
        out.append(
            expand_qsa_block_indices(blocks.int(), pos, lengths, RATIO, TOKEN_TOPK)
        )
    return torch.cat(out)


def run_case(rows, prefix, pattern, layers, reps, device):
    total = rows + prefix
    q = torch.randn(rows, HEADS, DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(total + RATIO * reps, KV_HEADS, DIM, device=device).bfloat16()
    v = torch.randn_like(k)
    times = {"triton": [], "flydsl": []}
    for rep in range(reps + 1):  # rep 0 compiles and is not recorded
        p = prefix + RATIO * rep if prefix else 0  # a new layout per chunk
        n = rows + p
        indices = selection(rows, p, pattern, device)
        cu_q = torch.tensor([0, rows], dtype=torch.int32, device=device)
        cu_k = torch.tensor([0, n], dtype=torch.int32, device=device)
        kv_lens = torch.tensor([n], dtype=torch.int32, device=device)

        def triton():
            if p:
                return sparse_gqa_fwd_interface_triton_ck(
                    q, k[:n], v[:n], indices, cu_q, cu_k, kv_lens, SCALE
                )
            return sparse_gqa_fwd_interface_triton(
                q, k[:n], v[:n], n, indices, cu_q, SCALE
            )

        def flydsl():
            out = backend._flydsl_sparse_prefill(
                q, k[:n], v[:n], indices, (rows,), (p,), SCALE
            )
            assert out is not None, "the op refused the call; see the log"
            return out

        for name in sorted(times, reverse=rep % 2 == 1):
            fn = triton if name == "triton" else flydsl
            events = [torch.cuda.Event(enable_timing=True) for _ in range(layers + 1)]
            torch.cuda.synchronize()
            events[0].record()
            for i in range(layers):
                fn()
                events[i + 1].record()
            torch.cuda.synchronize()
            if rep:
                times[name].append(
                    [events[i].elapsed_time(events[i + 1]) for i in range(layers)]
                )
    result = {"rows": rows, "prefix": prefix, "selection": pattern}
    for name, chunks in times.items():
        result[f"{name}_layer_ms"] = statistics.median(t for c in chunks for t in c[1:])
        result[f"{name}_first_ms"] = statistics.median(c[0] for c in chunks)
        result[f"{name}_chunk_ms"] = statistics.median(sum(c) for c in chunks)
    result["layer_ratio"] = result["flydsl_layer_ms"] / result["triton_layer_ms"]
    result["chunk_ratio"] = result["flydsl_chunk_ms"] / result["triton_chunk_ms"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--prefix", type=int, default=83968)
    parser.add_argument("--selection", choices=("shared", "random"), default="shared")
    parser.add_argument(
        "--sweep", action="store_true", help="every layout, both selections"
    )
    parser.add_argument("--layers", type=int, default=12, help="QSA layers per chunk")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", help="write the results as JSON")
    args = parser.parse_args()

    os.environ["SGLANG_ROCM_QSA_FLYDSL_PREFILL"] = "1"
    if backend._resolve_flydsl_sparse_prefill() is None:
        raise SystemExit("needs gfx950 and an aiter with ROCm/aiter#7556")
    torch.manual_seed(0)
    cases = (
        [(r, p, s) for r, p in SWEEP for s in ("shared", "random")]
        if args.sweep
        else [(args.rows, args.prefix, args.selection)]
    )
    print("rows  prefix  selection  Triton ms  FlyDSL ms  ratio  (layer | chunk)")
    results = []
    for rows, prefix, pattern in cases:
        r = run_case(rows, prefix, pattern, args.layers, args.reps, "cuda")
        results.append(r)
        print(
            f"{rows:5d} {prefix:7d}  {pattern:9s}"
            f" {r['triton_layer_ms']:8.3f} {r['flydsl_layer_ms']:9.3f} {r['layer_ratio']:6.2f}"
            f" | {r['triton_chunk_ms']:8.2f} {r['flydsl_chunk_ms']:8.2f} {r['chunk_ratio']:5.2f}",
            flush=True,
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
