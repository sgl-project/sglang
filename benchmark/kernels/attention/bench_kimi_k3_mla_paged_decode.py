"""Compare Kimi-K3 Gluon MLA decode with current Triton standard/Lean decode.

The native arm follows CUDA-graph policy: standard split-KV for M < 8 and
Lean attention for M >= 8. Both arms consume the production 1M-context graph
metadata layout and the same packed FP8 MLA cache.
"""

import argparse
import csv
import statistics

import torch
import triton

from sglang.kernels.ops.attention import mla_paged_decode_gluon_hip
from sglang.kernels.ops.attention.decode_attention import (
    _LEAN_BLOCK_M,
    _lean_decode_launch_params,
    decode_attention_fwd,
    lean_capture_policy,
)
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton


MAX_CONTEXT = 1048576
MAX_GRAPH_ROWS = 256
MAX_KV_SPLITS = 256
CACHE_SLOTS = 655361


def make_shared():
    torch.manual_seed(1701)
    cache = torch.randn((CACHE_SLOTS, 1, 576), dtype=torch.bfloat16, device="cuda").to(
        torch.float8_e4m3fn
    )
    indices = torch.empty(
        (MAX_GRAPH_ROWS * MAX_CONTEXT,), dtype=torch.int64, device="cuda"
    )
    return cache, cache[..., :512], indices


def compare(rows, context, rounds, rep, key_cache, value_cache, kv_indices):
    torch.manual_seed(1701 + rows + context)
    q = torch.randn((rows, 12, 576), dtype=torch.bfloat16, device="cuda")
    native_out = torch.empty((rows, 12, 512), dtype=torch.bfloat16, device="cuda")
    gluon_out = torch.empty_like(native_out)
    kv_indptr = torch.arange(
        0, (rows + 1) * context, context, dtype=torch.int32, device="cuda"
    )
    slots = torch.arange(1, context + 1, dtype=torch.int64, device="cuda")
    kv_indices[: rows * context].view(rows, context).copy_(slots)
    seq_lens = torch.full((rows,), context, dtype=torch.int32, device="cuda")
    num_kv_splits = torch.empty((rows,), dtype=torch.int32, device="cuda")
    get_num_kv_splits_triton[(1,)](
        num_kv_splits,
        seq_lens,
        rows,
        1,
        12,
        1,
        MAX_KV_SPLITS,
        256,
        MAX_NUM_SEQ=256,
    )
    attn_logits = torch.empty(
        (rows, 12, MAX_KV_SPLITS, 512), dtype=torch.float32, device="cuda"
    )
    attn_lse = torch.empty(
        (rows, 12, MAX_KV_SPLITS), dtype=torch.float32, device="cuda"
    )
    total_programs, _, _ = _lean_decode_launch_params(1, 12)
    lean_mp = torch.zeros(
        (total_programs, _LEAN_BLOCK_M), dtype=torch.float32, device="cuda"
    )
    lean_lp = torch.zeros_like(lean_mp)
    lean_op = torch.zeros(
        (total_programs, _LEAN_BLOCK_M, 512), dtype=torch.float32, device="cuda"
    )
    lean_locks = torch.zeros((total_programs,), dtype=torch.int32, device="cuda")
    enable_lean = lean_capture_policy(12, 12, rows, is_mla=True)

    def native():
        decode_attention_fwd(
            q,
            key_cache,
            value_cache,
            native_out,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            MAX_KV_SPLITS,
            192**-0.5,
            1.0,
            1.0,
            has_mla=True,
            page_size=1,
            enable_lean=enable_lean,
            lean_Mp=lean_mp,
            lean_Lp=lean_lp,
            lean_Op=lean_op,
            lean_locks=lean_locks,
        )
        return native_out

    def gluon():
        return mla_paged_decode_gluon_hip.run(
            q,
            key_cache,
            value_cache,
            kv_indptr,
            kv_indices,
            scale=192**-0.5,
            max_context=MAX_CONTEXT,
            out=gluon_out,
        )

    expected = native().clone()
    actual = gluon().clone()
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
        for name, fn in order:
            times[name].append(triton.testing.do_bench_cudagraph(fn, rep=rep))
    native_ms = statistics.median(times["native"])
    gluon_ms = statistics.median(times["gluon"])
    return dict(
        rows=rows,
        context=context,
        native_backend="lean" if enable_lean else "split_kv",
        native_ms=native_ms,
        gluon_ms=gluon_ms,
        speedup=native_ms / gluon_ms,
        cosine=cosine,
        max_abs=max_abs,
        rounds=rounds,
        rep=rep,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[4096, 65536, 350000]
    )
    args = parser.parse_args()
    key_cache, value_cache, kv_indices = make_shared()
    results = []
    for context in args.contexts:
        for rows in args.rows:
            result = compare(
                rows,
                context,
                args.rounds,
                args.rep,
                key_cache,
                value_cache,
                kv_indices,
            )
            results.append(result)
            print(result, flush=True)
    if args.output:
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0])
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
