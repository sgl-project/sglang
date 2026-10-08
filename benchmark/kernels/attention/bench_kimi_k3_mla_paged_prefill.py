"""Compare Kimi-K3 Gluon MLA paged prefill with current Triton extend."""

import argparse
import csv
import statistics

import torch
import triton

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.mla_paged_prefill_gluon_hip import run


CACHE_SLOTS = 655361
MAX_CONTEXT = 1048576


def split_rows(rows, requests):
    lengths = [rows // requests] * requests
    lengths[-1] += rows - sum(lengths)
    return lengths


def indptr(lengths, *, dtype):
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=dtype, device="cuda"
    )


def compare(rows, requests, prefix, rounds, warmup, rep, cache):
    torch.manual_seed(1701 + rows + requests + prefix)
    lengths = split_rows(rows, requests)
    q = torch.randn((rows, 12, 576), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((rows, 1, 576), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((rows, 1, 512), dtype=torch.bfloat16, device="cuda")
    key_cache = cache
    value_cache = cache[..., :512]
    qo_indptr = indptr(lengths, dtype=torch.int32)
    prefix_lengths = [prefix] * requests
    kv_indptr = indptr(prefix_lengths, dtype=torch.int32)
    kv_indices = torch.arange(requests * prefix, dtype=torch.int64, device="cuda")
    native_out = torch.empty((rows, 12, 512), dtype=torch.bfloat16, device="cuda")
    gluon_out = torch.empty_like(native_out)

    def native():
        extend_attention_fwd(
            q,
            k,
            v,
            native_out,
            key_cache,
            value_cache,
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
            page_size=1,
            extend_seq_lens_cpu=lengths,
        )
        return native_out

    def gluon():
        return run(
            q,
            k,
            v,
            key_cache,
            value_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            scale=192**-0.5,
            max_query_length=max(lengths),
            out=gluon_out,
        )

    expected = native().clone()
    actual = gluon().clone()
    # Both paths accumulate attention from an FP8 cache in a different order.
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.1)
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
            times[name].append(triton.testing.do_bench(fn, warmup=warmup, rep=rep))
    native_ms = statistics.median(times["native"])
    gluon_ms = statistics.median(times["gluon"])
    return dict(
        rows=rows,
        requests=requests,
        prefix_per_request=prefix,
        native_ms=native_ms,
        gluon_ms=gluon_ms,
        speedup=native_ms / gluon_ms,
        cosine=cosine,
        max_abs=max_abs,
        rounds=rounds,
        warmup=warmup,
        rep=rep,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--prefix", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    torch.manual_seed(1701)
    cache = torch.randn((CACHE_SLOTS, 1, 576), dtype=torch.bfloat16, device="cuda").to(
        torch.float8_e4m3fn
    )
    results = []
    for requests in args.requests:
        for rows in args.rows:
            result = compare(
                rows,
                requests,
                args.prefix,
                args.rounds,
                args.warmup,
                args.rep,
                cache,
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
