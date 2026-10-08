"""Compare PR17 Gluon MLA KC/cache with the current SGLang native chain.

The baseline is the production operation sequence: BF16 absorbed-key BMM,
query/key concatenation, two BF16-to-FP8 casts, and the current MLA paged-cache
writer.  Both arms use CUDA graphs and alternating timing order.
"""

import argparse
import csv
import statistics

import torch
import triton

from sglang.kernels.ops.attention import mla_kc_cache_gluon_hip
from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton


def compare(rows, rounds):
    torch.manual_seed(700 + rows)
    device = "cuda"
    dtype = torch.bfloat16
    query = torch.randn((rows, 12, 192), dtype=dtype, device=device)
    latent = torch.randn((rows, 512), dtype=dtype, device=device)
    key_tail = torch.randn((rows, 64), dtype=dtype, device=device)
    weight = torch.randn((12, 512, 128), dtype=dtype, device=device).transpose(1, 2)
    locations = torch.randperm(rows, dtype=torch.int64, device=device) + 1
    # MLATokenToKVPool stores FP8 physically as uint8. Its native writer casts
    # each source to FP8, reinterprets the bytes, then scatters into this buffer.
    native_cache = torch.zeros((rows + 1, 576), dtype=torch.uint8, device=device)
    gluon_cache = torch.zeros((rows + 1, 576), dtype=torch.float8_e4m3fn, device=device)

    def native():
        absorbed = torch.bmm(query[..., :128].transpose(0, 1), weight).transpose(0, 1)
        qcat = torch.cat((absorbed, query[..., 128:]), dim=-1)
        fresh = torch.cat((latent, key_tail), dim=-1)
        latent_fp8 = latent.to(torch.float8_e4m3fn).view(torch.uint8)
        tail_fp8 = key_tail.to(torch.float8_e4m3fn).view(torch.uint8)
        set_mla_kv_buffer_triton(native_cache, locations, latent_fp8, tail_fp8)
        return qcat, fresh

    def gluon():
        return mla_kc_cache_gluon_hip.run(
            query,
            latent,
            key_tail,
            weight,
            locations,
            gluon_cache,
        )

    expected_q, expected_fresh = native()
    actual_q, actual_fresh = gluon()
    torch.testing.assert_close(actual_q, expected_q, rtol=0.02, atol=0.02)
    torch.testing.assert_close(actual_fresh, expected_fresh, rtol=0, atol=0)
    torch.testing.assert_close(
        gluon_cache.view(torch.uint8)[locations],
        native_cache[locations],
        rtol=0,
        atol=0,
    )
    cosine = torch.nn.functional.cosine_similarity(
        expected_q.float().flatten(), actual_q.float().flatten(), dim=0
    ).item()
    assert cosine > 0.99999, cosine
    max_abs = (expected_q.float() - actual_q.float()).abs().max().item()

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
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 6144, 8192],
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
