"""Benchmark the MLX deferred radix prefill kernels: per-token vs tiled.

Apple Silicon only (MLX on Metal). Times one attention layer per call, with the
two kernels interleaved sample by sample so thermal drift hits both alike, and
checks the two outputs agree before timing.

    python test/manual/kernels/bench_deferred_radix_prefill_mlx.py
    python test/manual/kernels/bench_deferred_radix_prefill_mlx.py --heads 6,2,64
"""

import argparse
import math
import time

import mlx.core as mx

from sglang.kernels.ops.attention import _deferred_radix_attention_mlx as radix

POOL_SLOTS = TABLE_STRIDE = 32768
# Back-to-back calls rotate through distinct pools so one call's KV is not
# cache-warm for the next, as consecutive layers would not be.
NUM_POOLS = 4

SINGLE_REQUEST_SHAPES = [(p, n) for p in (0, 512, 2048, 8192) for n in (16, 128, 512)]
# (prefix_lens, extend_lens) of one packed multi-request batch.
RAGGED_BATCHES = [((2048, 512, 0, 8192), (128, 64, 256, 16))]


def make_pools(spec):
    shape = (POOL_SLOTS, spec.num_kv_heads, spec.head_dim)
    pools = [
        (
            mx.random.normal(shape).astype(mx.bfloat16),
            mx.random.normal(shape).astype(mx.bfloat16),
        )
        for _ in range(NUM_POOLS)
    ]
    mx.eval(pools)
    return pools


def make_batch(spec, prefix_lens, extend_lens):
    num_tokens = sum(extend_lens)
    slots = (mx.random.permutation(POOL_SLOTS - 1)[: sum(prefix_lens)] + 1).astype(
        mx.int32
    )
    rows, start = [], 0
    for prefix_len in prefix_lens:
        rows.append(
            mx.concatenate(
                [
                    slots[start : start + prefix_len],
                    mx.zeros((TABLE_STRIDE - prefix_len,), dtype=mx.int32),
                ]
            )
        )
        start += prefix_len
    batch = dict(
        q=mx.random.normal((num_tokens, spec.num_q_heads, spec.head_dim)),
        current_k=mx.random.normal((num_tokens, spec.num_kv_heads, spec.head_dim)),
        current_v=mx.random.normal((num_tokens, spec.num_kv_heads, spec.head_dim)),
    )
    batch = {name: array.astype(mx.bfloat16) for name, array in batch.items()}
    batch["req_to_token"] = mx.stack(rows)
    batch["req_pool_indices"] = mx.arange(len(extend_lens), dtype=mx.int64)
    batch["extend_prefix_lens"] = mx.array(prefix_lens, dtype=mx.int64)
    batch["extend_seq_lens"] = mx.array(extend_lens, dtype=mx.int64)
    mx.eval(batch)
    return batch


def launcher(launch, spec, batch, pools):
    request_count = batch["req_pool_indices"].shape[0]

    def call(pool_index):
        k_pool, v_pool = pools[pool_index]
        arrays = (
            batch["q"],
            batch["current_k"],
            batch["current_v"],
            k_pool,
            v_pool,
            batch["req_to_token"],
            batch["req_pool_indices"],
            batch["extend_prefix_lens"],
            batch["extend_seq_lens"],
        )
        return launch(arrays, spec=spec, request_count=request_count)

    return call


def bench(calls, samples, min_ms=25.0):
    repeats = []
    for call in calls:
        mx.eval(call(0), call(1))
        start = time.perf_counter()
        mx.eval(call(0))
        once_ms = (time.perf_counter() - start) * 1e3
        repeats.append(max(1, min(64, int(min_ms / max(once_ms, 1e-3)))))
    times = [[] for _ in calls]
    for _ in range(samples):
        for call, repeat, out in zip(calls, repeats, times):
            start = time.perf_counter()
            mx.eval([call(r % NUM_POOLS) for r in range(repeat)])
            out.append((time.perf_counter() - start) * 1e3 / repeat)
    stats = []
    for out in times:
        out.sort()
        p95 = out[min(len(out) - 1, math.ceil(0.95 * len(out)) - 1)]
        stats.append((sum(out) / len(out), p95))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", default="16,8,128", help="q_heads,kv_heads,head_dim")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    num_q_heads, num_kv_heads, head_dim = map(int, args.heads.split(","))
    spec = radix.DeferredAttentionSpec(
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    mx.random.seed(0)
    pools = make_pools(spec)

    print(f"{mx.device_info()['device_name']}, mlx {mx.__version__}, {spec}")
    print(
        f"{'prefix_lens':>22} {'extend_lens':>20} {'per-token ms':>13} {'p95':>8}"
        f" {'tiled ms':>9} {'p95':>8} {'speedup':>8} {'max diff':>9}"
    )
    batches = [((p,), (n,)) for p, n in SINGLE_REQUEST_SHAPES] + RAGGED_BATCHES
    for prefix_lens, extend_lens in batches:
        batch = make_batch(spec, prefix_lens, extend_lens)
        per_token = launcher(radix._launch_per_token_prefill, spec, batch, pools)
        tiled = launcher(radix._launch_tiled_prefill, spec, batch, pools)
        diff = mx.abs(
            per_token(0).astype(mx.float32) - tiled(0).astype(mx.float32)
        ).max()
        (base, base_p95), (ours, ours_p95) = bench([per_token, tiled], args.samples)
        print(
            f"{str(prefix_lens):>22} {str(extend_lens):>20} {base:>13.3f} {base_p95:>8.3f}"
            f" {ours:>9.3f} {ours_p95:>8.3f} {base / ours:>7.1f}x {diff.item():>9.1e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
