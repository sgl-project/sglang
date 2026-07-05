"""Benchmark the AOT Metal ``rms_norm`` kernel against ``mx.fast.rms_norm``.

Sweeps (dtype, batch, hidden) and prints per-config microseconds and the
speedup over the MLX baseline. Run on Apple Silicon with the sgl-kernel Metal
extension installed:

    ~/venvs/sglang/bin/python sgl-kernel/benchmark/bench_metal_rmsnorm.py

Unlike sgl-kernel/benchmark/bench_rmsnorm.py (CUDA, torch/triton), timing here
uses MLX's ``mx.eval`` barrier because MLX is lazy: a kernel call only records a
graph node and runs nothing until evaluation is forced.
"""

import argparse
import itertools
import time

import mlx.core as mx

from sgl_kernel import metal

if metal._metal is None:  # pragma: no cover - guards a missing/failed build
    raise SystemExit(f"sgl_kernel.metal not available: {metal._IMPORT_ERROR}")

EPS = 1e-6
DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16}


def rms_norm_ref_fp32(x, w, eps=EPS):
    """Reference RMSNorm computed in fp32 then cast back, the correctness oracle."""
    orig = x.dtype
    xf = x.astype(mx.float32)
    var = mx.mean(xf * xf, axis=-1, keepdims=True)
    xf = xf * mx.rsqrt(var + eps)
    return (xf * w.astype(mx.float32)).astype(orig)


def bench(fn, warmup=10, iters=100):
    """Return mean seconds per call for the thunk ``fn``.

    ``fn`` must build and return a fresh output array each call (do not reuse a
    single evaluated array, or the barrier below becomes a no-op).

    MLX is lazy: calling the kernel only records a graph node. You MUST force the
    GPU to run and finish, or you measure graph construction (roughly nothing).
    """
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    end = time.perf_counter()
    return (end - start) / iters

def bench_chain(step, x0, chain=200, warmup=3):
    for _ in range(warmup):
        mx.eval(step(x0))
    start = time.perf_counter()
    for _ in range(chain):
        x0 = step(x0)
    mx.eval(x0)
    end = time.perf_counter()
    return (end - start) / chain


def bench_pool(step, x_pool, repeats=7, warmup=3):
    """Best amortized seconds per call over a pool of DISTINCT inputs.

    Distinct inputs aren't common-subexpression-eliminated, so all of them run,
    and one ``mx.eval`` amortizes the barrier over the whole pool. No output
    feeds a later input, so there is no numerical drift (the dependent-chain
    version let f16 wander into slow denormals).

    Returns the MINIMUM over ``repeats`` timed passes: transient interference
    (allocation stalls, thermal, OS scheduling) only ever makes a pass slower,
    so the minimum is the uncontended time and is stable across runs.
    """
    n = len(x_pool)
    for _ in range(warmup):
        mx.eval(step(x_pool[0]))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        outs = [step(x) for x in x_pool]
        mx.eval(outs)
        best = min(best, (time.perf_counter() - start) / n)
    return best


def verify(x, w):
    """Max abs diff of the Metal kernel vs the fp32 reference, for a sanity gate."""
    y = metal.rms_norm(x, w, eps=EPS)
    ref = rms_norm_ref_fp32(x, w)
    mx.eval(y, ref)
    return mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))).item()


def run(batch_sizes, hidden_sizes, dtype_names, warmup, iters):
    header = (
        f"{'dtype':9s} {'batch':>6s} {'hidden':>7s} "
        f"{'metal_us':>10s} {'mxfast_us':>10s} {'speedup':>8s} {'max|diff|':>10s}"
    )
    print(header)
    print("-" * len(header))
    for dtype_name in dtype_names:
        dt = DTYPES[dtype_name]
        for B, H in itertools.product(batch_sizes, hidden_sizes):
            # Clear MLX's buffer cache so pressure from the previous config does
            # not leak into this one's timing, then build a distinct-input pool
            # (bounded to ~128 MB) and materialize it outside the timed region.
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            pool = max(8, min(32, (128 << 20) // max(1, B * H * 2)))
            x_pool = [mx.random.normal((B, H)).astype(dt) for _ in range(pool)]
            w = mx.random.normal((H,)).astype(dt)
            mx.eval(w, *x_pool)

            diff = verify(x_pool[0], w)  # sanity gate before we trust the timing

            t_metal = bench_pool(lambda xx: metal.rms_norm(xx, w, eps=EPS), x_pool)
            t_fast = bench_pool(lambda xx: mx.fast.rms_norm(xx, w, EPS), x_pool)
            speedup = t_fast / t_metal if t_metal > 0 else float("nan")

            print(
                f"{dtype_name:9s} {B:6d} {H:7d} "
                f"{t_metal * 1e6:10.2f} {t_fast * 1e6:10.2f} "
                f"{speedup:7.2f}x {diff:10.3e}"
            )


def _int_list(s):
    return [int(v) for v in s.split(",") if v]


def _str_list(s):
    return [v for v in s.split(",") if v]


if __name__ == "__main__":
    p = argparse.ArgumentParser("Metal rms_norm benchmark")
    p.add_argument("--batch_sizes", type=_int_list, default=[1, 8, 32, 128, 512, 2048])
    p.add_argument("--hidden_sizes", type=_int_list, default=[1024, 4096])
    p.add_argument("--dtypes", type=_str_list, default=["float16", "bfloat16"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    run(args.batch_sizes, args.hidden_sizes, args.dtypes, args.warmup, args.iters)
