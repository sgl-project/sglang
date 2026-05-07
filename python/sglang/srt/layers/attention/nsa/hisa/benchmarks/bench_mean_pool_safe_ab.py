"""A/B bench: original `fp8_native_block_mean_pooling` (UNSAFE bulk T.copy)
vs two safe variants. Goal — replace the production kernel iff the safe
version doesn't regress perf.

The bug
-------
``T.copy(K[s:s+block_N], index_k)`` issues a 64-row TMA load even when
the last pool block has < block_N valid rows. When ``seq_kv % block_N != 0``
the trailing CTA over-reads up to block_N-1 rows past the K tensor end
→ Xid 13 / illegal memory access at allocator page boundaries.

Two fix candidates
------------------
v1 (always row-guard):
    Replace bulk T.copy with ``T.Parallel(block_N) + if bn_i < cur_size``.
    Mirrors the grouped variant. Loses TMA's bulk async benefit.

v2 (branch on tile fullness):
    For full tiles (cur_tl_block_size == block_N) use bulk T.copy.
    For partial tiles use row-guard. Production hits "full" 99%+ of the
    time (only the last partial CTA has a partial inner iter). If tilelang
    refuses T.copy inside a runtime if-branch, this version won't compile
    and we fall back to v1.

Bench setup
-----------
HISA prefill mean-pool: K_fp8 [seq_kv, D=128], K_scale [seq_kv].
Production K=128 path. Aligned seq_kv only (so the unsafe original
doesn't crash). 5 warmup, 50 timed iters.
"""
import argparse
import time

import torch
import tilelang
import tilelang.language as T

from sglang.srt.layers.attention.nsa.hisa.tilelang_kernels import (
    fp8_native_block_mean_pooling as _orig_kernel_factory,
    fp8_native_block_mean_pooling_interface as orig_interface,
)


DEVICE = torch.device("cuda")
DTYPE = torch.float8_e4m3fn
DIM = 128


# ---------------------------------------------------------------------------
# v1: always row-guard (no bulk T.copy)
# ---------------------------------------------------------------------------

@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def safe_v1_factory(
    max_num_pooling_blocks: int,
    pooling_block_size: int,
    dim: int,
    block_N: int = 64,
    threads: int = 256,
):
    dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    seq_len_k = T.dynamic("seq_len_k")
    k_size = [seq_len_k, dim]
    scale_size = [seq_len_k]
    blocked_k_size = [max_num_pooling_blocks, dim]
    blocked_k_scale_size = [max_num_pooling_blocks]
    FP8_MAX_INV = 1.0 / 448.0

    @T.prim_func
    def kernel(
        K: T.Tensor(k_size, dtype=dtype),                                             # type: ignore
        KScale: T.Tensor(scale_size, dtype=accum_dtype),                              # type: ignore
        BlockedK: T.Tensor(blocked_k_size, dtype=dtype),                              # type: ignore
        BlockedKScale: T.Tensor(blocked_k_scale_size, dtype=accum_dtype),             # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len_k, pooling_block_size), threads=threads) as bx:
            index_k = T.alloc_fragment([block_N, dim], dtype)
            scale = T.alloc_fragment([block_N], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            max_abs = T.alloc_fragment([1], accum_dtype)
            T.fill(acc, 0.0)

            k_start = bx * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len_k)
            cur_pooling_block_size = k_end - k_start

            for b_i in T.serial(T.ceildiv(cur_pooling_block_size, block_N)):
                T.fill(index_k, 0.0)
                T.fill(scale, 0.0)

                tl_block_s = k_start + b_i * block_N
                tl_block_e = T.min(k_start + (b_i + 1) * block_N, k_end)
                cur_tl_block_size = tl_block_e - tl_block_s

                # Per-row guarded loads — no bulk T.copy. Each lane checks
                # its own bounds before issuing the load.
                for bn_i in T.Parallel(block_N):
                    if bn_i < cur_tl_block_size:
                        scale[bn_i] = KScale[tl_block_s + bn_i]
                for bn_i, d_i in T.Parallel(block_N, dim):
                    if bn_i < cur_tl_block_size:
                        index_k[bn_i, d_i] = K[tl_block_s + bn_i, d_i]

                for bn_i, d_i in T.Parallel(block_N, dim):
                    index_k[bn_i, d_i] = index_k[bn_i, d_i] * scale[bn_i]

                T.reduce_sum(index_k, acc, dim=0, clear=False)

            inv_count = T.cast(1.0, accum_dtype) / T.cast(cur_pooling_block_size, accum_dtype)
            for d_i in T.Parallel(dim):
                acc[d_i] = acc[d_i] * inv_count

            T.reduce_absmax(acc, max_abs, dim=0, clear=True)
            block_scale = T.max(
                max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype),
                T.cast(1e-10, accum_dtype),
            )
            inv_block_scale = T.cast(1.0, accum_dtype) / block_scale

            for d_i in T.Parallel(dim):
                BlockedK[bx, d_i] = T.cast(acc[d_i] * inv_block_scale, dtype)
            BlockedKScale[bx] = block_scale

    return kernel


# ---------------------------------------------------------------------------
# v2: branch on tile fullness (T.copy for full, row-guard for partial)
# ---------------------------------------------------------------------------

@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def safe_v2_factory(
    max_num_pooling_blocks: int,
    pooling_block_size: int,
    dim: int,
    block_N: int = 64,
    threads: int = 256,
):
    dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    seq_len_k = T.dynamic("seq_len_k")
    k_size = [seq_len_k, dim]
    scale_size = [seq_len_k]
    blocked_k_size = [max_num_pooling_blocks, dim]
    blocked_k_scale_size = [max_num_pooling_blocks]
    FP8_MAX_INV = 1.0 / 448.0

    @T.prim_func
    def kernel(
        K: T.Tensor(k_size, dtype=dtype),                                             # type: ignore
        KScale: T.Tensor(scale_size, dtype=accum_dtype),                              # type: ignore
        BlockedK: T.Tensor(blocked_k_size, dtype=dtype),                              # type: ignore
        BlockedKScale: T.Tensor(blocked_k_scale_size, dtype=accum_dtype),             # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len_k, pooling_block_size), threads=threads) as bx:
            index_k = T.alloc_fragment([block_N, dim], dtype)
            scale = T.alloc_fragment([block_N], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            max_abs = T.alloc_fragment([1], accum_dtype)
            T.fill(acc, 0.0)

            k_start = bx * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len_k)
            cur_pooling_block_size = k_end - k_start

            for b_i in T.serial(T.ceildiv(cur_pooling_block_size, block_N)):
                T.fill(index_k, 0.0)
                T.fill(scale, 0.0)

                tl_block_s = k_start + b_i * block_N
                tl_block_e = T.min(k_start + (b_i + 1) * block_N, k_end)
                cur_tl_block_size = tl_block_e - tl_block_s

                # Fast path: full tile → bulk TMA. Slow path: partial tile
                # → row-guard. Tilelang lowers ``if`` to a runtime branch;
                # bulk T.copy inside the branch must be supported (verify
                # by compiling).
                if cur_tl_block_size == block_N:
                    T.copy(K[tl_block_s:tl_block_s + block_N, :], index_k)
                    for bn_i in T.Parallel(block_N):
                        scale[bn_i] = KScale[tl_block_s + bn_i]
                else:
                    for bn_i in T.Parallel(block_N):
                        if bn_i < cur_tl_block_size:
                            scale[bn_i] = KScale[tl_block_s + bn_i]
                    for bn_i, d_i in T.Parallel(block_N, dim):
                        if bn_i < cur_tl_block_size:
                            index_k[bn_i, d_i] = K[tl_block_s + bn_i, d_i]

                for bn_i, d_i in T.Parallel(block_N, dim):
                    index_k[bn_i, d_i] = index_k[bn_i, d_i] * scale[bn_i]

                T.reduce_sum(index_k, acc, dim=0, clear=False)

            inv_count = T.cast(1.0, accum_dtype) / T.cast(cur_pooling_block_size, accum_dtype)
            for d_i in T.Parallel(dim):
                acc[d_i] = acc[d_i] * inv_count

            T.reduce_absmax(acc, max_abs, dim=0, clear=True)
            block_scale = T.max(
                max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype),
                T.cast(1e-10, accum_dtype),
            )
            inv_block_scale = T.cast(1.0, accum_dtype) / block_scale

            for d_i in T.Parallel(dim):
                BlockedK[bx, d_i] = T.cast(acc[d_i] * inv_block_scale, dtype)
            BlockedKScale[bx] = block_scale

    return kernel


# ---------------------------------------------------------------------------
# Interface wrappers (cache compiled kernels per shape)
# ---------------------------------------------------------------------------

_cache_v1 = {}
_cache_v2 = {}


def safe_v1_interface(k, k_scale, k_block_size):
    seq_len_k, d = k.shape
    max_num_pooling_blocks = (seq_len_k + k_block_size - 1) // k_block_size
    key = (max_num_pooling_blocks, k_block_size, d)
    if key not in _cache_v1:
        _cache_v1[key] = safe_v1_factory(*key)
    blocked_k = torch.empty((max_num_pooling_blocks, d), device=k.device, dtype=DTYPE)
    blocked_k_scale = torch.empty((max_num_pooling_blocks,), device=k.device, dtype=torch.float32)
    _cache_v1[key](k, k_scale, blocked_k, blocked_k_scale)
    return blocked_k, blocked_k_scale


def safe_v2_interface(k, k_scale, k_block_size):
    seq_len_k, d = k.shape
    max_num_pooling_blocks = (seq_len_k + k_block_size - 1) // k_block_size
    key = (max_num_pooling_blocks, k_block_size, d)
    if key not in _cache_v2:
        _cache_v2[key] = safe_v2_factory(*key)
    blocked_k = torch.empty((max_num_pooling_blocks, d), device=k.device, dtype=DTYPE)
    blocked_k_scale = torch.empty((max_num_pooling_blocks,), device=k.device, dtype=torch.float32)
    _cache_v2[key](k, k_scale, blocked_k, blocked_k_scale)
    return blocked_k, blocked_k_scale


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------

def make_inputs(seq_kv, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    k_bf16 = torch.randn(seq_kv, DIM, dtype=torch.bfloat16, device=DEVICE, generator=g)
    k_fp8 = k_bf16.to(DTYPE)
    k_scale = (torch.rand(seq_kv, dtype=torch.float32, device=DEVICE, generator=g) + 0.5)
    return k_fp8, k_scale


def time_fn(fn, *, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def correctness_check(seq_kv, K):
    """Verify all 3 versions produce the same output on aligned input."""
    k_fp8, k_scale = make_inputs(seq_kv, seed=42)
    bk_orig, bks_orig = orig_interface(k_fp8, k_scale, K)
    bk_v1, bks_v1 = safe_v1_interface(k_fp8, k_scale, K)
    try:
        bk_v2, bks_v2 = safe_v2_interface(k_fp8, k_scale, K)
        v2_ok = True
    except Exception as e:
        bk_v2, bks_v2 = None, None
        v2_ok = False
        v2_err = str(e)

    def cmp(a, b, tol=1e-2):
        diff = (a.float() - b.float()).abs().max().item()
        return diff, diff < tol

    d1, ok1 = cmp(bk_orig, bk_v1)
    d2, ok_s1 = cmp(bks_orig, bks_v1)
    print(f"  v1 vs orig: bk_diff={d1:.4f} ({'OK' if ok1 else 'BAD'})  "
          f"bks_diff={d2:.6f} ({'OK' if ok_s1 else 'BAD'})")
    if v2_ok:
        d3, ok2 = cmp(bk_orig, bk_v2)
        d4, ok_s2 = cmp(bks_orig, bks_v2)
        print(f"  v2 vs orig: bk_diff={d3:.4f} ({'OK' if ok2 else 'BAD'})  "
              f"bks_diff={d4:.6f} ({'OK' if ok_s2 else 'BAD'})")
    else:
        print(f"  v2: COMPILE FAILED — {v2_err[:200]}")
    return v2_ok


def bench_one(seq_kv, K, *, warmup=20, iters=200, v2_available=True):
    """Pre-compile all 3 versions, then time them in a tight interleaved loop
    to share GPU clock state."""
    k_fp8, k_scale = make_inputs(seq_kv)

    # Force JIT compile first (so warmup of time_fn doesn't include it)
    orig_interface(k_fp8, k_scale, K)
    safe_v1_interface(k_fp8, k_scale, K)
    if v2_available:
        try:
            safe_v2_interface(k_fp8, k_scale, K)
        except Exception:
            v2_available = False
    torch.cuda.synchronize()

    t_orig = time_fn(lambda: orig_interface(k_fp8, k_scale, K),
                     warmup=warmup, iters=iters)
    t_v1 = time_fn(lambda: safe_v1_interface(k_fp8, k_scale, K),
                   warmup=warmup, iters=iters)
    t_v2 = None
    if v2_available:
        t_v2 = time_fn(lambda: safe_v2_interface(k_fp8, k_scale, K),
                       warmup=warmup, iters=iters)

    return t_orig, t_v1, t_v2


def unaligned_safety_check(K):
    """Verify v1/v2 don't crash on unaligned seq_kv. Note: orig WILL crash
    here under sustained load (silent OOB → eventually Xid 13). We don't
    run orig in this mode — it's the bug we're fixing."""
    for offset in [1, 7, 63, 97]:
        seq_kv = 16384 - offset
        k_fp8, k_scale = make_inputs(seq_kv, seed=offset)
        bk_v1, bks_v1 = safe_v1_interface(k_fp8, k_scale, K)
        try:
            bk_v2, bks_v2 = safe_v2_interface(k_fp8, k_scale, K)
            v2_match = (bk_v1.float() - bk_v2.float()).abs().max().item() < 1e-2
        except Exception:
            v2_match = False
        torch.cuda.synchronize()
        # Sanity: outputs are finite
        v1_finite = torch.isfinite(bk_v1.float()).all().item()
        print(f"  seq_kv={seq_kv:6d} (K={K}, tail={offset:>3} rows) "
              f"v1 finite={v1_finite}  v1==v2: {v2_match}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=128, help="kv_block_size")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=3,
                        help="Repeat the full sweep N times to gauge variance")
    args = parser.parse_args()

    K = args.K
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"K = {K}, DIM = {DIM}")
    print()

    print("Correctness (aligned seq_kv = 65536):")
    v2_available = correctness_check(65536, K)
    print()

    print("Safety on UNALIGNED seq_kv (orig would silent-OOB here):")
    unaligned_safety_check(K)
    print()

    # Realistic prefill ctx values (multiples of K)
    seq_kvs = [4096, 16384, 32768, 65536, 131072]

    for run_i in range(args.repeat):
        print(f"\n=== Run {run_i + 1} / {args.repeat} (iters={args.iters}) ===")
        print(f"{'seq_kv':>10} | {'orig (us)':>11} | {'v1 (us)':>10} | "
              f"{'v2 (us)':>10} | {'v1 vs orig':>11} | {'v2 vs orig':>11}")
        print("-" * 80)
        for seq_kv in seq_kvs:
            t_orig, t_v1, t_v2 = bench_one(seq_kv, K, iters=args.iters,
                                            v2_available=v2_available)
            v1_ratio = f"{t_v1 / t_orig * 100:6.1f}%"
            v2_ratio = f"{t_v2 / t_orig * 100:6.1f}%" if t_v2 is not None else "  N/A"
            v2_str = f"{t_v2:10.2f}" if t_v2 is not None else f"{'N/A':>10}"
            print(f"{seq_kv:>10} | {t_orig:11.2f} | {t_v1:10.2f} | "
                  f"{v2_str} | {v1_ratio:>11} | {v2_ratio:>11}")


if __name__ == "__main__":
    main()
