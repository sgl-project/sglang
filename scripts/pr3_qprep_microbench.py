#!/usr/bin/env python3
"""Microbenchmark: Q8KV8 sparse-prefill q-prep — old path vs born-fp8 fused path.

Old path (production default):
    1. q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)
       (cublas bf16 bmm, writes bf16 [H, T, N] to DRAM)
    2. concat_and_cast_q_fp8_pad(q_fp8, q_nope_out, q_rope, H)
       (Triton: re-reads the bf16 bmm output + q_rope, writes fp8 [T, H, N+R])

New path (SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q):
    absorbed_bmm_concat_cast_q_fp8(q_fp8, q_nope, w_kc, q_rope, H)
    (one Triton kernel: bmm + concat + fp8 cast; the bf16 q_nope_out and the
    standalone concat-cast launch disappear)

Non-power-of-2 K (GLM 192) K-dimension codegen variants (A/B'd here; all keep
the identical fp32 -> bf16 -> fp8 epilogue, see cache_ops.py):
    loop      : split-K loop, BLOCK_K=64 x 3 (the original K=192 path)
    two_dot   : preload a once as 128+64 tiles, two chained tl.dot, no K-loop
    three_dot : preload a once as 3 x 64 tiles, three chained tl.dot
                (same fp32 add order as `loop`, loads hoisted)
    pad       : single tl.dot at BLOCK_K=256 with zero-masked k tail
    single_k  : single tl.dot at BLOCK_K=192 -- documents the Triton
                non-power-of-2 tl.arange limitation (compile fails <= 3.5.x)
Power-of-2 K (DeepSeek 128) collapses every variant to the same single-dot
fast path, so only one NEW row is shown there.

Shapes (both models: N = kv_lora_rank = 512, R = qk_rope_head_dim = 64;
K = qk_nope_head_dim differs per model):
    GLM-5.2: heads = 64,  K = 192  (w_kc [64, 192, 512]; DP attention,
             per-rank full heads; K=192 exercises the kernel's split-K path)
    DS-V3.2: heads = 128, K = 128  (power-of-2 K, preload-once fast path)

Metric conventions (per repo rule):
    * time is reported in microseconds per call (us/call) — LOWER = FASTER.
    * bandwidth is analytic-bytes / time in GB/s — HIGHER = BETTER.
    * "speedup x" = old_time / new_time — >1.0 means the NEW path is faster.

Correctness:
    * rope half must be BIT-EXACT (same bf16 source, same Triton conversion).
    * nope half: same rounding stages (fp32 accum -> bf16 -> fp8) but a
      different GEMM accumulation order than cublas -> near- but not
      guaranteed bit-exact.  We report the bitwise-match fraction, the max
      dequantized |diff|, and which path lands closer to an fp64 reference.

Usage (single GPU):
    python scripts/pr3_qprep_microbench.py                # both model shapes
    python scripts/pr3_qprep_microbench.py --tokens 8192 --iters 300
    python scripts/pr3_qprep_microbench.py --variants two_dot,pad
    python scripts/pr3_qprep_microbench.py --sweep        # + tile/warp sweep
    python scripts/pr3_qprep_microbench.py --rounding-study
"""

import argparse

import torch

from sglang.kernels.ops.kvcache.cache_ops import (
    absorbed_bmm_concat_cast_q_fp8,
    concat_and_cast_q_fp8_pad,
)

N_LORA = 512  # kv_lora_rank (post-absorb q_nope dim; "d_nope" at the kernel)
R_ROPE = 64  # qk_rope_head_dim


def make_inputs(
    num_tokens: int,
    num_heads: int,
    k_nope: int,
    device,
    seed: int,
    magnitude: float,
):
    g = torch.Generator(device=device).manual_seed(seed)
    # Production layout: q = q_b_proj output [T, H, K+R] bf16; q_nope/q_rope are
    # strided views of it (rope applied in-place on the q_rope slice).
    q = (
        torch.randn(
            (num_tokens, num_heads, k_nope + R_ROPE),
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        * magnitude
    ).to(torch.bfloat16)
    q_nope = q[..., :k_nope]
    q_rope = q[..., k_nope:]
    # Production w_kc layout: [H, K, N] with strides (K*N, 1, K) (N-major), the
    # result of w_kc.transpose(1, 2).contiguous().transpose(1, 2) at load.
    w_base = (
        torch.randn(
            (num_heads, N_LORA, k_nope),
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        / (k_nope**0.5)
    ).to(torch.bfloat16)
    w_kc = w_base.transpose(1, 2)
    return q, q_nope, q_rope, w_kc


def old_path(q_fp8, q_nope, w_kc, q_rope, num_heads):
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)
    concat_and_cast_q_fp8_pad(q_fp8, q_nope_out, q_rope, num_heads)


def old_path_bmm_only(q_nope, w_kc):
    return torch.bmm(q_nope.transpose(0, 1), w_kc)


def new_path(q_fp8, q_nope, w_kc, q_rope, num_heads, **kw):
    absorbed_bmm_concat_cast_q_fp8(q_fp8, q_nope, w_kc, q_rope, num_heads, **kw)


# Non-power-of-2-K variants, in bench order (power-of-2 K collapses to "auto").
ALL_VARIANTS = ["loop", "two_dot", "three_dot", "pad", "single_k"]

# (block_m, block_n, num_warps, num_stages) sweep grid; num_stages 0 = Triton
# default.  N=512 is a multiple of every block_n here; block_m stays power of 2.
SWEEP_TILES = [
    (64, 128, 4, 0),  # kernel default
    (64, 128, 8, 0),
    (64, 128, 4, 2),
    (64, 128, 4, 4),
    (128, 128, 4, 0),
    (128, 128, 8, 0),
    (64, 256, 8, 0),
    (128, 256, 8, 0),
    (32, 128, 4, 0),
    (64, 64, 4, 0),
    (128, 64, 8, 0),
]


def time_fn(fn, iters: int, warmup: int) -> float:
    """Median wall time of fn() in microseconds per call (lower = faster)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    chunk = 10
    for _ in range(max(1, iters // chunk)):
        start.record()
        for _ in range(chunk):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / chunk)  # ms -> us
    times.sort()
    return times[len(times) // 2]


def analytic_bytes(num_tokens: int, num_heads: int, k_nope: int):
    """(old_bytes, new_bytes) of DRAM traffic per call, analytic lower bound."""
    t, h = num_tokens, num_heads
    a_read = t * h * k_nope * 2  # q_nope bf16
    w_read = h * k_nope * N_LORA * 2  # w_kc bf16
    nope_bf16 = t * h * N_LORA * 2  # bmm bf16 out (written then re-read)
    rope_read = t * h * R_ROPE * 2  # q_rope bf16
    fp8_write = t * h * (N_LORA + R_ROPE)  # q_fp8 out
    old = (a_read + w_read + nope_bf16) + (nope_bf16 + rope_read + fp8_write)
    new = a_read + w_read + rope_read + fp8_write
    return old, new


def make_check_ctx(num_tokens, num_heads, k_nope, device, seed, magnitude):
    """Fresh inputs + old-path fp8 output + fp64 bmm reference (once/config)."""
    q, q_nope, q_rope, w_kc = make_inputs(
        num_tokens, num_heads, k_nope, device, seed, magnitude
    )
    q_fp8_old = torch.zeros(
        (num_tokens, num_heads, N_LORA + R_ROPE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    old_path(q_fp8_old, q_nope, w_kc, q_rope, num_heads)
    torch.cuda.synchronize()
    # fp64 reference: which path's fp8 lands closer to the exact bmm?
    ref = torch.bmm(
        q_nope.transpose(0, 1).to(torch.float64), w_kc.to(torch.float64)
    ).transpose(0, 1)
    err_old = (q_fp8_old[..., :N_LORA].to(torch.float64) - ref).abs()
    return {
        "q_nope": q_nope,
        "q_rope": q_rope,
        "w_kc": w_kc,
        "num_heads": num_heads,
        "q_fp8_old": q_fp8_old,
        "ref": ref,
        "meanerr_old": err_old.mean().item(),
        "maxerr_old": err_old.max().item(),
    }


def check_variant(ctx, **new_kwargs):
    """Correctness of one new-path variant vs the old path + fp64 reference."""
    q_fp8_old = ctx["q_fp8_old"]
    q_fp8_new = torch.zeros_like(q_fp8_old)
    new_path(
        q_fp8_new,
        ctx["q_nope"],
        ctx["w_kc"],
        ctx["q_rope"],
        ctx["num_heads"],
        **new_kwargs,
    )
    torch.cuda.synchronize()

    rope_old = q_fp8_old[..., N_LORA:].view(torch.uint8)
    rope_new = q_fp8_new[..., N_LORA:].view(torch.uint8)
    rope_bitexact = bool(torch.equal(rope_old, rope_new))

    nope_old = q_fp8_old[..., :N_LORA]
    nope_new = q_fp8_new[..., :N_LORA]
    match = (
        (nope_old.view(torch.uint8) == nope_new.view(torch.uint8)).float().mean().item()
    )
    diff = (nope_old.to(torch.float32) - nope_new.to(torch.float32)).abs()
    max_diff = diff.max().item()

    err_new = (nope_new.to(torch.float64) - ctx["ref"]).abs()
    return {
        "rope_bitexact": rope_bitexact,
        "nope_bitwise_match_frac": match,
        "nope_max_dequant_absdiff": max_diff,
        "nope_meanerr_old_vs_fp64": ctx["meanerr_old"],
        "nope_meanerr_new_vs_fp64": err_new.mean().item(),
        "nope_maxerr_old_vs_fp64": ctx["maxerr_old"],
        "nope_maxerr_new_vs_fp64": err_new.max().item(),
    }


def rounding_study(device, seed):
    """Task-2a quantification: fp8(bf16(x)) double round vs fp8(x) single round.

    (Informational only — the born-fp8 kernel deliberately keeps the
    fp32->bf16->fp8 double round to match the default path's rounding stages.)
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((1 << 22,), generator=g, device=device, dtype=torch.float32) * 8.0
    double = x.to(torch.bfloat16).to(torch.float8_e4m3fn)
    single = x.to(torch.float8_e4m3fn)
    mismatch = (
        (double.view(torch.uint8) != single.view(torch.uint8)).float().mean().item()
    )
    err_double = (double.to(torch.float32) - x).abs()
    err_single = (single.to(torch.float32) - x).abs()
    print("\n=== rounding study: fp32->bf16->fp8 (double) vs fp32->fp8 (single) ===")
    print(f"elements                      : {x.numel()}")
    print(f"byte-mismatch fraction        : {mismatch:.3e} (fraction, lower = closer)")
    print(
        f"mean |err| vs fp32 (double)   : {err_double.mean().item():.6e} (lower = more accurate)"
    )
    print(
        f"mean |err| vs fp32 (single)   : {err_single.mean().item():.6e} (lower = more accurate)"
    )
    winner = (
        "single (direct fp32->fp8)"
        if err_single.mean() <= err_double.mean()
        else "double (via bf16)"
    )
    print(f"more accurate on average      : {winner}")
    print(
        "NOTE: the born-fp8 kernel keeps the DOUBLE round on purpose to match "
        "the default path's rounding stages."
    )


def run_config(
    name,
    num_tokens,
    num_heads,
    k_nope,
    iters,
    warmup,
    device,
    seed,
    magnitude,
    variants,
    sweep,
):
    print(f"\n=== {name}: tokens={num_tokens} heads={num_heads} K={k_nope} ===")
    print(
        f"    (K={k_nope} nope-in, N={N_LORA} nope-out, R={R_ROPE} rope; "
        "us/call LOWER = FASTER; GB/s HIGHER = BETTER; speedup >1 = new faster)"
    )
    q, q_nope, q_rope, w_kc = make_inputs(
        num_tokens, num_heads, k_nope, device, seed, magnitude
    )
    q_fp8 = torch.zeros(
        (num_tokens, num_heads, N_LORA + R_ROPE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    t_old = time_fn(
        lambda: old_path(q_fp8, q_nope, w_kc, q_rope, num_heads), iters, warmup
    )
    t_bmm = time_fn(lambda: old_path_bmm_only(q_nope, w_kc), iters, warmup)
    # standalone concat-cast (reads a fresh bf16 bmm out, like production)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)
    t_cast = time_fn(
        lambda: concat_and_cast_q_fp8_pad(q_fp8, q_nope_out, q_rope, num_heads),
        iters,
        warmup,
    )

    b_old, b_new = analytic_bytes(num_tokens, num_heads, k_nope)
    print(
        f"OLD  bmm (cublas bf16)        : {t_bmm:10.1f} us/call"
        f"  (component of OLD total)"
    )
    print(f"OLD  concat_and_cast_q_fp8_pad: {t_cast:10.1f} us/call  (component)")
    print(
        f"OLD  total (bmm + concat-cast): {t_old:10.1f} us/call"
        f"  ({b_old / 1e6:8.1f} MB analytic, {b_old / t_old / 1e3:7.0f} GB/s)"
    )

    # Power-of-2 K collapses every variant to the same single-dot codegen.
    pow2 = k_nope & (k_nope - 1) == 0
    run_variants = ["auto"] if pow2 else variants
    ctx = make_check_ctx(num_tokens, num_heads, k_nope, device, seed + 1, magnitude)
    results = {}
    for v in run_variants:
        kw = {"variant": v}
        try:
            t_new = time_fn(
                lambda: new_path(q_fp8, q_nope, w_kc, q_rope, num_heads, **kw),
                iters,
                warmup,
            )
        except Exception as e:
            msg = (str(e).splitlines() or [type(e).__name__])[0]
            print(f"NEW  {v:<24}: COMPILE/RUN FAIL — {msg[:100]}")
            continue
        c = check_variant(ctx, **kw)
        results[v] = t_new
        faster = "NEW FASTER" if t_new < t_old else "OLD FASTER"
        print(
            f"NEW  {v:<24}: {t_new:10.1f} us/call"
            f"  ({b_new / 1e6:8.1f} MB analytic, {b_new / t_new / 1e3:7.0f} GB/s,"
            f" speedup {t_old / t_new:5.2f}x vs OLD, {faster})"
        )
        rope = "PASS (bitwise identical)" if c["rope_bitexact"] else "FAIL (BUG)"
        print(
            f"     rope bit-exact: {rope}; nope bitwise match vs OLD"
            f" {c['nope_bitwise_match_frac'] * 100:9.4f}% (100% = bit-exact);"
            f" max |dequant diff| {c['nope_max_dequant_absdiff']:.4f}"
        )
        print(
            f"     nope |err| vs fp64 ref: mean old {c['nope_meanerr_old_vs_fp64']:.3e}"
            f" / new {c['nope_meanerr_new_vs_fp64']:.3e}; max old"
            f" {c['nope_maxerr_old_vs_fp64']:.3e} / new"
            f" {c['nope_maxerr_new_vs_fp64']:.3e}  (lower = more accurate)"
        )
    if results:
        best = min(results, key=results.get)
        print(
            f"BEST variant                  : {best} @ {results[best]:.1f} us/call"
            f" (speedup {t_old / results[best]:.2f}x vs OLD total)"
        )

    if sweep and results:
        print(
            f"\n--- tile sweep: {name} (us/call LOWER = FASTER;"
            " stages=0 -> Triton default) ---"
        )
        rows = []
        for v in results:
            for bm, bn, nw, ns in SWEEP_TILES:
                kw = dict(
                    variant=v, block_m=bm, block_n=bn, num_warps=nw, num_stages=ns
                )
                try:
                    t = time_fn(
                        lambda: new_path(q_fp8, q_nope, w_kc, q_rope, num_heads, **kw),
                        max(iters // 2, 20),
                        warmup,
                    )
                except Exception as e:
                    msg = (str(e).splitlines() or [type(e).__name__])[0]
                    print(
                        f"  {v:<10} bm={bm:<3} bn={bn:<3} warps={nw} stages={ns}:"
                        f"     FAIL — {msg[:70]}"
                    )
                    continue
                rows.append((t, v, bm, bn, nw, ns))
                print(
                    f"  {v:<10} bm={bm:<3} bn={bn:<3} warps={nw} stages={ns}:"
                    f" {t:8.1f} us/call ({b_new / t / 1e3:5.0f} GB/s,"
                    f" {t_old / t:5.2f}x vs OLD)"
                )
        rows.sort()
        print("  -- top 5 (fastest first) --")
        for t, v, bm, bn, nw, ns in rows[:5]:
            print(
                f"  {v:<10} bm={bm:<3} bn={bn:<3} warps={nw} stages={ns}:"
                f" {t:8.1f} us/call ({t_old / t:5.2f}x vs OLD)"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096, help="s_q per call")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--magnitude",
        type=float,
        default=1.0,
        help="input scale multiplier (q amax stress)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=None,
        help="run a single head count instead of the GLM(64,K192)+DS(128,K128) pair",
    )
    parser.add_argument(
        "--k-nope",
        type=int,
        default=128,
        help="qk_nope_head_dim for --heads runs (ignored for the default pair)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help=(
            "comma list of non-power-of-2-K variants to bench "
            f"(default: all = {','.join(ALL_VARIANTS)}); power-of-2-K configs "
            "always run the single collapsed 'auto' variant"
        ),
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also sweep (block_m, block_n, num_warps, num_stages) per variant",
    )
    parser.add_argument("--rounding-study", action="store_true")
    args = parser.parse_args()

    if args.variants == "all":
        variants = ALL_VARIANTS
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = set(variants) - set(ALL_VARIANTS) - {"auto"}
        assert not unknown, f"unknown variants: {sorted(unknown)}"

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    print(f"device: {name}; torch {torch.__version__}")

    if args.heads is not None:
        configs = [(f"custom h={args.heads} K={args.k_nope}", args.heads, args.k_nope)]
    else:
        configs = [
            ("GLM-5.2 (h=64, K=192)", 64, 192),
            ("DS-V3.2 (h=128, K=128)", 128, 128),
        ]
    for cfg_name, heads, k_nope in configs:
        run_config(
            cfg_name,
            args.tokens,
            heads,
            k_nope,
            args.iters,
            args.warmup,
            device,
            args.seed,
            args.magnitude,
            variants,
            args.sweep,
        )

    if args.rounding_study:
        rounding_study(device, args.seed)


if __name__ == "__main__":
    main()
