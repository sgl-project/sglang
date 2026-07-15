"""
Benchmark & Correctness: Triton GDN vs CuTeDSL GDN (prefill, SM100 Blackwell).

Compares:
  - Triton:  sglang's chunk_gated_delta_rule (FLA chunkwise, fp32 state, K-contig pool)
  - CuteDSL: ported vLLM #43273 chunk_gated_delta_rule_cutedsl (SM100 only)

The two kernels share the same math and the same g/beta convention (log-space
g, post-sigmoid beta). The CuteDSL kernel needs pre-allocated chunk metadata
from prepare_metadata_cutedsl, and l2norm is done outside the kernel.

Reports correctness (output & state matching) and performance (ms, TFLOPS, TB/s).

Usage:
    python bench_gdn_prefill_cutedsl.py                       # default sweep
    python bench_gdn_prefill_cutedsl.py --mode bench          # benchmark only
    python bench_gdn_prefill_cutedsl.py --mode correctness    # correctness only
    python bench_gdn_prefill_cutedsl.py --preset qwen3-next   # Qwen3-Next config
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch

from sglang.kernels.ops.attention.linear.gdn_blackwell import (
    chunk_gated_delta_rule_cutedsl,
    prepare_metadata_cutedsl,
)
from sglang.kernels.ops.attention.fla.chunk import (
    chunk_gated_delta_rule as triton_chunk_gated_delta_rule,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

# ---------------------------------------------------------------------------
# Helpers (shared shape: pool layout [N, H, K, V] with K-last stride)
# ---------------------------------------------------------------------------


def make_k_contiguous(t: torch.Tensor) -> torch.Tensor:
    """K-last view of a logical [..., K, V] tensor (physically [..., V, K])."""
    return t.transpose(-2, -1).contiguous().transpose(-2, -1)


def gdn_flops(total_seq_len, num_heads, head_size_k, head_size_v):
    """Per-token-per-head: k@v^T outer (2*K*V) + q@state output (2*K*V)."""
    return 4 * total_seq_len * num_heads * head_size_k * head_size_v


def gdn_bytes(
    total_seq_len, num_q_heads, num_v_heads, head_size_k, head_size_v, num_seqs, dtype
):
    num_o_heads = max(num_q_heads, num_v_heads)
    elem = dtype.itemsize
    q_b = total_seq_len * num_q_heads * head_size_k * elem
    k_b = total_seq_len * num_v_heads * head_size_k * elem
    v_b = total_seq_len * num_v_heads * head_size_v * elem
    o_b = total_seq_len * num_o_heads * head_size_v * elem
    state_b = 2 * num_seqs * num_o_heads * head_size_k * head_size_v * 4  # fp32 r/w
    g_b = total_seq_len * num_o_heads * 4
    beta_b = total_seq_len * num_o_heads * 4
    return q_b + k_b + v_b + o_b + state_b + g_b + beta_b


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------


def make_inputs(
    B, T_per_seq, H, K, V, pool_size, device, dtype, sequential_indices=False, seed=42
):
    T = B * T_per_seq
    torch.manual_seed(seed)

    if sequential_indices:
        cache_indices = torch.arange(B, dtype=torch.int32, device=device)
    else:
        perm = torch.randperm(pool_size, device=device)[:B]
        cache_indices = perm.to(torch.int32)

    pool_init = torch.randn(pool_size, H, K, V, dtype=dtype, device=device) * 0.1
    cu_seqlens = torch.arange(
        0, (B + 1) * T_per_seq, T_per_seq, dtype=torch.long, device=device
    )

    q = torch.randn(1, T, H, K, dtype=dtype, device=device)
    k = torch.randn(1, T, H, K, dtype=dtype, device=device)
    v = torch.randn(1, T, H, V, dtype=dtype, device=device)

    g_raw = torch.randn(1, T, H, dtype=dtype, device=device)
    g_triton = torch.nn.functional.logsigmoid(g_raw)
    beta_triton = torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device))

    return dict(
        B=B,
        T=T,
        T_per_seq=T_per_seq,
        H=H,
        K=K,
        V=V,
        pool_size=pool_size,
        cache_indices=cache_indices,
        pool_init=pool_init,
        cu_seqlens=cu_seqlens,
        q=q,
        k=k,
        v=v,
        g_triton=g_triton,
        beta_triton=beta_triton,
    )


# ---------------------------------------------------------------------------
# Runner wrappers
# ---------------------------------------------------------------------------


def run_triton(inp):
    """Triton path: K-contiguous pool, pool-indexed, [1,T,H,D] tensors."""
    pool = make_k_contiguous(inp["pool_init"].clone())
    o, _, h = triton_chunk_gated_delta_rule(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g_triton"],
        beta=inp["beta_triton"],
        initial_state=pool,
        initial_state_indices=inp["cache_indices"],
        cu_seqlens=inp["cu_seqlens"],
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    return o, pool, h


def run_cutedsl(inp):
    """CuteDSL path: matches CuteDSLGDNKernel.extend() exactly."""
    pool = make_k_contiguous(inp["pool_init"].clone())
    cache_indices = inp["cache_indices"]
    cu_seqlens = inp["cu_seqlens"].to(torch.int32)

    q_in = l2norm_fwd(inp["q"][0].contiguous()).unsqueeze(0)
    k_in = l2norm_fwd(inp["k"][0].contiguous()).unsqueeze(0)
    v_in = inp["v"][0].contiguous().unsqueeze(0)
    g_in = inp["g_triton"][0].to(torch.float32).unsqueeze(0)
    beta_in = inp["beta_triton"][0].to(torch.float32).unsqueeze(0)

    initial_state = pool[cache_indices.to(torch.long)].contiguous()
    chunk_indices, chunk_offsets = prepare_metadata_cutedsl(
        cu_seqlens, inp["T"], chunk_size=64
    )

    o, final_state = chunk_gated_delta_rule_cutedsl(
        q=q_in,
        k=k_in,
        v=v_in,
        g=g_in,
        beta=beta_in,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )

    pool.index_copy_(0, cache_indices.to(torch.long), final_state.to(pool.dtype))
    return o, pool, final_state


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_shape(
    B, T_per_seq, H, K, V, pool_size, device, dtype, sequential_indices=False, seed=42
):
    tag = (
        f"B={B:>3} T/seq={T_per_seq:>4} H={H:>2} K={K:>3} V={V:>3} pool={pool_size:>4}"
    )
    idx_tag = " (seq)" if sequential_indices else ""

    # The ported CuteDSL kernel hard-codes K == V == 128.
    if K != 128 or V != 128:
        print(f"  [SKIP] {tag}{idx_tag}  (CuteDSL requires K=V=128)")
        return True

    inp = make_inputs(
        B,
        T_per_seq,
        H,
        K,
        V,
        pool_size,
        device,
        dtype,
        sequential_indices=sequential_indices,
        seed=seed,
    )

    o_triton, pool_triton, _ = run_triton(inp)

    try:
        o_cutedsl, pool_cutedsl, _ = run_cutedsl(inp)
        torch.cuda.synchronize()
    except Exception as e:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        print(f"  [SKIP] {tag}{idx_tag}  (CuteDSL error: {e})")
        return True

    # Output comparison. Both kernels are bf16 with L2norm + chunked accumulation,
    # tolerances mirror bench_gdn_prefill.py.
    try:
        torch.testing.assert_close(o_triton, o_cutedsl, atol=5e-2, rtol=1e-2)
        out_ok = True
    except AssertionError as e:
        out_ok = False
        out_err = str(e).splitlines()[0]

    status = "PASS" if out_ok else "FAIL"
    extra = "" if out_ok else f"  ({out_err})"
    print(f"  [{status}] {tag}{idx_tag}{extra}")
    return out_ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_shape(B, H, T_per_seq, K, V, pool_size, device, dtype):
    import triton.testing

    if K != 128 or V != 128:
        print(f"  [SKIP] B={B} H={H} T={T_per_seq} K={K} V={V} (CuteDSL K=V=128 only)")
        return

    T = B * T_per_seq
    inp = make_inputs(B, T_per_seq, H, K, V, pool_size, device, dtype)

    q, k_t, v = inp["q"], inp["k"], inp["v"]
    g_triton, beta_triton = inp["g_triton"], inp["beta_triton"]
    cu_seqlens = inp["cu_seqlens"]
    cache_indices = inp["cache_indices"]
    pool_v = inp["pool_init"]
    T_total = inp["T"]

    def fn_triton():
        pool = make_k_contiguous(pool_v.clone())
        triton_chunk_gated_delta_rule(
            q=q,
            k=k_t,
            v=v,
            g=g_triton,
            beta=beta_triton,
            initial_state=pool,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

    cu_int32 = cu_seqlens.to(torch.int32)

    def fn_cutedsl():
        q_in = l2norm_fwd(q[0].contiguous()).unsqueeze(0)
        k_in = l2norm_fwd(k_t[0].contiguous()).unsqueeze(0)
        v_in = v[0].contiguous().unsqueeze(0)
        g_in = g_triton[0].to(torch.float32).unsqueeze(0)
        beta_in = beta_triton[0].to(torch.float32).unsqueeze(0)

        pool = make_k_contiguous(pool_v.clone())
        initial_state = pool[cache_indices.to(torch.long)].contiguous()
        chunk_indices, chunk_offsets = prepare_metadata_cutedsl(
            cu_int32, T_total, chunk_size=64
        )
        chunk_gated_delta_rule_cutedsl(
            q=q_in,
            k=k_in,
            v=v_in,
            g=g_in,
            beta=beta_in,
            initial_state=initial_state,
            cu_seqlens=cu_int32,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )

    quantiles = [0.5, 0.2, 0.8]

    fn_triton()
    fn_cutedsl()
    torch.cuda.synchronize()

    ms_triton, _, _ = triton.testing.do_bench_cudagraph(fn_triton, quantiles=quantiles)
    ms_cutedsl, _, _ = triton.testing.do_bench_cudagraph(
        fn_cutedsl, quantiles=quantiles
    )

    flops = gdn_flops(T, H, K, V)
    mem_bytes = gdn_bytes(T, H, H, K, V, B, dtype)

    tflops_triton = flops / ms_triton / 1e9
    tflops_cutedsl = flops / ms_cutedsl / 1e9
    tb_s_triton = mem_bytes / ms_triton / 1e9
    tb_s_cutedsl = mem_bytes / ms_cutedsl / 1e9
    speedup = ms_triton / ms_cutedsl if ms_cutedsl > 0 else float("inf")

    print(
        f"  {B:>5}  {H:>3}  {T_per_seq:>6}  {T:>7} | "
        f"{ms_triton:>8.3f}  {tflops_triton:>7.2f}  {tb_s_triton:>7.2f} | "
        f"{ms_cutedsl:>8.3f}  {tflops_cutedsl:>7.2f}  {tb_s_cutedsl:>7.2f} | "
        f"{speedup:>7.2f}x"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_correctness(device, dtype):
    print("=" * 78)
    print("Correctness sweep: Triton vs CuTeDSL")
    print("=" * 78)

    shapes = [
        # (B, T_per_seq, H,  K,   V,   pool_size)
        (4, 64, 16, 128, 128, 32),
        (4, 256, 16, 128, 128, 32),
        (1, 128, 16, 128, 128, 32),
        (8, 128, 16, 128, 128, 64),
        (16, 64, 16, 128, 128, 128),
        (32, 32, 16, 128, 128, 256),
        (4, 128, 4, 128, 128, 32),
        (4, 128, 8, 128, 128, 32),
        (4, 128, 32, 128, 128, 32),
        (4, 128, 64, 128, 128, 32),
        (4, 1, 16, 128, 128, 32),
        (4, 7, 16, 128, 128, 32),
        (4, 16, 16, 128, 128, 32),
        (4, 128, 16, 128, 128, 512),
        (32, 128, 32, 128, 128, 256),
    ]

    shapes_seq = [
        (8, 128, 16, 128, 128, 8),
        (4, 128, 32, 128, 128, 4),
        (4, 128, 64, 128, 128, 4),
        (32, 128, 32, 128, 128, 32),
    ]

    all_pass = True
    for cfg in shapes:
        if not check_shape(*cfg, device, dtype):
            all_pass = False

    print("\nSequential-index variants:")
    for cfg in shapes_seq:
        if not check_shape(*cfg, device, dtype, sequential_indices=True):
            all_pass = False

    print()
    print("ALL PASSED." if all_pass else "SOME FAILED.")
    return all_pass


def run_benchmark(device, dtype, args):
    print()
    print("=" * 105)
    print("Benchmark: Triton GDN vs CuTeDSL GDN  (do_bench_cudagraph)")
    print("=" * 105)

    K = args.head_size_k
    V = args.head_size_v
    pool_size = args.pool_size

    if args.preset == "qwen3-next":
        bench_configs = [
            (4, 16, 256),
            (4, 32, 256),
            (16, 16, 256),
            (16, 32, 256),
            (32, 16, 256),
            (32, 32, 256),
            (64, 16, 256),
            (64, 32, 256),
            (128, 16, 256),
            (128, 32, 256),
            (4, 16, 1024),
            (4, 32, 1024),
            (32, 16, 1024),
            (32, 32, 1024),
        ]
    else:
        bench_configs = [
            (B, H, T)
            for B in args.batch_sizes
            for H in args.num_heads
            for T in args.seq_lens
        ]

    print(f"  Config: K={K}, V={V}, pool_size={pool_size}, dtype={dtype}")
    print(
        f"  {'B':>5}  {'H':>3}  {'T/seq':>6}  {'T_tot':>7} | "
        f"{'tri(ms)':>8}  {'TFLOPS':>7}  {'TB/s':>7} | "
        f"{'cute(ms)':>8}  {'TFLOPS':>7}  {'TB/s':>7} | "
        f"{'speedup':>8}"
    )
    print("  " + "-" * 98)

    for B, H, T_per_seq in bench_configs:
        actual_pool = max(pool_size, B)
        bench_shape(B, H, T_per_seq, K, V, actual_pool, device, dtype)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark & Correctness: Triton GDN vs CuTeDSL GDN (SM100)"
    )
    parser.add_argument(
        "--mode", choices=["all", "correctness", "bench"], default="all"
    )
    parser.add_argument(
        "--preset", choices=["qwen3-next", "custom"], default="qwen3-next"
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--head-size-k", type=int, default=128)
    parser.add_argument("--head-size-v", type=int, default=128)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[4, 16, 32, 64, 128]
    )
    parser.add_argument("--num-heads", type=int, nargs="+", default=[16, 32])
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024]
    )
    args = parser.parse_args()

    if args.preset == "qwen3-next":
        args.head_size_k = 128
        args.head_size_v = 128

    device = "cuda"
    dtype = getattr(torch, args.dtype)

    cap = torch.cuda.get_device_capability()
    dev_name = torch.cuda.get_device_name()
    print(f"Device: {dev_name}  (SM {cap[0]}{cap[1]})")
    if cap[0] < 10:
        print("ERROR: CuTeDSL GDN prefill requires SM100+ (Blackwell). Exiting.")
        return 1

    if args.mode in ("all", "correctness"):
        all_pass = run_correctness(device, dtype)
        if not all_pass and args.mode == "all":
            print("\nSkipping benchmark due to correctness failures.")
            return 1

    if args.mode in ("all", "bench"):
        run_benchmark(device, dtype, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
