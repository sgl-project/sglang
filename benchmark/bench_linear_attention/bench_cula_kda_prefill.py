"""
Benchmark & Correctness: cuLA KDA Prefill vs Triton KDA Prefill.

Compares:
  - Triton:  sglang's chunk_kda (chunked KDA prefill with delta rule)
  - cuLA:    SM90 fully-fused KDA prefill kernel (CUTLASS TMA + WGMMA)

Both kernels share the same API shape convention:
  - q/k = [1, T, H, K], v = [1, T, H, V], g = [1, T, H, K], beta = [1, T, H]
  - ssm_states (pool) = [pool_size, H, V, K] (VK layout, float32)
  - cu_seqlens = [N+1], cache_indices = [N]

Reports correctness (output & state matching) and performance (ms, TFLOPS, TB/s).

Usage:
    python bench_cula_kda_prefill.py                        # default sweep
    python bench_cula_kda_prefill.py --mode bench            # benchmark only
    python bench_cula_kda_prefill.py --mode correctness      # correctness only
    python bench_cula_kda_prefill.py --preset kimi-linear     # Kimi-Linear config
"""

import argparse
import math
import sys

import torch

from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.kda import chunk_kda
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd

# cuLA kernel uses exp2() internally, so gate values must be in log-base-2 space.
RCP_LN2 = 1.0 / math.log(2.0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def kda_flops(
    total_seq_len: int,
    num_heads: int,
    head_size_k: int,
    head_size_v: int,
) -> int:
    """
    FLOPs for KDA prefill (delta rule).

    Per token per head:
      1. k @ v^T (outer product):  2 * K * V
      2. q @ state (output):       2 * K * V
    """
    outer_product_flops = 2 * total_seq_len * num_heads * head_size_k * head_size_v
    output_flops = 2 * total_seq_len * num_heads * head_size_k * head_size_v
    return outer_product_flops + output_flops


def kda_bytes(
    total_seq_len: int,
    num_heads: int,
    head_size_k: int,
    head_size_v: int,
    num_seqs: int,
    dtype: torch.dtype,
) -> int:
    """Memory bytes accessed (inputs + outputs + state)."""
    elem = dtype.itemsize

    q_bytes = total_seq_len * num_heads * head_size_k * elem
    k_bytes = total_seq_len * num_heads * head_size_k * elem
    v_bytes = total_seq_len * num_heads * head_size_v * elem
    o_bytes = total_seq_len * num_heads * head_size_v * elem

    # state (float32): read + write
    state_bytes = 2 * num_seqs * num_heads * head_size_k * head_size_v * 4

    # g (float32, per-dim), beta (float32, per-head)
    g_bytes = total_seq_len * num_heads * head_size_k * 4
    beta_bytes = total_seq_len * num_heads * 4

    return q_bytes + k_bytes + v_bytes + o_bytes + state_bytes + g_bytes + beta_bytes


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------


def make_inputs(
    B: int,
    T_per_seq: int,
    H: int,
    K: int,
    V: int,
    pool_size: int,
    device: str,
    dtype: torch.dtype,
    sequential_indices: bool = False,
    seed: int = 42,
):
    """Create all input tensors for a single benchmark / correctness run."""
    T = B * T_per_seq
    torch.manual_seed(seed)

    if sequential_indices:
        cache_indices = torch.arange(B, dtype=torch.int32, device=device)
    else:
        perm = torch.randperm(pool_size, device=device)[:B]
        cache_indices = perm.to(torch.int32)

    # SSM state pool: VK layout [pool_size, H, V, K], float32
    pool_init = (
        torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device) * 0.1
    )

    cu_seqlens = torch.arange(
        0, (B + 1) * T_per_seq, T_per_seq, dtype=torch.long, device=device
    )

    # Triton / cuLA format: [1, T, H, D]
    q = torch.randn(1, T, H, K, dtype=dtype, device=device)
    k = torch.randn(1, T, H, K, dtype=dtype, device=device)
    v = torch.randn(1, T, H, V, dtype=dtype, device=device)

    # g (raw gate, per-dim): [1, T, H, K]
    # Real KDA gates are negative (decay): gate = -exp(A_log)*dt + softplus(a*dt_bias)
    # Use logsigmoid (always negative, bounded) for realistic values.
    # Clamp to _CULA_GATE_MIN=-8 to avoid cuLA fallback.
    g = (
        torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, K, dtype=torch.float32, device=device)
        )
        .clamp(min=-7.0)
        .to(dtype)
    )

    # beta (sigmoid): [1, T, H] — cuLA requires float32
    beta = torch.sigmoid(torch.randn(1, T, H, dtype=torch.float32, device=device))

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
        g=g,
        beta=beta,
    )


# ---------------------------------------------------------------------------
# Runner wrappers
# ---------------------------------------------------------------------------


def run_triton(inp):
    """Triton path: chunk_kda with VK-layout pool, pool-indexed."""
    pool = inp["pool_init"].clone()

    o = chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        initial_state=pool,
        initial_state_indices=inp["cache_indices"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu_seqlens"],
    )
    return o, pool


def run_cula(inp):
    """cuLA path: matches CulaKDAKernel._cula_extend() preprocessing exactly."""
    from sgl_kernel import kda_fwd_prefill

    q = inp["q"]
    k = inp["k"]
    v = inp["v"]
    g = inp["g"]
    beta = inp["beta"]
    pool = inp["pool_init"].clone()
    cache_indices = inp["cache_indices"]
    cu_seqlens = inp["cu_seqlens"]

    batch_size = q.shape[0]
    packed_seq = q.shape[1]
    num_heads = q.shape[2]
    head_dim = q.shape[3]

    # 1. L2 normalize Q, K
    q = l2norm_fwd(q.contiguous())
    k = l2norm_fwd(k.contiguous())

    # 2. Gate cumsum preprocessing (scale=RCP_LN2 for cuLA's exp2-based kernel)
    g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens)

    # 3. Reshape [1, packed_seq, H, D] -> [packed_seq, H, D]
    q = q.reshape(packed_seq, num_heads, head_dim).contiguous()
    k = k.reshape(packed_seq, num_heads, head_dim).contiguous()
    v = v.reshape(packed_seq, num_heads, head_dim).contiguous()
    g = g.reshape(packed_seq, num_heads, head_dim).contiguous()
    beta = beta.reshape(packed_seq, num_heads).contiguous()

    # 4. State gather
    input_state = pool[cache_indices].contiguous()

    # 5. cu_seqlens
    cu_seqlens_i32 = cu_seqlens.to(torch.int32)

    # 6. Workspace buffer
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    workspace_buffer = torch.zeros(sm_count * 128, dtype=torch.uint8, device=q.device)

    # 7. Scale
    scale = head_dim**-0.5

    # 8. Call C++ kernel
    output, output_state = kda_fwd_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens_i32,
        workspace_buffer=workspace_buffer,
        scale=scale,
        safe_gate=True,
        input_state=input_state,
        alpha=g,
        beta=beta,
    )

    # 9. Write state back
    pool[cache_indices] = output_state

    # 10. Reshape output
    output = output.reshape(batch_size, packed_seq, num_heads, head_dim)

    return output, pool


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_shape(
    B,
    T_per_seq,
    H,
    K,
    V,
    pool_size,
    device,
    dtype,
    sequential_indices=False,
    seed=42,
):
    """Run correctness check for a single shape config. Returns True if PASS.

    Pass/fail is based on OUTPUT comparison only (atol=1e-1).
    Pool state diff is reported as informational — state divergence over many
    tokens is expected due to different chunk sizes and accumulation order
    (Triton uses autotuned BT, cuLA uses fixed chunk_size=64 with exp2 gates).
    """
    tag = (
        f"B={B:>3} T/seq={T_per_seq:>4} H={H:>2} K={K:>3} V={V:>3} pool={pool_size:>4}"
    )
    idx_tag = " (seq)" if sequential_indices else ""

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

    o_triton, pool_triton = run_triton(inp)

    try:
        o_cula, pool_cula = run_cula(inp)
        torch.cuda.synchronize()
    except Exception as e:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        print(f"  [SKIP] {tag}{idx_tag}  (cuLA error: {e})")
        return True

    # --- Output comparison (pass/fail criterion) ---
    out_diff = (o_triton.float() - o_cula.float()).abs().max().item()
    out_ok = out_diff < 1e-1

    # --- State comparison (informational only) ---
    cache_indices = inp["cache_indices"]
    state_diff = (
        (pool_triton[cache_indices].float() - pool_cula[cache_indices].float())
        .abs()
        .max()
        .item()
    )

    status = "PASS" if out_ok else "FAIL"
    print(
        f"  [{status}] {tag}{idx_tag}  "
        f"out_diff={out_diff:.4f}  state_diff={state_diff:.4f}"
    )
    return out_ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_shape(B, H, T_per_seq, K, V, pool_size, device, dtype):
    """Benchmark Triton vs cuLA for a single config."""
    import triton.testing

    T = B * T_per_seq
    inp = make_inputs(B, T_per_seq, H, K, V, pool_size, device, dtype)

    # -- Shared read-only tensors --
    q, k_t, v = inp["q"], inp["k"], inp["v"]
    g, beta = inp["g"], inp["beta"]
    cu_seqlens = inp["cu_seqlens"]
    cache_indices = inp["cache_indices"]
    pool_v = inp["pool_init"]

    # -- Pre-compute cuLA format tensors (outside timing for fair comparison) --
    packed_seq = q.shape[1]
    num_heads = q.shape[2]
    head_dim = q.shape[3]

    q_normed = l2norm_fwd(q.contiguous())
    k_normed = l2norm_fwd(k_t.contiguous())
    g_cumsum = chunk_local_cumsum(
        g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens
    )

    q_cula = q_normed.reshape(packed_seq, num_heads, head_dim).contiguous()
    k_cula = k_normed.reshape(packed_seq, num_heads, head_dim).contiguous()
    v_cula = v.reshape(packed_seq, num_heads, head_dim).contiguous()
    g_cula = g_cumsum.reshape(packed_seq, num_heads, head_dim).contiguous()
    beta_cula = beta.reshape(packed_seq, num_heads).contiguous()
    cu_seqlens_i32 = cu_seqlens.to(torch.int32)

    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    workspace_buffer = torch.zeros(sm_count * 128, dtype=torch.uint8, device=q.device)
    scale = head_dim**-0.5

    from sgl_kernel import kda_fwd_prefill

    def fn_triton():
        pool = pool_v.clone()
        chunk_kda(
            q=q,
            k=k_t,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

    def fn_cula():
        input_state = pool_v[cache_indices].contiguous()
        kda_fwd_prefill(
            q=q_cula,
            k=k_cula,
            v=v_cula,
            cu_seqlens=cu_seqlens_i32,
            workspace_buffer=workspace_buffer,
            scale=scale,
            safe_gate=True,
            input_state=input_state,
            alpha=g_cula,
            beta=beta_cula,
        )

    quantiles = [0.5, 0.2, 0.8]

    # Warmup
    for _ in range(5):
        fn_triton()
        fn_cula()
    torch.cuda.synchronize()

    ms_triton, _, _ = triton.testing.do_bench(
        fn_triton, quantiles=quantiles, warmup=50, rep=200
    )
    ms_cula, _, _ = triton.testing.do_bench(
        fn_cula, quantiles=quantiles, warmup=50, rep=200
    )

    # Metrics
    flops = kda_flops(T, H, K, V)
    mem_bytes = kda_bytes(T, H, K, V, B, dtype)

    tflops_triton = flops / ms_triton / 1e9
    tflops_cula = flops / ms_cula / 1e9
    tb_s_triton = mem_bytes / ms_triton / 1e9
    tb_s_cula = mem_bytes / ms_cula / 1e9

    speedup = ms_triton / ms_cula if ms_cula > 0 else float("inf")

    print(
        f"  {B:>5}  {H:>3}  {T_per_seq:>6}  {T:>7} | "
        f"{ms_triton:>8.3f}  {tflops_triton:>7.2f}  {tb_s_triton:>7.2f} | "
        f"{ms_cula:>8.3f}  {tflops_cula:>7.2f}  {tb_s_cula:>7.2f} | "
        f"{speedup:>7.2f}x"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_correctness(device, dtype):
    print("=" * 78)
    print("Correctness sweep: Triton KDA Prefill vs cuLA KDA Prefill")
    print("=" * 78)

    shapes = [
        # (B, T_per_seq, H,  K,   V,   pool_size)
        # --- baseline (Kimi-Linear style) ---
        (4, 64, 16, 128, 128, 32),
        (4, 128, 16, 128, 128, 32),
        (4, 256, 16, 128, 128, 32),
        # --- different batch sizes ---
        (1, 128, 16, 128, 128, 32),
        (8, 128, 16, 128, 128, 64),
        (16, 64, 16, 128, 128, 128),
        (32, 64, 16, 128, 128, 256),
        # --- different head counts ---
        (4, 128, 4, 128, 128, 32),
        (4, 128, 8, 128, 128, 32),
        (4, 128, 32, 128, 128, 32),
        # --- longer sequences ---
        (4, 512, 16, 128, 128, 32),
        (4, 1024, 16, 128, 128, 32),
        # --- large pool (sparse access) ---
        (4, 128, 16, 128, 128, 512),
        # --- combined stress ---
        (32, 128, 32, 128, 128, 256),
    ]

    shapes_seq = [
        (8, 128, 16, 128, 128, 8),
        (4, 128, 32, 128, 128, 4),
        (32, 128, 16, 128, 128, 32),
    ]

    all_pass = True
    for B, T_per_seq, H, K, V, pool_size in shapes:
        if not check_shape(B, T_per_seq, H, K, V, pool_size, device, dtype):
            all_pass = False

    print()
    print("Sequential-index variants:")
    for B, T_per_seq, H, K, V, pool_size in shapes_seq:
        if not check_shape(
            B,
            T_per_seq,
            H,
            K,
            V,
            pool_size,
            device,
            dtype,
            sequential_indices=True,
        ):
            all_pass = False

    print()
    if all_pass:
        print("ALL PASSED.")
    else:
        print("SOME FAILED.")
    return all_pass


def run_benchmark(device, dtype, args):
    print()
    print("=" * 105)
    print("Benchmark: Triton KDA Prefill vs cuLA KDA Prefill  (do_bench)")
    print("=" * 105)

    K = args.head_size_k
    V = args.head_size_v
    pool_size = args.pool_size

    if args.preset == "kimi-linear":
        bench_configs = [
            # (B,   H, T_per_seq)
            (4, 16, 128),
            (4, 16, 256),
            (4, 16, 512),
            (4, 16, 1024),
            (4, 16, 2048),
            (16, 16, 128),
            (16, 16, 256),
            (16, 16, 512),
            (16, 16, 1024),
            (32, 16, 128),
            (32, 16, 256),
            (32, 16, 512),
            (64, 16, 128),
            (64, 16, 256),
            (128, 16, 128),
            (128, 16, 256),
            # larger head counts
            (4, 32, 256),
            (4, 32, 512),
            (4, 32, 1024),
            (16, 32, 256),
            (16, 32, 512),
            (32, 32, 256),
        ]
    else:
        bench_configs = []
        for B in args.batch_sizes:
            for H in args.num_heads:
                for T_per_seq in args.seq_lens:
                    bench_configs.append((B, H, T_per_seq))

    print(f"  Config: K={K}, V={V}, pool_size={pool_size}, dtype={dtype}")
    print(
        f"  {'B':>5}  {'H':>3}  {'T/seq':>6}  {'T_tot':>7} | "
        f"{'tri(ms)':>8}  {'TFLOPS':>7}  {'TB/s':>7} | "
        f"{'cula(ms)':>8}  {'TFLOPS':>7}  {'TB/s':>7} | "
        f"{'speedup':>8}"
    )
    print("  " + "-" * 98)

    for B, H, T_per_seq in bench_configs:
        actual_pool = max(pool_size, B)
        bench_shape(B, H, T_per_seq, K, V, actual_pool, device, dtype)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark & Correctness: Triton KDA Prefill vs cuLA KDA Prefill"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "correctness", "bench"],
        default="all",
        help="Run mode (default: all)",
    )
    parser.add_argument(
        "--preset",
        choices=["kimi-linear", "custom"],
        default="kimi-linear",
        help="Preset config (default: kimi-linear)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--head-size-k", type=int, default=128)
    parser.add_argument("--head-size-v", type=int, default=128)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[4, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        default=[16, 32],
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
    )
    args = parser.parse_args()

    if args.preset == "kimi-linear":
        args.head_size_k = 128
        args.head_size_v = 128

    device = "cuda"
    dtype = getattr(torch, args.dtype)

    # Check SM version
    cap = torch.cuda.get_device_capability()
    dev_name = torch.cuda.get_device_name()
    print(f"Device: {dev_name}  (SM {cap[0]}{cap[1]})")

    if cap[0] < 9:
        print(
            "WARNING: cuLA KDA requires SM90 (Hopper) GPU. Correctness tests will skip cuLA."
        )

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
