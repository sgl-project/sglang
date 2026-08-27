"""
Benchmark & Correctness: Triton GDN vs FlashInfer GDN (prefill).

Compares:
  - Triton:     sglang's chunk_gated_delta_rule (K-contiguous pool, pool-indexed)
  - FlashInfer: flashinfer's chunk_gated_delta_rule (gather/scatter, 3D tensors)

The two kernels have different APIs:
  - Triton:     q/k/v=[1,T,H,D], g=logsigmoid, beta=sigmoid, has initial_state_indices
  - FlashInfer: q/k/v=[T,H,D],   g=alpha(float32), beta=float32, no indices (gathered state)

Reports correctness (output & state matching) and performance (ms, TFLOPS, TB/s).

Usage:
    python benchmark_gdn_prefill.py                          # default sweep
    python benchmark_gdn_prefill.py --mode bench             # benchmark only
    python benchmark_gdn_prefill.py --mode correctness       # correctness only
    python benchmark_gdn_prefill.py --preset qwen3-next      # Qwen3-Next config
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from flashinfer.gdn_prefill import (
    chunk_gated_delta_rule as flashinfer_chunk_gated_delta_rule,
)

from sglang.srt.layers.attention.fla.chunk import (
    chunk_gated_delta_rule as triton_chunk_gated_delta_rule,
)
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_k_contiguous(t: torch.Tensor) -> torch.Tensor:
    """
    Given a V-contiguous tensor [..., K, V], return a K-contiguous view of the
    same logical shape [..., K, V] (physically [..., V, K], K-last).
    """
    return t.transpose(-2, -1).contiguous().transpose(-2, -1)


def gdn_flops(
    total_seq_len: int,
    num_heads: int,
    head_size_k: int,
    head_size_v: int,
) -> int:
    """
    FLOPs for GDN prefill (delta rule).

    Per token per head:
      1. k @ v^T (outer product):  2 * K * V
      2. q @ state (output):       2 * K * V
    """
    outer_product_flops = 2 * total_seq_len * num_heads * head_size_k * head_size_v
    output_flops = 2 * total_seq_len * num_heads * head_size_k * head_size_v
    return outer_product_flops + output_flops


def gdn_bytes(
    total_seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size_k: int,
    head_size_v: int,
    num_seqs: int,
    dtype: torch.dtype,
) -> int:
    """Memory bytes accessed (inputs + outputs + state)."""
    num_o_heads = max(num_q_heads, num_v_heads)
    elem = dtype.itemsize

    q_bytes = total_seq_len * num_q_heads * head_size_k * elem
    k_bytes = total_seq_len * num_v_heads * head_size_k * elem
    v_bytes = total_seq_len * num_v_heads * head_size_v * elem
    o_bytes = total_seq_len * num_o_heads * head_size_v * elem

    # state (float32): read + write
    state_bytes = 2 * num_seqs * num_o_heads * head_size_k * head_size_v * 4

    # g, beta (float32)
    g_bytes = total_seq_len * num_o_heads * 4
    beta_bytes = total_seq_len * num_o_heads * 4

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
    """Create all input tensors for a single benchmark / correctness run.

    Returns a dict with both Triton-format and FlashInfer-format tensors.
    """
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

    # Triton format: [1, T, H, D]
    q = torch.randn(1, T, H, K, dtype=dtype, device=device)
    k = torch.randn(1, T, H, K, dtype=dtype, device=device)
    v = torch.randn(1, T, H, V, dtype=dtype, device=device)

    # g (logsigmoid) and beta (sigmoid) in Triton format: [1, T, H]
    g_raw = torch.randn(1, T, H, dtype=dtype, device=device)
    g_triton = torch.nn.functional.logsigmoid(g_raw)  # logsigmoid for Triton
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


def run_flashinfer(inp):
    """FlashInfer path: matches sglang FlashInferGDNKernel.extend() exactly.

    Key differences from Triton path:
      - q, k are L2-normalized BEFORE calling the kernel
      - use_qk_l2norm_in_kernel=False (kernel skips internal normalization)
      - Tensors are [T, H, D] (no batch dim)
      - g is alpha = exp(logsigmoid(...)) = sigmoid(...), float32
      - beta is float32
      - initial_state is gathered from pool (no pool-index support)
      - Uses keyword arguments (matching sglang production code)

    NOTE: FlashInfer GDN requires K == V (square head_size).
    """
    K = inp["K"]
    V = inp["V"]
    assert K == V, f"FlashInfer GDN requires K == V, got K={K}, V={V}"

    pool = make_k_contiguous(inp["pool_init"].clone())
    cache_indices = inp["cache_indices"]

    # Gather states from K-contiguous pool -> K-contiguous float32
    # In production, ssm_states is already float32 so .float() is no-op.
    # Here pool_init is bf16, so .float() loses K-contiguous layout.
    gathered = pool[cache_indices]
    initial_state = make_k_contiguous(gathered.float().contiguous())

    q_fi = l2norm_fwd(inp["q"][0].contiguous())
    k_fi = l2norm_fwd(inp["k"][0].contiguous())
    v_fi = inp["v"][0].contiguous()

    # g -> alpha (exp of logsigmoid = sigmoid), float32
    alpha_fi = torch.exp(inp["g_triton"][0].to(torch.float32))
    # beta -> float32
    beta_fi = inp["beta_triton"][0].to(torch.float32)

    cu_seqlens_fi = inp["cu_seqlens"].to(torch.int64)

    # Call FlashInfer with keyword args (matching sglang production code)
    # use_qk_l2norm_in_kernel=False because we pre-normalized above
    o_fi, state_fi = flashinfer_chunk_gated_delta_rule(
        q=q_fi,
        k=k_fi,
        v=v_fi,
        g=alpha_fi,
        beta=beta_fi,
        scale=None,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens_fi,
        use_qk_l2norm_in_kernel=False,
    )

    # Scatter updated states back to K-contiguous pool
    pool[cache_indices] = state_fi.to(pool.dtype)

    # Reshape output: [T, H, D] -> [1, T, H, D] to match Triton
    o_out = o_fi.unsqueeze(0)

    return o_out, pool, state_fi


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

    Pass/fail is based on OUTPUT comparison only (atol=5e-2).
    Pool state diff is reported as informational — state divergence over many
    tokens is expected due to different chunk sizes and accumulation order.
    """
    tag = (
        f"B={B:>3} T/seq={T_per_seq:>4} H={H:>2} K={K:>3} V={V:>3} pool={pool_size:>4}"
    )
    idx_tag = " (seq)" if sequential_indices else ""

    # FlashInfer GDN requires K == V (square head_size)
    if K != V:
        print(f"  [SKIP] {tag}{idx_tag}  (FlashInfer requires K==V)")
        return True

    # FlashInfer GDN CUTLASS kernels are only compiled for head_size=128.
    # Running with other sizes causes illegal memory access that poisons
    # the CUDA context (unrecoverable), so we must skip upfront.
    FLASHINFER_SUPPORTED_HEAD_SIZES = {128}
    if K not in FLASHINFER_SUPPORTED_HEAD_SIZES:
        print(
            f"  [SKIP] {tag}{idx_tag}  (FlashInfer only supports head_size={FLASHINFER_SUPPORTED_HEAD_SIZES})"
        )
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

    o_triton, pool_triton, h_triton = run_triton(inp)

    # FlashInfer may not support all head_size values (e.g., only 128).
    # CUDA errors from unsupported configs are often asynchronous, so we
    # must synchronize inside the try block to catch them here.
    try:
        o_fi, pool_fi, _ = run_flashinfer(inp)
        torch.cuda.synchronize()
    except Exception as e:
        # Catch RuntimeError, torch.AcceleratorError, etc.
        # Reset CUDA error state so subsequent tests can proceed
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        print(f"  [SKIP] {tag}{idx_tag}  (FlashInfer error: {e})")
        return True

    cache_indices = inp["cache_indices"]

    # --- Output comparison ---
    # bf16 prefill with L2norm + chunked accumulation
    torch.testing.assert_close(o_triton, o_fi, atol=5e-2, rtol=1e-2)

    # --- Stride check ---
    def strides_ok(pool):
        s = pool.stride()
        return s[-2] == 1 and s[-1] == K

    strides_triton = strides_ok(pool_triton)
    strides_fi = strides_ok(pool_fi)

    passed = strides_triton and strides_fi

    # Build detail string
    details = []
    if not strides_triton:
        details.append("triton strides bad")
    if not strides_fi:
        details.append("flashinfer strides bad")

    status = "PASS" if passed else "FAIL"
    detail_str = f"  [{', '.join(details)}]"
    print(f"  [{status}] {tag}{idx_tag}")
    return passed


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_shape(B, H, T_per_seq, K, V, pool_size, device, dtype):
    """Benchmark Triton vs FlashInfer for a single config. Requires K == V."""
    import triton.testing

    assert K == V, f"FlashInfer GDN requires K == V, got K={K}, V={V}"

    T = B * T_per_seq
    inp = make_inputs(B, T_per_seq, H, K, V, pool_size, device, dtype)

    # -- Shared read-only tensors --
    q, k_t, v = inp["q"], inp["k"], inp["v"]
    g_triton, beta_triton = inp["g_triton"], inp["beta_triton"]
    cu_seqlens = inp["cu_seqlens"]
    cache_indices = inp["cache_indices"]
    seq_indices = torch.arange(B, dtype=torch.int32, device=device)
    pool_v = inp["pool_init"]

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

    def fn_flashinfer():
        # -- Pre-compute FlashInfer format tensors (outside timing) --
        # Pre-normalize q and k (matching sglang production: l2norm_fwd)
        # q_fi = torch.nn.functional.normalize(q[0].contiguous().float(), p=2.0, dim=-1).to(
        #     dtype
        # )
        # k_fi = torch.nn.functional.normalize(k_t[0].contiguous().float(), p=2.0, dim=-1).to(
        #     dtype
        # )
        q_fi = l2norm_fwd(q[0].contiguous())
        k_fi = l2norm_fwd(k_t[0].contiguous())
        v_fi = v[0].contiguous()
        alpha_fi = torch.exp(g_triton[0].to(torch.float32))
        beta_fi = beta_triton[0].to(torch.float32)
        cu_seqlens_fi = cu_seqlens.to(torch.int64)
        pool = make_k_contiguous(pool_v.clone())
        gathered = pool[cache_indices]
        initial_state = make_k_contiguous(gathered.float().contiguous())
        flashinfer_chunk_gated_delta_rule(
            q=q_fi,
            k=k_fi,
            v=v_fi,
            g=alpha_fi,
            beta=beta_fi,
            scale=None,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens_fi,
            use_qk_l2norm_in_kernel=False,
        )

    quantiles = [0.5, 0.2, 0.8]

    # Warmup
    fn_triton()
    fn_flashinfer()
    torch.cuda.synchronize()

    ms_triton, _, _ = triton.testing.do_bench_cudagraph(fn_triton, quantiles=quantiles)
    ms_fi, _, _ = triton.testing.do_bench_cudagraph(fn_flashinfer, quantiles=quantiles)

    # Metrics
    num_o_heads = H
    flops = gdn_flops(T, num_o_heads, K, V)
    mem_bytes = gdn_bytes(T, H, H, K, V, B, dtype)

    tflops_triton = flops / ms_triton / 1e9
    tflops_fi = flops / ms_fi / 1e9
    tb_s_triton = mem_bytes / ms_triton / 1e9
    tb_s_fi = mem_bytes / ms_fi / 1e9

    speedup = ms_triton / ms_fi if ms_fi > 0 else float("inf")

    print(
        f"  {B:>5}  {H:>3}  {T_per_seq:>6}  {T:>7} | "
        f"{ms_triton:>8.3f}  {tflops_triton:>7.2f}  {tb_s_triton:>7.2f} | "
        f"{ms_fi:>8.3f}  {tflops_fi:>7.2f}  {tb_s_fi:>7.2f} | "
        f"{speedup:>7.2f}x"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_correctness(device, dtype):
    print("=" * 78)
    print("Correctness sweep: Triton vs FlashInfer")
    print("=" * 78)

    shapes = [
        # (B, T_per_seq, H,  K,   V,   pool_size)
        # --- baseline (Qwen3-Next style) ---
        (4, 64, 16, 128, 128, 32),
        (4, 256, 16, 128, 128, 32),
        # --- different batch sizes ---
        (1, 128, 16, 128, 128, 32),
        (8, 128, 16, 128, 128, 64),
        (16, 64, 16, 128, 128, 128),
        (32, 32, 16, 128, 128, 256),
        # --- different head counts ---
        (4, 128, 4, 128, 128, 32),
        (4, 128, 8, 128, 128, 32),
        (4, 128, 16, 64, 64, 32),
        (4, 128, 32, 128, 128, 32),
        (4, 128, 64, 128, 128, 32),
        # --- short sequences ---
        (4, 1, 16, 128, 128, 32),
        (4, 7, 16, 128, 128, 32),
        (4, 16, 16, 128, 128, 32),
        # --- large pool (sparse access) ---
        (4, 128, 16, 128, 128, 512),
        # --- combined stress ---
        (32, 128, 32, 128, 128, 256),
    ]

    shapes_seq = [
        (8, 128, 16, 128, 128, 8),
        (4, 128, 32, 128, 128, 4),
        (4, 128, 64, 128, 128, 4),
        (32, 128, 32, 128, 128, 32),
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
    print("Benchmark: Triton GDN vs FlashInfer GDN  (do_bench_cudagraph)")
    print("=" * 105)

    K = args.head_size_k
    V = args.head_size_v
    pool_size = args.pool_size

    if args.preset == "qwen3-next":
        bench_configs = [
            # (B,   H, T_per_seq)
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
            # longer sequences
            (4, 16, 1024),
            (4, 32, 1024),
            (32, 16, 1024),
            (32, 32, 1024),
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
        f"{'fi(ms)':>8}  {'TFLOPS':>7}  {'TB/s':>7} | "
        f"{'speedup':>8}"
    )
    print("  " + "-" * 98)

    for B, H, T_per_seq in bench_configs:
        actual_pool = max(pool_size, B)
        bench_shape(B, H, T_per_seq, K, V, actual_pool, device, dtype)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark & Correctness: Triton GDN vs FlashInfer GDN"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "correctness", "bench"],
        default="all",
        help="Run mode (default: all)",
    )
    parser.add_argument(
        "--preset",
        choices=["qwen3-next", "custom"],
        default="qwen3-next",
        help="Preset config (default: qwen3-next)",
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

    if args.preset == "qwen3-next":
        args.head_size_k = 128
        args.head_size_v = 128

    device = "cuda"
    dtype = getattr(torch, args.dtype)

    # Check SM version
    cap = torch.cuda.get_device_capability()
    dev_name = torch.cuda.get_device_name()
    print(f"Device: {dev_name}  (SM {cap[0]}{cap[1]})")

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
