"""
Benchmark: chunk_gated_delta_rule_fwd_intra
============================================================
Measures the intra-chunk forward pass (fused kkt-solve + recompute_w_u).
This is the dominant compute stage during GDN prefill.

Model: Qwen3.5-4B (H=32, Hg=16, K=128, V=128, BT=64).
The "real-model" shape collected from a B=1/T=1024 run is included.

Works on both CUDA and XPU.

Usage:
    # Full sweep (default)
    python bench_intra.py

    # Single config (for single / quick check)
    python bench_intra.py --mode single --batch 1 --seq-len 1024

    # Short run for short profiling windows
    python bench_intra.py --mode single --batch 4 --seq-len 256
"""

import argparse

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra

CHUNK_SIZE = 64  # must match BT constant in chunk_fwd.py


# ---------------------------------------------------------------------------
# Memory bandwidth measurement
# ---------------------------------------------------------------------------


def measure_memory_bandwidth_gbs(
    device: str,
    dtype: torch.dtype = torch.float32,
    tensor_mb: int = 256,
    warmup: int = 20,
    rep: int = 50,
) -> float:
    """Measure sustained device memory bandwidth (GB/s) via a large copy."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    device_mod = getattr(torch, device)
    curr_mb = tensor_mb
    while curr_mb >= 32:
        try:
            numel = (curr_mb * 1024 * 1024) // elem_size
            src = torch.empty(numel, device=device, dtype=dtype)
            dst = torch.empty_like(src)
            src.uniform_(-1.0, 1.0)
            ms = triton.testing.do_bench(
                lambda: dst.copy_(src), warmup=warmup, rep=rep, return_mode="median"
            )
            return 2 * src.numel() * elem_size / ms / 1e6
        except RuntimeError:
            if hasattr(device_mod, "empty_cache"):
                device_mod.empty_cache()
            curr_mb //= 2
    raise RuntimeError(f"Failed to measure memory bandwidth on device={device}.")


# ---------------------------------------------------------------------------
# Theoretical bytes accessed by chunk_gated_delta_rule_fwd_intra
#
# Two kernel launches (fused kkt+solve, then recompute_w_u):
#   kernel 1 reads:  k, g, beta          writes: A
#   kernel 2 reads:  k, v, A             writes: w, u
#
# k is read by both kernels → counted twice.
# g is float32, everything else is `dtype`.
# ---------------------------------------------------------------------------


def intra_bytes(
    B: int, T: int, H: int, Hg: int, K: int, V: int, dtype: torch.dtype
) -> int:
    BT = CHUNK_SIZE
    e = dtype.itemsize  # bytes per element (2 for bf16/fp16, 4 for fp32)

    reads = (
        2 * B * T * Hg * K * e  # k: read by both kernels
        + B * T * H * 4  # g: float32
        + B * T * H * e  # beta
        + B * T * H * V * e  # v: read by kernel 2
        + B * T * H * BT * e  # A: read by kernel 2
    )
    writes = (
        B * T * H * BT * e  # A: written by kernel 1
        + B * T * H * K * e  # w: written by kernel 2
        + B * T * H * V * e  # u: written by kernel 2
    )
    return reads + writes


# ---------------------------------------------------------------------------
# Theoretical FLOPs for chunk_gated_delta_rule_fwd_intra
#
# kkt_solve (per chunk, per head):
#   Build lower-triangular A = k_i @ k_j^T  (BC=16 sub-blocks, BT=64)
#   10 blocks (4 diag + 6 off-diag) × 2 × BC × BC × K FMACs
#   + triangular solve: O(BC^3) per block (dominated by K-loop)
#
# recompute_w_u (per chunk, per head):
#   w = A[BT,BT] @ k[BT,K]:  2 × BT × BT × K  FMACs
#   u = A[BT,BT] @ v[BT,V]:  2 × BT × BT × V  FMACs
# ---------------------------------------------------------------------------


def intra_flops(B: int, T: int, H: int, Hg: int, K: int, V: int) -> int:
    BT = CHUNK_SIZE
    BC = 16
    NT = (T + BT - 1) // BT
    n_lower = (BT // BC) * ((BT // BC) + 1) // 2  # = 10 for BT=64, BC=16

    # kkt: each of the n_lower blocks is a BC×BC matrix from K-dim dot products
    kkt_flops = n_lower * 2 * BC * BC * K
    # recompute_w_u: two gemm per chunk, w and u
    wu_flops = 2 * BT * BT * K + 2 * BT * BT * V

    return B * NT * H * (kkt_flops + wu_flops)


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------


def make_inputs(
    B: int,
    T_per_seq: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    device: str,
    dtype: torch.dtype,
):
    """Create inputs following the actual SGLang convention:
    - With cu_seqlens: batch_dim=1, all B sequences concatenated in T dim.
    - cu_seqlens = [0, T_per_seq, 2*T_per_seq, ..., B*T_per_seq].
    """
    T = B * T_per_seq
    torch.manual_seed(42)
    k = torch.randn(1, T, Hg, K, device=device, dtype=dtype)
    v = torch.randn(1, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(
        torch.randn(1, T, H, device=device, dtype=torch.float32)
    )  # must be fp32
    beta = torch.sigmoid(torch.randn(1, T, H, device=device, dtype=dtype))
    cu_seqlens = torch.arange(
        0, (B + 1) * T_per_seq, T_per_seq, device=device, dtype=torch.int32
    )
    return k, v, g, beta, cu_seqlens


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------


def bench_config(
    B: int,
    T_per_seq: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    device: str,
    dtype: torch.dtype,
    peak_bw_gbs: float,
    warmup: int,
    rep: int,
):
    k, v, g, beta, cu_seqlens = make_inputs(B, T_per_seq, H, Hg, K, V, device, dtype)

    def fn():
        chunk_gated_delta_rule_fwd_intra(
            k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE
        )

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    nb = intra_bytes(B, T_per_seq, H, Hg, K, V, dtype)
    nf = intra_flops(B, T_per_seq, H, Hg, K, V)
    gbs = nb / ms / 1e6  # GB/s  (bytes / ms = bytes * 1e3/s → /1e9 → /1e6 net)
    tflops = nf / ms / 1e9  # TFLOPS (flops / ms = flops * 1e3/s → /1e12 → /1e9 net)
    eff = gbs / peak_bw_gbs * 100
    return ms, gbs, tflops, eff


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    device: str, dtype: torch.dtype, peak_bw_gbs: float, warmup: int, rep: int
):
    # Qwen3.5-4B model config (confirmed from real-model run):
    #   H=32, Hg=16, K=128, V=128
    H, Hg, K, V = 32, 16, 128, 128

    print(f"\nchunk_gated_delta_rule_fwd_intra — isolated benchmark")
    print(f"Device: {getattr(torch, device).get_device_name(0)}")
    print(f"dtype={dtype}  H={H}, Hg={Hg}, K={K}, V={V}  (Qwen3.5-4B config)")
    print(f"Peak BW measured: {peak_bw_gbs:.1f} GB/s")
    print()
    print(
        f"  {'B':>4}  {'T/seq':>5}  {'NT':>3}  | {'ms':>8}  {'GB/s':>7}  {'TFLOPS':>7}  {'%peak':>7}"
    )
    print("  " + "-" * 62)

    # Configurations
    configs = [
        (1, 64),
        (1, 128),
        (1, 256),
        (1, 512),
        (1, 1024),
        (1, 2048),
        (4, 256),
        (4, 512),
        (4, 1024),
        (8, 256),
        (8, 512),
        (16, 256),
        (16, 512),
        (32, 256),
    ]

    for B, T_per_seq in configs:
        NT = (T_per_seq + CHUNK_SIZE - 1) // CHUNK_SIZE
        try:
            ms, gbs, tflops, eff = bench_config(
                B, T_per_seq, H, Hg, K, V, device, dtype, peak_bw_gbs, warmup, rep
            )
            print(
                f"  {B:>4}  {T_per_seq:>5}  {NT:>3}  | {ms:>8.3f}  {gbs:>7.1f}  {tflops:>7.3f}  {eff:>6.1f}%"
            )
        except Exception as exc:
            print(f"  {B:>4}  {T_per_seq:>5}  {NT:>3}  | ERROR: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    default_device = "xpu" if torch.xpu.is_available() else "cuda"
    p = argparse.ArgumentParser(
        description="Benchmark chunk_gated_delta_rule_fwd_intra on CUDA/XPU"
    )
    p.add_argument("--device", choices=["xpu", "cuda"], default=default_device)
    p.add_argument(
        "--mode",
        choices=["sweep", "single"],
        default="sweep",
        help="sweep: full sweep; single: single config (2 warmup + 2 rep)",
    )
    p.add_argument("--batch", type=int, default=1, help="batch size (single mode)")
    p.add_argument(
        "--seq-len", type=int, default=1024, help="sequence length (single mode)"
    )
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument(
        "--H", type=int, default=32, help="number of GDN heads (default: Qwen3.5-4B)"
    )
    p.add_argument(
        "--Hg",
        type=int,
        default=16,
        help="number of GDN key heads (default: Qwen3.5-4B)",
    )
    p.add_argument("--K", type=int, default=128, help="head dim K")
    p.add_argument("--V", type=int, default=128, help="head dim V")
    args = p.parse_args()

    device = args.device
    device_mod = getattr(torch, device)
    dtype = getattr(torch, args.dtype)

    print(f"Device: {device_mod.get_device_name(0)}")
    peak_bw_gbs = measure_memory_bandwidth_gbs(device=device)
    print(f"Measured memory bandwidth: {peak_bw_gbs:.1f} GB/s")

    if args.mode == "single":
        B, T_per_seq = args.batch, args.seq_len
        H, Hg, K, V = args.H, args.Hg, args.K, args.V
        NT = (T_per_seq + CHUNK_SIZE - 1) // CHUNK_SIZE
        ms, gbs, tflops, eff = bench_config(
            B, T_per_seq, H, Hg, K, V, device, dtype, peak_bw_gbs, warmup=2, rep=2
        )
        print(
            f"B={B} T/seq={T_per_seq} NT={NT} H={H} Hg={Hg} K={K} V={V}: "
            f"{ms:.3f} ms  {gbs:.1f} GB/s  {tflops:.3f} TFLOPS  {eff:.1f}% of peak"
        )
    else:
        run_sweep(device, dtype, peak_bw_gbs, warmup=25, rep=100)


if __name__ == "__main__":
    main()
