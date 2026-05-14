"""
Benchmark: chunk_gated_delta_rule_fwd_h
=====================================================================
Measures ONLY the fwd_h stage (stage 5 of chunk_gated_delta_rule).
Bypasses all other pipeline stages to isolate this dominant cost.

Shapes follow Qwen3.5 preset: H=8, Hg=8, K=128, V=128.
Uses fixed-length (non-varlen) format for simplicity.

Usage:
    # Full sweep
    python bench_fwd_h.py

    # Single config
    python bench_fwd_h.py --batch 4 --seq-len 256

    # Short run for single (2 warmup + 2 rep)
    python bench_fwd_h.py --mode single --batch 4 --seq-len 256
"""

import argparse

import torch
import torch.nn.functional as F
import triton.testing

from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

CHUNK_SIZE = 64  # must match CHUNK_SIZE constant in chunk_delta_h.py


def measure_memory_bandwidth_gbs(
    device: str,
    dtype: torch.dtype = torch.float32,
    tensor_mb: int = 256,
    warmup: int = 20,
    rep: int = 50,
):
    """
    Measure sustained device memory bandwidth via a large device-to-device copy.

    Works for both CUDA and XPU backends by using ``dst.copy_(src)`` and Triton
    benchmarking utilities. Returns GB/s.
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    device_mod = getattr(torch, device)
    curr_tensor_mb = tensor_mb

    while curr_tensor_mb >= 32:
        try:
            numel = (curr_tensor_mb * 1024 * 1024) // elem_size
            src = torch.empty(numel, device=device, dtype=dtype)
            dst = torch.empty_like(src)
            src.uniform_(-1.0, 1.0)

            def copy_fn():
                dst.copy_(src)

            ms = triton.testing.do_bench(
                copy_fn,
                warmup=warmup,
                rep=rep,
                return_mode="median",
            )
            bytes_moved = 2 * src.numel() * elem_size  # read src + write dst
            return bytes_moved / ms / 1e6
        except RuntimeError:
            # If allocation fails on a smaller device, retry with smaller buffers.
            if hasattr(device_mod, "empty_cache"):
                device_mod.empty_cache()
            curr_tensor_mb //= 2

    raise RuntimeError(
        f"Failed to measure memory bandwidth on device={device}. "
        "Could not allocate/copy buffers down to 32MB."
    )


def fwd_h_bytes(B, T, H, Hg, K, V, NT, dtype):
    """
    Theoretical bytes accessed by chunk_gated_delta_rule_fwd_h.

    Accounts for:
      Reads:  k, w, u (bf16) + g (bf16) + initial_state (f32, B states)
      Writes: h per-chunk states (bf16) + v_new (bf16) + initial_state write-back (f32)
    """
    e = torch.finfo(dtype).bits // 8 if hasattr(torch.finfo(dtype), "bits") else 2
    # bf16 = 2 bytes, f32 = 4 bytes
    try:
        e = torch.tensor([], dtype=dtype).element_size()
    except Exception:
        e = 2  # fallback: bf16

    reads = (
        B * T * Hg * K * e  # k
        + B * T * H * K * e  # w
        + B * T * H * V * e  # u
        + B * T * H * 4  # g (float32)
        + B * H * V * K * 4  # initial_state read (float32)
    )
    writes = (
        B * NT * H * V * K * e  # h output (per-chunk states)
        + B * T * H * V * e  # v_new output
        + B * H * V * K * 4  # initial_state write-back (float32)
    )
    return reads + writes


def make_inputs(B, T, H, Hg, K, V, pool_size, device, dtype):
    torch.manual_seed(42)
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    w = torch.randn(B, T, H, K, device=device, dtype=dtype)
    u = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    st = torch.randn(pool_size, H, V, K, device=device, dtype=torch.float32) * 0.1
    ix = torch.arange(B, device=device, dtype=torch.int32)
    return k, w, u, g, st, ix


def bench_config(B, T, H, Hg, K, V, pool_size, device, dtype, peak_bw_gbs, warmup, rep):
    NT = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    k, w, u, g, st, ix = make_inputs(B, T, H, Hg, K, V, pool_size, device, dtype)
    cu_seqlens = torch.arange(
        0, (T + 1) * B, T, device=device, dtype=torch.int32
    )  # [0, T, 2T, ..., BT]

    def fn():
        chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=st,
            initial_state_indices=ix,
            save_new_value=True,
            cu_seqlens=cu_seqlens,
        )

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    nb = fwd_h_bytes(B, T, H, Hg, K, V, NT, dtype)
    gbs = (
        nb / ms / 1e6
    )  # bytes / ms = bytes / (s * 1e-3) = bytes * 1e3 / s → /1e9 for GB → /1e6 net
    eff = gbs / peak_bw_gbs * 100
    return ms, gbs, eff, NT


def run_sweep(device, dtype, peak_bw_gbs, warmup, rep):
    H = Hg = 8
    K = V = 128

    print(f"\nchunk_gated_delta_rule_fwd_h — isolated benchmark")
    print(f"Device: {getattr(torch, device).get_device_name(0)}")
    print(f"dtype={dtype}  H={H}, Hg={Hg}, K={K}, V={V}")
    print(f"Peak BW measured: {peak_bw_gbs:.1f} GB/s")
    print()
    print(f"  {'B':>4}  {'T':>5}  {'NT':>3}  | {'ms':>8}  {'GB/s':>7}  {'%peak':>7}")
    print("  " + "-" * 52)

    configs = [
        (1, 64, 32),
        (1, 256, 32),
        (1, 512, 32),
        (1, 1024, 32),
        (4, 64, 32),
        (4, 128, 32),
        (4, 256, 32),
        (4, 512, 32),
        (4, 1024, 32),
        (8, 256, 32),
        (16, 256, 32),
        (32, 256, 64),
    ]

    for B, T, pool in configs:
        pool_size = max(pool, B)
        ms, gbs, eff, NT = bench_config(
            B, T, H, Hg, K, V, pool_size, device, dtype, peak_bw_gbs, warmup, rep
        )
        print(f"  {B:>4}  {T:>5}  {NT:>3}  | {ms:>8.3f}  {gbs:>7.1f}  {eff:>6.1f}%")


def main():
    # get default device based on availability
    default_device = "cuda"
    if not torch.cuda.is_available():
        assert (
            torch.xpu.is_available()
        ), "Neither CUDA nor XPU is available on this system."
        default_device = "xpu"
    p = argparse.ArgumentParser(
        description="Benchmark chunk_gated_delta_rule_fwd_h on CUDA/XPU"
    )
    p.add_argument("--device", choices=["xpu", "cuda"], default=default_device)
    p.add_argument(
        "--mode",
        choices=["sweep", "single"],
        default="sweep",
        help="sweep: full sweep (25 warmup, 100 rep); "
        "single: single config with 2 warmup + 2 rep for short traces",
    )
    p.add_argument("--batch", type=int, default=4, help="batch size (single mode)")
    p.add_argument(
        "--seq-len", type=int, default=256, help="sequence length (single mode)"
    )
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = p.parse_args()

    device = args.device
    device_mod = getattr(torch, device)
    dtype = getattr(torch, args.dtype)
    peak_bw_gbs = measure_memory_bandwidth_gbs(device=device)

    print(f"Device: {device_mod.get_device_name(0)}")
    print(f"Measured memory bandwidth: {peak_bw_gbs:.1f} GB/s")

    if args.mode == "single":
        H, Hg = 32, 16
        K = V = 128
        B, T = args.batch, args.seq_len
        pool_size = max(B, 128)
        ms, gbs, eff, NT = bench_config(
            B, T, H, Hg, K, V, pool_size, device, dtype, peak_bw_gbs, warmup=2, rep=2
        )
        print(
            f"B={B} T={T} NT={NT}:  {ms:.3f} ms  {gbs:.1f} GB/s  {eff:.1f}% of {peak_bw_gbs:.1f} GB/s peak"
        )
    else:
        run_sweep(device, dtype, peak_bw_gbs, warmup=25, rep=100)


if __name__ == "__main__":
    main()
