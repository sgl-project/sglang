import argparse
import math
from typing import Callable, Dict, Tuple

import torch

from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
)


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp32", "float32"}:
        return torch.float32
    if s in {"fp64", "float64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {s}")


@torch.no_grad()
def _make_cos_sin(
    seqlen: int,
    half_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Standard RoPE frequencies (only for generating consistent inputs; perf is what we care about).
    pos = torch.arange(seqlen, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim)
    )
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    return cos, sin


@torch.no_grad()
def _build_cos_sin_cache_like_wanvideo(
    cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # Mirror the current WanVideo code path: float32 + contiguous before cat.
    return torch.cat(
        [cos.to(dtype=torch.float32).contiguous(), sin.to(dtype=torch.float32).contiguous()],
        dim=-1,
    )


def _time_cuda_events(
    fn: Callable[[], None], *, warmup: int, iters: int
) -> Tuple[float, float]:
    # Returns (ms_total, ms_avg)
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

    total_ms = start.elapsed_time(end)
    return total_ms, total_ms / iters


def _format_bw(gb_per_s: float) -> str:
    if math.isfinite(gb_per_s):
        return f"{gb_per_s:,.2f} GB/s"
    return "nan"


def _estimate_bytes_rope_rw(q: torch.Tensor, k: torch.Tensor) -> int:
    # RoPE conceptually reads and writes q, and reads and writes k.
    # This is a lower-bound estimate that ignores intermediate/cache effects.
    return 2 * q.numel() * q.element_size() + 2 * k.numel() * k.element_size()


def _estimate_bytes_cos_sin_read(cos: torch.Tensor, sin: torch.Tensor) -> int:
    return cos.numel() * cos.element_size() + sin.numel() * sin.element_size()


def _estimate_bytes_cos_sin_cache_read(cache: torch.Tensor) -> int:
    return cache.numel() * cache.element_size()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=9180)
    parser.add_argument("--nheads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--cos-dtype", type=str, default="fp32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check numerical agreement between implementations.",
    )
    args = parser.parse_args()

    if args.device != "cuda":
        raise ValueError("This benchmark is intended to run on CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    bsz = args.bsz
    seqlen = args.seqlen
    nheads = args.nheads
    head_dim = args.head_dim
    half_dim = head_dim // 2

    dtype = _dtype_from_str(args.dtype)
    cos_dtype = _dtype_from_str(args.cos_dtype)

    device = torch.device(args.device)

    # Inputs
    q = torch.randn((bsz, seqlen, nheads, head_dim), device=device, dtype=dtype)
    k = torch.randn((bsz, seqlen, nheads, head_dim), device=device, dtype=dtype)

    # Ensure the exact memory layout used in your prints.
    assert q.shape == (1, 9180, 40, 128) if (bsz, seqlen, nheads, head_dim) == (1, 9180, 40, 128) else True

    cos, sin = _make_cos_sin(seqlen, half_dim, device=device, dtype=cos_dtype)

    # Pre-built variants
    cos_sin_cache = _build_cos_sin_cache_like_wanvideo(cos, sin)
    positions_gpu = torch.arange(bsz * seqlen, device=device, dtype=torch.long)

    # 1) Baseline: Triton via _apply_rotary_emb
    def triton_path() -> None:
        _ = _apply_rotary_emb(q, cos, sin, is_neox_style=False)
        _ = _apply_rotary_emb(k, cos, sin, is_neox_style=False)

    # 2) FlashInfer: mimic current WanVideo path (build cache inside the timed region, positions=None)
    def flashinfer_like_wanvideo() -> None:
        q2 = q
        k2 = k
        cache = _build_cos_sin_cache_like_wanvideo(cos, sin)
        _ = apply_flashinfer_rope_qk_inplace(q2, k2, cache, is_neox=False)

    # 3) FlashInfer: prebuilt cache + prebuilt GPU positions (isolates kernel cost)
    def flashinfer_prebuilt() -> None:
        q2 = q
        k2 = k
        _ = apply_flashinfer_rope_qk_inplace(
            q2, k2, cos_sin_cache, is_neox=False, positions=positions_gpu
        )

    # Validate correctness (optional)
    if args.validate:
        q_triton = _apply_rotary_emb(q, cos, sin, is_neox_style=False)
        k_triton = _apply_rotary_emb(k, cos, sin, is_neox_style=False)

        q_fi, k_fi = apply_flashinfer_rope_qk_inplace(
            q, k, cos_sin_cache, is_neox=False, positions=positions_gpu
        )
        q_diff = (q_triton - q_fi).abs()
        k_diff = (k_triton - k_fi).abs()
        print("[validate] q max abs diff:", q_diff.max().item())
        print("[validate] q mean abs diff:", q_diff.mean().item())
        print("[validate] k max abs diff:", k_diff.max().item())
        print("[validate] k mean abs diff:", k_diff.mean().item())

    # Timings
    results: Dict[str, Dict[str, float]] = {}

    total_ms, avg_ms = _time_cuda_events(triton_path, warmup=args.warmup, iters=args.iters)
    results["triton__apply_rotary_emb_x2"] = {"total_ms": total_ms, "avg_ms": avg_ms}

    total_ms, avg_ms = _time_cuda_events(
        flashinfer_like_wanvideo, warmup=args.warmup, iters=args.iters
    )
    results["flashinfer__build_cache_each_iter__pos_none"] = {
        "total_ms": total_ms,
        "avg_ms": avg_ms,
    }

    total_ms, avg_ms = _time_cuda_events(flashinfer_prebuilt, warmup=args.warmup, iters=args.iters)
    results["flashinfer__prebuilt_cache__prebuilt_positions"] = {
        "total_ms": total_ms,
        "avg_ms": avg_ms,
    }

    # Bandwidth estimates
    # Lower-bound bytes: q/k read+write once + minimal cos/sin read.
    base_bytes_rw = _estimate_bytes_rope_rw(q, k)

    # Triton reads cos/sin twice (q and k) from separate tensors.
    triton_cos_sin_bytes = 2 * _estimate_bytes_cos_sin_read(cos, sin)

    # FlashInfer reads cos_sin_cache once (conceptually). In practice, it might be re-read per head.
    fi_cache_bytes = _estimate_bytes_cos_sin_cache_read(cos_sin_cache)

    print("\n=== Rope Micro-benchmark ===")
    print(f"shape: q/k = [{bsz}, {seqlen}, {nheads}, {head_dim}]  dtype={dtype}  cos_dtype={cos_dtype}")
    print(f"iters={args.iters} warmup={args.warmup}")

    # Report
    triton_avg = results["triton__apply_rotary_emb_x2"]["avg_ms"]
    for name, m in results.items():
        avg_ms = m["avg_ms"]
        speedup = triton_avg / avg_ms if avg_ms > 0 else float("nan")

        if name.startswith("triton"):
            total_bytes = base_bytes_rw + triton_cos_sin_bytes
        else:
            total_bytes = base_bytes_rw + fi_cache_bytes

        gbps = (total_bytes / 1e9) / (avg_ms / 1e3)

        print(f"\n[{name}]")
        print(f"avg: {avg_ms * 1e3:,.2f} us/iter  (speedup vs triton: {speedup:,.3f}x)")
        print(f"bytes (est): {total_bytes/1e6:,.2f} MB/iter")
        print(f"bandwidth (est): {_format_bw(gbps)}")

    print("\nNotes:")
    print("- flashinfer__...__pos_none includes positions construction inside apply_flashinfer_rope_qk_inplace (CPU arange + H2D copy).")
    print("- flashinfer__prebuilt_cache__prebuilt_positions isolates the RoPE kernel better (recommended for fair kernel comparison).")


if __name__ == "__main__":
    main()
