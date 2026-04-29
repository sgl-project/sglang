"""Logits-level alignment: are hisa's per-block logits numerically equal to
baseline's dense logits at the same K positions?

hisa's output layout:
  topk_block_indices[m, s]  = the absolute block id of the s-th selected block
                              (NOT sorted — sorted by score desc)
  block_sparse_logits[m, s * k_block_size + j]
                            = logit for K position
                              topk_block_indices[m, s] * k_block_size + j

Baseline dense:
  dense_logits[m, p]        = logit for K position p

We gather baseline at hisa-selected positions and compare to hisa's logits
element-wise. The two kernels compute the same math in different order, so we
allow small fp tolerances; relative ordering (which this test *doesn't*
evaluate) is what matters for topk recall (see test_hisa_vs_baseline_recall.py).
"""
from __future__ import annotations

import sys

import torch
import deep_gemm

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_hierarchy_mqa_logits_tilelang_legacy,
)


DEVICE = torch.device("cuda")


def build_inputs(seq_len, H=64, D=128, seed=0):
    torch.manual_seed(seed)
    q_bf16 = torch.randn(seq_len, H, D, device=DEVICE, dtype=torch.bfloat16)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    k_bf16 = torch.randn(seq_len, D, device=DEVICE, dtype=torch.bfloat16)
    k_fp8 = k_bf16.to(torch.float8_e4m3fn)

    k_scale_f32 = 0.1 + 0.01 * torch.rand(seq_len, device=DEVICE, dtype=torch.float32)
    k_scale_uint8 = k_scale_f32.view(torch.uint8).clone().reshape(seq_len, 4)

    weights = torch.randn(seq_len, H, device=DEVICE, dtype=torch.float32)
    ks = torch.zeros(seq_len, device=DEVICE, dtype=torch.int32)
    ke = (torch.arange(seq_len, device=DEVICE, dtype=torch.int32) + 1)
    return dict(
        q_fp8=q_fp8, k_fp8=k_fp8,
        k_scale_f32=k_scale_f32, k_scale_uint8=k_scale_uint8,
        weights=weights, ks=ks, ke=ke, seq_len=seq_len,
    )


@torch.inference_mode()
def compare_logits(seq_len=8192, k_block_size=128, block_topk=64, H=64, D=128):
    print(f"\n{'='*72}")
    print(f"seq_len={seq_len}  k_block_size={k_block_size}  block_topk={block_topk}")
    print(f"{'='*72}")

    inp = build_inputs(seq_len, H=H, D=D)
    M = seq_len
    K = seq_len

    # ---- Baseline dense logits ----
    dense_logits = deep_gemm.fp8_mqa_logits(
        inp["q_fp8"], (inp["k_fp8"], inp["k_scale_f32"]),
        inp["weights"], inp["ks"], inp["ke"],
        clean_logits=False,
    )
    assert dense_logits.shape == (M, K), f"unexpected dense shape {dense_logits.shape}"

    # ---- Hisa sparse logits + block indices ----
    block_sparse_logits, topk_block_indices = fp8_native_hierarchy_mqa_logits_tilelang_legacy(
        inp["q_fp8"], (inp["k_fp8"], inp["k_scale_uint8"]),
        inp["weights"], inp["ks"], inp["ke"],
        k_block_size, block_topk,
    )
    # Shape may clamp when num_blocks < block_topk; here seq=8192 → 64 blocks == block_topk.
    actual_bt = topk_block_indices.shape[-1]
    assert block_sparse_logits.shape == (M, actual_bt * k_block_size), (
        f"unexpected hisa shape: block_sparse_logits={block_sparse_logits.shape}, "
        f"expected (M={M}, {actual_bt * k_block_size})"
    )

    # ---- Gather baseline at hisa-selected positions ----
    # For each (m, s, j):  k_pos = topk_block_indices[m, s] * k_block_size + j
    # Build [M, actual_bt, k_block_size] index tensor, gather, reshape.
    block_starts = topk_block_indices.to(torch.int64) * k_block_size          # [M, actual_bt]
    j_range = torch.arange(k_block_size, device=DEVICE, dtype=torch.int64)     # [k_block_size]
    indices = block_starts.unsqueeze(-1) + j_range.view(1, 1, -1)             # [M, actual_bt, k_block_size]
    indices_flat = indices.view(M, actual_bt * k_block_size)                  # [M, actual_bt*k_block_size]

    # Clamp indices within [0, K) for gather safety; out-of-range positions
    # (block crosses ke boundary) are handled via valid_mask below.
    indices_clamped = indices_flat.clamp(0, K - 1)
    gathered = torch.gather(dense_logits, -1, indices_clamped)                # [M, actual_bt*k_block_size]

    # ---- Valid mask (exclude positions past each query's ke) ----
    # A gathered position is valid only if pos < ke[m].
    ke_col = inp["ke"].unsqueeze(-1).to(torch.int64)                          # [M, 1]
    valid = indices_flat < ke_col                                             # [M, actual_bt*k_block_size]
    # Also: OOB-clamped entries (indices_flat >= K) need to be excluded too.
    valid &= indices_flat < K

    # ---- Compare ----
    diff = (block_sparse_logits - gathered).abs()
    masked_diff = diff.where(valid, torch.zeros_like(diff))
    n_valid = int(valid.sum().item())

    print(f"valid positions compared:   {n_valid:,} / {M * actual_bt * k_block_size:,} "
          f"({100 * n_valid / (M * actual_bt * k_block_size):.1f}%)")
    # Statistics over the valid positions only.
    flat_diff = masked_diff[valid]
    flat_gathered = gathered[valid]
    flat_hisa = block_sparse_logits[valid]

    # torch.quantile has a ~16M-element limit; subsample for percentile estimates.
    if flat_diff.numel() > 5_000_000:
        idx = torch.randint(
            0, flat_diff.numel(), (5_000_000,), device=flat_diff.device
        )
        sample = flat_diff[idx]
    else:
        sample = flat_diff
    print(f"\nAbsolute difference (hisa vs gathered baseline):")
    print(f"  mean abs diff:    {flat_diff.mean().item():.6e}")
    print(f"  median abs diff:  {flat_diff.median().item():.6e}")
    print(f"  p99 abs diff:     {torch.quantile(sample, 0.99).item():.6e} "
          f"(sampled)")
    print(f"  max abs diff:     {flat_diff.max().item():.6e}")

    # Relative to baseline magnitude.
    scale = flat_gathered.abs().mean().item()
    print(f"  baseline |logit| mean: {scale:.6e}")
    print(f"  mean rel diff:    {flat_diff.mean().item() / max(scale, 1e-9):.4%}")

    # Correlation — do hisa and baseline rank values similarly?
    corr = torch.corrcoef(torch.stack([flat_hisa, flat_gathered]).float())[0, 1]
    print(f"  pearson correlation: {corr.item():.6f}")

    # Bucket: % of positions with |diff| < threshold
    for thresh in [0.01, 0.05, 0.1, 0.5, 1.0]:
        rel = flat_diff / (flat_gathered.abs() + 1e-6)
        frac_abs = (flat_diff < thresh).float().mean().item()
        frac_rel = (rel < thresh).float().mean().item()
        print(f"  |diff| < {thresh:5.2f}:  abs={frac_abs:.4f}   rel={frac_rel:.4f}")

    # ---- Sample a few worst positions ----
    print("\nTop 5 positions with largest abs diff:")
    k = 5
    _, top_idx = flat_diff.topk(min(k, flat_diff.numel()))
    # We need to map back to (m, pos) — do it naively via valid positions list.
    valid_coords = valid.nonzero(as_tuple=False)  # [n_valid, 2]
    for i, local_i in enumerate(top_idx.tolist()):
        m, col = valid_coords[local_i].tolist()
        k_pos = int(indices_flat[m, col].item())
        hisa_v = block_sparse_logits[m, col].item()
        base_v = gathered[m, col].item()
        print(f"  #{i+1}: m={m:5d} k_pos={k_pos:6d}   hisa={hisa_v:+.4f}  "
              f"baseline={base_v:+.4f}  diff={abs(hisa_v - base_v):.4e}")


def main() -> int:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Primary: 8192 (what the user asked about).
    compare_logits(seq_len=8192)
    # Secondary: shorter samsum-ish shape.
    compare_logits(seq_len=4096)
    return 0


if __name__ == "__main__":
    sys.exit(main())
