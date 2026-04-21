"""Smoke test: at seq_len=8192, how much of baseline's top-2048 does hisa recover?

Runs both the baseline (``deep_gemm.fp8_mqa_logits`` + ``fast_topk_v2``) and
the full hisa pipeline (``fp8_native_hierarchy_mqa_logits`` + ``fast_topk_v2``
+ ``hisa_coord_transform``) on identical inputs, and measures per-row set
overlap ("recall@2048").

Each row of topk_indices is a set of ks-relative K positions. We report:
- mean / median / p10 / min recall across rows with K range >= index_topk
  (rows with K range < index_topk are always a perfect match — baseline and
  hisa both select all available positions)
- a few example rows where the sets diverge significantly
"""
from __future__ import annotations

import sys

import torch
from sgl_kernel import fast_topk_v2
import deep_gemm

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_hierarchy_mqa_logits,
)
from sglang.srt.layers.attention.nsa.hisa.triton_kernel import hisa_coord_transform


DEVICE = torch.device("cuda")


def build_inputs(seq_len, H=64, D=128):
    torch.manual_seed(0)
    q_bf16 = torch.randn(seq_len, H, D, device=DEVICE, dtype=torch.bfloat16)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    k_bf16 = torch.randn(seq_len, D, device=DEVICE, dtype=torch.bfloat16)
    k_fp8 = k_bf16.to(torch.float8_e4m3fn)

    # Per-token float32 scale, stored as [N, 4] uint8 for hisa, [N] float32 for baseline.
    k_scale_f32 = 0.1 + 0.01 * torch.rand(seq_len, device=DEVICE, dtype=torch.float32)
    k_scale_uint8 = k_scale_f32.view(torch.uint8).clone().reshape(seq_len, 4)

    weights = torch.randn(seq_len, H, device=DEVICE, dtype=torch.float32)

    # Single-seq causal: ks = 0, ke = 1..M.
    ks = torch.zeros(seq_len, device=DEVICE, dtype=torch.int32)
    ke = (torch.arange(seq_len, device=DEVICE, dtype=torch.int32) + 1)
    return dict(
        q_fp8=q_fp8, k_fp8=k_fp8,
        k_scale_f32=k_scale_f32, k_scale_uint8=k_scale_uint8,
        weights=weights, ks=ks, ke=ke,
    )


def baseline_topk(inp, index_topk):
    logits = deep_gemm.fp8_mqa_logits(
        inp["q_fp8"], (inp["k_fp8"], inp["k_scale_f32"]),
        inp["weights"], inp["ks"], inp["ke"],
        clean_logits=False,
    )
    seq_lens_topk = (inp["ke"] - inp["ks"]).to(torch.int32)
    # fast_topk_v2 with row_starts=ks → output is ks-relative positions.
    out = fast_topk_v2(logits, seq_lens_topk, index_topk, row_starts=inp["ks"])
    return out


def hisa_topk(inp, index_topk, k_block_size=128, block_topk=64):
    block_sparse_logits, topk_block_indices = fp8_native_hierarchy_mqa_logits(
        inp["q_fp8"], (inp["k_fp8"], inp["k_scale_uint8"]),
        inp["weights"], inp["ks"], inp["ke"],
        k_block_size, block_topk,
    )
    M = block_sparse_logits.shape[0]
    sparse_len = block_sparse_logits.shape[-1]
    full_lens = torch.full(
        (M,), sparse_len, dtype=torch.int32, device=DEVICE,
    )
    relevant = fast_topk_v2(block_sparse_logits, full_lens, index_topk)
    out = hisa_coord_transform(
        relevant, topk_block_indices,
        lens=inp["ke"], k_block_size=k_block_size, ks=inp["ks"],
    )
    return out


@torch.inference_mode()
def recall_per_row(a: torch.Tensor, b: torch.Tensor, valid_mask: torch.Tensor):
    """Per-row |a ∩ b| / |{a-valid}|, ignoring -1 entries.

    Args:
        a, b: [M, K] int32 topk sets (ks-relative; -1 = padding).
        valid_mask: [M] bool — rows to include (typically K >= index_topk).
    """
    M, K = a.shape
    # Treat -1 as non-matching by using a sentinel far outside both sets.
    # Use sorted bool approach:  for each row, how many a[i,:] values are in b[i, :] set?
    a_sorted, _ = torch.sort(a, dim=-1)
    b_sorted, _ = torch.sort(b, dim=-1)
    # Count overlap per row.
    overlaps = torch.zeros(M, dtype=torch.int32, device=a.device)
    for m in range(M):
        if not valid_mask[m]:
            continue
        a_set = set(a[m].tolist()) - {-1}
        b_set = set(b[m].tolist()) - {-1}
        if not a_set:
            continue
        overlaps[m] = len(a_set & b_set)
    return overlaps


def main() -> int:
    seq_len = 8192
    index_topk = 2048
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: seq_len={seq_len}, index_topk={index_topk}, "
          f"k_block_size=128, block_topk=64, H=64, D=128")

    inp = build_inputs(seq_len)

    print("Running baseline pipeline...")
    a = baseline_topk(inp, index_topk)      # [M, index_topk] int32
    print("Running hisa pipeline...")
    b = hisa_topk(inp, index_topk)          # [M, index_topk] int32

    # For rows where (ke - ks) <= index_topk, both pipelines should select
    # literally all positions (trivially recall=1). Focus on the non-trivial
    # rows where K range > index_topk.
    k_extent = inp["ke"] - inp["ks"]
    nontrivial = k_extent > index_topk
    n_nontrivial = int(nontrivial.sum().item())
    print(f"\nRows with (ke-ks) > index_topk: {n_nontrivial} / {seq_len}")

    if n_nontrivial == 0:
        print("No non-trivial rows — skipping recall check.")
        return 0

    # Per-row overlap count.
    overlaps = recall_per_row(a, b, nontrivial)

    # Rows we care about.
    rel_sizes = k_extent.clamp(max=index_topk)  # |baseline set| = min(ke-ks, index_topk)
    recall = overlaps.float() / rel_sizes.float()

    # Summary stats over non-trivial rows.
    r = recall[nontrivial]
    print("\nRecall@{} summary (non-trivial rows):".format(index_topk))
    print(f"  mean:    {r.mean().item():.4f}")
    print(f"  median:  {r.median().item():.4f}")
    print(f"  p10:     {torch.quantile(r, 0.10).item():.4f}")
    print(f"  p01:     {torch.quantile(r, 0.01).item():.4f}")
    print(f"  min:     {r.min().item():.4f}")
    print(f"  rows with recall < 0.95:  "
          f"{int((r < 0.95).sum().item())} / {n_nontrivial}")
    print(f"  rows with recall < 0.90:  "
          f"{int((r < 0.90).sum().item())} / {n_nontrivial}")
    print(f"  rows with recall < 0.80:  "
          f"{int((r < 0.80).sum().item())} / {n_nontrivial}")

    # Show a few sample rows with lowest recall (most divergent).
    idx = torch.where(nontrivial)[0]
    r_sorted_idx = idx[torch.argsort(recall[idx])]
    print("\nWorst 3 rows (lowest recall):")
    for m in r_sorted_idx[:3].tolist():
        ke_val = int(inp["ke"][m].item())
        base_set = set(a[m].tolist()) - {-1}
        hisa_set = set(b[m].tolist()) - {-1}
        inter = base_set & hisa_set
        missed = base_set - hisa_set
        extra = hisa_set - base_set
        print(f"  row m={m} (ke={ke_val}): "
              f"|baseline|={len(base_set)} |hisa|={len(hisa_set)} "
              f"overlap={len(inter)} recall={len(inter) / max(len(base_set), 1):.4f}")
        if len(missed) <= 8 and len(missed) > 0:
            print(f"    baseline-only: {sorted(missed)[:8]}")
        if len(extra) <= 8 and len(extra) > 0:
            print(f"    hisa-only:     {sorted(extra)[:8]}")

    # Binary expectation: typical hisa should recover >= 0.95 mean recall.
    mean_r = r.mean().item()
    if mean_r >= 0.95:
        verdict = "PASS (approximation quality healthy)"
    elif mean_r >= 0.90:
        verdict = "MARGINAL (expected >= 0.95)"
    else:
        verdict = "POOR (unexpected — investigate)"
    print(f"\nVerdict: mean recall = {mean_r:.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
