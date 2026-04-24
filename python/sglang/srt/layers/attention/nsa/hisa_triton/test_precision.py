"""Precision verification: triton-port outputs vs tilelang reference.

Uses ``torch.testing.assert_close`` with a ladder of tolerances to
characterise where the two kernels agree and where fp8 ULP noise dominates.

For each (kernel, shape) we also check what downstream actually cares about:
the *topk_2048 set* induced by the logits. If the two kernels pick the same
2048 K positions (IoU = 1.0), then kernel-level ULP drift is harmless e2e.

Run:
    python -m sglang.srt.layers.attention.nsa.hisa_triton.test_precision
"""
from __future__ import annotations

import sys

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    batch_pool_mqa_attn_return_logits_fp8_interface,
    batch_pool_mqa_attn_return_logits_fp8_v3_interface,
    fp8_native_paged_block_sparse_mqa_attn_return_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.benchmark import (
    MqaDims,
    _make_batch_pool_mqa_inputs,
    _make_batch_pool_mqa_v3_inputs,
    _make_sparse_paged_mqa_inputs,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    batch_decode_pool_mqa_v3_triton,
    batch_pool_mqa_triton,
    sparse_paged_mqa_triton,
)


DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# Tolerance ladder — strictest first.
# ---------------------------------------------------------------------------

TOLERANCE_LADDER = [
    ("default",       None,    None),     # torch defaults (tight, rtol=rtol=1.3e-6 for f32)
    ("strict",        1e-5,    1e-3),
    ("fp8-strict",    1e-3,    1e-2),
    ("fp8-reasonable", 5e-3,   1e-1),
    ("fp8-loose",     5e-2,    1.0),
    ("fp8-very-loose", 2e-1,   100.0),
]


def _finite_only(a: torch.Tensor, b: torch.Tensor):
    """Mask out non-finite (±inf) positions in BOTH tensors. Both kernels
    set the same positions to ±inf (force_maintain + ctx_len mask), so we
    verify that separately and only ``assert_close`` the finite subset.
    """
    m = torch.isfinite(a) & torch.isfinite(b)
    return a[m], b[m]


def _inf_match(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    """Check that ±inf positions in a match ±inf positions in b bit-for-bit."""
    pos_inf_a = a == float("inf")
    pos_inf_b = b == float("inf")
    neg_inf_a = a == float("-inf")
    neg_inf_b = b == float("-inf")
    if not torch.equal(pos_inf_a, pos_inf_b):
        return False, f"+inf masks differ: a={pos_inf_a.sum().item()} b={pos_inf_b.sum().item()}"
    if not torch.equal(neg_inf_a, neg_inf_b):
        return False, f"-inf masks differ: a={neg_inf_a.sum().item()} b={neg_inf_b.sum().item()}"
    return True, "+inf/-inf masks match"


def _strictest_passing_tolerance(tl_out: torch.Tensor, tr_out: torch.Tensor) -> tuple[str, str]:
    """Return (label of strictest passing level, diagnostic string)."""
    a, b = _finite_only(tl_out, tr_out)
    if a.numel() == 0:
        return "n/a (no finite)", "no finite elements to compare"

    strictest_passing = "FAIL (none)"
    for label, rtol, atol in TOLERANCE_LADDER:
        try:
            if rtol is None:
                torch.testing.assert_close(tr_out, tl_out, equal_nan=False)
            else:
                torch.testing.assert_close(
                    tr_out, tl_out, rtol=rtol, atol=atol, equal_nan=False,
                )
            strictest_passing = label
            break
        except AssertionError:
            continue

    abs_d = (a - b).abs()
    rel_d = abs_d / (a.abs() + 1e-6)
    diag = (
        f"max|abs|={abs_d.max().item():.3e}, "
        f"max|rel|={rel_d.max().item():.3e}, "
        f"mean|abs|={abs_d.mean().item():.3e}, "
        f"n_finite={a.numel()}"
    )
    return strictest_passing, diag


def _topk_iou(
    tl_logits: torch.Tensor, tr_logits: torch.Tensor, k: int,
) -> tuple[float, float, int]:
    """Return (mean_iou, min_iou, n_rows) of top-k index sets induced by the
    two logit tensors. tl_logits / tr_logits should be 2D [rows, cols]; we
    take top-k per row and compute set-IoU."""
    assert tl_logits.shape == tr_logits.shape and tl_logits.ndim == 2
    rows = tl_logits.shape[0]
    # Replace -inf with very small value so topk is stable.
    tl_clean = torch.where(torch.isfinite(tl_logits), tl_logits, torch.tensor(-1e30, device=tl_logits.device))
    tr_clean = torch.where(torch.isfinite(tr_logits), tr_logits, torch.tensor(-1e30, device=tr_logits.device))
    k_use = min(k, tl_logits.shape[1])
    tl_idx = torch.topk(tl_clean, k_use, dim=-1).indices  # [rows, k]
    tr_idx = torch.topk(tr_clean, k_use, dim=-1).indices
    ious = []
    for r in range(rows):
        a = set(tl_idx[r].tolist())
        b = set(tr_idx[r].tolist())
        inter = a & b
        union = a | b
        ious.append(len(inter) / max(len(union), 1))
    import statistics as st
    return st.mean(ious), min(ious), rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@torch.inference_mode()
def test_batch_pool_mqa():
    dims = MqaDims(heads=64, dim=128)
    print("\n" + "=" * 90)
    print("batch_pool_mqa — triton vs tilelang precision")
    print("=" * 90)
    print(f"{'B':>4} | {'nb':>6} | {'inf mask':>12} | {'strictest passing':>22} | diag")
    print("-" * 120)
    for B in [1, 8, 32, 64]:
        for nb in [128, 512, 1024]:
            inp = _make_batch_pool_mqa_inputs(B, nb, dims, seed=0)
            out_tl = batch_pool_mqa_attn_return_logits_fp8_interface(
                q_fp8=inp["q_fp8"],
                blocked_kv_fp8=inp["blocked_k_fp8"],
                blocked_kv_scale=inp["blocked_k_scale"],
                weights_f32=inp["weights"],
                context_lens=inp["context_lens"],
                kv_block_size=128,
            ).squeeze(1)  # [B, nb]
            out_tr = batch_pool_mqa_triton(
                q_fp8=inp["q_fp8"],
                blocked_k_fp8=inp["blocked_k_fp8"],
                blocked_k_scale=inp["blocked_k_scale"],
                weights_f32=inp["weights"],
                context_lens=inp["context_lens"],
            ).squeeze(1)

            inf_ok, inf_msg = _inf_match(out_tl, out_tr)
            level, diag = _strictest_passing_tolerance(out_tl, out_tr)
            print(f"{B:>4} | {nb:>6} | {'OK' if inf_ok else 'FAIL':>12} | {level:>22} | {diag}")


@torch.inference_mode()
def test_sparse_paged_mqa():
    dims = MqaDims(heads=64, dim=128)
    print("\n" + "=" * 90)
    print("sparse_paged_mqa — triton vs tilelang precision (THE 80% HOTSPOT)")
    print("=" * 90)
    print(f"{'B':>4} | {'ctx':>6} | {'inf mask':>12} | {'strictest passing':>22} | topk IoU (mean/min) | diag")
    print("-" * 150)
    for B in [1, 8, 32, 64]:
        for ctx in [16384, 32768, 65536]:
            inp = _make_sparse_paged_mqa_inputs(B, ctx, topk=64, dims=dims, seed=0)
            out_tl = fp8_native_paged_block_sparse_mqa_attn_return_logits_interface(
                q_fp8=inp["q_fp8"],
                kv_cache_fp8=inp["kv_cache_fp8"],
                topk_block_index=inp["topk_block_index"],
                kv_block_size=inp["kv_block_size"],
                weights=inp["weights"],
                context_lens=inp["context_lens"],
                block_tables=inp["block_tables"],
            )  # [B, 1, topk * kv_block_size]
            out_tr = sparse_paged_mqa_triton(
                q_fp8=inp["q_fp8"],
                kv_cache_fp8=inp["kv_cache_fp8"],
                topk_block_index=inp["topk_block_index"],
                kv_block_size=inp["kv_block_size"],
                weights=inp["weights"],
                context_lens=inp["context_lens"],
                block_tables=inp["block_tables"],
            )

            inf_ok, inf_msg = _inf_match(out_tl, out_tr)
            level, diag = _strictest_passing_tolerance(out_tl, out_tr)
            # topk IoU — pick top-2048 (matches production index_topk)
            tl_flat = out_tl.squeeze(1)
            tr_flat = out_tr.squeeze(1)
            mean_iou, min_iou, rows = _topk_iou(tl_flat, tr_flat, k=2048)
            iou_str = f"{mean_iou:.4f}/{min_iou:.4f} ({rows} rows)"
            print(
                f"{B:>4} | {ctx:>6} | {'OK' if inf_ok else 'FAIL':>12} | "
                f"{level:>22} | {iou_str:>20} | {diag}"
            )


@torch.inference_mode()
def test_batch_decode_pool_mqa_v3():
    """v4 HOT KERNEL #2: paged pool_k_pages block-MQA."""
    dims = MqaDims(heads=64, dim=128)
    print("\n" + "=" * 90)
    print("batch_decode_pool_mqa_v3 — triton vs tilelang precision (V4 HOT #2)")
    print("=" * 90)
    print(f"{'B':>4} | {'num_pool':>8} | {'inf mask':>12} | {'strictest passing':>22} | topk IoU (mean/min) | diag")
    print("-" * 150)
    for B in [1, 8, 32, 64]:
        for num_pool in [128, 512, 1024]:
            inp = _make_batch_pool_mqa_v3_inputs(B, num_pool, dims, seed=0)
            out_tl = batch_pool_mqa_attn_return_logits_fp8_v3_interface(
                q_fp8=inp["q"], pool_k_pages=inp["pool_k_pages"],
                pool_page_tables=inp["pool_page_tables"],
                weights_f32=inp["weights"],
                context_lens_pool=inp["context_lens_pool"],
                pool_page_size=inp["pool_page_size"],
            )  # [B, 1, max_pool_pages * PP]
            out_tr = batch_decode_pool_mqa_v3_triton(
                q_fp8=inp["q"], pool_k_pages=inp["pool_k_pages"],
                pool_page_tables=inp["pool_page_tables"],
                weights_f32=inp["weights"],
                context_lens_pool=inp["context_lens_pool"],
                pool_page_size=inp["pool_page_size"],
            )
            inf_ok, _ = _inf_match(out_tl, out_tr)
            level, diag = _strictest_passing_tolerance(out_tl, out_tr)
            # topk IoU — pick top-64 (matches production block_topk).
            tl_flat = out_tl.squeeze(1)
            tr_flat = out_tr.squeeze(1)
            mean_iou, min_iou, rows = _topk_iou(tl_flat, tr_flat, k=64)
            iou_str = f"{mean_iou:.4f}/{min_iou:.4f} ({rows} rows)"
            print(
                f"{B:>4} | {num_pool:>8} | {'OK' if inf_ok else 'FAIL':>12} | "
                f"{level:>22} | {iou_str:>20} | {diag}"
            )


def main() -> int:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\nTolerance ladder:")
    for label, rtol, atol in TOLERANCE_LADDER:
        if rtol is None:
            print(f"  {label:>18}: torch defaults (rtol=atol=1.3e-6 for f32)")
        else:
            print(f"  {label:>18}: rtol={rtol:<6g}  atol={atol}")
    test_batch_pool_mqa()
    test_sparse_paged_mqa()
    test_batch_decode_pool_mqa_v3()
    return 0


if __name__ == "__main__":
    sys.exit(main())
