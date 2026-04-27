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
    _make_paged_kv_cache_soa,
    _make_sparse_paged_mqa_inputs,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    batch_decode_pool_mqa_v3_triton,
    batch_pool_mqa_triton,
    block_mean_pooling_triton,
    block_sparse_mqa_triton,
    paged_mean_pooling_triton,
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


# ---------------------------------------------------------------------------
# Cross-k_block_size correctness — torch references for all 4 hisa kernels.
#
# These references mirror the kernel's fp32-accumulator + fp8-input arithmetic
# (cast fp8→fp32, do reductions in fp32, cast outputs back to fp8). They are
# slow (Python loops) but unambiguous: any disagreement at fp8-strict
# tolerance (rtol=1e-3, atol=1e-2) is a real bug, not fp8 ULP noise.
# ---------------------------------------------------------------------------


def _ref_block_mean_pooling(k_fp8, k_scale, k_block_size):
    """Ragged mean-pool reference: pool every k_block_size tokens of K."""
    seq_kv, D = k_fp8.shape
    n_pool = (seq_kv + k_block_size - 1) // k_block_size
    out_k = torch.zeros(n_pool, D, dtype=torch.float8_e4m3fn, device=k_fp8.device)
    out_s = torch.zeros(n_pool, dtype=torch.float32, device=k_fp8.device)
    k_f32 = k_fp8.to(torch.float32) * k_scale[:, None]   # [seq_kv, D]
    for p in range(n_pool):
        s, e = p * k_block_size, min((p + 1) * k_block_size, seq_kv)
        if s >= e:
            continue
        mean = k_f32[s:e].sum(dim=0) / float(e - s)      # [D]
        mx = mean.abs().max().item()
        scale = max(mx * (1.0 / 448.0), 1e-10)
        out_k[p] = (mean / scale).to(torch.float8_e4m3fn)
        out_s[p] = scale
    return out_k, out_s


def _ref_paged_mean_pooling(kv, block_tables, ctx_lens, k_block_size, paged_block_size):
    """Paged mean-pool reference (SoA byte layout)."""
    num_phys, _, _, DPlus4 = kv.shape
    D = DPlus4 - 4
    B, max_log = block_tables.shape
    kv_flat = kv.reshape(num_phys, paged_block_size * DPlus4).contiguous()
    kv_fp8 = (
        kv_flat[:, : paged_block_size * D].contiguous()
        .view(torch.float8_e4m3fn).reshape(num_phys, paged_block_size, D)
    )
    kv_scale = (
        kv_flat[:, paged_block_size * D :].contiguous()
        .view(torch.float32).reshape(num_phys, paged_block_size)
    )
    max_n_pool = max(int((int(ctx_lens[b].item()) + k_block_size - 1) // k_block_size)
                      for b in range(B))
    out_k = torch.zeros(B, max_n_pool, D, dtype=torch.float8_e4m3fn, device=kv.device)
    out_s = torch.zeros(B, max_n_pool, dtype=torch.float32, device=kv.device)
    for b in range(B):
        seq = int(ctx_lens[b].item())
        n_pool = (seq + k_block_size - 1) // k_block_size
        for p in range(n_pool):
            s, e = p * k_block_size, min((p + 1) * k_block_size, seq)
            if s >= e:
                continue
            acc = torch.zeros(D, dtype=torch.float32, device=kv.device)
            for t in range(s, e):
                logical = t // paged_block_size
                row = t % paged_block_size
                phys = int(block_tables[b, logical].item())
                acc += kv_fp8[phys, row].to(torch.float32) * kv_scale[phys, row].item()
            mean = acc / float(e - s)
            mx = mean.abs().max().item()
            scale = max(mx * (1.0 / 448.0), 1e-10)
            out_k[b, p] = (mean / scale).to(torch.float8_e4m3fn)
            out_s[b, p] = scale
    return out_k, out_s


def _ref_block_sparse_mqa(q_fp8, k_fp8, k_scale, topk_idx, kv_block_size,
                          weights, cu_ks, cu_ke):
    """Ragged block-sparse MQA reference."""
    seq_q, H, D = q_fp8.shape
    seq_kv = k_fp8.shape[0]
    topk = topk_idx.shape[1]
    out = torch.full((seq_q, topk * kv_block_size), float("-inf"),
                      dtype=torch.float32, device=q_fp8.device)
    q_f32 = q_fp8.to(torch.float32)                       # [seq_q, H, D]
    k_f32 = k_fp8.to(torch.float32)                       # [seq_kv, D]
    for s_i in range(seq_q):
        ks_min = int(cu_ks[s_i].item())
        ke_max = int(cu_ke[s_i].item())
        w = weights[s_i]                                  # [H]
        for n_i in range(topk):
            tid = int(topk_idx[s_i, n_i].item())
            for j in range(kv_block_size):
                t_abs = tid * kv_block_size + j
                if t_abs < 0 or t_abs >= seq_kv:
                    continue
                if t_abs < ks_min or t_abs >= ke_max:
                    continue
                # GEMM row: [H] = q_f32[s_i] @ k_f32[t_abs] * scale
                kv = k_f32[t_abs]
                s = (q_f32[s_i] @ kv) * k_scale[t_abs].item()  # [H]
                s = torch.clamp(s, min=0)
                logit = (s * w).sum().item()
                out[s_i, n_i * kv_block_size + j] = logit
    return out


def _ref_sparse_paged_mqa(q_fp8, kv, topk_idx, kv_block_size,
                           weights, ctx_lens, block_tables, paged_block_size):
    """Decode paged sparse MQA reference (SoA byte layout)."""
    B, seq_q, H, D = q_fp8.shape
    topk = topk_idx.shape[2]
    num_phys, _, _, DPlus4 = kv.shape
    kv_flat = kv.reshape(num_phys, paged_block_size * DPlus4).contiguous()
    kv_fp8 = (
        kv_flat[:, : paged_block_size * D].contiguous()
        .view(torch.float8_e4m3fn).reshape(num_phys, paged_block_size, D)
    )
    kv_scale = (
        kv_flat[:, paged_block_size * D :].contiguous()
        .view(torch.float32).reshape(num_phys, paged_block_size)
    )
    weights = weights.view(B, seq_q, H)
    out = torch.full((B, seq_q, topk * kv_block_size), float("-inf"),
                      dtype=torch.float32, device=q_fp8.device)
    q_f32 = q_fp8.to(torch.float32)
    max_blocks = block_tables.shape[1]
    for b in range(B):
        ctx = int(ctx_lens[b].item())
        for s_i in range(seq_q):
            w = weights[b, s_i]
            for n_i in range(topk):
                tid = int(topk_idx[b, s_i, n_i].item())
                for j in range(kv_block_size):
                    t_abs = tid * kv_block_size + j
                    if t_abs < 0 or t_abs >= ctx:
                        continue
                    logical = t_abs // paged_block_size
                    if logical < 0 or logical >= max_blocks:
                        continue
                    row = t_abs % paged_block_size
                    phys = int(block_tables[b, logical].item())
                    kv_vec = kv_fp8[phys, row].to(torch.float32)
                    s = (q_f32[b, s_i] @ kv_vec) * kv_scale[phys, row].item()
                    s = torch.clamp(s, min=0)
                    out[b, s_i, n_i * kv_block_size + j] = (s * w).sum().item()
    return out


@torch.inference_mode()
def test_cross_k_block_size():
    """Verify all 4 hisa kernels at k ∈ {8, 16, 32, 64, 128} agree with a
    torch reference (fp8 → fp32 → ops → fp8) at fp8-strict tolerance.
    Small sizes for speed; the goal is correctness coverage of grouped
    paths, not perf."""
    print("\n" + "=" * 90)
    print("CROSS-k_block_size correctness (torch ref @ fp8-strict tolerance)")
    print("=" * 90)
    print(f"{'kernel':>22} | {'k':>4} | {'config':>20} | {'level':>22} | diag")
    print("-" * 130)

    K_VALS = [8, 16, 32, 64, 128]
    dims = MqaDims(heads=64, dim=128)
    H, D = dims.heads, dims.dim

    # --- 1. block_mean_pooling ---
    for k in K_VALS:
        seq_kv = 256  # small for speed
        torch.manual_seed(0)
        k_fp8 = torch.randn(seq_kv, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        k_scale = 0.05 + 0.02 * torch.rand(seq_kv, device=DEVICE, dtype=torch.float32)
        bk_t, bks_t = block_mean_pooling_triton(k_fp8, k_scale, k)
        bk_r, bks_r = _ref_block_mean_pooling(k_fp8, k_scale, k)
        # Compare in fp32 space for fp8 tensors (cast both to fp32).
        level, diag = _strictest_passing_tolerance(bk_t.to(torch.float32), bk_r.to(torch.float32))
        cfg = f"seq_kv={seq_kv}"
        print(f"{'block_mean_pooling':>22} | {k:>4} | {cfg:>20} | {level:>22} | {diag}")

    # --- 2. paged_mean_pooling ---
    paged = 64
    for k in K_VALS:
        B = 2
        ctx = 256
        max_logical = (ctx + paged - 1) // paged
        num_phys = max_logical * B + 4
        kv = _make_paged_kv_cache_soa(num_phys, paged, D, seed=0)
        block_tables = torch.arange(max_logical * B, dtype=torch.int32, device=DEVICE).reshape(B, max_logical)
        ctx_lens = torch.full((B,), ctx, dtype=torch.int32, device=DEVICE)
        max_n_pool = (ctx + k - 1) // k
        bk_t, bks_t, _ = paged_mean_pooling_triton(max_n_pool, kv, ctx_lens, block_tables, k)
        bk_r, bks_r = _ref_paged_mean_pooling(kv, block_tables, ctx_lens, k, paged)
        level, diag = _strictest_passing_tolerance(bk_t.to(torch.float32), bk_r.to(torch.float32))
        cfg = f"B={B} ctx={ctx}"
        print(f"{'paged_mean_pooling':>22} | {k:>4} | {cfg:>20} | {level:>22} | {diag}")

    # --- 3. block_sparse_mqa (prefill ragged) ---
    # SK11: GEMM_TILE=256 → GROUP_SIZE=256/k (max=32 for k=8). topk must be
    # divisible by GROUP_SIZE; use topk=32 to satisfy all k.
    for k in K_VALS:
        seq_q = 8
        seq_kv = 4096
        topk = 32
        torch.manual_seed(0)
        q = torch.randn(seq_q, H, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        k_fp8 = torch.randn(seq_kv, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        k_scale = 0.05 + 0.02 * torch.rand(seq_kv, device=DEVICE, dtype=torch.float32)
        w = torch.randn(seq_q, H, device=DEVICE, dtype=torch.float32)
        cu_ks = torch.zeros(seq_q, device=DEVICE, dtype=torch.int32)
        cu_ke = torch.full((seq_q,), seq_kv, device=DEVICE, dtype=torch.int32)
        max_block = seq_kv // k
        topk_idx = torch.stack([torch.randperm(max_block, device=DEVICE)[:topk] for _ in range(seq_q)]).to(torch.int64)
        out_t = block_sparse_mqa_triton(q, k_fp8, k_scale, topk_idx, k, w, cu_ks, cu_ke)
        out_r = _ref_block_sparse_mqa(q, k_fp8, k_scale, topk_idx, k, w, cu_ks, cu_ke)
        inf_ok, inf_msg = _inf_match(out_t, out_r)
        level, diag = _strictest_passing_tolerance(out_t, out_r)
        cfg = f"sq={seq_q} skv={seq_kv}"
        status = "OK" if inf_ok else "INF-MISMATCH"
        print(f"{'block_sparse_mqa':>22} | {k:>4} | {cfg:>20} | {level:>22} | [{status}] {diag}")

    # --- 4. sparse_paged_mqa (decode) — INCLUDES SK12's k=128 cross-page path ---
    # SK3: GEMM_TILE=64 → GROUP_SIZE=64/k (max=8 for k=8). Use topk=8.
    for k in K_VALS:
        B = 2
        ctx = 1024
        topk = 8
        torch.manual_seed(0)
        q = torch.randn(B, 1, H, D, device=DEVICE, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        max_logical = (ctx + paged - 1) // paged
        num_phys = max_logical * B + 4
        kv = _make_paged_kv_cache_soa(num_phys, paged, D, seed=0)
        block_tables = torch.arange(max_logical * B, dtype=torch.int32, device=DEVICE).reshape(B, max_logical)
        ctx_lens = torch.full((B,), ctx, dtype=torch.int32, device=DEVICE)
        weights = torch.randn(B, H, device=DEVICE, dtype=torch.float32)
        max_block = ctx // k
        topk_idx = torch.stack([torch.randperm(max_block, device=DEVICE)[:topk] for _ in range(B)]).unsqueeze(1).to(torch.int64)
        out_t = sparse_paged_mqa_triton(
            q_fp8=q, kv_cache_fp8=kv, topk_block_index=topk_idx,
            kv_block_size=k, weights=weights, context_lens=ctx_lens,
            block_tables=block_tables,
        )
        out_r = _ref_sparse_paged_mqa(q, kv, topk_idx, k, weights, ctx_lens, block_tables, paged)
        inf_ok, inf_msg = _inf_match(out_t, out_r)
        level, diag = _strictest_passing_tolerance(out_t, out_r)
        cfg = f"B={B} ctx={ctx}"
        path_note = "k=128 cross-page" if k == 128 else f"G={64//k if k<64 else 1}"
        status = "OK" if inf_ok else "INF-MISMATCH"
        print(f"{'sparse_paged_mqa':>22} | {k:>4} | {cfg:>20} | {level:>22} | [{status}] [{path_note}] {diag}")


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
    test_cross_k_block_size()
    return 0


if __name__ == "__main__":
    sys.exit(main())
