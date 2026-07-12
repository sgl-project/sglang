"""Unit tests for fp8 (fp8 attn-GEMM mode) support in the M3 sparse Triton kernels.

Strategy: quantize random bf16 tensors to fp8_e4m3fn, then compare the fp8
kernel run against the SAME kernel run in bf16 on the *dequantized* tensors.
Both runs see numerically identical Q/K values, so the QK GEMMs match closely
and top-k selection is stable; the only intended divergence is the fp8 PV MMA
(P quantized to e4m3), which the tolerances cover. This isolates kernel
arithmetic from quantization error.

Covers: step-3 decode/prefill (gqa-share sparse), step-1 decode/prefill
indexer, non-unit k_scale/v_scale semantics, and bf16-path regression.
"""

import pytest
import torch

from sglang.srt.layers.attention.minimax_sparse_ops.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.srt.layers.attention.minimax_sparse_ops.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.srt.layers.attention.minimax_sparse_ops.prefill.flash_with_topk_idx import (
    flash_prefill_with_topk_index,
)
from sglang.srt.layers.attention.minimax_sparse_ops.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (Triton fp8 kernels)"
)
# fp8 PV (P quantized to e4m3, ~3-bit mantissa on [0,1] weights) dominates the
# fp8-vs-dequantized-ref error; QK matches to fp32-accumulation noise.
FP8_ATOL = 6e-2
FP8_RTOL = 6e-2
# widening mode (bf16 Q, fp8 KV) computes on exactly the dequantized values.
WIDEN_ATOL = 1e-3
WIDEN_RTOL = 1e-3


def qdq(x: torch.Tensor):
    """Quantize to e4m3 and return (fp8, dequantized-bf16) views of it."""
    x8 = x.to(FP8)
    return x8, x8.to(torch.bfloat16)


def build_decode_inputs(
    batch_size=4,
    num_q_heads=8,
    num_kv_heads=1,
    head_dim=128,
    block_size=128,
    topk=8,
    seq_lens_list=(513, 1023, 257, 769),
):
    seq_lens_list = list(seq_lens_list)[:batch_size]
    max_kv_len = max(seq_lens_list)
    max_slots = batch_size * max_kv_len
    q = torch.randn(
        batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    k = torch.randn(
        max_slots, num_kv_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    v = torch.randn(
        max_slots, num_kv_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    req_to_token = torch.zeros(batch_size, max_kv_len, dtype=torch.int32, device=DEVICE)
    slot_ids = torch.arange(batch_size, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    for i in range(batch_size):
        base = i * max_kv_len
        req_to_token[i] = (torch.randperm(max_kv_len, device=DEVICE) + base).to(
            torch.int32
        )
    topk_idx = torch.full(
        (num_kv_heads, batch_size, topk), -1, dtype=torch.int32, device=DEVICE
    )
    for kh in range(num_kv_heads):
        for b in range(batch_size):
            nb = (seq_lens_list[b] + block_size - 1) // block_size
            ak = min(topk, nb)
            # sorted ascending, matching the production topk contract
            sel = torch.randperm(nb, device=DEVICE)[:ak].sort().values
            topk_idx[kh, b, :ak] = sel.to(torch.int32)
    return q, k, v, req_to_token, seq_lens, slot_ids, topk_idx


def build_prefill_inputs(
    batch_size=2,
    num_q_heads=8,
    num_kv_heads=1,
    head_dim=128,
    seq_lens_list=(513, 769),
    prefix_lens_list=(0, 257),
):
    seq_lens_list = list(seq_lens_list)[:batch_size]
    prefix_lens_list = list(prefix_lens_list)[:batch_size]
    q_lens = [s - p for s, p in zip(seq_lens_list, prefix_lens_list)]
    total_q = sum(q_lens)
    max_kv_len = max(seq_lens_list)
    max_slots = batch_size * max_kv_len
    q = torch.randn(total_q, num_q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(
        max_slots, num_kv_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    v = torch.randn(
        max_slots, num_kv_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    req_to_token = torch.zeros(batch_size, max_kv_len, dtype=torch.int32, device=DEVICE)
    slot_ids = torch.arange(batch_size, dtype=torch.int64, device=DEVICE)
    for i in range(batch_size):
        base = i * max_kv_len
        req_to_token[i] = (torch.randperm(max_kv_len, device=DEVICE) + base).to(
            torch.int32
        )
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    cu_seqlens[1:] = torch.tensor(q_lens, device=DEVICE).cumsum(0)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, device=DEVICE)
    return (
        q,
        k,
        v,
        req_to_token,
        slot_ids,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max(q_lens),
        max(seq_lens_list),
    )


def run_step3_decode(q, k, v, req_to_token, seq_lens, slot_ids, topk_idx, **kw):
    return flash_decode_with_gqa_share_sparse(
        q, None, k, v, req_to_token, seq_lens, slot_ids, 128, topk_idx, **kw
    )


# ---------------------------------------------------------------------------
# step-3 decode (gqa-share sparse)
# ---------------------------------------------------------------------------


def test_step3_decode_all_fp8_vs_dequant_ref():
    torch.manual_seed(0)
    q, k, v, r2t, seq_lens, sids, tidx = build_decode_inputs()
    q8, qr = qdq(q)
    k8, kr = qdq(k)
    v8, vr = qdq(v)
    out8 = run_step3_decode(q8, k8, v8, r2t, seq_lens, sids, tidx)
    ref = run_step3_decode(qr, kr, vr, r2t, seq_lens, sids, tidx)
    assert out8.dtype == torch.bfloat16
    torch.testing.assert_close(out8.float(), ref.float(), atol=FP8_ATOL, rtol=FP8_RTOL)


def test_step3_decode_widening_mode():
    torch.manual_seed(1)
    q, k, v, r2t, seq_lens, sids, tidx = build_decode_inputs()
    k8, kr = qdq(k)
    v8, vr = qdq(v)
    out = run_step3_decode(q, k8, v8, r2t, seq_lens, sids, tidx)
    ref = run_step3_decode(q, kr, vr, r2t, seq_lens, sids, tidx)
    torch.testing.assert_close(
        out.float(), ref.float(), atol=WIDEN_ATOL, rtol=WIDEN_RTOL
    )


def test_step3_decode_scales():
    torch.manual_seed(2)
    q, k, v, r2t, seq_lens, sids, tidx = build_decode_inputs()
    q8, qr = qdq(q)
    k8, kr = qdq(k)
    v8, vr = qdq(v)
    q_scale, k_scale, v_scale = 1.5, 0.5, 2.0
    out8 = run_step3_decode(
        q8,
        k8,
        v8,
        r2t,
        seq_lens,
        sids,
        tidx,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    # reference: bf16 kernel on pre-scaled dequantized Q/K/V (scale semantics:
    # the tensor stores value/scale; attention runs on value = stored * scale)
    ref = run_step3_decode(
        (qr * q_scale).to(torch.bfloat16),
        (kr * k_scale).to(torch.bfloat16),
        (vr * v_scale).to(torch.bfloat16),
        r2t,
        seq_lens,
        sids,
        tidx,
    )
    torch.testing.assert_close(
        out8.float(), ref.float(), atol=FP8_ATOL * v_scale, rtol=FP8_RTOL * v_scale
    )


def test_step3_decode_bf16_regression():
    torch.manual_seed(3)
    q, k, v, r2t, seq_lens, sids, tidx = build_decode_inputs()
    out = run_step3_decode(q, k, v, r2t, seq_lens, sids, tidx)
    out_scaled = run_step3_decode(
        q, k, v, r2t, seq_lens, sids, tidx, q_scale=1.0, k_scale=1.0, v_scale=1.0
    )
    assert out.dtype == q.dtype
    torch.testing.assert_close(out, out_scaled, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# step-3 prefill (gqa-share sparse)
# ---------------------------------------------------------------------------


def run_step3_prefill(q, k, v, r2t, sids, tidx, cu, seq_lens, prefix, max_q, **kw):
    return flash_prefill_with_gqa_share_sparse(
        q=q,
        k_cache=k,
        v_cache=v,
        sink=None,
        req_to_token=r2t,
        slot_ids=sids,
        topk_idx=tidx,
        block_size_q=1,
        block_size_k=128,
        cu_seqlens=cu,
        seq_lens=seq_lens,
        prefix_lens=prefix,
        max_seqlen_q=max_q,
        **kw,
    )


def _prefill_topk_idx(cu_seqlens, seq_lens, prefix_lens, num_kv_heads, topk, block):
    # per-token (block_size_q=1) causal topk: for query at absolute position p,
    # any blocks with start <= p, sorted ascending, -1 padded.
    total_q = int(cu_seqlens[-1].item())
    tidx = torch.full(
        (num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=DEVICE
    )
    row = 0
    for b in range(len(seq_lens)):
        q_len = int(cu_seqlens[b + 1] - cu_seqlens[b])
        prefix = int(prefix_lens[b])
        for j in range(q_len):
            nb = (prefix + j) // block + 1  # blocks fully/partially before pos
            ak = min(topk, nb)
            sel = torch.randperm(nb, device=DEVICE)[:ak].sort().values
            for kh in range(num_kv_heads):
                tidx[kh, row, :ak] = sel.to(torch.int32)
            row += 1
    return tidx


def test_step3_prefill_all_fp8_vs_dequant_ref():
    torch.manual_seed(4)
    q, k, v, r2t, sids, cu, seq_lens, prefix, max_q, _ = build_prefill_inputs()
    tidx = _prefill_topk_idx(cu.cpu(), seq_lens.cpu(), prefix.cpu(), 1, 8, 128)
    q8, qr = qdq(q)
    k8, kr = qdq(k)
    v8, vr = qdq(v)
    out8 = run_step3_prefill(q8, k8, v8, r2t, sids, tidx, cu, seq_lens, prefix, max_q)
    ref = run_step3_prefill(qr, kr, vr, r2t, sids, tidx, cu, seq_lens, prefix, max_q)
    assert out8.dtype == torch.bfloat16
    torch.testing.assert_close(out8.float(), ref.float(), atol=FP8_ATOL, rtol=FP8_RTOL)


def test_step3_prefill_scales():
    torch.manual_seed(5)
    q, k, v, r2t, sids, cu, seq_lens, prefix, max_q, _ = build_prefill_inputs()
    tidx = _prefill_topk_idx(cu.cpu(), seq_lens.cpu(), prefix.cpu(), 1, 8, 128)
    q8, qr = qdq(q)
    k8, kr = qdq(k)
    v8, vr = qdq(v)
    q_scale, k_scale, v_scale = 1.5, 0.5, 2.0
    out8 = run_step3_prefill(
        q8,
        k8,
        v8,
        r2t,
        sids,
        tidx,
        cu,
        seq_lens,
        prefix,
        max_q,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    ref = run_step3_prefill(
        (qr * q_scale).to(torch.bfloat16),
        (kr * k_scale).to(torch.bfloat16),
        (vr * v_scale).to(torch.bfloat16),
        r2t,
        sids,
        tidx,
        cu,
        seq_lens,
        prefix,
        max_q,
    )
    # v_scale multiplies the fp8 kernel's quantized-P PV output, amplifying
    # the P-quantization error by the same factor vs the pre-scaled bf16 ref.
    torch.testing.assert_close(
        out8.float(), ref.float(), atol=FP8_ATOL * v_scale, rtol=FP8_RTOL * v_scale
    )


# ---------------------------------------------------------------------------
# step-1 decode indexer
# ---------------------------------------------------------------------------


def run_indexer_decode(q, k, v, r2t, seq_lens, sids, **kw):
    return flash_decode_with_topk_idx(
        q=q,
        sink=None,
        k_cache=k,
        v_cache=v,
        req_to_token=r2t,
        seq_lens=seq_lens,
        slot_ids=sids,
        max_seqlen=int(seq_lens.max().item()),
        block_size=128,
        topk=4,
        init_blocks=1,
        local_blocks=1,
        **kw,
    )


def _topk_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-row overlap of the selected (non-negative) block-id sets."""
    total, hit = 0, 0
    af, bf = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    for i in range(af.shape[0]):
        sa = set(af[i][af[i] >= 0].tolist())
        sb = set(bf[i][bf[i] >= 0].tolist())
        if not sa and not sb:
            continue
        total += max(len(sa), len(sb))
        hit += len(sa & sb)
    return hit / max(total, 1)


def test_indexer_decode_all_fp8_vs_dequant_ref():
    torch.manual_seed(6)
    q, k, v, r2t, seq_lens, sids, _ = build_decode_inputs(
        num_q_heads=1, num_kv_heads=1, seq_lens_list=(1023, 769, 513, 257)
    )
    q8, qr = qdq(q)
    k8, kr = qdq(k)
    v8, vr = qdq(v)
    o8, tidx8, _ = run_indexer_decode(q8, k8, v8, r2t, seq_lens, sids)
    oref, tidxref, _ = run_indexer_decode(qr, kr, vr, r2t, seq_lens, sids)
    assert o8.dtype == torch.bfloat16
    # o is the full (non-sparse) indexer attention output — independent of the
    # topk side-channel — so it must track the dequantized reference.
    torch.testing.assert_close(o8.float(), oref.float(), atol=FP8_ATOL, rtol=FP8_RTOL)
    # topk selection runs on QK scores that only differ by fp32-accumulation
    # noise; require near-total agreement (ties may flip an occasional block).
    assert _topk_overlap(tidx8, tidxref) >= 0.9


def test_indexer_decode_score_only_fp8():
    torch.manual_seed(7)
    q, k, _, r2t, seq_lens, sids, _ = build_decode_inputs(
        batch_size=2, num_q_heads=1, num_kv_heads=1, seq_lens_list=(1023, 769)
    )
    q8, qr = qdq(q)
    k8, kr = qdq(k)
    o8, tidx8, _ = run_indexer_decode(
        q8, k8, None, r2t, seq_lens, sids, disable_index_value=True
    )
    oref, tidxref, _ = run_indexer_decode(
        qr, kr, None, r2t, seq_lens, sids, disable_index_value=True
    )
    assert o8 is None and oref is None
    assert _topk_overlap(tidx8, tidxref) >= 0.9


# ---------------------------------------------------------------------------
# step-1 prefill indexer
# ---------------------------------------------------------------------------


def run_indexer_prefill(q, k, v, r2t, sids, cu, seq_lens, prefix, max_q, max_k, **kw):
    return flash_prefill_with_topk_index(
        q=q,
        k_cache=k,
        v_cache=v,
        sink=None,
        req_to_token=r2t,
        slot_ids=sids,
        cu_seqlens=cu,
        seq_lens=seq_lens,
        prefix_lens=prefix,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        block_size_q=1,
        block_size_k=128,
        topk=4,
        init_blocks=1,
        local_blocks=1,
        **kw,
    )


def test_indexer_prefill_all_fp8_vs_dequant_ref():
    torch.manual_seed(8)
    q, k, v, r2t, sids, cu, seq_lens, prefix, max_q, max_k = build_prefill_inputs(
        num_q_heads=1, num_kv_heads=1
    )
    q8, qr = qdq(q)
    k8, kr = qdq(k)
    v8, vr = qdq(v)
    o8, tidx8 = run_indexer_prefill(
        q8, k8, v8, r2t, sids, cu, seq_lens, prefix, max_q, max_k
    )
    oref, tidxref = run_indexer_prefill(
        qr, kr, vr, r2t, sids, cu, seq_lens, prefix, max_q, max_k
    )
    assert o8.dtype == torch.bfloat16
    torch.testing.assert_close(o8.float(), oref.float(), atol=FP8_ATOL, rtol=FP8_RTOL)
    assert _topk_overlap(tidx8, tidxref) >= 0.9


def test_dtype_contract_rejects_e5m2_q():
    q = torch.randn(2, 8, 128, dtype=torch.bfloat16, device=DEVICE).to(
        torch.float8_e5m2
    )
    k = torch.randn(256, 1, 128, dtype=torch.bfloat16, device=DEVICE).to(FP8)
    from sglang.srt.layers.attention.minimax_sparse_ops.common.utils import (
        check_sparse_kv_fp8,
    )

    with pytest.raises(AssertionError):
        check_sparse_kv_fp8(q, k, None, label="test")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
