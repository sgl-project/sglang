"""Parity tests for MSA (fmha_sm100) all-fp8 sparse attention (fp8 attn-GEMM mode).

No upstream fp8 test exists for fmha_sm100's cutlass sparse-decode path, so
this is the reference check: MSA fp8 vs the Triton fp8 sparse kernels on the
same quantized tensors, plus fp8-vs-bf16 error bounds and CUDA-graph
capture/replay bit-exactness of the fp8 decode.

Both fp8 paths quantize the unnormalized softmax P to e4m3 before the PV MMA,
but their QK/accumulation orders differ, so MSA-fp8 vs Triton-fp8 tolerances
cover two independent P-quantization errors (~1e-1 worst-case elementwise).

Requires SM100 + fmha_sm100 (first run JIT-compiles the fp8 variants).
Run: pytest python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_msa_fp8_parity.py -v
"""

import pytest
import torch

from sglang.srt.layers.attention.minimax_sparse_ops.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
    build_msa_decode_cg_plan,
    msa_available,
    msa_sparse_decode_main,
    msa_sparse_prefill_main,
    update_msa_decode_cg_meta,
)
from sglang.srt.layers.attention.minimax_sparse_ops.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn
P = 128  # sparse block == page size
# two independent e4m3-P quantizations (MSA + Triton)
X_ATOL = 1e-1
X_RTOL = 1e-1

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not msa_available(),
    reason="requires SM100 + fmha_sm100",
)


def qdq(x):
    x8 = x.to(FP8)
    return x8, x8.to(torch.bfloat16)


def build_paged_inputs(seq_lens_list, num_q_heads=16, num_kv_heads=1, head_dim=128):
    """Page-aligned pool: each logical 128-token page maps to one physical page
    (contiguous 128 slots), as MSA's page-table builder requires."""
    batch = len(seq_lens_list)
    pages_per_req = [(s + P - 1) // P for s in seq_lens_list]
    max_pages = max(pages_per_req)
    total_pages = batch * max_pages
    max_slots = total_pages * P
    page_perm = torch.randperm(total_pages, device=DEVICE)
    req_to_token = torch.zeros(batch, max_pages * P, dtype=torch.int32, device=DEVICE)
    for b in range(batch):
        for p in range(max_pages):
            phys = page_perm[b * max_pages + p]
            req_to_token[b, p * P : (p + 1) * P] = phys * P + torch.arange(
                P, device=DEVICE
            )
    k = torch.randn(
        max_slots, num_kv_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    v = torch.randn(
        max_slots, num_kv_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    slot_ids = torch.arange(batch, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    return k, v, req_to_token, slot_ids, seq_lens


def make_topk_idx(seq_lens_list, num_kv_heads, rows_per_req, topk):
    """Sorted-ascending causal-safe topk over each request's block count.
    rows_per_req[b] = number of query rows for request b (1 for decode)."""
    total_rows = sum(rows_per_req)
    tidx = torch.full(
        (num_kv_heads, total_rows, topk), -1, dtype=torch.int32, device=DEVICE
    )
    row = 0
    for b, s in enumerate(seq_lens_list):
        nb = (s + P - 1) // P
        for _ in range(rows_per_req[b]):
            ak = min(topk, nb)
            sel = torch.randperm(nb, device=DEVICE)[:ak].sort().values
            for kh in range(num_kv_heads):
                tidx[kh, row, :ak] = sel.to(torch.int32)
            row += 1
    return tidx


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def _decode_setup(seq_lens_list=(1023, 769, 513, 130), topk=4):
    """topk*P = 512 < 1023/769: exercises the degenerate partial-last-block
    masking (the historical unsorted-topk failure shape)."""
    torch.manual_seed(0)
    batch = len(seq_lens_list)
    k, v, r2t, sids, seq_lens = build_paged_inputs(seq_lens_list)
    q = torch.randn(batch, 16, 128, dtype=torch.bfloat16, device=DEVICE)
    tidx = make_topk_idx(seq_lens_list, 1, [1] * batch, topk)
    return q, k, v, r2t, sids, seq_lens, tidx


def test_msa_fp8_decode_vs_triton_fp8():
    q, k, v, r2t, sids, seq_lens, tidx = _decode_setup()
    q8, qr = qdq(q)
    k8, kr = qdq(k)
    v8, vr = qdq(v)
    o_msa = msa_sparse_decode_main(
        q8, k8, v8, tidx, r2t, sids, seq_lens, block_size_k=P
    )
    o_triton = flash_decode_with_gqa_share_sparse(
        q8, None, k8, v8, r2t, seq_lens, sids, P, tidx
    )
    assert o_msa.dtype == torch.bfloat16
    torch.testing.assert_close(
        o_msa.float(), o_triton.float(), atol=X_ATOL, rtol=X_RTOL
    )
    # bf16 MSA on the dequantized tensors bounds the pure fp8-kernel error
    o_bf16 = msa_sparse_decode_main(
        qr, kr, vr, tidx, r2t, sids, seq_lens, block_size_k=P
    )
    err = (o_msa.float() - o_bf16.float()).abs().mean()
    ref = o_bf16.float().abs().mean()
    assert err / ref < 0.06, f"mean rel err {err/ref:.4f} too high vs bf16 MSA"


def test_msa_fp8_decode_scales():
    q, k, v, r2t, sids, seq_lens, tidx = _decode_setup()
    q8, qr = qdq(q)
    k8, kr = qdq(k)
    v8, vr = qdq(v)
    q_scale, k_scale, v_scale = 1.5, 0.5, 2.0
    o = msa_sparse_decode_main(
        q8,
        k8,
        v8,
        tidx,
        r2t,
        sids,
        seq_lens,
        block_size_k=P,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    # reference: bf16 MSA on pre-scaled dequantized Q/K/V
    ref = msa_sparse_decode_main(
        (qr * q_scale).to(torch.bfloat16),
        (kr * k_scale).to(torch.bfloat16),
        (vr * v_scale).to(torch.bfloat16),
        tidx,
        r2t,
        sids,
        seq_lens,
        block_size_k=P,
    )
    torch.testing.assert_close(
        o.float(), ref.float(), atol=X_ATOL * v_scale, rtol=X_RTOL * v_scale
    )


def test_msa_fp8_decode_capture_replay_bitexact():
    q, k, v, r2t, sids, seq_lens, tidx = _decode_setup()
    q8, _ = qdq(q)
    k8, _ = qdq(k)
    v8, _ = qdq(v)
    bs = q8.shape[0]
    nb_max = r2t.shape[1] // P
    plan = build_msa_decode_cg_plan(
        16, 1, P, tidx.shape[-1], bs, device=q8.device, is_fp8=True
    )
    kv_indices = torch.zeros(bs * nb_max, dtype=torch.int32, device=DEVICE)
    update_msa_decode_cg_meta(
        plan, kv_indices, r2t, sids, seq_lens, P, tidx.shape[-1], 16, 1
    )

    def run():
        return msa_sparse_decode_main(
            q8,
            k8,
            v8,
            tidx,
            r2t,
            sids,
            seq_lens,
            block_size_k=P,
            kv_indices=kv_indices,
            plan=plan,
        )

    # eager warmups (also pays the fp8 JIT) on a side stream, as capture does
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            o_eager = run()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o_graph = run()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(o_graph, o_eager), "fp8 MSA decode capture/replay not bit-exact"


# ---------------------------------------------------------------------------
# prefill (cutlass short-q and cute long-q branches)
# ---------------------------------------------------------------------------


def _prefill_setup(seq_lens_list, prefix_lens_list, topk=4):
    torch.manual_seed(1)
    k, v, r2t, sids, seq_lens = build_paged_inputs(seq_lens_list)
    q_lens = [s - p for s, p in zip(seq_lens_list, prefix_lens_list)]
    total_q = sum(q_lens)
    q = torch.randn(total_q, 16, 128, dtype=torch.bfloat16, device=DEVICE)
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(q_lens, device=DEVICE).cumsum(0)
    prefix = torch.tensor(prefix_lens_list, dtype=torch.int32, device=DEVICE)
    # causal-valid per-token topk (block_size_q == 1)
    tidx = torch.full((1, total_q, topk), -1, dtype=torch.int32, device=DEVICE)
    row = 0
    for b, (s, p) in enumerate(zip(seq_lens_list, prefix_lens_list)):
        for j in range(s - p):
            nb = (p + j) // P + 1
            ak = min(topk, nb)
            sel = torch.randperm(nb, device=DEVICE)[:ak].sort().values
            tidx[0, row, :ak] = sel.to(torch.int32)
            row += 1
    return q, k, v, r2t, sids, cu, seq_lens, prefix, tidx, max(q_lens)


@pytest.mark.parametrize(
    "seq_lens,prefix_lens,branch",
    [
        ((530, 700), (500, 680), "cutlass_short_q"),  # qo <= 32
        ((513, 769), (0, 257), "cute_long_q"),  # qo > 32
    ],
    ids=["cutlass_short_q", "cute_long_q"],
)
def test_msa_fp8_prefill_vs_triton_fp8(seq_lens, prefix_lens, branch):
    q, k, v, r2t, sids, cu, seq_lens_t, prefix, tidx, max_q = _prefill_setup(
        list(seq_lens), list(prefix_lens)
    )
    q8, _ = qdq(q)
    k8, _ = qdq(k)
    v8, _ = qdq(v)
    o_msa = msa_sparse_prefill_main(
        q8, k8, v8, tidx, r2t, sids, cu, seq_lens_t, prefix, block_size_k=P
    )
    o_triton = flash_prefill_with_gqa_share_sparse(
        q=q8,
        k_cache=k8,
        v_cache=v8,
        sink=None,
        req_to_token=r2t,
        slot_ids=sids,
        topk_idx=tidx,
        block_size_q=1,
        block_size_k=P,
        cu_seqlens=cu,
        seq_lens=seq_lens_t,
        prefix_lens=prefix,
        max_seqlen_q=max_q,
    )
    assert o_msa.dtype == torch.bfloat16
    torch.testing.assert_close(
        o_msa.float(), o_triton.float(), atol=X_ATOL, rtol=X_RTOL
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
