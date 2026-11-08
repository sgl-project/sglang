"""Muse Spark full-layer short extend on aiter's 3D kernel with 32-row q-blocks against ``unified_attention`` itself:
bit-identical output wherever ``unified_attention`` picks its 3D kernel (one and several requests, a one-token request
in the batch, contexts off the 64-key tile grid, 2 and 4 KV heads, 16- and 64-token pages, BF16 and FP8 Q), and no
launch where it picks the 2D kernel (short context, many query rows) or the batch is all decode."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)

F8 = torch.float8_e4m3fn
SCALE = 0.125


def _inputs(new_ctx, kv_heads, page, q_fp8, seed=0):
    torch.manual_seed(seed)
    lens = [n + c for n, c in new_ctx]
    pages = [(n + page - 1) // page for n in lens]
    nb = sum(pages) + 8
    kc = (torch.randn(nb, page, kv_heads, 64, device="cuda") * 2).to(F8)
    vc = torch.randn(nb, page, kv_heads, 64, device="cuda").to(F8)
    # a few spare columns, as the served page table is wider than the longest request
    table = torch.zeros(len(lens), max(pages) + 3, dtype=torch.int32, device="cuda")
    perm = torch.randperm(nb, device="cuda").to(torch.int32)
    start = 0
    for i, p in enumerate(pages):
        table[i, :p] = perm[start : start + p]
        start += p
    news = [n for n, _ in new_ctx]
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.cumsum(torch.tensor(news, device="cuda"), 0)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q = torch.randn(sum(news), 8 * kv_heads, 64, device="cuda").to(torch.bfloat16)
    q_descale = None
    if q_fp8:
        q_descale = (q.abs().amax().float() / 448.0).clamp_min(1e-12).view(1)
        q = (q.float() / q_descale).to(F8)
    kd = torch.tensor([0.7], device="cuda")
    vd = torch.tensor([1.3], device="cuda")
    return q, kc, vc, cu, seq_lens, table, max(news), q_descale, kd, vd


def _run_both(new_ctx, kv_heads, page, q_fp8):
    from aiter.ops.triton.attention.unified_attention import unified_attention

    from sglang.kernels.ops.attention.muse_extend_short_3d import (
        muse_extend_short_3d,
    )

    q, kc, vc, cu, seq_lens, table, max_q, qd, kd, vd = _inputs(
        new_ctx, kv_heads, page, q_fp8
    )
    max_k = table.shape[1] * page
    shape = (q.shape[0], q.shape[1], 64)
    ref = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    unified_attention(
        q=q,
        k=kc,
        v=vc,
        out=ref,
        cu_seqlens_q=cu,
        seqused_k=seq_lens,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softmax_scale=SCALE,
        causal=True,
        window_size=(-1, -1),
        block_table=table,
        softcap=0.0,
        q_descale=qd,
        k_descale=kd,
        v_descale=vd,
    )
    out = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    took = muse_extend_short_3d(
        q, kc, vc, out, cu, seq_lens, table, max_q, max_k, SCALE, qd, kd, vd
    )
    return took, out, ref


@pytest.mark.parametrize("q_fp8", [False, True])
@pytest.mark.parametrize("page", [16, 64])
@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize(
    "new_ctx",
    [
        [(9, 4000)],
        [(64, 30000)],
        [(200, 60000)],
        [(255, 1000)],
        [(2, 700)],
        [(10, 3000), (50, 600), (3, 20000)],
        [(1, 9000), (17, 5000), (1, 800)],
    ],
)
def test_matches_unified_attention(new_ctx, kv_heads, page, q_fp8):
    took, out, ref = _run_both(new_ctx, kv_heads, page, q_fp8)
    assert took
    assert not ref.isnan().any()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize(
    "new_ctx",
    [
        [(100, 200)],  # context <= 512 keys: 2D kernel
        [(3000, 5000)],  # more q-blocks than programs: 2D kernel
        [(1, 4000), (1, 9000)],  # all decode
    ],
)
def test_declines_where_unified_attention_does_not_use_3d(new_ctx):
    took, out, _ = _run_both(new_ctx, 2, 16, False)
    assert not took
    assert out.isnan().all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
