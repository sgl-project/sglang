"""Muse Spark full-attention FP8 extend on the gfx950 Gluon kernel against a torch reference over the same FP8 Q
and dequantized paged cache: requests with and without a cached prefix, query lengths and contexts off the 16-token
query block and 128-key tile grids (partial blocks, diagonal and tail tiles), 2 and 4 KV heads (TP8 / TP4)."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)

F8 = torch.float8_e4m3fn
SCALE = 0.125


def _inputs(new_ctx, kv_heads, seed=0):
    torch.manual_seed(seed)
    lens = [n + c for n, c in new_ctx]
    pages = [(n + 15) // 16 for n in lens]
    nb = sum(pages) + 8
    kc = (torch.randn(nb, 16, kv_heads, 64, device="cuda") * 2).to(F8)
    vc = torch.randn(nb, 16, kv_heads, 64, device="cuda").to(F8)
    table = torch.zeros(len(lens), max(pages), dtype=torch.int32, device="cuda")
    perm = torch.randperm(nb, device="cuda").int()
    start = 0
    for i, p in enumerate(pages):
        table[i, :p] = perm[start : start + p]
        start += p
    q = torch.randn(sum(n for n, _ in new_ctx), kv_heads * 8, 64, device="cuda")
    qd = (q.abs().amax() / 448.0).view(1)
    cu = torch.tensor([0] + [n for n, _ in new_ctx], device="cuda").cumsum(0)
    return (
        (q / qd).to(F8),
        kc,
        vc,
        cu.to(torch.int32),
        torch.tensor(lens, device="cuda"),
        table,
        qd,
        torch.tensor([0.75], device="cuda"),
        torch.tensor([1.25], device="cuda"),
    )


def _reference(q8, kc, vc, cu, lens, table, qd, kd, vd):
    out = torch.empty(q8.shape, device="cuda", dtype=torch.float32)
    kv_heads = kc.shape[2]
    for i, n in enumerate(lens.tolist()):
        lo, hi = cu[i].item(), cu[i + 1].item()
        pages = table[i, : (n + 15) // 16].long()
        k = kc[pages].float().reshape(-1, kv_heads, 64)[:n] * kd
        v = vc[pages].float().reshape(-1, kv_heads, 64)[:n] * vd
        k, v = k.repeat_interleave(8, dim=1), v.repeat_interleave(8, dim=1)
        q = q8[lo:hi].float() * qd
        s = torch.einsum("qhd,khd->hqk", q, k) * SCALE
        pos = torch.arange(hi - lo, device="cuda") + n - (hi - lo)
        s = s.masked_fill(
            torch.arange(n, device="cuda")[None, None, :] > pos[None, :, None],
            float("-inf"),
        )
        out[lo:hi] = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v)
    return out


@pytest.mark.parametrize("kv_heads", [2, 4])
def test_extend_matches_reference(kv_heads):
    from sglang.kernels.ops.attention.muse_extend_gluon import (
        muse_extend_applies,
        muse_extend_attention,
    )

    # (new tokens, cached prefix): a prefix-free prompt, partial query blocks, contexts off the 128-key tiles.
    new_ctx = [(37, 0), (200, 3000), (130, 17000), (256, 1)]
    q8, kc, vc, cu, lens, table, qd, kd, vd = _inputs(new_ctx, kv_heads)
    assert muse_extend_applies(q8, kc)
    out = torch.empty(q8.shape, device="cuda", dtype=torch.bfloat16)
    muse_extend_attention(q8, kc, vc, out, cu, lens, table, qd, kd, vd, SCALE)
    ref = _reference(q8, kc, vc, cu, lens, table, qd, kd, vd)
    for i in range(len(new_ctx)):
        lo, hi = cu[i].item(), cu[i + 1].item()
        # FP8 P in the PV MFMA: a few percent of relative L2 error, as the unified kernel's FP8 path.
        err = (out[lo:hi].float() - ref[lo:hi]).norm() / ref[lo:hi].norm()
        assert err < 0.04, (i, err.item())
