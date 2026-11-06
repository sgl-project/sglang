"""Muse Spark full-attention FP8 extend on the gfx950 Gluon kernel against a torch reference over the same FP8 Q
and dequantized paged cache: requests with and without a cached prefix, query lengths and contexts off the 16-token
query block and 128-key tile grids (partial blocks, diagonal and tail tiles), 2 and 4 KV heads (TP8 / TP4), 16- and
128-token pages, and the 2047-token sliding window (NaN codes in the keys before each window); full layers also on the FlyDSL
kernel."""

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


def _inputs(new_ctx, kv_heads, seed=0, page=16, window=0):
    torch.manual_seed(seed)
    lens = [n + c for n, c in new_ctx]
    pages = [(n + page - 1) // page for n in lens]
    nb = sum(pages) + 8
    kc = (torch.randn(nb, page, kv_heads, 64, device="cuda") * 2).to(F8)
    vc = torch.randn(nb, page, kv_heads, 64, device="cuda").to(F8)
    table = torch.zeros(len(lens), max(pages), dtype=torch.int32, device="cuda")
    perm = torch.randperm(nb, device="cuda").int()
    start = 0
    for i, p in enumerate(pages):
        table[i, :p] = perm[start : start + p]
        start += p
    if window:  # keys no query of the request sees can sit in freed slots
        for i, (n, c) in enumerate(new_ctx):
            for key in range(max(c - window + 1, 0)):
                slot = int(table[i, key // page])
                kc.view(torch.uint8)[slot, key % page] = 0x7F
                vc.view(torch.uint8)[slot, key % page] = 0x7F
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


def _reference(q8, kc, vc, cu, lens, table, qd, kd, vd, window=0):
    out = torch.empty(q8.shape, device="cuda", dtype=torch.float32)
    kv_heads, page = kc.shape[2], kc.shape[1]
    for i, n in enumerate(lens.tolist()):
        lo, hi = cu[i].item(), cu[i + 1].item()
        pages = table[i, : (n + page - 1) // page].long()
        k = kc[pages].float().reshape(-1, kv_heads, 64)[:n] * kd
        v = vc[pages].float().reshape(-1, kv_heads, 64)[:n] * vd
        k, v = k.repeat_interleave(8, dim=1), v.repeat_interleave(8, dim=1)
        q = q8[lo:hi].float() * qd
        s = torch.einsum("qhd,khd->hqk", q, k) * SCALE
        pos = torch.arange(hi - lo, device="cuda") + n - (hi - lo)
        key = torch.arange(n, device="cuda")
        hidden = key[None, :] > pos[:, None]
        if window:
            hidden |= key[None, :] <= pos[:, None] - window
            seen = (~hidden).any(0)[:, None, None]
            k, v = torch.where(seen, k, 0), torch.where(seen, v, 0)
            s = torch.einsum("qhd,khd->hqk", q, k) * SCALE
        s = s.masked_fill(hidden[None], float("-inf"))
        out[lo:hi] = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v)
    return out


@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize("kernel", ["gluon", "flydsl"])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_extend_matches_reference(kv_heads, kernel, page):
    from sglang.kernels.ops.attention.muse_extend_flydsl import (
        muse_extend_attention_fly,
    )
    from sglang.kernels.ops.attention.muse_extend_gluon import muse_extend_applies
    from sglang.kernels.ops.attention.muse_extend_gluon import (
        muse_extend_attention as gluon,
    )

    muse_extend_attention = muse_extend_attention_fly if kernel == "flydsl" else gluon

    # (new tokens, cached prefix): a prefix-free prompt, partial query blocks, contexts off the 128-key tiles.
    new_ctx = [(37, 0), (200, 3000), (130, 17000), (256, 1)]
    q8, kc, vc, cu, lens, table, qd, kd, vd = _inputs(new_ctx, kv_heads, page=page)
    assert muse_extend_applies(q8, kc)
    out = torch.empty(q8.shape, device="cuda", dtype=torch.bfloat16)
    muse_extend_attention(q8, kc, vc, out, cu, lens, table, qd, kd, vd, SCALE)
    ref = _reference(q8, kc, vc, cu, lens, table, qd, kd, vd)
    for i in range(len(new_ctx)):
        lo, hi = cu[i].item(), cu[i + 1].item()
        # FP8 P in the PV MFMA: a few percent of relative L2 error, as the unified kernel's FP8 path.
        err = (out[lo:hi].float() - ref[lo:hi]).norm() / ref[lo:hi].norm()
        assert err < 0.04, (i, err.item())


@pytest.mark.parametrize("page", [16, 128])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_sliding_window_extend(kv_heads, page):
    from sglang.kernels.ops.attention.muse_extend_gluon import muse_extend_attention

    window = 2047
    # windows that do not bind, start mid-tile, and cover only the tail of a long prefix
    new_ctx = [(37, 0), (200, 2100), (130, 17000), (2048, 3000), (16, 2047)]
    q8, kc, vc, cu, lens, table, qd, kd, vd = _inputs(
        new_ctx, kv_heads, page=page, window=window
    )
    out = torch.empty(q8.shape, device="cuda", dtype=torch.bfloat16)
    muse_extend_attention(
        q8, kc, vc, out, cu, lens, table, qd, kd, vd, SCALE, window=window
    )
    ref = _reference(q8, kc, vc, cu, lens, table, qd, kd, vd, window=window)
    assert torch.isfinite(out.float()).all()
    for i in range(len(new_ctx)):
        lo, hi = cu[i].item(), cu[i + 1].item()
        err = (out[lo:hi].float() - ref[lo:hi]).norm() / ref[lo:hi].norm()
        assert err < 0.04, (i, err.item())
