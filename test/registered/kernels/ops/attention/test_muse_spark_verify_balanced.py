"""Muse Spark target verify on the length-balanced FlyDSL kernel (full layers) and the fixed-split FlyDSL sliding
kernel, against an fp64 reference: uneven batches, 2 and 4 KV heads per rank, NaN codes in stale cache rows, and the
deferred merge with the fused norm-gate."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx95_supported(),
    reason="gfx950 FlyDSL verify kernels",
)

WINDOW = 2047
SCALE = 0.125


def _make(lens, kvh, seed, poison_window=False, page=16):
    gen = torch.Generator().manual_seed(seed)
    pages = [(n + page - 1) // page for n in lens]
    total = sum(pages) + 16
    perm = torch.randperm(total, generator=gen)
    table = torch.zeros(len(lens), max(pages), dtype=torch.int32)
    start = 0
    for i, p in enumerate(pages):
        table[i, :p] = perm[start : start + p]
        start += p
    q = (torch.randn(len(lens) * 8, kvh * 8, 64, generator=gen) * 2).to(torch.bfloat16)
    k = torch.randn(total, page, kvh, 64, generator=gen).to(torch.float8_e4m3fn)
    v = torch.randn(total, page, kvh, 64, generator=gen).to(torch.float8_e4m3fn)
    for r, n in enumerate(lens):
        if n % page:  # NaN codes past each request's length
            last = int(table[r, (n - 1) // page])
            k.view(torch.uint8)[last, n % page :] = 0x7F
            v.view(torch.uint8)[last, n % page :] = 0x7F
        if poison_window:  # and before each window (freed sliding-window slots)
            for key in range(max(n - 8 + 1 - WINDOW, 0)):
                slot = int(table[r, key // page])
                k.view(torch.uint8)[slot, key % page] = 0x7F
                v.view(torch.uint8)[slot, key % page] = 0x7F
    seq = torch.tensor(lens, dtype=torch.int64)
    return [t.cuda() for t in (q, k, v, table, seq)]


def _reference(q, k, v, table, seq, kvh, window, page=16):
    out = []
    for r in range(seq.numel()):
        n = int(seq[r])
        lo = max(n - 8 + 1 - window, 0) // page * page if window else 0
        rows = table[r, lo // page : (n + page - 1) // page].long()
        kk = k[rows].reshape(-1, kvh, 64)[: n - lo].double()
        vv = v[rows].reshape(-1, kvh, 64)[: n - lo].double()
        qq = q[r * 8 : (r + 1) * 8].double().view(8, kvh, 8, 64)
        pos = lo + torch.arange(kk.shape[0], device=q.device)
        qpos = n - 8 + torch.arange(8, device=q.device)
        vis = pos[None, :] <= qpos[:, None]
        if window:
            vis &= pos[None, :] > qpos[:, None] - window
        seen = vis.any(0)[:, None, None]
        kk, vv = torch.where(seen, kk, 0), torch.where(seen, vv, 0)
        s = torch.einsum("tgqd,kgd->tgqk", qq, kk) * SCALE
        s = s.masked_fill(~vis[:, None, None, :], float("-inf"))
        out.append(
            torch.einsum("tgqk,kgd->tgqd", torch.softmax(s, -1), vv).reshape(
                8, kvh * 8, 64
            )
        )
    return torch.cat(out)


def _rel(out, ref):
    assert torch.isfinite(out.float()).all()
    return ((out.double() - ref).norm() / ref.norm()).item()


BATCHES = [
    [33],
    [8, 47, 65, 130],
    [3077, 140000],
    [600, 70000, 3000, 21000, 701, 9, 64, 250000],
    [5000] * 64,
]


@pytest.mark.parametrize("kvh", [2, 4])
@pytest.mark.parametrize("lens", BATCHES)
def test_balanced_full(kvh, lens):
    from sglang.kernels.ops.attention.muse_verify_balanced import (
        gqa_verify_flydsl_balanced,
    )

    q, k, v, table, seq = _make(lens, kvh, seed=len(lens) + kvh)
    out = torch.empty_like(q)
    gqa_verify_flydsl_balanced(q, k, v, out, seq, table, SCALE, None, None)
    assert _rel(out, _reference(q, k, v, table, seq, kvh, 0)) < 5e-3


@pytest.mark.parametrize("kvh", [2, 4])
@pytest.mark.parametrize(
    "lens", [[33], [8, 2047, 2055, 2056, 4100], [600, 70000, 3000, 21000] * 5]
)
def test_flydsl_sliding(kvh, lens):
    from sglang.kernels.ops.attention.muse_verify_balanced import swa_verify_flydsl

    q, k, v, table, seq = _make(lens, kvh, seed=len(lens) * 3 + kvh, poison_window=True)
    out = torch.empty_like(q)
    swa_verify_flydsl(q, k, v, out, seq, table, SCALE, None, None, window=WINDOW)
    assert _rel(out, _reference(q, k, v, table, seq, kvh, WINDOW)) < 5e-3


@pytest.mark.parametrize("kvh", [2, 4])
def test_deferred_norm_gate_matches_merge(kvh):
    from sglang.kernels.ops.attention.muse_verify_balanced import (
        gqa_verify_flydsl_balanced,
    )
    from sglang.kernels.ops.attention.verify_reduce_norm_gate import (
        verify_reduce_norm_gate,
    )

    lens = [600, 70000, 3000, 21000, 701, 9, 64, 250000]
    q, k, v, table, seq = _make(lens, kvh, seed=11)
    heads = q.shape[1]
    merged = torch.empty_like(q)
    gqa_verify_flydsl_balanced(q, k, v, merged, seq, table, SCALE, None, None)
    state = gqa_verify_flydsl_balanced(
        q, k, v, None, seq, table, SCALE, None, None, defer_reduce=True
    )
    gate = torch.randn(q.shape[0], 1, heads * 64, device=q.device, dtype=torch.bfloat16)
    got = verify_reduce_norm_gate(
        (*state, None), (q.shape[0], heads * 64), torch.bfloat16, gate, 1e-5
    )
    x = merged.float()
    want = (
        x
        * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        * torch.sigmoid(gate.view_as(x).float())
    )
    torch.testing.assert_close(got.view_as(want).float(), want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("page", [32, 128])
@pytest.mark.parametrize("kvh", [2, 4])
def test_physical_pages(page, kvh):
    from sglang.kernels.ops.attention.muse_verify_balanced import (
        gqa_verify_flydsl_balanced,
        swa_verify_flydsl,
    )

    lens = [600, 70000, 3000, 21000, 701, 9, 64, 25000] * 2
    q, k, v, table, seq = _make(
        lens, kvh, seed=page + kvh, poison_window=True, page=page
    )
    out = torch.empty_like(q)
    swa_verify_flydsl(q, k, v, out, seq, table, SCALE, None, None, window=WINDOW)
    assert _rel(out, _reference(q, k, v, table, seq, kvh, WINDOW, page)) < 5e-3
    q, k, v, table, seq = _make(lens, kvh, seed=page + kvh, page=page)
    gqa_verify_flydsl_balanced(q, k, v, out, seq, table, SCALE, None, None)
    assert _rel(out, _reference(q, k, v, table, seq, kvh, 0, page)) < 5e-3
