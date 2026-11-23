"""Muse Spark SM90 fused attention vs fp32 torch: pfa (extend / mixed batches) and attn_srv (chain / tree verify)."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="1-gpu-large")

F8, W = torch.float8_e4m3fn, 2048
# verify tree slots: 0-1-2-3 is a chain, 4..7 branch off it
PARENT = [-1, 0, 1, 2, 1, 4, 0, 2]
# past lengths around 64-key tiles and the window
LENS = [0, 1, 63, 64, 65, 2039, 2040, 2047, 2048, 2100, 9000, 33000]
sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90",
)


def _rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


def _tree():
    anc, dep = [], []
    for i, p in enumerate(PARENT):
        anc.append(1 << i | (anc[p] if p >= 0 else 0))
        dep.append(dep[p] + 1 if p >= 0 else 0)
    return anc, dep


def _ref(x, reqs, pt, loc, kc, vc, window, scale, cs):
    """fp32 attention per request (past, new-token position offsets, new-token visibility) + expected cache."""
    kr, vr = kc.float(), vc.float()
    pos = torch.cat([p + d for p, d, _ in reqs])
    y = _rms(x[:, :18]).bfloat16().float()  # 16 q | 2 k | 2 v | 16 gate heads
    if window:
        a, b = y.unflatten(-1, (32, 2)).unbind(-1)
        c, s = cs[pos, None].chunk(2, -1)
        y = torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2).bfloat16()
    y = y.float().to(F8).float()
    kr[loc], vr[loc] = y[:, 16:], x[:, 18:20].to(F8).float()
    o, t = [], 0
    for r, (p, d, A) in enumerate(reqs):
        e, i = len(d), pos[t : t + len(d), None]
        ks = torch.cat([pt[r, :p].long(), loc[t : t + e]])
        m = torch.cat([torch.ones(e, p, dtype=torch.bool), A], 1)
        if window:
            m &= i - torch.cat([torch.arange(p), i[:, 0]]) < window
        s = torch.einsum("tgjd,kgd->gjtk", y[t : t + e, :16].view(e, 2, 8, 64), kr[ks])
        s = (s * scale).masked_fill(~m, -torch.inf).softmax(-1)
        o.append(torch.einsum("gjtk,kgd->tgjd", s, vr[ks]).flatten(1, 2))
        t += e
    return _rms(torch.cat(o).bfloat16().float()) * x[:, 20:].sigmoid(), kr, vr


def _run(past, new, tails, window, tree, sink=False, x=None):
    """pfa over the extend requests, attn_srv over the last `tails` 8-token verify requests (glue.py's mixed
    layout), vs fp32; also checks the cache writes, untouched rows, determinism and CUDA graph replay."""
    from sglang.srt.layers.muse_spark.fused import attn_srv, pfa

    torch.manual_seed(0)
    B, T, R = len(past), sum(new), len(past) - tails
    seq, n, qsf = [p + e for p, e in zip(past, new)], sum(new[:R]), 2.0 if sink else 1.0
    N = sum(seq) + 64  # 64 rows are never written
    kf, vf, perm = torch.randn(N, 2, 64), torch.randn(N, 2, 64), torch.randperm(N)
    # drawn even when x is given: the same cache for the same lengths
    x0, u = torch.randn(T, 36, 64), torch.randn(64)
    x = x0 if x is None else x
    if sink:  # every q leans to key 0 (NoPE layers): rows dominated by the oldest key
        x[:, :16] += 4 * u
    pt = torch.zeros(B, max(seq), dtype=torch.int32)
    for b, (p, e, k) in enumerate(zip(past, new, seq)):
        pt[b, :k] = perm[sum(seq[:b]) :][:k]
        if p:  # heavy V at key 0 and at each row's oldest visible / first hidden key
            hv = pt[b, [0, *range(max(p - W, 0), max(p + e - W + 1, 0))]].long()
            vf[hv] = 32 * torch.randn(len(hv), 2, 64)
            if sink:
                kf[hv[0]] = _rms(u)
    kc, vc, qkv = kf.to(F8), vf.to(F8), x.flatten(1).bfloat16()
    loc = torch.cat([pt[b, p:k] for b, (p, k) in enumerate(zip(past, seq))]).long()
    anc, dep = _tree()
    tr = [torch.tensor([v] * tails, dtype=torch.uint8) for v in (anc, dep)]
    A8 = (torch.tensor(anc)[:, None] >> torch.arange(8) & 1).bool()
    d8, tril = torch.tensor(dep), torch.ones(1024, 1024, dtype=torch.bool).tril()
    if not tree:
        tr, A8, d8 = (None, None), tril[:8, :8], torch.arange(8)
    reqs = [
        (p, torch.arange(e), tril[:e, :e]) if b < R else (p, d8, A8)
        for b, (p, e) in enumerate(zip(past, new))
    ]
    pos = torch.cat([p + d for p, d, _ in reqs])
    # kernel's angle: fp32 pos * (1 / theta ** (2i / 64))
    inv = 1.0 / 1e4 ** (torch.arange(0, 64, 2).float() / 64)
    f = torch.arange(max(seq) + 8)[:, None] * inv
    cs, sl = torch.cat([f.cos(), f.sin()], -1), torch.tensor(seq).int()
    ref, kr, vr = _ref(
        qkv.float().view(T, 36, 64), reqs, pt, loc, kc, vc, window, 0.125 * qsf, cs
    )
    cfg = dict(window=window, rope_theta=1e4, qk_scale_factor=qsf, sm_scale=0.125)

    def go():
        out = torch.full((T + 8, 1024), 7.0, dtype=torch.bfloat16)
        if n:
            cu = torch.tensor([0] + new[:R]).cumsum(0).int()
            kvb = sum(min(k, e + window + 128) if window else k for e, k in zip(new, seq[:R]))  # fmt: skip
            pfa.forward(qkv[:n], loc[:n], kc, vc, pt[:R], sl[:R], cu, kvb, 0.125, qsf, 1e-5, out[:n], window, cs if window else None, pos[:n] if window else None)  # fmt: skip
        if tails:
            o = out[n:T].view(tails, 8, -1)
            attn_srv.kernel_fn(qkv[n:].view(tails, 8, -1), sl[R:], loc[n:], kc, vc, pt[R:], cfg, tr, o)  # fmt: skip
        return out

    out = go()
    # FP8 P costs a few percent per element; a wrong key block moves the mean far more
    o32 = out[:T].float().view(T, 16, 64)
    torch.testing.assert_close(o32, ref, atol=0.25, rtol=0.1)
    assert (o32 - ref).abs().mean() < 0.04
    # new K rows may be 1 E4M3 step off (bf16 RoPE rounding); other rows untouched
    torch.testing.assert_close(kc.float(), kr, atol=0.125, rtol=0.125)
    assert torch.equal(vc.float(), vr) and bool((out[T:] == 7).all())
    assert torch.equal(go(), out)
    if not n:  # verify runs inside decode CUDA graphs
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o2 = go()
        g.replay()
        assert torch.equal(o2, out)
    return out[:T], x


# (past, new, verify tails, window, sink): prefix lengths at 128-key chunk / window edges, mixed extend + tree
# verify batches, a sink-dominated NoPE batch
PFA = [([0, 2600], [40, 30], 0, W, False), ([500, 3000], [64, 100], 0, 0, True)]
PFA += [([0, 1, 127, 128, 2047, 2048, 3001], [1, 127, 129, 64, 2, 300, 1000], 0, w, False) for w in (W, 0)]  # fmt: skip
PFA += [([100, 2600, 2047, 5000, 70], [300, 50, 8, 8, 8], 3, w, False) for w in (W, 0)]
# (verify batch, window, tree, sink): the batch picks the cluster size (sliding) / combine width (full);
# 1 long request splits over every SM, [200000, 0] leaves the short one on a single CTA
SRV = [(B, w, B in (2, 20, 32), not w and B == 16) for B in (1, 2, 16, 20, 28, 32) for w in (W, 0)]  # fmt: skip


@sm90
@pytest.mark.parametrize("past,new,tails,window,sink", PFA)
def test_pfa_extend_and_mixed(past, new, tails, window, sink):
    with torch.device("cuda"):
        _run(past, new, tails, window, tails > 0, sink)


@sm90
@pytest.mark.parametrize("B,window,tree,sink", SRV)
def test_attn_srv_verify(B, window, tree, sink):
    past = [LENS[(3 * i + B) % 12] for i in range(B)]
    if B < 3:
        past = [[70000], [200000, 0]][B - 1]
    with torch.device("cuda"):
        _run(past, [8] * B, B, window, tree, sink)


@sm90
@pytest.mark.parametrize("window", [W, 0])
def test_tree_verify_equals_chain(window):
    """Tree node i == the last row of a chain verify over i's root path (same past, so the same split)."""
    anc, dep = _tree()
    paths = [[j for j in range(8) if anc[i] >> j & 1] for i in range(8)]
    with torch.device("cuda"):
        tree, x = _run([3000] * 8, [8] * 8, 8, window, True)
        x = x.view(8, 8, 36, 64)
        x = torch.stack([x[i, p + [j for j in range(8) if j not in p]] for i, p in enumerate(paths)])  # fmt: skip
        chain, _ = _run([3000] * 8, [8] * 8, 8, window, False, x=x.flatten(0, 1))
    tree, chain = tree.view(8, 8, -1), chain.view(8, 8, -1)
    for i, d in enumerate(dep):
        if paths[i] == list(range(d + 1)):  # same keys in the same slots
            assert torch.equal(tree[i, i], chain[i, d])
        else:  # same P, only the fp32 sum order differs: <= 1 bf16 step before the norm
            torch.testing.assert_close(
                tree[i, i], chain[i, d], atol=1 / 64, rtol=1 / 64
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
