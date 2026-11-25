"""Fused NVFP4 W4A16 MoE vs fp64 torch and Marlin; dynamic guests; CUDA graph replay."""

import functools
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe import muse_spark_nvfp4_w4a16_moe as m
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=420, stage="base-b-kernel-unit", runner_config="1-gpu-large")

E, H, G, D = 32, 4096, "cuda", torch.float64  # the only supported shape
V = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
sm90 = pytest.mark.skipif(not m.is_supported(H, H, E, 8), reason="SM90")


@functools.cache
def _experts(n, rank=0):
    torch.manual_seed(n + rank)
    q = torch.randint(0, 256, (E, n, H // 2), dtype=torch.uint8, device=G)
    s = torch.randint(64, 127, (E, n, H // 16), dtype=torch.uint8, device=G)
    s[s < 66] = 0  # e4m3 0 or [2, 448]
    s = s.view(torch.float8_e4m3fn)
    return q, s, m.repack_nvfp4_experts(q, s, gate_up=n > H)


@functools.cache
def _gs():  # BF16-exact global scales of 256 experts: gate-up, down
    return 2.0 ** -(12 + torch.arange(512, device=G) % 3).float().view(2, 256)


@functools.cache
def _ranks():  # 8 ranks: global expert 32 r + i is rank r's local expert i
    w = [[_experts.__wrapped__(n, r)[2] for r in range(1, 8)] for n in (2 * H, H)]
    return (*(torch.cat([_experts(n)[2], *t]) for n, t in zip((2 * H, H), w)), *_gs())


@functools.cache
def _guests():
    w = _ranks()
    peer = [[t[32 * r :].data_ptr() for t in w] for r in range(8)]
    stage = tuple(t.new_empty((n, *t.shape[1:])) for t, n in zip(w, (2, 2, 4)))
    gsrc = torch.full((2,), -1, dtype=torch.int32, device=G)
    return m.DynamicGuests(stage, gsrc, torch.tensor(peer, device=G))


@functools.cache
def _marlin():  # Hopper's stock NVFP4 MoE (Marlin) on the same weights
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_nvfp4_layer_for_marlin,
    )

    (q13, s13, _), (q2, s2, _) = _experts(2 * H), _experts(H)
    k = "w13_weight w2_weight w13_weight_scale w2_weight_scale w13_weight_scale_2"
    layer = SimpleNamespace(**dict(zip(k.split(), (q13, q2, s13, s2, _gs()[0, :E]))))
    layer.__dict__.update(w2_weight_scale_2=_gs()[1, :E], params_dtype=torch.bfloat16)
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.intermediate_size_per_partition, layer.num_local_experts = H, E
    layer.global_num_experts, layer._expert_map = E, None
    prepare_moe_nvfp4_layer_for_marlin(layer, group_size=16)
    return layer


def _inputs(M, n, skew, seed):
    torch.manual_seed(seed)
    ids = torch.rand(M, n, device=G).argsort(1)[:, :8]
    if skew == 1:  # experts 0-5 take every token, odd experts above 5 none
        odd = 6 + 2 * torch.rand(M, n // 2 - 3, device=G).argsort(1)[:, :2]
        ids = torch.cat([torch.arange(6, device=G).expand(M, 6), odd], 1)
    elif skew == 2:  # repeated ids: experts 0 and n - 1 take two slots of every token
        ids[:, :2], ids[:, -2:] = 0, n - 1
    x = torch.randn(M, H, device=G, dtype=torch.bfloat16)
    tw, nw = torch.rand(M, 8, device=G), (torch.rand(H, device=G) + 0.5).bfloat16()
    return x, ids.int(), tw, nw


def _w(q, s, gs):  # e2m1 x e4m3 per 16 x global
    c = torch.stack((q & 15, q >> 4), -1).flatten(1).long()
    return torch.cat([V, -V]).to(G, D)[c] * s.float().repeat_interleave(16, 1) * gs


def _ref(x, ids, tw, nw, exact=False):
    """-> (out, sum_j tw_j |y_j|); h rounded to BF16 as in the kernels unless exact."""
    (q13, s13, _), (q2, s2, _), (g13, g2) = _experts(2 * H), _experts(H), _gs()
    ref = torch.zeros(x.shape, dtype=D, device=G)
    mag = torch.zeros_like(ref)
    for e in ids[ids >= 0].unique().tolist():
        t, j = (ids == e).nonzero(as_tuple=True)
        g, u = (x[t].to(D) @ _w(q13[e], s13[e], g13[e]).T).chunk(2, -1)
        h = g * g.sigmoid() * u
        y = (h if exact else h.bfloat16().to(D)) @ _w(q2[e], s2[e], g2[e]).T * nw.to(D)
        y *= tw[t, j, None] * (y.square().mean(-1, True) + 1e-8).rsqrt()
        ref.index_add_(0, t, y)
        mag.index_add_(0, t, y.abs())
    return ref, mag


@sm90  # split-K tiles to 512, 1024-token chunks, SP_MIN, BIG_MIN, 8192-token big chunks
@pytest.mark.parametrize(
    "M,sp,skew",
    [(1, 0, 0), (100, 0, 1), (512, 0, 0), (513, 0, 1), (1025, 0, 0), (65, 1, 0)]
    + [(300, 1, 1), (2048, 1, 0), (2049, 1, 1), (8192, 1, 0), (8193, 1, 1)]
    + [(16384, 1, 0)]
    # repeated ids: more routes per expert than tokens in the large-M route lists
    + [(100, 0, 2), (2049, 1, 2), (8193, 1, 2)],
)
def test_matches_reference(M, sp, skew):
    (w13, w2), (g13, g2) = (_experts(n)[2] for n in (2 * H, H)), _gs()[:, :E]
    x, ids, tw, nw = _inputs(M, E, skew, M)
    if not sp:  # -1: remote (token 1: all)
        ids[torch.rand(M, 8, device=G) < 0.25] = ids[1:2] = -1
    a, outs = (ids, tw, w13, g13, w2, g2, nw.float()), []
    for _ in range(2):  # NaN sentinel row after the outputs
        r = x.new_full((8 * M + 1 if sp else M + 1, H), float("nan"))
        if sp:  # one rank: route j of t -> r[j, t]
            ptrs = dict(shard_ptrs=[x.data_ptr()], recv_ptrs=[r.data_ptr()])
            m.nvfp4_w4a16_moe_sp(*a, shard=x, **ptrs)
            outs.append(m.nvfp4_w4a16_moe_sp_finalize(r, tw, 0, M))
        else:
            outs.append(m.nvfp4_w4a16_moe_with_post_norm(x, *a, out=r[:-1]))
        assert r[-1].isnan().all()
    assert torch.equal(*outs)  # deterministic
    ref, mag = _ref(x, ids, tw, nw)
    # each route's y and the output are rounded to BF16 (2^-9 each); h's BF16 rounding
    # spreads through W2 as noise of ~2^-9 of the row's typical |y|, ~5 sigma at the
    # max over millions of outputs (a wrong or missing route is of the order of |y|)
    tol = 2**-7 * mag + 2**-6 * mag.mean(-1, keepdim=True)
    assert ((outs[0].to(D) - ref).abs() <= tol).all()


@sm90
@pytest.mark.parametrize("M", [1, 200])
def test_not_less_accurate_than_marlin(M):
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import (
        fused_post_expert_norm_weighted_sum as finalize,
    )
    from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts

    (w13, w2), (g13, g2) = (_experts(n)[2] for n in (2 * H, H)), _gs()[:, :E]
    x, ids, tw, nw = _inputs(M, E, 0, M)
    y, lid = MuseSpark1xFp4Experts._marlin_expert_outputs(_marlin(), x, tw, ids)
    stock = finalize(y, tw, nw, 1e-8, True, slot_expert_ids=lid)
    ours = m.nvfp4_w4a16_moe_with_post_norm(x, ids, tw, w13, g13, w2, g2, nw.float())
    ref = _ref(x, ids, tw, nw, exact=True)[0]
    err = lambda o: (o.to(D) - ref).norm().item()
    # Marlin also rounds gate / up to BF16 before the SiLU
    assert err(ours) <= err(stock) <= 2**-7 * ref.norm().item()


def _all_ranks(x, ids, tw, nw, guests_on):
    """8 SP ranks on one GPU; with ``guests_on`` busy ranks' experts run as dynamic guests."""
    (w13, w2, g13, g2), M = _ranks(), x.shape[0]
    xs, ptrs, guests = x.view(8, -1, H), lambda t: [p.data_ptr() for p in t], 0
    res = x.new_full((8, 8, M // 8, H), float("nan"))  # recv: [rank, route j, row]
    for r in range(8):
        own, dg = slice(32 * r, 32 * r + 32), None
        local = torch.where(ids // 32 == r, ids % 32, -1)
        if guests_on:  # local ids 32 + j
            dg, dur = _guests(), torch.arange(1, 9, dtype=torch.int32, device=G)
            local = m.sp_dynamic_guest_ids(ids, r, dg.gsrc, dur)
            guests += int((local >= 32).sum())
        a = (local, tw, w13[own], g13[own], w2[own], g2[own], nw)
        kw = dict(shard_ptrs=ptrs(xs), recv_ptrs=ptrs(res), dg=dg)
        m.nvfp4_w4a16_moe_sp(*a, shard=xs[r], **kw)
    return res, guests


@sm90  # guests from DG_MIN, big kernel from BIG_MIN
@pytest.mark.parametrize("M,skew", [(776, 1), (2048, 1), (2056, 0), (8200, 1)])
def test_guests_same_math(M, skew):
    x, ids, tw, nw = _inputs(M, 256, skew, M)
    run = lambda on: _all_ranks(x, ids, tw, nw.float(), on)
    (base, _), (moved, guests), (again, _) = run(False), run(True), run(True)
    assert torch.equal(moved, again)  # deterministic
    if skew:  # positive control: work did move
        assert guests
    # each route's row, NaN-initialized, written once, same bits wherever it ran
    assert torch.equal(base, moved)


@sm90
@pytest.mark.parametrize("M", [1, 256])
def test_cuda_graph_replay(M):
    """Decode on rank 0 of 8, captured once, replayed on new inputs: as eager."""
    (w13, w2, g13, g2), local = _ranks(), lambda i: torch.where(i < E, i, -1)
    x, ids, tw, nw = _inputs(M, 256, 0, 0)
    ids = local(ids)
    a = (x, ids, tw, w13[:E], g13[:E], w2[:E], g2[:E], nw.float())
    run = lambda out=None: m.nvfp4_w4a16_moe_with_post_norm(*a, out=out)
    out, graph = run(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out)
    for seed, skew in ((1, 0), (2, 1), (3, 0)):
        v = _inputs(M, 256, skew, seed)
        for t, u in zip((x, ids, tw), (v[0], local(v[1]), v[2])):
            t.copy_(u)
        graph.replay()
        assert torch.equal(out, run())


_PARTITION_SCRIPT = textwrap.dedent(
    """
    import sys
    import torch
    from flashinfer.green_ctx import split_device_green_ctx
    sys.path.insert(0, sys.argv[1])
    import test_muse_spark_nvfp4_w4a16_moe as t

    (w13, w2), (g13, g2) = (t._experts(n)[2] for n in (2 * t.H, t.H)), t._gs()[:, : t.E]

    def run(M):  # the small-M kernel, then the large-M one (two 8192-token launches)
        x, ids, tw, nw = t._inputs(M, t.E, 0, M)
        a = (ids, tw, w13, g13, w2, g2, nw.float())
        if M <= t.m.MAX_TOKENS:
            return t.m.nvfp4_w4a16_moe_with_post_norm(x, *a)
        r = x.new_zeros((8 * M, t.H))
        ptrs = dict(shard_ptrs=[x.data_ptr()], recv_ptrs=[r.data_ptr()])
        t.m.nvfp4_w4a16_moe_sp(*a, shard=x, **ptrs)
        return r

    full = [run(M) for M in (256, 8193)]
    streams, resources = split_device_green_ctx(torch.device("cuda", 0), 1, 16)
    torch.cuda.synchronize()
    with torch.cuda.stream(streams[0]):
        part = [run(M) for M in (256, 8193)]
    streams[0].synchronize()
    assert all(torch.equal(a, b) for a, b in zip(full, part))
    print("partition SMs", resources[0].sm.smCount)
    """
)


@sm90
def test_kernels_complete_on_sm_partition():
    """Both kernels launch one CTA per SM. On a green-context SM partition only part
    of that grid is resident at a time: no CTA may wait on work owned by a CTA that
    has not started. A subprocess with a timeout turns a hang into a failure."""
    pytest.importorskip("flashinfer.green_ctx")
    if torch.cuda.get_device_properties(0).multi_processor_count < 64:
        pytest.skip("needs a device much larger than the 16-SM partition")
    try:
        res = subprocess.run(
            [sys.executable, "-c", _PARTITION_SCRIPT, os.path.dirname(__file__)],
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the W4A16 MoE kernels did not complete on a 16-SM partition")
    if res.returncode != 0 and "Failed to split device" in res.stderr:
        pytest.skip("green contexts are not available")
    assert res.returncode == 0, res.stderr[-4000:]


@sm90
def test_big_workspace_holds_padded_chunk():
    """A prefill chunk that is not a multiple of 128 or of tp: SP pads it to tp * ceil(chunk / tp)
    tokens, and the large-M kernel writes 64 counters for every started 128-token unit."""
    chunk = 2 * m._BIG_MAXT + 1
    m.init_runtime(torch.device(G, 0), chunk)
    ucnt = m._big_workspace(0)["ucnt"]
    for tp in (1, 2, 4, 8):
        M = tp * -(-chunk // tp)
        assert M <= m._BIG_MAXT and ucnt.numel() >= -(-M // 128) * 64


def test_wide_ids_stay_skipped():
    """int64 ids are narrowed to int32 for the kernels; one that would wrap into a
    valid local id (2**32 + e -> e) must stay outside [0, 32 + guests)."""
    wide = [2**32, 2**32 + 3, -(2**32), -(2**32) + 1, 2**31, -(2**31) - 1, 2**63 - 1]
    ids = torch.tensor([wide + [5], [0, 1, 31, 32, 33, -1, 7, 2]], dtype=torch.int64)
    out = m._local_ids(ids)
    valid = (ids >= -1) & (ids < m._BIG_NE)
    assert out.dtype == torch.int32 and torch.equal(out[valid], ids[valid].int())
    assert bool(((out[~valid] < 0) | (out[~valid] >= m._BIG_NE)).all())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
