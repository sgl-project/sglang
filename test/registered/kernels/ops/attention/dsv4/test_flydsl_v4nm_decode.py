"""In-tree FlyDSL v4 nm MLA decode (ROCm/aiter#5882 copy, gfx950) vs an fp64 reference.

Ported from ROCm/aiter#5882 ``op_tests/test_flydsl_mla_v4_decode.py`` (commit
03359c43df), trimmed to the DeepSeek-V4 shapes sglang runs: 16 / 32 / 64 / 128
local heads at one q row (TP8 / TP4 / TP2 / DP attention), 16 heads at four q
rows (grouped asm verify blocks), the fused inverse-RoPE + wo_a mxfp8 epilogue,
and grouped DSpark verify (q_kv_bounds) at 4 and 7 drafts. The asm comparison
and perf tables of the original are dropped; the assertions are kept.
Rows are packed like the live KV pool, 0xFF padding at bytes 462..511.

TODO(aiter#5882): remove once ROCm/aiter#5882 is in the aiter pin
"""

from __future__ import annotations

import itertools
import sys

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=600, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

_skip_reason = "FlyDSL v4 nm decode is gfx950-only"
if is_hip() and is_gfx95_supported():
    try:
        from sglang.kernels.ops.attention.dsv4 import flydsl_v4nm
        from sglang.kernels.ops.attention.dsv4.flydsl_v4nm import (
            mla_v4_decode as fly_v4,
        )
    except ImportError as e:  # no flydsl, or an aiter before ROCm/aiter#6377
        _skip_reason = f"in-tree FlyDSL v4 nm decode unavailable: {e}"
    else:
        _skip_reason = None
        flydsl_mla_decode_fwd_v4_nm = flydsl_v4nm.flydsl_mla_decode_fwd_v4_nm
        flydsl_mla_v4_decode_fused = flydsl_v4nm.flydsl_mla_v4_decode_fused
        flydsl_mla_v4_decode_supported = flydsl_v4nm.flydsl_mla_v4_decode_supported
        flydsl_mla_v4_grouped_verify_meta = (
            flydsl_v4nm.flydsl_mla_v4_grouped_verify_meta
        )

# CI runs this file as a script; module-level pytest.skip would escape __main__.
pytestmark = pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")

FP8, BF16, I32 = torch.float8_e4m3fn, torch.bfloat16, torch.int32
NOPE, ROPE, DV, ROW, QB = 448, 64, 512, 512, 128
PAD = NOPE + 14
WIN, RING, CRATIO = 128, 256, 128  # SWA window, ring slots per request, HCA ratio


@pytest.fixture(autouse=True)
def _cuda_default_device():
    with torch.device("cuda"):
        yield


def pack_rows(n, gen, pad=0xFF):
    """fp8 rows [n, 512] u8 (448 e4m3 + 7 e8m0 scales, each twice) + bf16 rope."""
    nope = torch.randn(n, NOPE // 64, 64, generator=gen)
    rope = torch.randn(n, ROPE, generator=gen).to(BF16)
    scale = torch.exp2(torch.ceil(torch.log2((nope.abs().amax(-1) / 448).clamp(1e-4))))
    row = torch.full((n, ROW), pad, dtype=torch.uint8)
    row[:, :NOPE] = (nope / scale[..., None]).to(FP8).view(n, NOPE).view(torch.uint8)
    e8 = (torch.log2(scale).to(torch.int32) + 127).to(torch.uint8)
    row[:, NOPE:PAD] = e8.repeat_interleave(2, dim=-1)
    return row, rope


def dequant_rows(row, rope):
    vals = row[..., :NOPE].view(FP8).double()
    sc = torch.exp2(row[..., NOPE:PAD:2].double() - 127)
    return torch.cat(
        [(vals.unflatten(-1, (7, 64)) * sc[..., None]).flatten(-2), rope.double()], -1
    )


def make_inputs(H, msq, batch, kv_len, ragged, seed=0):
    """batch sequences of msq q rows; ragged: kv lengths in [0, 2 kv_len], every 5th
    empty (2: the first one 16 kv_len)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    gcpu = torch.Generator().manual_seed(seed)
    kv_lens = [kv_len] * batch
    if ragged:
        kv_lens = torch.randint(
            0, 2 * kv_len + 1, (batch,), generator=gcpu, device="cpu"
        )
        kv_lens = [0 if b % 5 == 3 else n for b, n in enumerate(kv_lens.tolist())]
        kv_lens[0] = 16 * kv_len if ragged == 2 else kv_lens[0]
    rows = max(1024, sum(kv_lens) + 64)
    kv, kv_rope = pack_rows(rows, gen)
    kv_indptr = torch.tensor([0] + kv_lens).cumsum(0).to(I32)
    kv_indices = torch.randperm(rows, generator=gen)[: max(1, sum(kv_lens))].to(I32)
    q, q_rope = pack_rows(batch * msq * H, gen)
    return {
        "q": q.view(-1, H, ROW),
        "q_rope": q_rope.view(-1, H, ROPE),
        "kv": kv,
        "kv_rope": kv_rope,
        "qo_indptr": torch.arange(batch + 1, dtype=I32) * msq,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "sink": torch.randn(H, generator=gen),
    }, kv_lens


def run_torch(q, q_rope, kv, kv_rope, qo_indptr, kv_indptr, kv_indices, sink, **_):
    """fp64 softmax([q k^T / sqrt(512), sink]) @ kv per sequence; rows without keys NaN."""
    qd = dequant_rows(q, q_rope)
    out = torch.full(qd.shape, float("nan"), dtype=torch.float64)
    qo, ki = qo_indptr.tolist(), kv_indptr.tolist()
    for b in range(len(qo) - 1):
        rows = kv_indices[ki[b] : ki[b + 1]].long()
        if rows.numel() == 0 or qo[b + 1] == qo[b]:
            continue
        kd = dequant_rows(kv[rows], kv_rope[rows])
        s = torch.einsum("qhd,kd->qhk", qd[qo[b] : qo[b + 1]], kd) * DV**-0.5
        sk = sink.double()[None, :, None].expand(s.shape[0], -1, 1)
        p = torch.softmax(torch.cat([s, sk], -1), -1)[..., :-1]
        out[qo[b] : qo[b + 1]] = torch.einsum("qhk,kd->qhd", p, kd)
    return out


def inv_rope(ref, positions, freqs):
    T, H = ref.shape[:2]
    fc = torch.view_as_complex(freqs[positions].double().view(T, 1, ROPE // 2, 2))
    tail = torch.view_as_complex(ref[..., NOPE:].contiguous().view(T, H, ROPE // 2, 2))
    out = ref.clone()
    out[..., NOPE:] = torch.view_as_real(tail * fc.conj()).view(T, H, ROPE)
    return out


def rope_freqs(max_pos=1 << 18):
    inv = 10000.0 ** (-torch.arange(0, ROPE, 2, dtype=torch.float64) / ROPE)
    ang = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv
    return torch.stack([ang.cos(), ang.sin()], -1).flatten(1).float()


def dequant_mxfp8(xq, xs):
    T = xq.shape[0]
    x = xq.view(FP8).view(T, -1, QB).double()
    return x * torch.exp2(xs.view(torch.uint8).view(T, -1, 1).double() - 127)


def mxfp8_rel_l2(ref, xq, xs):
    blk = ref.reshape(ref.shape[0], -1, QB)
    return ((dequant_mxfp8(xq, xs) - blk).norm() / blk.norm()).item()


def rel_l2(o, ref):
    return ((o.double() - ref).norm() / ref.norm()).item()


def decode(inp, out, msq, **kw):
    rows = inp["kv"].shape[0]
    return flydsl_mla_decode_fwd_v4_nm(
        inp["q"].view(FP8),
        inp["q_rope"],
        inp["kv"].view(FP8).view(rows, 1, 1, ROW),
        inp["kv_rope"].view(rows, 1, 1, ROPE),
        out,
        inp["qo_indptr"],
        inp["kv_indptr"],
        inp["kv_indices"],
        msq,
        sink=inp["sink"],
        **kw,
    )


def graph_replay(fn, n=3, st=None):
    """Warm up eagerly on a side stream (as sglang does), capture, replay n times."""
    st = st or torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        fn()
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    return g


def bits(x):
    return x.view(torch.int16) if x.element_size() == 2 else x.view(torch.uint8)


# DeepSeek-V4 shapes: (H, msq, hints) x batches x kv lengths x ragged; sglang always
# passes a stream hint (csa / hca / swa), which only H16 msq1 plans read.
_BATCHES = (1, 9, 37, 129, 300)
_DECODE_CASES = [
    (H, msq, batch, kv_len, ragged, hint)
    for (H, msq, hints) in (
        (16, 1, ("csa", "hca", "swa")),
        (16, 4, ("hca",)),
        (32, 1, ("hca",)),
        (64, 1, ("hca",)),
        (128, 1, ("hca",)),
    )
    for batch, kv_len, ragged, hint in itertools.product(
        _BATCHES, (128, 1152), (0, 1), hints
    )
] + [
    (16, 1, 37, 4096, 0, "csa"),
    (128, 1, 129, 4096, 0, "hca"),
    # one long stream among short ones: the key-balanced grid at short runs
    (16, 1, 96, 256, 2, "hca"),
]


@pytest.mark.parametrize("H, msq, batch, kv_len, ragged, hint", _DECODE_CASES)
def test_decode(H, msq, batch, kv_len, ragged, hint):
    inp, kv_lens = make_inputs(H, msq, batch, kv_len, ragged)
    ref = run_torch(**inp)
    ok = torch.tensor(
        [b * msq + r for b, n in enumerate(kv_lens) if n for r in range(msq)],
        dtype=torch.long,
    )
    empty = torch.tensor(
        [b * msq + r for b, n in enumerate(kv_lens) if not n for r in range(msq)],
        dtype=torch.long,
    )
    out = torch.empty((batch * msq, H, DV), dtype=BF16)
    decode(inp, out, msq, hint=hint)
    assert rel_l2(out[ok], ref[ok]) < 3e-3
    assert torch.isnan(out[empty].float()).all(), "kv_len == 0 rows must be NaN"
    base = out.clone()
    p = dict(inp, q=inp["q"].clone(), kv=inp["kv"].clone())
    for k in ("q", "kv"):  # padding bytes are never read
        p[k][..., PAD:] = torch.randint(
            0, 256, p[k][..., PAD:].shape, dtype=torch.uint8
        )
    decode(p, out, msq, hint=hint)
    assert torch.equal(bits(out[ok]), bits(base[ok]))
    out.zero_()
    graph_replay(lambda: decode(inp, out, msq, hint=hint))
    assert torch.equal(bits(out), bits(base)), "graph replay"


_FUSED_CASES = list(
    itertools.product(
        (
            (16, 4, 1152),
            (16, 128, 659),
            (16, 0, 128),
            (32, 128, 659),
            (128, 128, 659),
            (128, 4, 1152),
        ),
        (3, 40, 100, 512),
        (0, 1),
    )
)


@pytest.mark.parametrize("shape, T, ragged", _FUSED_CASES)
def test_decode_fused(shape, T, ragged):
    H, compress_ratio, kv_len = shape
    inp, kv_lens = make_inputs(H, 1, T, kv_len, ragged)
    del inp["qo_indptr"]
    inp["positions"] = (
        65536
        + torch.randint(
            0, 4096, (T,), generator=torch.Generator().manual_seed(2), device="cpu"
        ).cuda()
    )
    inp["freqs"] = rope_freqs()
    ref = inv_rope(
        run_torch(**inp, qo_indptr=torch.arange(T + 1, dtype=I32)),
        inp["positions"],
        inp["freqs"],
    )
    ok = torch.tensor([t for t, n in enumerate(kv_lens) if n], dtype=torch.long)
    empty = torch.tensor([t for t, n in enumerate(kv_lens) if not n], dtype=torch.long)
    res = {}

    def fused():
        res["x"] = flydsl_mla_v4_decode_fused(**inp, compress_ratio=compress_ratio)
        return res["x"]

    q8, s8 = [x.view(torch.uint8).clone() for x in fused()]
    assert mxfp8_rel_l2(ref[ok], q8[ok], s8[ok]) < 0.05
    assert not (q8[ok] & 0x7F).eq(0x7F).any(), "NaN codes in xq"
    assert ((q8[empty] & 0x7F) == 0x7F).all() and (s8[empty] == 0xFF).all()
    graph_replay(fused)
    assert torch.equal(res["x"][0].view(torch.uint8), q8)
    assert torch.equal(res["x"][1], s8)


def make_dspark(num_draft, kind, requests, seed=0, H=16):
    """One DSpark verify step on a pool [SWA ring: R x RING rows][compressed rows]: the
    drafts of request r end at positions around the SWA window / block commits; per
    token (sglang's streams): its window + (hca) its committed rows (p + 1) // 128."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    base = [6, 10, 126, 127, 128, 131, 133, 135, 255, 256, 260, 4000, 70000, 131000]
    last = [base[r % len(base)] + 7 * (r // len(base)) for r in range(requests)]
    pos = [p - num_draft + 1 + d for p in last for d in range(num_draft)]
    ncomp = [(p + 1) // CRATIO if kind == "hca" else 0 for p in last]
    swa_pages = requests * RING
    kv, kv_rope = pack_rows(swa_pages + max(1, sum(ncomp)), gen)
    comp = torch.randperm(max(1, sum(ncomp)), generator=gen, device="cuda").tolist()
    lists = [comp[sum(ncomp[:r]) : sum(ncomp[: r + 1])] for r in range(requests)]
    tail_pages = torch.full((len(pos), max(1, max(ncomp))), -1, dtype=I32)
    tail_len, toks = [], []
    for t, p in enumerate(pos):
        r = t // num_draft
        c = (p + 1) // CRATIO if kind == "hca" else 0
        tail_len.append(c)
        tail_pages[t, :c] = torch.tensor(lists[r][:c], dtype=I32)
        swa = [r * RING + x % RING for x in range(max(0, p - WIN + 1), p + 1)]
        toks.append(swa + [x + swa_pages for x in lists[r][:c]])
    q, q_rope = pack_rows(len(pos) * H, gen)
    hca = kind == "hca"
    return {
        "q": q.view(-1, H, ROW),
        "q_rope": q_rope.view(-1, H, ROPE),
        "kv": kv,
        "kv_rope": kv_rope,
        "sink": torch.randn(H, generator=gen, device="cuda"),
        "meta": (
            torch.tensor([t // num_draft for t in range(len(pos))], dtype=I32),
            torch.tensor(pos),
            torch.tensor(tail_len, dtype=I32) if hca else None,
            tail_pages if hca else None,
        ),
        "swa_pages": swa_pages,
        "qo_indptr": torch.arange(len(pos) + 1, dtype=I32),
        "kv_indptr": torch.tensor([0] + [len(x) for x in toks]).cumsum(0).to(I32),
        "kv_indices": torch.tensor(list(itertools.chain(*toks)), dtype=I32),
    }


def grouped_meta_ref(slot, pos, tail_len, tail_pages, win, ring_stride, swa_pages, nd):
    """Torch reference of flydsl_mla_v4_grouped_verify_meta."""
    slot, pos = slot.tolist(), pos.tolist()
    tl_ = tail_len.tolist() if tail_len is not None else [0] * len(pos)
    idx, kvp, bnd = [], [0], []
    for g in range(len(pos) // nd):
        last = g * nd + nd - 1
        p, c = pos[last], tl_[last]
        n = min(p + 1, win + nd - 1)
        if c:
            idx += [
                x + swa_pages if x >= 0 else -1 for x in tail_pages[last, :c].tolist()
            ]
        idx += [
            slot[last] * ring_stride + (p - n + 1 + i) % ring_stride for i in range(n)
        ]
        s0 = p - n + 1
        for t in range(g * nd, last + 1):
            bnd.append(
                [0, tl_[t], c + max(0, pos[t] - win + 1 - s0), c + pos[t] - s0 + 1]
            )
        kvp.append(len(idx))
    return (
        torch.tensor(idx, dtype=I32),
        torch.tensor(kvp, dtype=I32),
        torch.tensor(bnd, dtype=I32),
    )


@pytest.mark.parametrize(
    "kind, num_draft, requests, epi",
    list(itertools.product(("hca", "swa"), (4, 7), (14, 32, 72), ("bf16", "mxfp8"))),
)
def test_decode_grouped(kind, num_draft, requests, epi):
    d = make_dspark(num_draft, kind, requests, seed=num_draft + requests)
    N, H = d["q"].shape[:2]
    ref = run_torch(**d)
    idx, kvp, qop, bnd = flydsl_mla_v4_grouped_verify_meta(
        *d["meta"],
        win=WIN,
        ring_stride=RING,
        swa_pages=d["swa_pages"],
        num_draft=num_draft,
    )
    r_idx, r_kvp, r_bnd = grouped_meta_ref(
        *d["meta"], WIN, RING, d["swa_pages"], num_draft
    )
    L = int(r_kvp[-1])
    assert torch.equal(kvp, r_kvp)
    assert torch.equal(bnd, r_bnd)
    assert torch.equal(idx[:L], r_idx)
    assert torch.equal(qop, torch.arange(requests + 1, dtype=I32) * num_draft)
    grp = dict(d, qo_indptr=qop, kv_indptr=kvp, kv_indices=idx)
    # a reversed interval (x1 < x0) is empty
    b_rev, b_emp = bnd.clone(), bnd.clone()
    b_rev[3] = torch.tensor([L + 7, 1, 97, 2], dtype=I32)
    b_emp[3] = 0
    hint = "hca" if kind == "hca" else "swa"
    if epi == "bf16":
        out_g, out_t = [torch.empty((N, H, DV), dtype=BF16) for _ in range(2)]

        def grouped(b=bnd):
            return decode(grp, out_g, num_draft, hint=hint, q_kv_bounds=b)

        grouped()
        decode(d, out_t, 1, hint=hint)
        assert rel_l2(out_g, ref) < 3e-3
        assert rel_l2(out_t, ref) < 3e-3
        g0 = out_g.clone()
        e0 = grouped(b_emp).clone()
        assert torch.isnan(e0[3].float()).all()
        assert torch.equal(bits(grouped(b_rev)), bits(e0))
        out_g.zero_()
        graph_replay(grouped)
        assert torch.equal(bits(out_g), bits(g0))
        return
    freqs = rope_freqs()
    positions = d["meta"][1]
    ref_f = inv_rope(ref, positions, freqs)
    fa = (d["q"], d["q_rope"], d["kv"], d["kv_rope"])
    cr = 128 if kind == "hca" else 0
    res = {}

    def fused_grouped(b=bnd):
        res["x"] = flydsl_mla_v4_decode_fused(
            *fa,
            kvp,
            idx,
            d["sink"],
            positions,
            freqs,
            cr,
            qo_indptr=qop,
            max_seqlen_q=num_draft,
            q_kv_bounds=b,
        )
        return res["x"]

    xt = flydsl_mla_v4_decode_fused(
        *fa, d["kv_indptr"], d["kv_indices"], d["sink"], positions, freqs, cr
    )
    assert mxfp8_rel_l2(ref_f, *xt) < 0.05
    q8, s8 = [x.view(torch.uint8).clone() for x in fused_grouped()]
    assert mxfp8_rel_l2(ref_f, q8, s8) < 0.05
    e0 = [x.view(torch.uint8).clone() for x in fused_grouped(b_emp)]
    assert (e0[1][3] == 0xFF).all()
    assert all(
        torch.equal(x.view(torch.uint8), y) for x, y in zip(fused_grouped(b_rev), e0)
    )
    graph_replay(fused_grouped)
    assert torch.equal(res["x"][0].view(torch.uint8), q8)
    assert torch.equal(res["x"][1], s8)


def test_contracts():
    """Unsupported pairs rejected; B = 0 is a no-op; > 4096 merge groups run unsplit;
    graphs sharing one stream's workspace replay interleaved; a graph captured on a
    fresh stream takes pre-zeroed protocol slots (no memset in the graph)."""
    for H, msq in ((32, 2), (64, 4), (16, 3), (8, 1)):
        assert not flydsl_mla_v4_decode_supported(H, msq)
    for H in (16, 32, 64, 128):
        assert flydsl_mla_v4_decode_supported(H, 1)
    assert flydsl_mla_v4_decode_supported(16, 4)
    assert flydsl_mla_v4_decode_supported(16, 7, grouped=True)
    inp, _ = make_inputs(16, 1, 1, 128, False)
    empty = dict(
        inp,
        q=inp["q"][:0],
        q_rope=inp["q_rope"][:0],
        qo_indptr=inp["qo_indptr"][:1],
        kv_indptr=inp["kv_indptr"][:1],
    )
    decode(empty, torch.empty((0, 16, DV), dtype=BF16), 1)
    B = 4097
    assert fly_v4._plan_for(16, 1, B, "bf16", None, 256, False).S > 1
    assert fly_v4._plan(16, 1, B).S == 1
    inp, kv_lens = make_inputs(16, 2, B, 300, True)
    inp = dict(
        inp, q=inp["q"][::2].contiguous(), q_rope=inp["q_rope"][::2].contiguous()
    )
    inp["qo_indptr"] = torch.arange(B + 1, dtype=I32)
    out = torch.empty((B, 16, DV), dtype=BF16)
    decode(inp, out, 1)
    seqs = [b for b in range(0, B, 97) if kv_lens[b]]
    sub = dict(inp)
    sub["q"], sub["q_rope"] = [inp[k][seqs] for k in ("q", "q_rope")]
    sub["qo_indptr"] = torch.arange(len(seqs) + 1, dtype=I32)
    kvi = inp["kv_indptr"].tolist()
    sub["kv_indptr"] = torch.tensor([0] + [kv_lens[b] for b in seqs]).cumsum(0).to(I32)
    sub["kv_indices"] = torch.cat(
        [inp["kv_indices"][kvi[b] : kvi[b + 1]] for b in seqs]
    )
    assert rel_l2(out[seqs], run_torch(**sub)) < 3e-3
    cases = []
    for batch in (3, 37, 200):
        inp, kv_lens = make_inputs(16, 1, batch, 659, True, seed=batch)
        out = torch.empty((batch, 16, DV), dtype=BF16)
        decode(inp, out, 1)
        cases.append((inp, out, out.clone()))
    st = torch.cuda.Stream()
    graphs = [
        graph_replay(lambda i=i, o=o: decode(i, o, 1), 1, st) for i, o, _ in cases
    ]
    for n in range(3000):
        graphs[n % 3].replay()
    torch.cuda.synchronize()
    assert all(torch.equal(bits(o), bits(r)) for _, o, r in cases)
    inp, out, ref = cases[1]
    spare = fly_v4._HDR_SPARE[out.device.index].data_ptr()
    # high-priority pool: a default-priority Stream() may reuse a handle that
    # already owns a workspace
    s3, g = torch.cuda.Stream(priority=-1), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s3):
        decode(inp, out, 1)
    with torch.cuda.stream(s3):
        assert fly_v4._workspace(out.device).hdr.data_ptr() == spare
    out.zero_()
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), bits(ref))


def _aiter_v4_decode():
    try:
        from aiter.ops import flydsl as fly

        fly.flydsl_mla_v4_decode_supported
        return fly
    except (ImportError, AttributeError):
        return None


def test_runtime_source_selection():
    """runtime prefers aiter's copy, falls back to the in-tree one; the env knobs
    force one copy or turn FlyDSL off."""
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime
    from sglang.srt.environ import envs

    aiter_fly = _aiter_v4_decode()
    expect = {
        "auto": aiter_fly or flydsl_v4nm,
        "aiter": aiter_fly,
        "intree": flydsl_v4nm,
    }
    try:
        for source, fly in expect.items():
            runtime._flydsl_v4_decode.cache_clear()
            with envs.SGLANG_DSV4_FLYDSL_DECODE_SOURCE.override(source):
                assert runtime._flydsl_v4_decode() is fly, source
        runtime._flydsl_v4_decode.cache_clear()
        with envs.SGLANG_DSV4_FLYDSL_DECODE.override(False):
            assert runtime._flydsl_v4_decode() is None
            assert not runtime.flydsl_decode_supported(16)
    finally:
        runtime._flydsl_v4_decode.cache_clear()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
