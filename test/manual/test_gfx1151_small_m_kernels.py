"""Correctness of the gfx115x small-batch kernels against torch references.

    python3 -m pytest test/manual/test_gfx1151_small_m_kernels.py -q

Needs a gfx115x GPU (RDNA3.5, e.g. Strix Halo gfx1151); every test skips elsewhere.
"""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx115_supported

gfx115 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx115_supported()),
    reason="needs a gfx115x GPU",
)


def _dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp32 ``[N, K]`` from the skinny int4 layout (zero point 8, group 128)."""
    n, k8 = w.shape
    lanes = torch.arange(8, device=w.device)
    q = (w[:, :, None] >> ((lanes // 2) * 4 + (lanes % 2) * 16)) & 0xF
    return (q - 8).reshape(n, k8 * 8).float() * s.float().repeat_interleave(128, dim=1)


def _rand_w4(n: int, k: int):
    w = torch.randint(
        -(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32, device="cuda"
    )
    s = (torch.rand(n, k // 128, device="cuda") * 0.02 + 0.005).to(torch.bfloat16)
    return w, s


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref.float()).norm() / ref.float().norm()).item()


@gfx115
@pytest.mark.parametrize("m", [1, 5, 8, 16])
@pytest.mark.parametrize(
    "n,k", [(34816, 5120), (5120, 17408), (5120, 6144), (16384, 5120)]
)
@pytest.mark.parametrize("padded", [False, True])
def test_w4a16_small_m_gemm(m, n, k, padded):
    from sglang.kernels.ops.quantization import w4a16_small_m_gfx1151 as w4
    from sglang.kernels.ops.quantization.w4a16_skinny_triton import pad_rows

    torch.manual_seed(0)
    w, s = _rand_w4(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    wp = pad_rows(w) if padded else w
    assert w4.covered(x, wp, s)
    got = w4.w4a16_small_m_gemm(x, wp, s)
    assert _rel_err(got, x.float() @ _dequant(w, s).T) < 1e-2
    if padded:
        assert torch.equal(got, w4.w4a16_small_m_gemm(x, w, s))


@gfx115
def test_quantize_w4a16_roundtrip():
    from sglang.kernels.ops.quantization import w4a16_small_m_gfx1151 as w4

    torch.manual_seed(0)
    weight = torch.randn(4096, 5120, device="cuda", dtype=torch.bfloat16) * 0.02
    w, s = w4.quantize_w4a16(weight)
    deq = _dequant(w, s)
    assert _rel_err(deq, weight) < 0.12
    x = torch.randn(8, 5120, device="cuda", dtype=torch.bfloat16)
    assert _rel_err(w4.w4a16_small_m_gemm(x, w, s), x.float() @ deq.T) < 1e-2


@gfx115
def test_w4a16_prefill_gemm():
    from sglang.kernels.ops.quantization import w4a16_prefill_gfx1151 as pf

    torch.manual_seed(0)
    from sglang.kernels.ops.quantization.w4a16_skinny_triton import pad_rows

    w, s = _rand_w4(5120, 6144)
    x = torch.randn(300, 6144, device="cuda", dtype=torch.bfloat16)
    assert pf.covered(x, w, s)
    out = pf.w4a16_prefill_gemm(x, w, s)
    assert _rel_err(out, x.float() @ _dequant(w, s).T) < 1e-2
    assert torch.equal(out, pf.w4a16_prefill_gemm(x, pad_rows(w), s))


@gfx115
@pytest.mark.parametrize("m", [8, 40, 300])
def test_skinny_gemm_padded_rows(m):
    from sglang.kernels.ops.quantization.w4a16_skinny_triton import (
        pad_rows,
        w4a16_skinny_gemm,
    )

    torch.manual_seed(0)
    w, s = _rand_w4(96, 5120)
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    out = w4a16_skinny_gemm(x, w, s, 128)
    assert _rel_err(out, x.float() @ _dequant(w, s).T) < 1e-2
    assert torch.equal(out, w4a16_skinny_gemm(x, pad_rows(w), s, 128))


@gfx115
@pytest.mark.parametrize("prefix", [1, 37, 1024, 5000])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize(
    "hq,hkv,d,t", [(24, 4, 256, 8), (32, 4, 128, 8), (12, 2, 128, 4), (8, 8, 64, 8)]
)
def test_verify_attention(prefix, bs, hq, hkv, d, t):
    from sglang.kernels.ops.attention import verify_attention_gfx1151 as va

    torch.manual_seed(0)
    slots = bs * (prefix + t) + 16
    kc = torch.randn(slots, hkv, d, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    perm = torch.randperm(slots, device="cuda").to(torch.int32)
    r2t = torch.zeros(bs + 1, prefix + t + 8, dtype=torch.int32, device="cuda")
    rows = []
    for b in range(bs):
        r2t[b + 1, : prefix + t] = perm[b * (prefix + t) : (b + 1) * (prefix + t)]
        rows.append(b + 1)
    rpi = torch.tensor(rows, device="cuda", dtype=torch.int64)
    seq_lens = torch.full((bs,), prefix, device="cuda", dtype=torch.int32)
    q = torch.randn(bs * t, hq, d, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    assert va.covered(q, o, kc, vc, t, hq, hkv, d)
    va.verify_attention_gfx1151(q, o, kc, vc, r2t, rpi, seq_lens, t, d**-0.5)
    for b in range(bs):
        idx = r2t[b + 1, : prefix + t].long()
        k = kc[idx].float().repeat_interleave(hq // hkv, dim=1)
        v = vc[idx].float().repeat_interleave(hq // hkv, dim=1)
        for i in range(t):
            qi = q[b * t + i].float()
            sc = torch.einsum("hd,thd->ht", qi, k[: prefix + i + 1]) * d**-0.5
            ref = torch.einsum("ht,thd->hd", sc.softmax(-1), v[: prefix + i + 1])
            cos = F.cosine_similarity(
                o[b * t + i].float().flatten(), ref.flatten(), dim=0
            )
            assert cos > 0.9999


@gfx115
def test_verify_attention_rejects_strided_q():
    from sglang.kernels.ops.attention import verify_attention_gfx1151 as va

    hq, hkv, d, t = 24, 4, 256, 8
    kc = torch.randn(64, hkv, d, device="cuda", dtype=torch.bfloat16)
    qkv = torch.randn(t, (hq + 2 * hkv) * d, device="cuda", dtype=torch.bfloat16)
    q = qkv[:, : hq * d].view(-1, hq, d)  # a slice of a fused qkv buffer
    o = torch.empty(t, hq, d, device="cuda", dtype=torch.bfloat16)
    assert not va.covered(q, o, kc, kc, t, hq, hkv, d)
    assert va.covered(q.contiguous(), o, kc, kc, t, hq, hkv, d)


def _gdn_inputs(T):
    torch.manual_seed(0)
    q = F.normalize(torch.randn(1, T, 16, 128, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, T, 16, 128, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(1, T, 48, 128, device="cuda", dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, T, 48, device="cuda")).float()
    beta = torch.sigmoid(torch.randn(1, T, 48, device="cuda")).bfloat16()
    cu = torch.tensor([0, T], dtype=torch.long, device="cuda")
    return q, k, v, g, beta, cu


def _gdn_forward(q, k, v, g, beta, cu):
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

    state = torch.zeros(1, 48, 128, 128, device="cuda", dtype=torch.float32)
    idx = torch.zeros(1, dtype=torch.int32, device="cuda")
    o = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=128**-0.5,
        initial_state=state,
        initial_state_indices=idx,
        cu_seqlens=cu,
    )
    return (o[0] if isinstance(o, tuple) else o), state


@gfx115
@pytest.mark.parametrize("T", [64, 777, 2048])
def test_gdn_head_major_and_grouped_chunk_o(T, monkeypatch):
    """Qwen3.8-27B GDN shapes (16 q/k heads, 48 value heads): the head-major grids are
    bitwise equal to the default order; the grouped three-head chunk_fwd_o gives the same
    final state and output within bf16 rounding."""
    import sglang.kernels.ops.attention.fla.chunk_fwd as cf
    import sglang.kernels.ops.attention.fla.chunk_o as co
    import sglang.kernels.ops.attention.fla.wy_fast as wf

    inputs = _gdn_inputs(T)
    o_all, s_all = _gdn_forward(*inputs)  # head-major + grouped (the gfx115x default)

    monkeypatch.setattr(co, "_use_grouped_three_head", lambda *a, **kw: False)
    o_hm, s_hm = _gdn_forward(*inputs)  # head-major only
    monkeypatch.setattr(cf, "use_gfx115_kernels", lambda: False)
    monkeypatch.setattr(wf, "use_gfx115_kernels", lambda: False)
    o_ref, s_ref = _gdn_forward(*inputs)  # default order, generic chunk_fwd_o

    assert torch.equal(o_hm, o_ref) and torch.equal(s_hm, s_ref)
    assert torch.equal(s_all, s_ref)
    assert (o_all.float() - o_ref.float()).abs().max().item() < 1e-3


@gfx115
def test_gluon_small_m_falls_back_without_triton_3_7(monkeypatch):
    """With a pre-3.7 ``AMDWMMALayout`` (no register / warp bases) the Quark scheme leaves
    small-M calls to the skinny GEMM instead of failing at launch."""
    import triton.experimental.gluon.language.amd as amd

    from sglang.srt.layers.quantization.quark.schemes import quark_w4a16_int4 as qs

    class OldLayout:  # Triton 3.6 signature
        def __init__(self, version, transposed, warps_per_cta, tiles_per_warp=None):
            pass

    w, s = _rand_w4(5120, 6144)
    x = torch.randn(8, 6144, device="cuda", dtype=torch.bfloat16)
    qs._gluon_small_m_available.cache_clear()
    assert qs._gluon_small_m_available()
    assert qs._gfx115_w4a16_gemm(x, w, s) is not None
    monkeypatch.setattr(amd, "AMDWMMALayout", OldLayout)
    qs._gluon_small_m_available.cache_clear()
    try:
        assert not qs._gluon_small_m_available()
        assert qs._gfx115_w4a16_gemm(x, w, s) is None  # 8 rows: skinny GEMM
    finally:
        qs._gluon_small_m_available.cache_clear()


@gfx115
def test_verify_attention_register_spills():
    """The split kernel streams q in chunks to stay near the register file; 12 spills with
    Triton 3.7 / ROCm 7.2.4 (139 when q was held as a whole operand). A compiler change that
    brings the spills back shows up here before it shows up as a slower verify step."""
    from sglang.kernels.ops.attention import verify_attention_gfx1151 as va

    hq, hkv, d, t = 24, 4, 256, 8
    kc = torch.randn(4096, hkv, d, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(t, hq, d, device="cuda", dtype=torch.bfloat16)
    r2t = torch.arange(4096, device="cuda", dtype=torch.int32)[None]
    compiled = {}
    run = va._verify_split_kernel.run

    def capture(*args, **kwargs):
        compiled["kernel"] = run(*args, **kwargs)
        return compiled["kernel"]

    va._verify_split_kernel.run = capture
    try:
        va.verify_attention_gfx1151(
            q,
            torch.empty_like(q),
            kc,
            torch.randn_like(kc),
            r2t,
            torch.zeros(1, dtype=torch.int64, device="cuda"),
            torch.tensor([2000], device="cuda"),
            t,
            d**-0.5,
        )
    finally:
        va._verify_split_kernel.run = run
    assert compiled["kernel"].n_spills <= 32, compiled["kernel"].n_spills
