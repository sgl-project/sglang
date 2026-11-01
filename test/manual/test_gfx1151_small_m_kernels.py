"""Correctness of the gfx1151 small-batch kernels against torch references.

    python3 -m pytest test/manual/test_gfx1151_small_m_kernels.py -q

Needs a gfx1151 GPU (Strix Halo); every test skips elsewhere.
"""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx1151_supported

gfx1151 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1151_supported()),
    reason="needs a gfx1151 GPU",
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


@gfx1151
@pytest.mark.parametrize("m", [1, 5, 8, 16])
@pytest.mark.parametrize(
    "n,k", [(34816, 5120), (5120, 17408), (5120, 6144), (16384, 5120)]
)
@pytest.mark.parametrize("padded", [False, True])
def test_w4a16_small_m_gemm(m, n, k, padded):
    from sglang.kernels.ops.quantization import w4a16_small_m_gfx1151 as w4

    torch.manual_seed(0)
    w, s = _rand_w4(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    wp = w4.pad_rows(w) if padded else w
    assert w4.covered(x, wp, s)
    got = w4.w4a16_small_m_gemm(x, wp, s)
    assert _rel_err(got, x.float() @ _dequant(w, s).T) < 1e-2
    if padded:
        assert torch.equal(got, w4.w4a16_small_m_gemm(x, w, s))


@gfx1151
def test_quantize_w4a16_roundtrip():
    from sglang.kernels.ops.quantization import w4a16_small_m_gfx1151 as w4

    torch.manual_seed(0)
    weight = torch.randn(4096, 5120, device="cuda", dtype=torch.bfloat16) * 0.02
    w, s = w4.quantize_w4a16(weight)
    deq = _dequant(w, s)
    assert _rel_err(deq, weight) < 0.12
    x = torch.randn(8, 5120, device="cuda", dtype=torch.bfloat16)
    assert _rel_err(w4.w4a16_small_m_gemm(x, w, s), x.float() @ deq.T) < 1e-2


@gfx1151
def test_w4a16_prefill_gemm():
    from sglang.kernels.ops.quantization import w4a16_prefill_gfx1151 as pf

    torch.manual_seed(0)
    from sglang.kernels.ops.quantization.w4a16_small_m_gfx1151 import pad_rows

    w, s = _rand_w4(5120, 6144)
    x = torch.randn(300, 6144, device="cuda", dtype=torch.bfloat16)
    assert pf.covered(x, w, s)
    out = pf.w4a16_prefill_gemm(x, w, s)
    assert _rel_err(out, x.float() @ _dequant(w, s).T) < 1e-2
    assert torch.equal(out, pf.w4a16_prefill_gemm(x, pad_rows(w), s))


@gfx1151
@pytest.mark.parametrize("m", [8, 40, 300])
def test_skinny_gemm_padded_rows(m):
    from sglang.kernels.ops.quantization.w4a16_skinny_triton import w4a16_skinny_gemm
    from sglang.kernels.ops.quantization.w4a16_small_m_gfx1151 import pad_rows

    torch.manual_seed(0)
    w, s = _rand_w4(96, 5120)
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    out = w4a16_skinny_gemm(x, w, s, 128)
    assert _rel_err(out, x.float() @ _dequant(w, s).T) < 1e-2
    assert torch.equal(out, w4a16_skinny_gemm(x, pad_rows(w), s, 128))


@gfx1151
@pytest.mark.parametrize("n,k", [(248320, 5120), (6144, 5120), (5120, 17408)])
def test_bf16_small_m_gemm(n, k):
    from sglang.kernels.ops.gemm import bf16_small_m_gfx1151 as bf

    torch.manual_seed(0)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    assert bf.covered(x, w)
    assert _rel_err(bf.bf16_small_m_gemm(x, w), F.linear(x.float(), w.float())) < 1e-2


@gfx1151
@pytest.mark.parametrize("prefix", [1, 37, 1024, 5000])
@pytest.mark.parametrize("bs", [1, 2])
def test_verify_attention(prefix, bs):
    from sglang.kernels.ops.attention import verify_attention_gfx1151 as va

    torch.manual_seed(0)
    hq, hkv, d, t = 24, 4, 256, 8
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
    assert va.covered(q, kc, t, hq, hkv, d)
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
