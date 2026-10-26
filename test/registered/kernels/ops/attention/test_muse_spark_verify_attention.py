"""Muse Spark 8-token target-verify kernels (gfx950, head_dim 64, GQA 8, FP8 KV in 16-token pages) and the DFlash
draft block split-KV, against a torch reference over the dequantized paged cache."""

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


def _paged_inputs(bs, kv_heads=2, seed=0):
    torch.manual_seed(seed)
    nb = 2048
    kc = (torch.randn(nb, 16, kv_heads, 64, device="cuda") * 2).to(F8)
    vc = (torch.randn(nb, 16, kv_heads, 64, device="cuda") * 2).to(F8)
    seq = torch.randint(300, 3000, (bs,), device="cuda") + 8
    table = torch.randint(0, nb, (bs, 200), device="cuda", dtype=torch.int32)
    q = torch.randn(bs * 8, kv_heads * 8, 64, device="cuda").to(torch.bfloat16)
    kd = torch.tensor([0.75], device="cuda")
    vd = torch.tensor([1.25], device="cuda")
    return q, kc, vc, seq.to(torch.int64), table, kd, vd


def _reference(q, kc, vc, seq, table, kd, vd, window):
    out = torch.empty_like(q, dtype=torch.float32)
    kv_heads = kc.shape[2]
    for i, n in enumerate(seq.tolist()):
        pages = table[i, : (n + 15) // 16].long()
        k = (kc[pages].float().reshape(-1, kv_heads, 64)[:n] * kd).repeat_interleave(
            8, dim=1
        )
        v = (vc[pages].float().reshape(-1, kv_heads, 64)[:n] * vd).repeat_interleave(
            8, dim=1
        )
        qi = q[i * 8 : (i + 1) * 8].float()
        s = torch.einsum("qhd,khd->hqk", qi, k) * SCALE
        qpos = torch.arange(n - 8, n, device="cuda")[:, None]
        kpos = torch.arange(n, device="cuda")[None, :]
        mask = kpos <= qpos
        if window:
            mask &= kpos > qpos - window
        s = s.masked_fill(~mask, float("-inf"))
        out[i * 8 : (i + 1) * 8] = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v)
    return out


@pytest.mark.parametrize("bs", [1, 4, 12, 32])
@pytest.mark.parametrize("window", [0, 2047])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_gqa_verify_splitkv(bs, window, kv_heads):
    from sglang.kernels.ops.attention.gqa_verify_splitkv import gqa_verify_splitkv

    q, kc, vc, seq, table, kd, vd = _paged_inputs(bs, kv_heads=kv_heads, seed=bs)
    cu = torch.arange(0, bs * 8 + 1, 8, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    gqa_verify_splitkv(q, kc, vc, out, cu, seq, table, SCALE, window, kd, vd)
    torch.testing.assert_close(
        out.float(),
        _reference(q, kc, vc, seq, table, kd, vd, window),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("bs", [1, 8, 48])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_gqa_verify_splitkv_2kv(bs, kv_heads, fp8):
    from sglang.kernels.ops.attention.gqa_verify_splitkv_2kv import (
        gqa_verify_splitkv_2kv,
    )

    q, kc, vc, seq, table, kd, vd = _paged_inputs(bs, kv_heads=kv_heads, seed=100 + bs)
    out = torch.empty_like(q)
    gqa_verify_splitkv_2kv(q, kc, vc, out, seq, table, SCALE, kd, vd, fp8=fp8)
    ref = _reference(q, kc, vc, seq, table, kd, vd, 0)
    if fp8:
        # FP8 Q / P: 3-4.5% relative L2 error with isolated larger elements (e4m3 P), as the prefill FP8 path.
        assert (out.float() - ref).norm() / ref.norm() < 0.06
        torch.testing.assert_close(out.float(), ref, rtol=0.5, atol=0.25)
    else:
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("bs", [1, 8, 12])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_gqa_verify_flydsl(bs, kv_heads):
    flydsl = pytest.importorskip("flydsl")
    from sglang.kernels.ops.attention.muse_verify_flydsl import gqa_verify_flydsl

    q, kc, vc, seq, table, kd, vd = _paged_inputs(bs, kv_heads=kv_heads, seed=200 + bs)
    out = torch.empty_like(q)
    gqa_verify_flydsl(q, kc, vc, out, seq, table, SCALE, kd, vd)
    # exact BF16 QK / PV: BF16 rounding only, as the Triton pair kernel
    torch.testing.assert_close(
        out.float(), _reference(q, kc, vc, seq, table, kd, vd, 0), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("bs", [1, 4, 8])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_swa_verify_splitkv(bs, kv_heads):
    from sglang.kernels.ops.attention.swa_verify_splitkv import swa_verify_splitkv

    q, kc, vc, seq, table, kd, vd = _paged_inputs(bs, kv_heads=kv_heads, seed=200 + bs)
    cu = torch.arange(0, bs * 8 + 1, 8, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    swa_verify_splitkv(q, kc, vc, out, cu, 8, seq, 200 * 16, SCALE, True, (2046, 0), table, 0, None, kd, vd,
                       num_splits=16 // (kv_heads // 2))  # fmt: skip
    torch.testing.assert_close(
        out.float(),
        _reference(q, kc, vc, seq, table, kd, vd, 2047),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("bs", [1, 3, 8])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_dflash_block_splitkv_through_extend_attention(bs, kv_heads):
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

    torch.manual_seed(bs)
    prefix = torch.randint(100, 1500, (bs,))
    q = torch.randn(bs * 8, 8 * kv_heads, 128, device="cuda").to(torch.bfloat16)
    ke = torch.randn(bs * 8, kv_heads, 128, device="cuda").to(torch.bfloat16)
    ve = torch.randn(bs * 8, kv_heads, 128, device="cuda").to(torch.bfloat16)
    kb = (torch.randn(20000, kv_heads, 128, device="cuda") * 2).to(F8)
    vb = (torch.randn(20000, kv_heads, 128, device="cuda") * 2).to(F8)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(prefix, 0).to(torch.int32).cuda()
    kv_indices = torch.randperm(20000, device="cuda")[: int(prefix.sum())]
    qo = torch.arange(0, bs * 8 + 1, 8, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    extend_attention_fwd(q, ke, ve, out, kb, vb, qo, kv_indptr, kv_indices, None, False, None, 8, 0.8, 1.1,
                         sliding_window_size=-1)  # fmt: skip
    # The kernel runs the prefix Q.K on FP8 operands (Q cast unscaled, K from the FP8 cache), the block's own
    # K/V in BF16, and the prefix P.V on BF16 probabilities; the reference follows the same operand precision.
    for i in range(bs):
        idx = kv_indices[kv_indptr[i] : kv_indptr[i + 1]].long()
        rows = slice(i * 8, (i + 1) * 8)
        for h in range(kv_heads):
            heads = slice(h * 8, (h + 1) * 8)
            qi = q[rows, heads].float()
            q8 = q[rows, heads].to(F8).float()
            s_prefix = torch.einsum("qhd,kd->hqk", q8, kb[idx, h].float()) * 0.8
            s_block = torch.einsum("qhd,kd->hqk", qi, ke[rows, h].float())
            p = torch.softmax(torch.cat([s_prefix, s_block], -1) * 128**-0.5, -1)
            n = idx.numel()
            ref = torch.einsum("hqk,kd->qhd", p[..., :n], vb[idx, h].float() * 1.1)
            ref += torch.einsum("hqk,kd->qhd", p[..., n:], ve[rows, h].float())
            torch.testing.assert_close(
                out[rows, heads].float(), ref, rtol=3e-2, atol=3e-2
            )
