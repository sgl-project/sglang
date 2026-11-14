"""gfx950 Gluon q-absorb + RoPE + KV-write vs the aiter kernel it replaces.

The KV-cache rows must match aiter bit for bit (later tokens read them back);
the RoPE half of q_cat is checked against aiter, the absorbed half against an
fp32 GEMM.
"""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")

_ON_GFX950 = is_hip() and is_gfx95_supported()
pytestmark = pytest.mark.skipif(not _ON_GFX950, reason="gfx950 Gluon kernels")

if _ON_GFX950:
    from sglang.kernels.ops.attention.gluon_mla_absorb_rope_cache_gfx950 import (
        _ROWS,
        gluon_mla_absorb_rope_cache,
        gluon_mla_absorb_rope_cache_supported,
    )
    from sglang.srt.layers.rocm_linear_utils import (
        fused_fp8_bmm_rope_cat_and_cache_mla,
    )

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
NUM_SLOTS, MAX_POS = 1 << 16, 1 << 17
# Every exact-M row plus generic M with and without a partial last tile.
SHAPES = [(8, m) for m in (1, 5, 6, 7, 12, 16, 17, 18, 24, 30, 36, 42, 48, 60, 128)]
SHAPES += [(16, m) for m in (1, 6, 12, 18, 24, 30, 36, 42, 64, 120, 255, 384)]


def _inputs(h, m, with_k_scale, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(m, h, 256, device="cuda", generator=g).to(BF16)
    q_nope, q_pe = q.split([192, 64], dim=-1)
    latent = torch.randn(m, 2048 + 576, device="cuda", generator=g).to(BF16)
    k_pe = latent[:, 2048 + 512 :].unsqueeze(1)
    k_nope = torch.randn(m, 1, 512, device="cuda", generator=g).to(BF16)
    w_kc = (torch.randn(h, 512, 192, device="cuda", generator=g) * 0.1).to(FP8)
    w_scale = torch.tensor([0.02], device="cuda")
    angle = torch.rand(MAX_POS, 1, 1, 32, device="cuda", generator=g) * 6.28
    cos, sin = angle.cos().to(BF16), angle.sin().to(BF16)
    positions = torch.randint(0, MAX_POS, (m,), device="cuda", generator=g)
    slots = torch.randperm(NUM_SLOTS, device="cuda", generator=g)[:m]
    k_scale = torch.tensor([0.5], device="cuda") if with_k_scale else None
    return (
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        w_kc,
        w_scale,
        positions,
        slots,
        cos,
        sin,
        k_scale,
    )


def _run(fn_name, inputs):
    q_nope, q_pe, k_nope, k_pe, w_kc, w_scale, positions, slots, cos, sin, k_scale = (
        inputs
    )
    kv = torch.zeros(NUM_SLOTS, 1, 576, dtype=FP8, device="cuda")
    if fn_name == "gluon":
        q_cat = gluon_mla_absorb_rope_cache(
            q_nope, w_kc, w_scale, q_pe, k_nope, k_pe, kv, slots, positions,
            cos, sin, k_scale,
        )  # fmt: skip
    else:
        q_cat = fused_fp8_bmm_rope_cat_and_cache_mla(
            q_nope.transpose(0, 1), w_kc, w_scale, q_pe, k_nope, k_pe, kv, slots,
            positions, cos, sin, group_size=128, k_scale=k_scale, is_neox=False,
            q_out_dtype=FP8,
        )[0]  # fmt: skip
    return q_cat, kv


@pytest.mark.parametrize("with_k_scale", [False, True])
@pytest.mark.parametrize("h,m", SHAPES)
def test_matches_aiter(h, m, with_k_scale):
    inputs = _inputs(h, m, with_k_scale)
    q_nope, q_pe, k_nope, k_pe, w_kc, w_scale, *_, k_scale = inputs
    kv_probe = torch.empty(1, 1, 576, dtype=FP8, device="cuda")
    assert gluon_mla_absorb_rope_cache_supported(
        q_nope, q_pe, k_nope, k_pe, kv_probe, w_kc, w_scale, inputs[8], k_scale
    )
    q_cat, kv = _run("gluon", inputs)
    q_ref, kv_ref = _run("aiter", inputs)
    assert torch.equal(kv.view(torch.uint8), kv_ref.view(torch.uint8))
    rope, rope_ref = q_cat[..., 512:].float(), q_ref[..., 512:].float()
    assert (rope - rope_ref).pow(2).mean() <= 1e-4 * rope_ref.pow(2).mean()
    absorbed = torch.einsum("mhk,hnk->mhn", q_nope.float(), w_kc.float() * w_scale)
    ref_rms = absorbed.pow(2).mean().sqrt()
    for q in (q_cat, q_ref):
        err = (q[..., :512].float() - absorbed).pow(2).mean().sqrt() / ref_rms
        assert err < 0.05, err


@pytest.mark.parametrize("h,m", [(8, 6), (8, 12), (8, 60), (16, 42)])
def test_cuda_graph_replay_matches_eager(h, m):
    inputs = _inputs(h, m, with_k_scale=False)
    q_eager, kv_eager = _run("gluon", inputs)
    q_nope, q_pe, k_nope, k_pe, w_kc, w_scale, positions, slots, cos, sin, _ = inputs
    kv = torch.zeros_like(kv_eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q_cat = gluon_mla_absorb_rope_cache(
            q_nope, w_kc, w_scale, q_pe, k_nope, k_pe, kv, slots, positions,
            cos, sin, None,
        )  # fmt: skip
    kv.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q_cat.view(torch.uint8), q_eager.view(torch.uint8))
    assert torch.equal(kv.view(torch.uint8), kv_eager.view(torch.uint8))


def test_supported_rejects_what_the_kernel_does_not_implement():
    q_nope, q_pe, k_nope, k_pe, w_kc, w_scale, _, _, cos, _, _ = _inputs(8, 6, False)
    kv = torch.empty(4, 1, 576, dtype=FP8, device="cuda")
    good = [q_nope, q_pe, k_nope, k_pe, kv, w_kc, w_scale, cos, None]
    assert gluon_mla_absorb_rope_cache_supported(*good)
    for i, bad in (
        (0, q_nope.float()),
        (4, kv.to(BF16)),
        (5, w_kc.to(BF16)),
        (5, w_kc.transpose(1, 2).contiguous().transpose(1, 2)),
        (6, 1.0),
        (7, cos[..., :16]),
        (8, torch.ones(2, device="cuda")),
    ):
        args = list(good)
        args[i] = bad
        assert not gluon_mla_absorb_rope_cache_supported(*args), i
    for h, m in ((12, 6), (8, 129), (16, 385)):
        q = torch.empty(m, h, 256, dtype=BF16, device="cuda")
        args = [q[..., :192], q[..., 192:], k_nope[:1].expand(m, 1, 512)]
        args += [k_pe[:1].expand(m, 1, 64), kv]
        args += [torch.empty(h, 512, 192, dtype=FP8, device="cuda"), w_scale, cos, None]
        assert not gluon_mla_absorb_rope_cache_supported(*args), (h, m)
    for m in _ROWS:
        assert (8, m) in SHAPES and (16, m) in SHAPES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
