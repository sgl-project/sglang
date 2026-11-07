"""gfx950 Gluon absorb_v BMM vs the aiter kernel ``rocm_absorb_v_bmm`` runs on main."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

_ON_GFX950 = is_hip() and is_gfx95_supported()
pytestmark = pytest.mark.skipif(not _ON_GFX950, reason="gfx950 Gluon kernels")

if _ON_GFX950:
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as aiter_bmm,
    )

    from sglang.kernels.ops.gemm.gluon_mla_absorb_v_gfx950 import (
        _ROWS,
        gluon_mla_absorb_v,
        gluon_mla_absorb_v_supported,
    )
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
        _absorb_bmm_config,
    )

FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
# Every exact-M row, both generic rows (M < 16 and M >= 16) and the 16-head edges.
SHAPES = [(8, m) for m in (1, 6, 7, 8, 12, 16, 18, 24, 30, 36, 42, 48, 49, 64, 128)]
SHAPES += [(16, m) for m in (1, 6, 12, 18, 24, 30, 36, 42, 72, 97, 120)]


def _inputs(h, m, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, h, 512, device="cuda", generator=g).to(BF16)
    w = (torch.randn(h, 256, 512, device="cuda", generator=g) * 0.1).to(FP8)
    w_scale = torch.tensor([0.02], device="cuda")
    return x, w, w_scale


def _aiter(x, w, w_scale):
    m, h, _ = x.shape
    out = torch.empty(m, h, 256, dtype=BF16, device="cuda")
    aiter_bmm(
        X=x,
        WQ=w,
        w_scale=w_scale,
        group_size=128,
        YQ=out,
        transpose_bm=True,
        transpose_bm_in=True,
        dtype=BF16,
        config=_absorb_bmm_config(h, m, 256),
    )
    return out


@pytest.mark.parametrize("h,m", SHAPES)
def test_matches_aiter_and_fp32(h, m):
    x, w, w_scale = _inputs(h, m)
    out = torch.empty(m, h, 256, dtype=BF16, device="cuda")
    assert gluon_mla_absorb_v_supported(x, w, w_scale, out)
    gluon_mla_absorb_v(x, w, w_scale, out)
    ref = _aiter(x, w, w_scale)
    # Same per-group quant and group order as aiter: equal up to one bf16 rounding.
    torch.testing.assert_close(out, ref, rtol=2**-7, atol=1e-3)
    exact = torch.einsum("mhk,hnk->mhn", x.float(), w.float() * w_scale)
    err = (out.float() - exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()
    assert err < 0.05, err


@pytest.mark.parametrize("h,m", [(8, 6), (8, 12), (8, 36), (16, 24)])
def test_cuda_graph_replay_matches_eager(h, m):
    x, w, w_scale = _inputs(h, m)
    eager = torch.empty(m, h, 256, dtype=BF16, device="cuda")
    gluon_mla_absorb_v(x, w, w_scale, eager)
    out = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gluon_mla_absorb_v(x, w, w_scale, out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@pytest.mark.parametrize(
    "h,m", [(h, m) for h in (8, 16) for m in (1, 6, 8, 16, 17, 32, 36)]
)
def test_epilogue_reads_finished_mfma_results(h, m):
    """The epilogue asm must not read an MFMA accumulator before it is written.

    The four 128-wide K groups contribute equally, so one stale group result
    moves the output by about half its RMS, far outside fp8 noise. Checked after
    warm-up and under graph replay.
    """
    x, w, w_scale = _inputs(h, m, seed=7)
    exact = torch.einsum("mhk,hnk->mhn", x.float(), w.float() * w_scale)
    out = torch.empty(m, h, 256, dtype=BF16, device="cuda")

    def check():
        torch.cuda.synchronize()
        err = (out.float() - exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()
        assert err < 0.05, err

    for _ in range(3):
        out.fill_(float("nan"))
        gluon_mla_absorb_v(x, w, w_scale, out)
        check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gluon_mla_absorb_v(x, w, w_scale, out)
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        check()


def test_supported_rejects_what_the_kernel_does_not_implement():
    x, w, w_scale = _inputs(8, 6)
    out = torch.empty(6, 8, 256, dtype=BF16, device="cuda")
    assert gluon_mla_absorb_v_supported(x, w, w_scale, out)
    for args in (
        (x.float(), w, w_scale, out),
        (x, w.to(BF16), w_scale, out),
        (x, w.transpose(1, 2).contiguous().transpose(1, 2), w_scale, out),
        (x, w, 0.02, out),
        (x, w, w_scale, out.float()),
        (x, w, w_scale, out[:5]),
    ):
        assert not gluon_mla_absorb_v_supported(*args)
    for h, m in ((12, 6), (8, 129), (16, 73), (16, 121)):
        x, w, w_scale = _inputs(h, m)
        out = torch.empty(m, h, 256, dtype=BF16, device="cuda")
        assert not gluon_mla_absorb_v_supported(x, w, w_scale, out), (h, m)
    for m in _ROWS:
        assert (8, m) in SHAPES and (16, m) in SHAPES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
