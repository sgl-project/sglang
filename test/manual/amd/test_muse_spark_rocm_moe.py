# ruff: noqa: C408, F841
"""CPU + GPU tests for the FlyDSL Muse MoE path."""

from __future__ import annotations

import pytest
import torch

from sglang.srt.rocm_muse_spark.model import _moe_backend, _want_fused_moe


def test_flydsl_backend_is_fused(monkeypatch):
    monkeypatch.setenv("MUSE_MOE_BACKEND", "flydsl")
    assert _moe_backend() == "flydsl"
    assert _want_fused_moe() is True


def _tiny_cfg():
    return dict(
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, GROUP_M=1, num_warps=4, num_stages=2
    )


def _make_bf16_case(device):
    torch.manual_seed(0)
    M, K, I, H, E, topk = 2, 64, 64, 64, 2, 2
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w13 = torch.randn(E, 2 * I, K, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device)
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    scores = torch.tensor(
        [[0.25, 0.75], [0.4, 0.6]], dtype=torch.float32, device=device
    )
    gamma = torch.randn(H, dtype=torch.bfloat16, device=device)
    return hidden, scores, ids, w13, w2, gamma


def _pack_mxfp4(w_bf16: torch.Tensor):
    """Pack BF16 [..., K] into OCP MX e2m1 + e8m0 using the Triton reference inverse."""
    from sglang.srt.rocm_muse_spark.muse_moe_triton import _E2M1_LUT

    k = w_bf16.shape[-1]
    assert k % 32 == 0
    w = w_bf16.float()
    groups = w.reshape(*w.shape[:-1], k // 32, 32)
    amax = groups.abs().amax(dim=-1).clamp_min(1e-12)
    # largest e2m1 mag is 6; choose e8m0 so 6 * 2^(e-127) >= amax
    exp = torch.ceil(torch.log2(amax / 6.0)).to(torch.int32) + 127
    exp = exp.clamp(0, 254)
    scale = torch.exp2(exp.float() - 127.0)
    q = groups / scale.unsqueeze(-1)
    lut = _E2M1_LUT.to(device=w.device, dtype=torch.float32)
    # nearest of 16 codes
    dist = (q.unsqueeze(-1) - lut).abs()
    codes = dist.argmin(dim=-1).to(torch.uint8)
    even = codes[..., 0::2]
    odd = codes[..., 1::2]
    packed = even | (odd << 4)
    packed = packed.reshape(*w_bf16.shape[:-1], k // 2)
    scales = exp.to(torch.uint8)
    return packed.contiguous(), scales.contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA required")
def test_flydsl_matches_triton_bf16(monkeypatch):
    from sglang.srt.rocm_muse_spark.muse_moe_flydsl import muse_moe_flydsl
    from sglang.srt.rocm_muse_spark.muse_moe_triton import muse_moe_triton

    device = torch.device("cuda")
    hidden, scores, ids, w13, w2, gamma = _make_bf16_case(device)
    kwargs = dict(
        local_expert_start=0,
        post_norm_weight=gamma,
        eps=1e-8,
        cfg=_tiny_cfg(),
        post_norm_gain=0.0,
        has_norm=True,
    )
    ref = muse_moe_triton(hidden, scores, ids, w13, w2, **kwargs)
    import sglang.srt.rocm_muse_spark.muse_moe_triton as triton_impl

    def forbidden(*args, **kwargs):
        raise AssertionError("FlyDSL must not call Triton grouped GEMM")

    monkeypatch.setattr(triton_impl, "_launch_gemm", forbidden)
    out = muse_moe_flydsl(hidden, scores, ids, w13, w2, **kwargs)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1.6e-2, atol=1.6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA required")
def test_flydsl_splitk_matches_triton_bf16():
    from sglang.srt.rocm_muse_spark.muse_moe_flydsl import muse_moe_flydsl
    from sglang.srt.rocm_muse_spark.muse_moe_triton import muse_moe_triton

    device = torch.device("cuda")
    hidden, scores, ids, w13, w2, gamma = _make_bf16_case(device)
    cfg = _tiny_cfg()
    cfg.update(BLOCK_N=16, FLYDSL_K_UNROLL=1, FLYDSL_SPLIT_K=2)
    kwargs = dict(
        local_expert_start=0,
        post_norm_weight=gamma,
        eps=1e-8,
        post_norm_gain=0.0,
        has_norm=True,
    )
    ref = muse_moe_triton(hidden, scores, ids, w13, w2, cfg=_tiny_cfg(), **kwargs)
    out = muse_moe_flydsl(hidden, scores, ids, w13, w2, cfg=cfg, **kwargs)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1.6e-2, atol=1.6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA required")
def test_flydsl_matches_triton_mxfp4():
    from sglang.srt.rocm_muse_spark.muse_moe_flydsl import muse_moe_flydsl
    from sglang.srt.rocm_muse_spark.muse_moe_triton import muse_moe_triton

    device = torch.device("cuda")
    hidden, scores, ids, w13, w2, gamma = _make_bf16_case(device)
    w13_q, w13_s = _pack_mxfp4(w13)
    w2_q, w2_s = _pack_mxfp4(w2)
    kwargs = dict(
        local_expert_start=0,
        post_norm_weight=gamma,
        eps=1e-8,
        cfg=_tiny_cfg(),
        post_norm_gain=1.0,
        has_norm=True,
        w13_scale=w13_s,
        w2_scale=w2_s,
    )
    ref = muse_moe_triton(hidden, scores, ids, w13_q, w2_q, **kwargs)
    out = muse_moe_flydsl(hidden, scores, ids, w13_q, w2_q, **kwargs)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def _torch_reference(x, scores, ids, w13, w2, start, gamma, gain, has_norm, eps=1e-8):
    """Independent FP32 GEMMs with the fused path's BF16 rounding points."""
    result = torch.zeros(x.shape[0], w2.shape[1], device=x.device)
    for m in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            e = int(ids[m, slot]) - start
            if 0 <= e < w13.shape[0]:
                gate, up = (w13[e].float() @ x[m].float()).chunk(2)
                inter = (torch.nn.functional.silu(gate) * up).bfloat16()
                y = (w2[e].float() @ inter.float()).bfloat16().float()
                if has_norm:
                    if gamma is not None:
                        scale = (gamma.float() + gain).to(gamma.dtype).float()
                        y = (y * scale).bfloat16().float()
                    y = y * torch.rsqrt(y.square().mean() + eps)
                result[m] += scores[m, slot] * y
    return result.bfloat16()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA required")
@pytest.mark.parametrize("mx", [False, True])
@pytest.mark.parametrize(
    "norm,scale_dtype",
    [
        (False, None),
        (True, None),
        (True, torch.bfloat16),
        (True, torch.float16),
        (True, torch.float32),
    ],
)
def test_sparse_routing_graph_and_reference(mx, norm, scale_dtype):
    from sglang.srt.rocm_muse_spark.muse_moe_flydsl import muse_moe_flydsl
    from sglang.srt.rocm_muse_spark.muse_moe_triton import dequant_mxfp4

    torch.manual_seed(23)
    M, H, I, E, topk, start = 3, 128, 128, 3, 4, 5
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.1
    s13 = s2 = None
    if mx:
        w13, s13 = _pack_mxfp4(w13)
        w2, s2 = _pack_mxfp4(w2)
        ref13, ref2 = dequant_mxfp4(w13, s13), dequant_mxfp4(w2, s2)
    else:
        ref13, ref2 = w13, w2
    gamma = (
        torch.randn(H, device="cuda", dtype=scale_dtype) * 0.1 if scale_dtype else None
    )
    ids = torch.tensor(
        [[5, 6, 40, -1], [7, 8, 0, 5], [40, 41, 42, 43]],
        device="cuda",
        dtype=torch.int32,
    )
    scores = torch.rand(M, topk, device="cuda").softmax(-1)

    def run():
        return muse_moe_flydsl(
            x,
            scores,
            ids,
            w13,
            w2,
            start,
            gamma,
            has_norm=norm,
            post_norm_gain=1.0,
            cfg=_tiny_cfg(),
            w13_scale=s13,
            w2_scale=s2,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for all_remote in (False, True, False):
        if all_remote:
            ids.fill_(99)
        else:
            ids.copy_(
                torch.tensor(
                    [[5, 6, 40, -1], [7, 8, 0, 5], [40, 41, 42, 43]],
                    device="cuda",
                    dtype=torch.int32,
                )
            )
        graph.replay()
        expected = _torch_reference(
            x, scores, ids, ref13, ref2, start, gamma, 1.0, norm
        )
        torch.testing.assert_close(out.float(), expected.float(), atol=3e-3, rtol=2e-2)
        assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA required")
@pytest.mark.parametrize("mx", [False, True])
def test_native_layout_large_k_and_multiple_row_tiles(mx):
    from sglang.srt.rocm_muse_spark.muse_moe_flydsl import muse_moe_flydsl
    from sglang.srt.rocm_muse_spark.muse_moe_triton import muse_moe_triton

    torch.manual_seed(93)
    M, H, I, E, topk = 35, 512, 256, 3, 2
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) / H**0.5
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) / I**0.5
    s13 = s2 = None
    if mx:
        w13, s13 = _pack_mxfp4(w13)
        w2, s2 = _pack_mxfp4(w2)
    ids = torch.tensor(
        [[3, 4], [4, 5], [3, 5], [99, -1], [3, 3]], device="cuda", dtype=torch.int32
    ).repeat(7, 1)
    scores = torch.rand(M, topk, device="cuda").softmax(-1)
    gamma = torch.randn(H, device="cuda", dtype=torch.bfloat16) * 0.1
    kwargs = dict(
        local_expert_start=3,
        post_norm_weight=gamma,
        post_norm_gain=1.0,
        w13_scale=s13,
        w2_scale=s2,
    )
    expected = muse_moe_triton(x, scores, ids, w13, w2, **kwargs)
    actual = muse_moe_flydsl(x, scores, ids, w13, w2, **kwargs)
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-3, rtol=2e-2)
    empty = muse_moe_flydsl(x[:0], scores[:0], ids[:0], w13, w2, **kwargs)
    assert empty.shape == (0, H)
