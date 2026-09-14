"""Fused DSV4.1 mHC boundary (post-mix + pre-combine + RMSNorm) vs the fp32
reference and the production Triton pair, eager and under CUDA graph."""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.mhc import (
    HC_WIDTH,
    HIDDEN_DIM,
    hc_boundary_fused,
    mhc_post_combine_norm,
    mhc_post_combine_norm_reference,
)
from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

EPS = 1e-6
# One bf16 ulp is 2^-7 of the value; every mismatch against the fp32 reference
# is a rounding tie flipped by a different fp32 summation order.
BF16_RTOL = 2.0**-7
BF16_ATOL = 2.0**-7
MAX_MISMATCH_FRACTION = 1e-3

VALID_CONFIGS = tuple(
    (c, v)
    for v in (8, 16)
    for c in range(1, 9)
    if (HIDDEN_DIM // c) * c == HIDDEN_DIM and (HIDDEN_DIM // c) % (v * 32) == 0
)
CONFIGS = [
    (c, v)
    for (c, v) in VALID_CONFIGS
    if v == 8 or torch.cuda.get_device_capability()[0] >= 10
]


def make_inputs(m: int, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x = torch.randn(m, HIDDEN_DIM, device=dev, dtype=torch.bfloat16, generator=gen)
    residual = torch.randn(
        m, HC_WIDTH, HIDDEN_DIM, device=dev, dtype=torch.bfloat16, generator=gen
    )
    # Realistic coefficient ranges: post = 2 sigmoid, comb row-stochastic, pre = sigmoid.
    post = 2.0 * torch.sigmoid(torch.randn(m, HC_WIDTH, device=dev, generator=gen))
    comb = torch.softmax(
        torch.randn(m, HC_WIDTH, HC_WIDTH, device=dev, generator=gen), dim=-1
    )
    pre = torch.sigmoid(torch.randn(m, HC_WIDTH, device=dev, generator=gen))
    weight = (torch.rand(HIDDEN_DIM, device=dev, generator=gen) + 0.5).to(
        torch.bfloat16
    )
    return x, residual, post, comb, pre, weight


def triton_pair(x, residual, post, comb, pre, weight):
    new_residual = mhc_post_split_h(x, residual, post, comb)
    if x.shape[0] <= 8:
        y = hc_combine_norm(new_residual.view(x.shape[0], -1), pre, weight, EPS)
    else:
        from sglang.kernels.ops.layernorm import rmsnorm
        from sglang.kernels.ops.layernorm.mhc import hc_combine

        y = rmsnorm(
            hc_combine(new_residual.view(x.shape[0], -1), pre, HC_WIDTH, x.dtype),
            weight,
            EPS,
        )
    return new_residual, y


def assert_bf16_close(actual, expected, what):
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=BF16_RTOL, atol=BF16_ATOL, msg=what
    )
    mismatch = (actual != expected).float().mean().item()
    assert mismatch <= MAX_MISMATCH_FRACTION, (
        f"{what}: {mismatch:.2e} of elements differ"
    )


@pytest.mark.parametrize("cluster_size,vec_size", CONFIGS)
@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 16, 33, 64, 128, 256])
def test_matches_reference_and_triton_pair(cluster_size, vec_size, m):
    x, residual, post, comb, pre, weight = make_inputs(m, seed=m)
    ref_residual, ref_y = mhc_post_combine_norm_reference(
        x, residual, post, comb, pre, weight, EPS
    )
    tri_residual, tri_y = triton_pair(x, residual, post, comb, pre, weight)

    inplace = residual.clone()
    y = mhc_post_combine_norm(
        x,
        inplace,
        post,
        comb,
        pre,
        weight,
        EPS,
        cluster_size=cluster_size,
        vec_size=vec_size,
    )
    torch.cuda.synchronize()
    assert y.shape == (m, HIDDEN_DIM) and y.dtype == torch.bfloat16
    assert y.data_ptr() != inplace.data_ptr()
    assert_bf16_close(inplace, ref_residual, "residual vs fp32 reference")
    assert_bf16_close(y, ref_y, "y vs fp32 reference")
    assert_bf16_close(inplace, tri_residual, "residual vs mhc_post_split_h")
    assert_bf16_close(y, tri_y, "y vs hc_combine_norm")


@pytest.mark.parametrize("cluster_size,vec_size", CONFIGS)
def test_cuda_graph_replay(cluster_size, vec_size):
    m = 8
    x, residual, post, comb, pre, weight = make_inputs(m, seed=100)
    inplace = residual.clone()
    out = torch.empty_like(x)
    # warm up the JIT module outside capture
    mhc_post_combine_norm(
        x,
        inplace.clone(),
        post,
        comb,
        pre,
        weight,
        EPS,
        cluster_size=cluster_size,
        vec_size=vec_size,
        out=out,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mhc_post_combine_norm(
            x,
            inplace,
            post,
            comb,
            pre,
            weight,
            EPS,
            cluster_size=cluster_size,
            vec_size=vec_size,
            out=out,
        )
    for seed in range(5):
        x2, residual2, post2, comb2, pre2, weight2 = make_inputs(m, seed=200 + seed)
        x.copy_(x2)
        inplace.copy_(residual2)
        post.copy_(post2)
        comb.copy_(comb2)
        pre.copy_(pre2)
        weight.copy_(weight2)
        expected_residual, expected_y = mhc_post_combine_norm_reference(
            x, inplace, post, comb, pre, weight, EPS
        )
        graph.replay()
        torch.cuda.synchronize()
        assert_bf16_close(inplace, expected_residual, "graph residual")
        assert_bf16_close(out, expected_y, "graph y")


def make_mixing_params(seed: int = 7):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mix = (2 + HC_WIDTH) * HC_WIDTH
    hc_fn = torch.randn(mix, HC_WIDTH * HIDDEN_DIM, device="cuda", generator=gen) * 0.01
    hc_base = torch.randn(mix, device="cuda", generator=gen) * 0.1
    hc_scale = torch.rand(3, device="cuda", generator=gen) + 0.5
    return hc_fn, hc_scale, hc_base


@pytest.mark.parametrize("use_stream", [False, True])
@pytest.mark.parametrize("m", [1, 8, 64])
def test_hc_boundary_fused_matches_unfused_sequence(use_stream, m):
    """hc_boundary_fused == mhc_post_split_h -> combine + norm -> hc_mix_stats_sinkhorn."""
    from sglang.kernels.ops.layernorm.mhc import hc_mix_stats_sinkhorn

    x, residual, post, comb, pre, weight = make_inputs(m, seed=500 + m)
    hc_fn, hc_scale, hc_base = make_mixing_params()
    ref_residual = mhc_post_split_h(x, residual, post, comb)
    _, ref_y = triton_pair(x, residual, post, comb, pre, weight)
    ref_pre, ref_post, ref_comb = hc_mix_stats_sinkhorn(
        ref_residual.flatten(1), hc_fn, hc_scale, hc_base, HC_WIDTH, 20, EPS, 1e-6
    )
    stream = torch.cuda.Stream() if use_stream else None
    inplace = residual.clone()
    y, nxt_pre, nxt_post, nxt_comb = hc_boundary_fused(
        x,
        inplace,
        post,
        comb,
        pre,
        weight,
        EPS,
        hc_fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        hc_mult=HC_WIDTH,
        sinkhorn_iters=20,
        rms_eps=EPS,
        hc_eps=1e-6,
        stats_stream=stream,
    )
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert_bf16_close(inplace, ref_residual, "boundary residual")
    assert_bf16_close(y, ref_y, "boundary y")
    # fp32 coefficients of a tiny GEMV + sinkhorn over streams that agree to 1 bf16 ulp
    for got, want, what in (
        (nxt_pre, ref_pre, "pre"),
        (nxt_post, ref_post, "post"),
        (nxt_comb, ref_comb, "comb"),
    ):
        torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-4, msg=what)


def test_zero_tokens():
    x, residual, post, comb, pre, weight = make_inputs(0)
    y = mhc_post_combine_norm(x, residual, post, comb, pre, weight, EPS, cluster_size=2)
    assert y.shape == (0, HIDDEN_DIM)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
