# SPDX-License-Identifier: Apache-2.0
"""Muse Spark FP32 router logits (MuseSpark1xRouter): FP64 semantics, dispatch and graph state.

Any-order FP32 dot products satisfy |fl(s) - s| <= gamma_K * sum_k |x_k w_k| plus an
underflow term; small-integer inputs keep every partial sum exact, so any summation
order must return the exact logits there. Decode-sized (T <= 8) FP32 inputs with
16-byte aligned rows run the CuTe DSL GEMV on SM100-family GPUs (only T = 1 when TF32
matmuls are allowed); everything else runs F.linear. Results are bitwise repeatable,
including side-stream CUDA-graph replay.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.gemm.muse_spark_router_gemv_cutedsl import (
    MAX_TOKENS,
    router_gemv_cute_eligible,
)
from sglang.srt.models.muse_spark import MuseSpark1xRouter
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
U = 2.0**-24


def _router(w):
    router = MuseSpark1xRouter(w.shape[1], w.shape[0], dtype=torch.float32).cuda()
    router.weight = torch.nn.Parameter(w, requires_grad=False)
    return router


def _inputs(tokens, hidden, experts, dist, pad=0, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    if dist == "integer":
        x = torch.randint(
            -8, 9, (tokens, hidden + pad), generator=g, device="cuda"
        ).float()
        w = torch.randint(-8, 9, (experts, hidden), generator=g, device="cuda").float()
    else:
        x = torch.randn(tokens, hidden + pad, generator=g, device="cuda")
        w = torch.randn(experts, hidden, generator=g, device="cuda") / hidden**0.5
        if dist == "wide":
            x = x * torch.exp2(
                torch.randint(-30, 31, x.shape, generator=g, device="cuda").float()
            )
    return x[:, :hidden], w


def _check(out, x, w, exact):
    assert out.shape == (x.shape[0], w.shape[0]) and out.dtype == torch.float32
    assert out.stride() == (w.shape[0], 1)
    xd, wd = x.double(), w.double()
    ref = xd @ wd.t()
    k = x.shape[1]
    bound = k * U / (1 - k * U) * (xd.abs() @ wd.abs().t()) + (2 * k + 1) * 2.0**-150
    err = (out.double() - ref).abs()
    if exact:
        assert torch.equal(out.double(), ref)
    else:
        assert bool((err <= bound).all()), float((err / bound).max())


@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 8, 9, 16])
@pytest.mark.parametrize("dist", ["normal", "wide", "integer"])
@pytest.mark.parametrize("pad", [0, 4, 3])
def test_router_logits_semantics(tokens, dist, pad):
    x, w = _inputs(tokens, 8192, 256, dist, pad, seed=tokens * 31 + pad)
    router = _router(w)
    before = (x.clone(), w.clone())
    eligible = router_gemv_cute_eligible(x, w)
    sm100 = torch.cuda.get_device_capability()[0] == 10
    assert torch.get_float32_matmul_precision() == "highest"
    assert eligible == (sm100 and tokens <= MAX_TOKENS and pad % 4 == 0)
    out = router(x)
    out2 = router(x)
    _check(out, x, w, dist == "integer")
    assert torch.equal(out.view(torch.int32), out2.view(torch.int32))
    assert torch.equal(before[0], x) and torch.equal(before[1], w)


@pytest.mark.parametrize("hidden,experts", [(2048, 64), (4096, 128), (1024, 32)])
def test_router_logits_other_shapes(hidden, experts):
    for tokens in (1, 4, 8):
        x, w = _inputs(tokens, hidden, experts, "integer", seed=hidden + tokens)
        _check(_router(w)(x), x, w, True)


@pytest.mark.parametrize("tokens", [1, 8])
def test_router_side_stream_graph_replay(tokens):
    x, w = _inputs(tokens, 8192, 256, "normal", seed=tokens)
    router = _router(w)
    sources = [
        _inputs(tokens, 8192, 256, "normal", seed=100 + i)[0].contiguous()
        for i in range(3)
    ]
    expected = [router(s).clone() for s in sources]
    src = sources[0].clone()
    main = torch.cuda.current_stream()
    capture, side = torch.cuda.Stream(), torch.cuda.Stream()
    capture.wait_stream(main)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture):
        x.copy_(src)
        side.wait_stream(capture)
        with torch.cuda.stream(side):
            out = router(x)
        capture.wait_stream(side)
    for rep in range(30):
        src.copy_(sources[rep % 3])
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int32), expected[rep % 3].view(torch.int32))


def test_router_tf32_precision_keeps_multi_row_on_cublas():
    """With TF32 matmuls allowed (the server default), only single-row logits move."""
    sm100 = torch.cuda.get_device_capability()[0] == 10
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        for tokens in (1, 2, 8):
            x, w = _inputs(tokens, 8192, 256, "integer", seed=tokens)
            assert router_gemv_cute_eligible(x, w) == (sm100 and tokens == 1)
            if tokens == 1:
                _check(_router(w)(x), x, w, True)
    finally:
        torch.set_float32_matmul_precision(previous)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
