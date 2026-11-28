# SPDX-License-Identifier: Apache-2.0
"""Muse Spark multi-row router logits under TF32 (MuseSpark1xRouter -> flashinfer.mm_tf32).

With TF32 matmuls allowed, 2 <= T <= MAX_TOKENS FP32 logits on SM100-family GPUs run
FlashInfer's TF32 cluster split-K GEMM; under "highest" they stay on the FP32 paths.
Small-integer inputs are exact in TF32 and keep every partial sum exact, so they must
return the exact logits; random inputs stay within TF32 error of the FP64 product and
of cuBLAS under TF32. Results are bitwise repeatable, including side-stream CUDA-graph
replay.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.gemm.muse_spark_router_tf32 import (
    MAX_TOKENS,
    _mm_tf32,
    router_tf32_eligible,
)
from sglang.srt.models.muse_spark import MuseSpark1xRouter
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _supported() -> bool:
    return torch.cuda.get_device_capability()[0] == 10 and _mm_tf32() is not None


@pytest.fixture
def tf32():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(previous)


def _router(w):
    router = MuseSpark1xRouter(w.shape[1], w.shape[0], dtype=torch.float32).cuda()
    router.weight = torch.nn.Parameter(w, requires_grad=False)
    return router


def _inputs(tokens, hidden, experts, dist, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    if dist == "integer":
        x = torch.randint(-8, 9, (tokens, hidden), generator=g, device="cuda").float()
        w = torch.randint(-8, 9, (experts, hidden), generator=g, device="cuda").float()
    else:
        x = torch.randn(tokens, hidden, generator=g, device="cuda")
        w = torch.randn(experts, hidden, generator=g, device="cuda") / hidden**0.5
    return x, w


def _check(out, x, w, exact):
    assert out.shape == (x.shape[0], w.shape[0]) and out.dtype == torch.float32
    assert out.is_contiguous()
    ref = x.double() @ w.double().t()
    if exact:
        assert torch.equal(out.double(), ref)
        return
    scale = ref.abs().max().item()
    err = (out.double() - ref).abs().max().item() / scale
    assert err < 2e-3, f"max |out - fp64| / max |fp64| = {err:.2e}"
    err = (out.double() - (x @ w.t()).double()).abs().max().item() / scale
    assert err < 2e-3, f"max |out - cuBLAS TF32| / max |fp64| = {err:.2e}"


def test_router_tf32_eligibility(tf32):
    for tokens in (1, 2, 8, MAX_TOKENS, MAX_TOKENS + 1):
        x, w = _inputs(tokens, 8192, 256, "integer", seed=tokens)
        assert router_tf32_eligible(x, w) == (
            _supported() and 2 <= tokens <= MAX_TOKENS
        )
    x, w = _inputs(8, 8192, 256, "integer")
    torch.set_float32_matmul_precision("highest")
    assert not router_tf32_eligible(x, w)


@pytest.mark.parametrize(
    "tokens", [2, 3, 8, 9, 32, 64, 128, 129, 160, 256, 384, 480, MAX_TOKENS]
)
@pytest.mark.parametrize("dist", ["normal", "integer"])
def test_router_tf32_logits(tf32, tokens, dist):
    if not _supported():
        pytest.skip("flashinfer.mm_tf32 needs an SM100-family GPU and CuTe DSL")
    x, w = _inputs(tokens, 8192, 256, dist, seed=tokens)
    router = _router(w)
    before = (x.clone(), w.clone())
    out = router(x)
    out2 = router(x)
    _check(out, x, w, dist == "integer")
    assert torch.equal(out.view(torch.int32), out2.view(torch.int32))
    assert torch.equal(before[0], x) and torch.equal(before[1], w)


@pytest.mark.parametrize("hidden,experts", [(5120, 160), (4096, 128)])
def test_router_tf32_other_shapes(tf32, hidden, experts):
    if not _supported():
        pytest.skip("flashinfer.mm_tf32 needs an SM100-family GPU and CuTe DSL")
    for tokens in (2, 16, 200):
        x, w = _inputs(tokens, hidden, experts, "integer", seed=hidden + tokens)
        _check(_router(w)(x), x, w, True)


@pytest.mark.parametrize("tokens", [8, 128, 512])
def test_router_tf32_side_stream_graph_replay(tf32, tokens):
    if not _supported():
        pytest.skip("flashinfer.mm_tf32 needs an SM100-family GPU and CuTe DSL")
    x, w = _inputs(tokens, 8192, 256, "normal", seed=tokens)
    router = _router(w)
    sources = [_inputs(tokens, 8192, 256, "normal", seed=100 + i)[0] for i in range(3)]
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
