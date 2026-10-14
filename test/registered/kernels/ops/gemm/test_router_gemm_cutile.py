"""Accuracy and CUDA graph tests for the prepared BF16x3 router GEMM."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytest.importorskip("cuda.tile")

import cuda.tile as ct  # noqa: E402

from sglang.kernels.ops.gemm.router_gemm_cutile import (  # noqa: E402
    router_gemm_bf16x3_cutile,
)
from sglang.kernels.ops.gemm.router_gemv import split_fp32_to_bf16x3  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="Hopper or Blackwell required",
)


def _check(out, x, w):
    ref = x.double() @ w.double().T
    error = out.double() - ref
    torch.testing.assert_close(out.double(), ref, atol=8e-6, rtol=3e-6)
    assert (error.square().mean() / ref.square().mean()).sqrt().item() < 1e-6


@pytest.mark.parametrize(
    "m",
    [
        1,
        2,
        8,
        16,
        32,
        64,
        65,
        128,
        129,
        512,
        513,
        1024,
        1025,
        2048,
        2049,
        4096,
        4097,
        8192,
        16384,
    ],
)
def test_router_shape(m):
    torch.manual_seed(1)
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 5120, device="cuda") * 0.02
    out = router_gemm_bf16x3_cutile(x, split_fp32_to_bf16x3(w))
    assert out.dtype == torch.float32
    _check(out, x, w)


@pytest.mark.parametrize(
    "m,k,n",
    [
        (2, 137, 160),
        (17, 2816, 256),
        (129, 5120, 160),
        (257, 5128, 160),
        (257, 6144, 128),
        (513, 5128, 160),
        (1025, 137, 160),
        (8193, 5120, 256),
    ],
)
def test_tail_tiles(m, k, n):
    torch.manual_seed(1)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda") / k**0.5
    _check(router_gemm_bf16x3_cutile(x, split_fp32_to_bf16x3(w)), x, w)


def test_strided_input():
    torch.manual_seed(1)
    x = torch.randn(129, 5144, device="cuda", dtype=torch.bfloat16)[:, :5128]
    w = torch.randn(160, 5128, device="cuda") * 0.02
    _check(router_gemm_bf16x3_cutile(x, split_fp32_to_bf16x3(w)), x, w)


@pytest.mark.parametrize("m,n", [(0, 256), (32, 0)])
def test_empty_output(m, n):
    x = torch.empty(m, 5120, device="cuda", dtype=torch.bfloat16)
    parts = torch.empty(3 * n, 5120, device="cuda", dtype=torch.bfloat16)
    out = router_gemm_bf16x3_cutile(x, parts)
    assert out.shape == (m, n)
    assert out.dtype == torch.float32
    assert out.device == x.device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two GPUs required")
def test_non_current_device():
    current = torch.cuda.current_device()
    with torch.cuda.device((current + 1) % torch.cuda.device_count()):
        x = torch.randn(32, 5120, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(256, 5120, device="cuda") * 0.02
        parts = split_fp32_to_bf16x3(w)
    assert torch.cuda.current_device() == current
    _check(router_gemm_bf16x3_cutile(x, parts), x, w)


@pytest.mark.parametrize("m", [1, 2, 8, 16, 32, 128, 512, 1024, 2048, 16384])
@pytest.mark.parametrize("k,distance", [(512, 256), (5120, 320), (5120, 2560)])
def test_cancellation(m, k, distance, record_property):
    torch.manual_seed(1)
    blocks = k // (2 * distance)
    half = torch.randn(m, blocks, distance, device="cuda", dtype=torch.bfloat16)
    x = torch.cat((half, half), dim=-1).reshape(m, k)
    main = torch.randn(
        256, blocks, distance, device="cuda", dtype=torch.bfloat16
    ).float()
    w = torch.cat((main, -main + torch.randn_like(main) * 1e-4), dim=-1).reshape(256, k)
    if k != 512:
        w = w / k**0.5
    ref = x.double() @ w.double().T
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        baseline_error = (x.float() @ w.T - ref).abs().max()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    error = (router_gemm_bf16x3_cutile(x, split_fp32_to_bf16x3(w)) - ref).abs()
    record_property("fp32_max_error", baseline_error.item())
    record_property("candidate_max_error", error.max().item())
    record_property("candidate_mae", error.mean().item())
    assert error.mean().item() < 1e-6


@pytest.mark.parametrize("m", [64, 512, 1024, 2048, 4096, 8192])
def test_cuda_graph_replay(m):
    torch.manual_seed(1)
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 5120, device="cuda") * 0.02
    parts = split_fp32_to_bf16x3(w)
    router_gemm_bf16x3_cutile(x, parts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = router_gemm_bf16x3_cutile(x, parts)
    for _ in range(2):
        x.normal_()
        w.normal_(std=0.02)
        split_fp32_to_bf16x3(w, out=parts)
        graph.replay()
        _check(out, x, w)


@ct.kernel
def _pdl_producer(source, output):
    ct.grid_dependency_control_launch_dependents()
    # Keep input production pending after releasing the dependent GEMM.
    delay = ct.astype(ct.load(source, (0,), (32,)), ct.float32)
    for _ in range(1024):
        delay = ct.sin(delay) + 0.001
    offset = ct.sum(delay, axis=0) / 32
    for block in range(ct.bid(0), ct.cdiv(source.shape[0], 1024), 32):
        value = ct.load(source, (block,), (1024,), padding_mode=ct.PaddingMode.ZERO)
        ct.store(output, (block,), ct.astype(value + offset, ct.bfloat16))


@pytest.mark.parametrize("m", [512, 1024])
def test_pdl_producer_graph_replay(m):
    torch.manual_seed(1)
    source = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    x = torch.empty_like(source)
    w = torch.randn(256, 5120, device="cuda") * 0.02
    parts = split_fp32_to_bf16x3(w)
    ct.launch(
        torch.cuda.current_stream(), (32,), _pdl_producer, (source.view(-1), x.view(-1))
    )
    torch.cuda.synchronize()
    router_gemm_bf16x3_cutile(x, parts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ct.launch(
            torch.cuda.current_stream(),
            (32,),
            _pdl_producer,
            (source.view(-1), x.view(-1)),
        )
        out = router_gemm_bf16x3_cutile(x, parts)
    for _ in range(3):
        source.normal_()
        x.fill_(float("nan"))
        graph.replay()
        _check(out, x, w)


def test_concurrent_graphs():
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    cases = []
    for stream in streams:
        with torch.cuda.stream(stream):
            x = torch.randn(1024, 5120, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(256, 5120, device="cuda") * 0.02
            parts = split_fp32_to_bf16x3(w)
            router_gemm_bf16x3_cutile(x, parts)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = router_gemm_bf16x3_cutile(x, parts)
        cases.append((graph, x, w, out))
    for _ in range(3):
        for stream, (graph, x, _, _) in zip(streams, cases):
            with torch.cuda.stream(stream):
                x.normal_()
                graph.replay()
        torch.cuda.synchronize()
        for _, x, w, out in cases:
            _check(out, x, w)


@pytest.mark.parametrize("m", [32, 1024, 8192])
def test_torch_compile(m):
    torch.manual_seed(1)
    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(256, 5120, device="cuda") * 0.02
    parts = split_fp32_to_bf16x3(w)
    compiled = torch.compile(router_gemm_bf16x3_cutile, fullgraph=True)
    _check(compiled(x, parts), x, w)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
