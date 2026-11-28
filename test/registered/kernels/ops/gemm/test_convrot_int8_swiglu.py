import sys

import pytest
import torch

from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding,
)
from sglang.kernels.ops.gemm import convrot_int8 as ops
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 TMA SwiGLU",
)


def _equal(actual, expected):
    assert (
        (actual.view(torch.int16) == expected.view(torch.int16))
        | (actual.isnan() & expected.isnan())
    ).all()


def _inputs(m, n, k, bias, group):
    torch.manual_seed(42)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    wq, ws = ops.convrot_rotate_quantize_activation(w, group)
    bias = torch.randn(2 * n, device="cuda", dtype=torch.bfloat16) if bias else None
    return x, wq, ws, bias


def _reference(x, wq, ws, bias, group):
    return silu_and_mul_with_activation_rounding(
        ops.convrot_int8_fused_linear(x, wq, ws, bias, group)
    )


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 128, 256),
        (65, 256, 512),
        (129, 384, 768),
        (511, 1024, 1024),
        (2049, 4096, 1536),
        (6151, 7168, 4096),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("group", [64, 128, 256])
def test_projection_and_activation_rounding(m, n, k, bias, group):
    args = _inputs(m, n, k, bias, group)
    _equal(ops.convrot_int8_swiglu(*args, group), _reference(*args, group))


@pytest.mark.parametrize(
    "m,n,k,group",
    [(1, 16, 64, 64), (65, 48, 192, 64), (513, 80, 320, 64), (31, 144, 384, 128)],
)
@pytest.mark.parametrize("bias", [False, True])
def test_projection_column_and_reduction_tails(m, n, k, group, bias):
    args = _inputs(m, n, k, bias, group)
    _equal(ops.convrot_int8_swiglu(*args, group), _reference(*args, group))


def test_all_bf16_bias_values_and_underflow():
    values = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    x = torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16)
    wq = torch.zeros(131072, 256, device="cuda", dtype=torch.int8)
    ws = torch.ones(131072, device="cuda")
    bias = torch.cat([values, torch.ones_like(values)])
    _equal(
        ops.convrot_int8_swiglu(x, wq, ws, bias, 256), _reference(x, wq, ws, bias, 256)
    )


@pytest.mark.parametrize("scale", [0.0, 1e-40, 1e-38, 1e-20, 1.0])
def test_scale_order(scale):
    x, wq, ws, bias = _inputs(129, 1024, 256, True, 256)
    ws.fill_(scale)
    _equal(
        ops.convrot_int8_swiglu(x, wq, ws, bias, 256), _reference(x, wq, ws, bias, 256)
    )


def test_current_stream_output_and_graph_replay():
    args = _inputs(513, 1024, 512, True, 256)
    out = torch.empty(513, 1024, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert ops.convrot_int8_swiglu(*args, 256, out=out) is out
    stream.synchronize()
    _equal(out, _reference(*args, 256))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.convrot_int8_swiglu(*args, 256, out=out)
    for value in [0.5, -0.25]:
        args[0].fill_(value)
        graph.replay()
        _equal(out, _reference(*args, 256))


def test_compiled_fullgraph():
    args = _inputs(129, 1024, 512, False, 256)
    compiled = torch.compile(ops.convrot_int8_swiglu, fullgraph=True)
    _equal(compiled(*args, 256), _reference(*args, 256))


def test_fallback(monkeypatch):
    args = _inputs(129, 1024, 512, False, 256)
    reference = _reference(*args, 256)
    monkeypatch.setattr(ops, "convrot_int8_local_capability", lambda: (9, 0))
    _equal(ops.convrot_int8_swiglu(*args, 256), reference)


def test_output_shape_is_checked():
    args = _inputs(129, 1024, 512, False, 256)
    out = torch.empty(129, 1023, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out must"):
        ops.convrot_int8_swiglu(*args, 256, out=out)


@pytest.mark.parametrize("n", [4, 8, 24, 40])
def test_projection_half_matches_existing_activation_alignment(n):
    args = _inputs(129, n, 512, True, 512)
    with pytest.raises(ValueError, match="N a multiple of 16"):
        ops.convrot_int8_swiglu(*args, 512)
    with pytest.raises(RuntimeError, match="hidden size must be divisible"):
        _reference(*args, 512)


def test_group_512_and_empty_batch():
    args = _inputs(65, 384, 1024, False, 512)
    _equal(ops.convrot_int8_swiglu(*args, 512), _reference(*args, 512))
    empty = ops.convrot_int8_swiglu(args[0][:0], *args[1:], 512)
    assert empty.shape == (0, 384) and empty.dtype == torch.bfloat16


def test_input_contract():
    args = _inputs(129, 384, 512, False, 256)
    for x in [args[0].half(), args[0].T, args[0].unsqueeze(0)]:
        with pytest.raises(ValueError, match="x must"):
            ops.convrot_int8_swiglu(x, *args[1:], 256)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
