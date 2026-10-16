import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.activation.gelu import new_gelu, quick_gelu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

OPS = [new_gelu, quick_gelu]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def reference(fn, x):
    x = x.float()
    result = (
        F.gelu(x, approximate="tanh")
        if fn is new_gelu
        else x * torch.sigmoid(1.702 * x)
    )
    return result


def assert_result(fn, x, result):
    expected = reference(fn, x).to(x.dtype)
    rtol, atol = (1e-5, 1e-6) if x.dtype == torch.float32 else (1e-2, 1e-3)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
    assert result.shape == x.shape


@pytest.mark.parametrize("fn", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (2, 0),
        (1,),
        (7,),
        (8,),
        (9,),
        (31,),
        (32,),
        (33,),
        (64,),
        (127,),
        (128,),
        (129,),
        (255,),
        (256,),
        (257,),
        (511,),
        (512,),
        (513,),
        (4097,),
        (65535,),
        (65536,),
        (65537,),
        (2, 3, 513),
    ],
)
def test_gelu_shapes_and_tail(fn, dtype, shape):
    # Odd lengths exercise scalar tails; zero dimensions must avoid a zero-grid launch.
    x = torch.randn(shape, dtype=dtype, device="cuda")
    assert_result(fn, x, fn(x))


@pytest.mark.parametrize("fn", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("size", [127, 65536, 65537])
def test_gelu_out_alignment(fn, dtype, offset, size):
    # Contiguous slices can still have pointers misaligned for vector loads/stores.
    x = torch.randn(size + 3, dtype=dtype, device="cuda")[offset : offset + size]
    backing = torch.full((size + 3,), float("nan"), dtype=dtype, device="cuda")
    out = backing[offset : offset + size]
    assert fn(x, out) is out
    assert_result(fn, x, out)
    assert torch.isnan(backing[:offset]).all()
    assert torch.isnan(backing[offset + size :]).all()


@pytest.mark.parametrize("fn", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_noncontiguous_input_and_extremes(fn, dtype):
    x = torch.linspace(-100, 100, 1026, device="cuda", dtype=dtype).view(2, 513).t()
    assert_result(fn, x, fn(x))


@pytest.mark.parametrize("fn", OPS)
def test_gelu_invalid_output(fn):
    x = torch.randn(2, 16, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="same shape"):
        fn(x, torch.empty(32, dtype=x.dtype, device=x.device))
    with pytest.raises(RuntimeError):
        fn(x, torch.empty(2, 32, dtype=x.dtype, device=x.device)[:, ::2])
    with pytest.raises(RuntimeError, match="dtype"):
        fn(x, torch.empty_like(x, dtype=torch.float32))


@pytest.mark.parametrize("fn", OPS)
def test_gelu_unsupported_dtype(fn):
    with pytest.raises(RuntimeError, match="dtype"):
        fn(torch.ones(16, dtype=torch.int32, device="cuda"))


@pytest.mark.parametrize("fn", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_compile_and_graph_replay(fn, dtype):
    x = torch.randn(3, 513, device="cuda", dtype=dtype)
    compiled = torch.compile(fn, fullgraph=True)
    assert_result(fn, x, compiled(x))
    out = torch.empty_like(x)
    fn(x, out)  # Warm up JIT compilation before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(x, out)
    x.neg_()
    graph.replay()
    assert_result(fn, x, out)


@pytest.mark.parametrize("fn", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [0, 1])
def test_gelu_large_out_and_graph_replay(fn, dtype, offset):
    # Large vectorized inputs, a partial final vector, and unaligned slices.
    size = 2**22 + 7
    x = torch.randn(size + 2, device="cuda", dtype=dtype)[offset : offset + size]
    backing = torch.full((size + 2,), float("nan"), device="cuda", dtype=dtype)
    out = backing[offset : offset + size]
    fn(x, out)
    assert_result(fn, x, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(x, out)
    x.neg_()
    graph.replay()
    assert_result(fn, x, out)
    assert torch.isnan(backing[:offset]).all()
    assert torch.isnan(backing[offset + size :]).all()


@pytest.mark.parametrize("fn", OPS)
@pytest.mark.parametrize("size", [2**24, 2**24 + 7])
def test_gelu_large_inputs(fn, size):
    x = torch.randn(size, device="cuda", dtype=torch.float16)
    assert_result(fn, x, fn(x))


@pytest.mark.parametrize(
    "layer_module",
    ["sglang.srt.layers.activation", "sglang.multimodal_gen.runtime.layers.activation"],
)
@pytest.mark.parametrize("name,fn", [("NewGELU", new_gelu), ("QuickGELU", quick_gelu)])
def test_gelu_runtime_cuda(layer_module, name, fn):
    import importlib

    layer = getattr(importlib.import_module(layer_module), name)()
    x = torch.randn(2, 513, dtype=torch.bfloat16, device="cuda")
    assert_result(fn, x, layer.forward_cuda(x))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
