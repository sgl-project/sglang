"""A strided output gate (a column slice of a fused projection, as in Kimi-K3
KDA) must give the same gated RMSNorm as a contiguous one, without a copy."""

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HEADS, _HEAD_DIM, _PROJ_WIDTH = 12, 128, 6144


def _strided_gate(num_tokens):
    proj = torch.randn(num_tokens, _PROJ_WIDTH, device="cuda", dtype=torch.bfloat16)
    return proj[:, _PROJ_WIDTH - _HEADS * _HEAD_DIM :].unflatten(-1, (-1, _HEAD_DIM))


@pytest.mark.parametrize("activation", ["sigmoid", "swish"])
@pytest.mark.parametrize("num_tokens", [1, 2, 8, 33, 1000])
@torch.inference_mode()
def test_strided_gate_matches_contiguous(activation, num_tokens):
    torch.manual_seed(num_tokens)
    norm = FusedRMSNormGated(
        _HEAD_DIM, eps=1e-6, activation=activation, device="cuda", dtype=torch.bfloat16
    )
    torch.nn.init.normal_(norm.weight, 1.0, 0.1)
    gate = _strided_gate(num_tokens)
    x = torch.randn(
        1, num_tokens, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    torch.testing.assert_close(
        norm(x.clone(), gate), norm(x.clone(), gate.contiguous()), rtol=0, atol=0
    )


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@torch.inference_mode()
def test_strided_gate_widths_and_dtypes(head_dim, dtype, activation):
    torch.manual_seed(0)
    tokens, heads = 11, 3
    proj = torch.randn(tokens, 4 * heads * head_dim, device="cuda", dtype=dtype)
    gate = proj[:, 3 * heads * head_dim :].unflatten(-1, (heads, head_dim))
    x = torch.randn(tokens, heads, head_dim, device="cuda", dtype=dtype)
    norm = FusedRMSNormGated(
        head_dim, activation=activation, device="cuda", dtype=dtype
    )
    torch.nn.init.normal_(norm.weight, 1.0, 0.1)
    torch.testing.assert_close(
        norm(x.clone(), gate), norm(x.clone(), gate.contiguous()), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "layout", ["2d", "inner_stride", "head_stride", "batch", "wide"]
)
@torch.inference_mode()
def test_unsupported_gate_layouts_fall_back(layout):
    torch.manual_seed(0)
    dim = 1024 if layout == "wide" else 128
    shape = (2, 3, 4, dim) if layout == "batch" else (3, 4, dim)
    if layout == "inner_stride":
        gate = torch.randn(*shape[:-1], 2 * dim, device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    elif layout == "head_stride":
        gate = torch.randn(3, 8, dim, device="cuda", dtype=torch.bfloat16)[:, ::2, :]
    else:
        proj = torch.randn(
            *shape[:-2], 4 * shape[-2] * dim, device="cuda", dtype=torch.bfloat16
        )
        gate = proj[..., -shape[-2] * dim :].unflatten(-1, (shape[-2], dim))
        if layout == "2d":
            gate = gate[:, 0, :]
    x = torch.randn_like(gate)
    norm = FusedRMSNormGated(
        dim, activation="sigmoid", device="cuda", dtype=torch.bfloat16
    )
    torch.nn.init.ones_(norm.weight)
    torch.testing.assert_close(
        norm(x.clone(), gate), norm(x.clone(), gate.contiguous()), rtol=0, atol=0
    )


@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_strided_gate_cuda_graph_and_prenorm(with_residual):
    norm = FusedRMSNormGated(
        _HEAD_DIM, activation="sigmoid", device="cuda", dtype=torch.bfloat16
    )
    torch.nn.init.ones_(norm.weight)
    gate = _strided_gate(8)
    x = torch.randn(1, 8, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None
    expected = norm(x.clone(), gate.contiguous(), residual=residual, prenorm=True)
    static_x = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        norm(static_x, gate, residual=residual, prenorm=True)
    torch.cuda.current_stream().wait_stream(stream)
    static_x.copy_(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = norm(static_x, gate, residual=residual, prenorm=True)
    static_x.copy_(x)
    graph.replay()
    torch.cuda.synchronize()
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@torch.inference_mode()
def test_strided_gate_launches_no_copy():
    norm = FusedRMSNormGated(
        _HEAD_DIM, eps=1e-6, activation="sigmoid", device="cuda", dtype=torch.bfloat16
    )
    gate = _strided_gate(8)
    x = torch.randn(1, 8, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    norm(x, gate)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        norm(x, gate)
        torch.cuda.synchronize()
    kernels = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert kernels and not any("copy" in k for k in kernels), kernels


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
