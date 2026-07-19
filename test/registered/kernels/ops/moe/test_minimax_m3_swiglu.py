# SPDX-License-Identifier: Apache-2.0
"""Correctness and compile-boundary tests for MiniMax-M3 SwiGLU-OAI."""

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=8, stage="jit-kernel-unit", runner_config="amd")

if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.kernels.ops.moe.minimax_m3_swiglu import (  # noqa: E402
    swiglu_oai_split,
)


def _reference(
    gate_up: torch.Tensor,
    alpha: float = 1.702,
    beta: float = 1.0,
    limit: float | None = 7.0,
) -> torch.Tensor:
    gate, up = gate_up.float().chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(gate_up.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1536),  # shared expert decode at TP=4
        (16, 6144),  # dense expert decode at TP=4
        (257, 6144),  # non-power-of-two prefill tile count
    ],
)
@torch.inference_mode()
def test_swiglu_oai_split_matches_reference(shape, dtype):
    torch.manual_seed(42)
    gate_up = 4 * torch.randn(shape, device="cuda", dtype=dtype)

    out = swiglu_oai_split(gate_up, alpha=1.702, beta=1.0, limit=7.0)
    ref = _reference(gate_up)

    assert out.shape == (*shape[:-1], shape[-1] // 2)
    assert out.dtype == dtype
    assert out.is_contiguous()
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(out, ref, rtol=tolerance, atol=tolerance)


@torch.inference_mode()
def test_swiglu_oai_split_supports_unclamped_fp32_output():
    gate_up = torch.randn((3, 2048), device="cuda", dtype=torch.bfloat16)
    out = swiglu_oai_split(
        gate_up,
        alpha=1.25,
        beta=0.5,
        limit=None,
        out_dtype=torch.float32,
    )
    gate, up = gate_up.float().chunk(2, dim=-1)
    ref = gate * torch.sigmoid(1.25 * gate) * (up + 0.5)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("limit", [7.0, 0.0])
def test_swiglu_oai_split_clamp_contract(limit):
    # gate has only an upper clamp; up is clamped on both sides.
    gate = torch.tensor([[-9.0, 9.0, 0.5, -0.5]], device="cuda")
    up = torch.tensor([[-9.0, 9.0, 0.5, -0.5]], device="cuda")
    gate_up = torch.cat((gate, up), dim=-1)

    out = swiglu_oai_split(gate_up, alpha=1.702, beta=1.0, limit=limit)
    ref = _reference(gate_up, limit=limit)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(2, 3, 1536), (3, 2050), (0, 6144)])
def test_swiglu_oai_split_shape_tails(shape):
    gate_up = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    out = swiglu_oai_split(gate_up, alpha=1.702, beta=1.0, limit=7.0)
    assert out.shape == (*shape[:-1], shape[-1] // 2)
    torch.testing.assert_close(out, _reference(gate_up), rtol=2e-2, atol=2e-2)


def test_swiglu_oai_split_is_fullgraph_compile_safe():
    gate_up = torch.randn((17, 6144), device="cuda", dtype=torch.bfloat16)

    def fn(x):
        return swiglu_oai_split(x, alpha=1.702, beta=1.0, limit=7.0)

    out = torch.compile(fn, fullgraph=True)(gate_up)
    torch.testing.assert_close(out, _reference(gate_up), rtol=2e-2, atol=2e-2)


def test_swiglu_oai_split_stays_opaque_in_fullgraph():
    gate_up = torch.randn((17, 6144), device="cuda", dtype=torch.bfloat16)
    captured_graphs = []

    def capture_backend(graph_module, _example_inputs):
        captured_graphs.append(graph_module)
        return graph_module.forward

    def fn(x):
        return swiglu_oai_split(x, alpha=1.702, beta=1.0, limit=7.0)

    out = torch.compile(fn, backend=capture_backend, fullgraph=True)(gate_up)
    torch.testing.assert_close(out, _reference(gate_up), rtol=2e-2, atol=2e-2)

    assert len(captured_graphs) == 1
    call_targets = [
        node.target
        for node in captured_graphs[0].graph.nodes
        if node.op == "call_function"
    ]
    custom_ops = {
        torch.ops.sglang.swiglu_oai_split_inplace,
        torch.ops.sglang.swiglu_oai_split_inplace.default,
    }
    assert sum(target in custom_ops for target in call_targets) == 1
    forbidden_names = (
        "aten.clamp",
        "aten.sigmoid",
        "aten.mul",
        "aten.split",
        "aten.chunk",
    )
    assert not any(
        name in str(target) for target in call_targets for name in forbidden_names
    )


def test_swiglu_oai_split_rejects_hidden_materialization():
    gate_up = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert not gate_up.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        swiglu_oai_split(gate_up, alpha=1.702, beta=1.0, limit=7.0)

    odd = torch.randn((4, 4095), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="last dimension is even"):
        swiglu_oai_split(odd, alpha=1.702, beta=1.0, limit=7.0)
