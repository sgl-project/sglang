import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.unquant import (
    Bf16GemmBackend,
)
from sglang.srt.models.nemotron_h import NemotronHMoE
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _make_projection(input_size: int, output_size: int):
    projection = ReplicatedLinear(
        input_size,
        output_size,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda()
    projection.weight.copy_(torch.randn_like(projection.weight) / (input_size**0.5))
    return projection, projection.quant_method


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_latent_projection_accumulates_into_shared_output_and_replays_graph(
    monkeypatch,
):
    """A graph replay must consume the newly produced shared buffer each time."""
    monkeypatch.setattr(unquant, "_BF16_GEMM_BACKEND", Bf16GemmBackend.CUTEDSL)
    monkeypatch.setattr(unquant, "_enable_bf16_splitk_gemm", False)
    monkeypatch.setattr(unquant, "_use_cutedsl_bf16_gemm", lambda *args: False)
    projection, _ = _make_projection(64, 128)
    moe = SimpleNamespace(fc2_latent_proj=projection)
    routed = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    reference = F.linear(routed, projection.weight) + shared

    shared_output = shared.clone()
    candidate = NemotronHMoE._apply_latent_projection(moe, routed, shared_output)
    torch.testing.assert_close(candidate, reference, rtol=1e-2, atol=3.125e-2)
    assert candidate.data_ptr() == shared_output.data_ptr()

    # The shared-expert producer rewrites this buffer on every real graph replay.
    graph_shared_input = shared.clone()
    NemotronHMoE._apply_latent_projection(moe, routed, graph_shared_input.clone())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_shared_output = graph_shared_input.clone()
        graph_output = NemotronHMoE._apply_latent_projection(
            moe, routed, graph_shared_output
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_output, reference, rtol=1e-2, atol=3.125e-2)

    routed.copy_(torch.randn_like(routed))
    graph_shared_input.copy_(torch.randn_like(graph_shared_input))
    replay_reference = F.linear(routed, projection.weight) + graph_shared_input
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_output, replay_reference, rtol=1e-2, atol=3.125e-2)


@pytest.mark.parametrize(
    "fallback",
    [
        "noncontiguous",
        "compiled",
        "aiter",
        "cutedsl",
        "hopper_gemv",
        "splitk",
        "rocm",
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_latent_projection_preserves_unfused_fallbacks(monkeypatch, fallback):
    """Unsupported platforms and custom kernels must retain separate-add semantics."""
    backend = {
        "cutedsl": Bf16GemmBackend.CUTEDSL,
        "hopper_gemv": Bf16GemmBackend.CUTEDSL,
        "splitk": Bf16GemmBackend.FLASHINFER_PR4266,
    }.get(fallback, Bf16GemmBackend.TORCH)
    monkeypatch.setattr(unquant, "_BF16_GEMM_BACKEND", backend)
    projection, method = _make_projection(64, 128)
    routed = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    shared = (
        torch.randn(128, 4, device="cuda", dtype=torch.bfloat16).t()
        if fallback == "noncontiguous"
        else torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    )
    shared_before = shared.clone()

    if fallback == "compiled":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif fallback == "aiter":
        monkeypatch.setattr(unquant, "_use_aiter", True)
        linear_output = F.linear(routed, projection.weight)
        monkeypatch.setattr(method, "apply", lambda layer, x, bias: linear_output)
    elif fallback == "cutedsl":
        monkeypatch.setattr(unquant, "_use_cutedsl_bf16_gemm", lambda *args: True)
        monkeypatch.setattr(
            unquant,
            "_cutedsl_bf16_gemm",
            lambda x, weight, bias: F.linear(x, weight, bias),
        )
    elif fallback == "hopper_gemv":
        monkeypatch.setattr(unquant, "_use_hopper_bf16_gemv", lambda *args: True)
        linear_output = F.linear(routed, projection.weight)
        monkeypatch.setattr(method, "apply", lambda layer, x, bias: linear_output)
    elif fallback == "splitk":
        monkeypatch.setattr(unquant, "_enable_bf16_splitk_gemm", True)
        monkeypatch.setattr(
            unquant, "use_flashinfer_pr4266_bf16_gemm", lambda *args: True
        )
        linear_output = F.linear(routed, projection.weight)
        monkeypatch.setattr(method, "apply", lambda layer, x, bias: linear_output)
    elif fallback == "rocm":
        # HIP tensors also report ``is_cuda``; the platform guard must win.
        monkeypatch.setattr(unquant, "_is_cuda", False)

    candidate = method.apply_with_addend(projection, routed, shared)
    reference = F.linear(routed, projection.weight) + shared_before

    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    torch.testing.assert_close(shared, shared_before, rtol=0, atol=0)
    assert candidate.data_ptr() != shared.data_ptr()


def test_latent_projection_preserves_non_unquantized_path():
    """A non-unquantized base projection must dispatch through its own method."""

    class DoubleMethod:
        def apply(self, layer, x, bias):
            return 2 * x

    routed = torch.randn(4, 8)
    shared = torch.randn_like(routed)
    projection = ReplicatedLinear(8, 8, bias=False)
    projection.quant_method = DoubleMethod()
    moe = SimpleNamespace(fc2_latent_proj=projection)

    output = NemotronHMoE._apply_latent_projection(moe, routed, shared)

    torch.testing.assert_close(output, 2 * routed + shared, rtol=0, atol=0)


def test_latent_projection_preserves_lora_wrapper_path():
    """A wrapped projection without quant_method must run its wrapper forward."""

    class LoraLikeProjection(nn.Module):
        # Production LoRA wrappers intentionally do not expose quant_method.
        def forward(self, x):
            return 2 * x + 1, None

    routed = torch.randn(4, 8)
    shared = torch.randn_like(routed)
    moe = SimpleNamespace(fc2_latent_proj=LoraLikeProjection())

    output = NemotronHMoE._apply_latent_projection(moe, routed, shared)

    torch.testing.assert_close(output, 2 * routed + 1 + shared, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
