from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.layers.quantization.unquant import (
    Bf16GemmBackend,
    UnquantizedLinearMethod,
)
from sglang.srt.models.nemotron_h import NemotronHMoE
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _make_projection(input_size: int, output_size: int):
    weight = torch.randn(
        output_size,
        input_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) / (input_size**0.5)
    method = UnquantizedLinearMethod()
    return SimpleNamespace(weight=weight, bias=None, quant_method=method), method


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_latent_projection_accumulates_into_shared_output_and_replays_graph(
    monkeypatch,
):
    monkeypatch.setattr(unquant, "_BF16_GEMM_BACKEND", Bf16GemmBackend.CUTEDSL)
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


@pytest.mark.parametrize("fallback", ["noncontiguous", "compiled", "aiter", "cutedsl"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_latent_projection_preserves_unfused_fallbacks(monkeypatch, fallback):
    backend = (
        Bf16GemmBackend.CUTEDSL if fallback == "cutedsl" else Bf16GemmBackend.TORCH
    )
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

    candidate = method.apply_with_addend(projection, routed, shared)
    reference = F.linear(routed, projection.weight) + shared_before

    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    torch.testing.assert_close(shared, shared_before, rtol=0, atol=0)
    assert candidate.data_ptr() != shared.data_ptr()


def test_latent_projection_preserves_non_unquantized_path():
    class Projection:
        quant_method = object()

        def __call__(self, x):
            return 2 * x, None

    routed = torch.randn(4, 8)
    shared = torch.randn_like(routed)
    moe = SimpleNamespace(fc2_latent_proj=Projection())

    output = NemotronHMoE._apply_latent_projection(moe, routed, shared)

    torch.testing.assert_close(output, 2 * routed + shared, rtol=0, atol=0)
