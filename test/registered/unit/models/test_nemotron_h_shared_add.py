import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.unquant import (
    Bf16GemmBackend,
)
from sglang.srt.lora.layers import ReplicatedLinearWithLoRA
from sglang.srt.models.nemotron_h import NemotronHMoE
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")


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
        "splitk",
        "batch_invariant",
        "rocm",
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_latent_projection_preserves_unfused_fallbacks(monkeypatch, fallback):
    """Custom routes and unsupported fused cases retain separate-add semantics."""
    backend = {
        "compiled": Bf16GemmBackend.CUTEDSL,
        "cutedsl": Bf16GemmBackend.CUTEDSL,
        "splitk": Bf16GemmBackend.FLASHINFER_PR4266,
    }.get(fallback, Bf16GemmBackend.TORCH)
    monkeypatch.setattr(unquant, "_BF16_GEMM_BACKEND", backend)
    monkeypatch.setattr(unquant, "_use_aiter", False)
    monkeypatch.setattr(unquant, "_enable_bf16_splitk_gemm", False)
    monkeypatch.setattr(unquant, "_use_cutedsl_bf16_gemm", None)
    monkeypatch.setattr(unquant, "_use_hopper_bf16_gemv", None)
    monkeypatch.setattr(
        unquant,
        "is_batch_invariant_mode_enabled",
        lambda: False,
        raising=False,
    )
    projection, method = _make_projection(64, 128)
    routed = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    shared = (
        torch.randn(128, 4, device="cuda", dtype=torch.bfloat16).t()
        if fallback == "noncontiguous"
        else torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    )
    shared_before = shared.clone()
    route_checks = []
    route_calls = []

    if fallback == "compiled":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

        def compiled_dispatch(x, weight, bias):
            route_calls.append("compiled")
            return F.linear(x, weight, bias)

        monkeypatch.setattr(unquant, "bf16_gemm_dispatch", compiled_dispatch)
    elif fallback == "aiter":
        monkeypatch.setattr(unquant, "_use_aiter", True)

        def aiter_mm(x, weight, bias, otype):
            route_calls.append("aiter")
            assert otype == x.dtype
            return F.linear(x, weight, bias)

        monkeypatch.setattr(
            unquant, "tgemm", SimpleNamespace(mm=aiter_mm), raising=False
        )
    elif fallback == "cutedsl":

        def use_cutedsl(*args):
            route_checks.append("cutedsl")
            return True

        def cutedsl_gemm(x, weight, bias):
            route_calls.append("cutedsl")
            return F.linear(x, weight, bias)

        monkeypatch.setattr(unquant, "_use_cutedsl_bf16_gemm", use_cutedsl)
        monkeypatch.setattr(
            unquant,
            "_cutedsl_bf16_gemm",
            cutedsl_gemm,
        )
    elif fallback == "splitk":
        monkeypatch.setattr(unquant, "_enable_bf16_splitk_gemm", True)

        def use_splitk(*args):
            route_checks.append("splitk")
            return True

        def splitk_gemm(x, weight, bias):
            route_calls.append("splitk")
            return F.linear(x, weight, bias)

        monkeypatch.setattr(unquant, "use_flashinfer_pr4266_bf16_gemm", use_splitk)
        monkeypatch.setattr(unquant, "_flashinfer_pr4266_bf16_gemm", splitk_gemm)
    elif fallback == "batch_invariant":
        monkeypatch.setattr(unquant, "is_batch_invariant_mode_enabled", lambda: True)
    elif fallback == "rocm":
        # HIP tensors also report ``is_cuda``; the platform guard must win.
        monkeypatch.setattr(unquant, "_is_cuda", False)

    candidate = method.apply_with_addend(projection, routed, shared)
    reference = F.linear(routed, projection.weight) + shared_before

    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    torch.testing.assert_close(shared, shared_before, rtol=0, atol=0)
    assert candidate.data_ptr() != shared.data_ptr()
    if fallback in {"compiled", "aiter", "cutedsl", "splitk"}:
        assert route_calls == [fallback]
    if fallback in {"cutedsl", "splitk"}:
        assert route_checks == [fallback]


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
    """An active production LoRA wrapper must retain its LoRA update."""

    class FakeLoRABackend:
        batch_info = object()

        def run_lora_a_sgemm(self, x, weights):
            return 2 * x

        def run_lora_b_sgemm(self, *, x, base_output, **kwargs):
            return base_output + x

    routed = torch.randn(4, 8)
    shared = torch.randn_like(routed)
    projection = ReplicatedLinear(8, 8, bias=False)
    with torch.no_grad():
        projection.weight.zero_()
    wrapped_projection = ReplicatedLinearWithLoRA(projection, FakeLoRABackend())
    wrapped_projection.set_lora_info(torch.empty(1), torch.empty(8, 1))
    moe = SimpleNamespace(fc2_latent_proj=wrapped_projection)

    output = NemotronHMoE._apply_latent_projection(moe, routed, shared)

    torch.testing.assert_close(output, 2 * routed + shared, rtol=0, atol=0)


def test_latent_projection_preserves_replicated_linear_forward_semantics():
    """The optimized call must preserve deferred bias and module hooks."""
    routed = torch.randn(4, 8)
    shared = torch.randn_like(routed)
    projection = ReplicatedLinear(8, 8, bias=True, skip_bias_add=True)
    with torch.no_grad():
        projection.weight.copy_(torch.randn_like(projection.weight))
        projection.bias.copy_(torch.randn_like(projection.bias))
    hook_calls = []
    projection.register_forward_hook(lambda *args: hook_calls.append(True))
    moe = SimpleNamespace(fc2_latent_proj=projection)

    output = NemotronHMoE._apply_latent_projection(moe, routed, shared)
    reference = F.linear(routed, projection.weight) + shared

    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    assert hook_calls == [True]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
