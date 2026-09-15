"""Batch partition invariance at the BF16-input, FP32-output router seam."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import gemm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def select_batch_invariant_router(monkeypatch):
    monkeypatch.setattr(gemm, "_linear_bf16_fp32_algo", "batch_invariant")


def inputs(rows=64, experts=64, hidden=2048, weight_dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(32774)
    x = torch.randn(rows, hidden, generator=generator, device="cuda").bfloat16()
    w = (torch.randn(experts, hidden, generator=generator, device="cuda") * 0.03).to(
        weight_dtype
    )
    return x, w


@pytest.mark.parametrize("rows", [8, 16, 32, 64])
def test_router_full_and_half_batch_are_identical(rows):
    x, w = inputs()
    x = x[:rows]
    whole = gemm.linear_bf16_fp32(x, w)
    split = torch.cat([gemm.linear_bf16_fp32(part, w) for part in x.chunk(2)])
    assert whole.dtype == torch.float32
    torch.testing.assert_close(whole, split, rtol=0, atol=0)


@pytest.mark.parametrize("experts,hidden", [(65, 2053), (64, 8192)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_tail_strides_permutations_and_high_precision_reference(
    experts, hidden, weight_dtype
):
    x, w = inputs(17, experts, hidden, weight_dtype)
    # Exercise token and hidden-dimension strides, including unaligned slices.
    x_storage = torch.empty((34, hidden * 2), dtype=x.dtype, device=x.device)
    x_storage[::2, ::2].copy_(x)
    x = x_storage[::2, ::2]
    w = w.T.contiguous().T
    full = gemm.linear_bf16_fp32(x, w)
    split = torch.cat([gemm.linear_bf16_fp32(p, w) for p in x.split(3)])
    order = torch.arange(len(x) - 1, -1, -1, device=x.device)
    permuted = gemm.linear_bf16_fp32(x[order], w)[order]
    torch.testing.assert_close(full, split, rtol=0, atol=0)
    torch.testing.assert_close(full, permuted, rtol=0, atol=0)
    reference = (x.double() @ w.double().T).float()
    # This is a local dot-product accuracy check, not an FP8-model logit bound.
    torch.testing.assert_close(full, reference, rtol=1e-5, atol=2e-6)


def test_changing_graph_inputs_preserves_full_half_equivalence():
    x, w = inputs()
    static = x[:32].clone()
    gemm.linear_bf16_fp32(static, w)
    for part in static.chunk(2):
        gemm.linear_bf16_fp32(part, w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        full = gemm.linear_bf16_fp32(static, w)
        split = torch.cat([gemm.linear_bf16_fp32(part, w) for part in static.chunk(2)])
    outputs = []
    for value in (x[:32], x[32:], -x[:32]):
        static.copy_(value)
        graph.replay()
        torch.testing.assert_close(full, split, rtol=0, atol=0)
        torch.testing.assert_close(
            full, gemm.linear_bf16_fp32(value, w), rtol=0, atol=0
        )
        outputs.append(full.clone())
    assert not torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("rows,experts,hidden", [(0, 64, 2048), (4, 0, 32), (4, 5, 0)])
def test_empty_dimensions(rows, experts, hidden):
    x, w = inputs(rows, experts, hidden)
    actual = gemm.linear_bf16_fp32(x, w)
    assert actual.shape == (rows, experts) and actual.dtype == torch.float32
    if not hidden:
        assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize(
    "algorithm,experts,environment_value",
    [
        ("batch_invariant", experts, value)
        for experts in (64, 256)
        for value in ("cublas", "batch_invariant")
    ]
    + [("cublas", 256, "batch_invariant")],
)
def test_deepseek_gate_uses_cached_algorithm(
    monkeypatch, algorithm, experts, environment_value
):
    from sglang.srt.environ import envs
    from sglang.srt.models import deepseek_v2

    monkeypatch.setattr(gemm, "_linear_bf16_fp32_algo", algorithm)
    monkeypatch.setattr(deepseek_v2, "_device_sm", 90)
    monkeypatch.setattr(
        deepseek_v2,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        ),
    )
    gate = deepseek_v2.MoEGate(
        SimpleNamespace(
            n_routed_experts=experts,
            hidden_size=2048,
            topk_method="group_limited_greedy",
        ),
        quant_config=None,
    )
    x, w = inputs(16 if algorithm == "cublas" else 32, experts)
    gate.weight = torch.nn.Parameter(w, requires_grad=False)
    sentinel = torch.empty((len(x), experts), device=x.device, dtype=torch.float32)

    def shape_specialization(*args, **kwargs):
        assert (
            algorithm == "cublas"
        ), "Token-dependent specialization bypassed invariant router"
        return sentinel

    monkeypatch.setattr(deepseek_v2, "_jit_dsv3_router_gemm", shape_specialization)
    # Editing the environment after import must not split the two decisions:
    # the gate's specialization and the shared helper use one cached algorithm.
    with envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.override(environment_value):
        full = gate(x)
        if algorithm == "cublas":
            assert full is sentinel
        else:
            split = torch.cat([gate(part) for part in x.chunk(2)])
            torch.testing.assert_close(full, split, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
