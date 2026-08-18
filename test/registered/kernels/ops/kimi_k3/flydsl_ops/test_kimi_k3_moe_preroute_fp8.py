# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx_runtime
from sglang.kernels.ops.kimi_k3.flydsl.kimi_k3_moe_preroute_fp8 import (
    is_kimi_k3_moe_preroute_fp8_available,
    kimi_k3_moe_dual_projection_fp8,
    kimi_k3_moe_tri_projection_fp8,
    kimi_k3_shared_down_fp8,
    supports_kimi_k3_moe_dual_projection_fp8,
    supports_kimi_k3_moe_tri_projection_fp8,
    supports_kimi_k3_shared_down_fp8,
    supports_kimi_k3_shared_down_fp8_weight,
)
from aiter.ops.flydsl.utils import is_flydsl_available

_FP8_MAX = 448.0


def _quantize_rows(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_f32 = weight.float()
    amax = weight_f32.abs().amax(dim=1)
    scale = torch.where(
        amax > 0,
        amax / _FP8_MAX,
        torch.ones_like(amax),
    )
    quantized = (
        (weight_f32 / scale[:, None])
        .clamp(min=-_FP8_MAX, max=_FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return quantized, scale.contiguous()


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    reference = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return (error / reference).item()


def test_support_predicates_fail_closed_on_cpu():
    hidden = torch.empty((1, 7168), dtype=torch.bfloat16)
    fp8 = torch.empty((1, 1), dtype=torch.float8_e4m3fn)
    scale = torch.empty((1,), dtype=torch.float32)

    assert not supports_kimi_k3_moe_dual_projection_fp8(
        hidden,
        fp8,
        scale,
        fp8,
        scale,
    )
    assert not supports_kimi_k3_moe_tri_projection_fp8(
        hidden,
        fp8,
        scale,
        fp8,
        scale,
        hidden,
    )
    assert not supports_kimi_k3_shared_down_fp8(hidden, fp8, scale)
    assert not supports_kimi_k3_shared_down_fp8_weight(fp8, scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU runtime")
def test_backend_availability_matches_flydsl_and_architecture():
    assert is_kimi_k3_moe_preroute_fp8_available() == (
        is_flydsl_available() and get_gfx_runtime() == "gfx950"
    )


@pytest.mark.parametrize(
    ("situ_beta", "situ_linear_beta"),
    [(0.0, 25.0), (4.0, -1.0), (float("nan"), 25.0)],
)
def test_shared_down_rejects_invalid_situ_parameters(
    situ_beta: float,
    situ_linear_beta: float,
):
    tensor = torch.empty((1,), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="finite and positive"):
        kimi_k3_shared_down_fp8(
            tensor,
            tensor,
            tensor.float(),
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_flydsl_available()
    or get_gfx_runtime() != "gfx950",
    reason="requires FlyDSL on gfx950",
)
def test_kimi_k3_preroute_fp8_matches_dequantized_reference():
    torch.manual_seed(20260729)
    device = torch.device("cuda")
    hidden = torch.randn((1, 7168), device=device, dtype=torch.bfloat16)
    routed_bf16 = torch.randn(
        (3584, 7168),
        device=device,
        dtype=torch.bfloat16,
    )
    shared_up_bf16 = torch.randn(
        (1536, 7168),
        device=device,
        dtype=torch.bfloat16,
    )
    shared_down_bf16 = torch.randn(
        (7168, 768),
        device=device,
        dtype=torch.bfloat16,
    )
    router_bf16 = torch.randn(
        (896, 7168),
        device=device,
        dtype=torch.bfloat16,
    )
    routed_weight, routed_scale = _quantize_rows(routed_bf16)
    shared_up_weight, shared_up_scale = _quantize_rows(shared_up_bf16)
    shared_down_weight, shared_down_scale = _quantize_rows(shared_down_bf16)

    routed, gate_up = kimi_k3_moe_dual_projection_fp8(
        hidden,
        routed_weight,
        routed_scale,
        shared_up_weight,
        shared_up_scale,
    )
    tri_routed, tri_gate_up, router_logits = kimi_k3_moe_tri_projection_fp8(
        hidden,
        routed_weight,
        routed_scale,
        shared_up_weight,
        shared_up_scale,
        router_bf16,
    )
    shared = kimi_k3_shared_down_fp8(
        gate_up,
        shared_down_weight,
        shared_down_scale,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    routed_dequant = routed_weight.float() * routed_scale[:, None]
    shared_up_dequant = shared_up_weight.float() * shared_up_scale[:, None]
    shared_down_dequant = shared_down_weight.float() * shared_down_scale[:, None]
    routed_ref = hidden.float() @ routed_dequant.t()
    gate_up_ref = hidden.float() @ shared_up_dequant.t()
    gate, up = gate_up_ref.to(torch.bfloat16).float().chunk(2, dim=-1)
    activated = (
        (
            4.0
            * torch.tanh(gate / 4.0)
            * torch.sigmoid(gate)
            * 25.0
            * torch.tanh(up / 25.0)
        )
        .to(torch.bfloat16)
        .float()
    )
    shared_ref = activated @ shared_down_dequant.t()

    assert _relative_rmse(routed, routed_ref) < 0.035
    assert _relative_rmse(gate_up, gate_up_ref) < 0.035
    # fdot2 may round in a different reduction order than the dual kernel.
    assert _relative_rmse(tri_routed, routed_ref) < 0.035
    assert _relative_rmse(tri_gate_up, gate_up_ref) < 0.035
    router_ref = F.linear(hidden, router_bf16).float()
    assert _relative_rmse(router_logits, router_ref) < 0.01
    router_topk = router_logits.topk(17, dim=-1)
    reference_topk = router_ref.topk(17, dim=-1)
    assert reference_topk.values[0, 15] > reference_topk.values[0, 16]
    # torch.topk does not define the order of equal values within the result.
    torch.testing.assert_close(
        router_topk.indices[:, :16].sort(dim=-1).values,
        reference_topk.indices[:, :16].sort(dim=-1).values,
        atol=0,
        rtol=0,
    )
    assert _relative_rmse(shared, shared_ref) < 0.06
    assert F.cosine_similarity(shared.float(), shared_ref.float()).item() > 0.998

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_routed, captured_gate_up = kimi_k3_moe_dual_projection_fp8(
            hidden,
            routed_weight,
            routed_scale,
            shared_up_weight,
            shared_up_scale,
        )
        captured_shared = kimi_k3_shared_down_fp8(
            captured_gate_up,
            shared_down_weight,
            shared_down_scale,
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )
    graph.replay()
    expected_routed = captured_routed.clone()
    expected_shared = captured_shared.clone()
    graph.replay()
    torch.testing.assert_close(captured_routed, expected_routed, atol=0, rtol=0)
    torch.testing.assert_close(captured_shared, expected_shared, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [2])
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_flydsl_available()
    or get_gfx_runtime() != "gfx950",
    reason="requires FlyDSL on gfx950",
)
def test_kimi_k3_tri_and_shared_projection_multitoken(num_tokens: int):
    torch.manual_seed(20260811 + num_tokens)
    hidden = torch.randn(
        (num_tokens, 7168), device="cuda", dtype=torch.bfloat16
    )
    routed_bf16 = torch.randn((3584, 7168), device="cuda", dtype=torch.bfloat16)
    shared_bf16 = torch.randn((1536, 7168), device="cuda", dtype=torch.bfloat16)
    shared_down_bf16 = torch.randn(
        (7168, 768), device="cuda", dtype=torch.bfloat16
    )
    router_bf16 = torch.randn((896, 7168), device="cuda", dtype=torch.bfloat16)
    routed_weight, routed_scale = _quantize_rows(routed_bf16)
    shared_weight, shared_scale = _quantize_rows(shared_bf16)
    shared_down_weight, shared_down_scale = _quantize_rows(shared_down_bf16)
    routed, shared, router = kimi_k3_moe_tri_projection_fp8(
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
        router_bf16,
    )
    routed_ref = hidden.float() @ (
        routed_weight.float() * routed_scale[:, None]
    ).t()
    shared_ref = hidden.float() @ (
        shared_weight.float() * shared_scale[:, None]
    ).t()
    router_ref = F.linear(hidden, router_bf16).float()
    shared_out = kimi_k3_shared_down_fp8(
        shared,
        shared_down_weight,
        shared_down_scale,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    gate, up = shared_ref.to(torch.bfloat16).float().chunk(2, dim=-1)
    activated = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    ).to(torch.bfloat16)
    shared_out_ref = activated.float() @ (
        shared_down_weight.float() * shared_down_scale[:, None]
    ).t()
    assert _relative_rmse(routed, routed_ref) < 0.035
    assert _relative_rmse(shared, shared_ref) < 0.035
    assert _relative_rmse(router, router_ref) < 0.01
    assert _relative_rmse(shared_out, shared_out_ref) < 0.06
