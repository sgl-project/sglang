import importlib
import sys

import pytest
import torch
import torch.nn.functional as F

if not (hasattr(torch, "musa") and torch.musa.is_available()):
    pytest.skip("MUSA device not available", allow_module_level=True)

sgl_kernel = importlib.import_module("sgl_kernel")
DEVICE = torch.device("musa:0")


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


def tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.bfloat16:
        return 1e-2, 2e-2
    return 2e-3, 2e-3


def gemv_reference(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    *,
    use_swigelu: bool = False,
    gamma: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    projected = torch.matmul(inputs.float(), weights.float().T)
    if use_swigelu:
        gate, up = projected.chunk(2, dim=-1)
        projected = F.silu(gate) * up
    elif gamma is not None:
        variance = projected.square().mean(dim=-1, keepdim=True)
        projected = projected * torch.rsqrt(variance + eps) * gamma.float()
    return projected.to(inputs.dtype)


def moe_gemv_reference(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    mul_routed_weight: bool,
    use_swigelu: bool,
) -> torch.Tensor:
    output_size = weights.shape[1] // (2 if use_swigelu else 1)
    expected = torch.zeros(
        inputs.shape[0],
        topk_ids.shape[1],
        output_size,
        dtype=inputs.dtype,
        device=inputs.device,
    )

    for token_idx in range(inputs.shape[0]):
        for route_idx in range(topk_ids.shape[1]):
            expert_idx = int(topk_ids[token_idx, route_idx].item())
            if expert_idx < 0 or expert_idx >= weights.shape[0]:
                continue

            projected = torch.matmul(
                inputs[token_idx].float(), weights[expert_idx].float().T
            )
            if mul_routed_weight:
                projected = projected * topk_weights[token_idx, route_idx].float()
            if use_swigelu:
                gate, up = projected.chunk(2, dim=-1)
                projected = F.silu(gate) * up
            expected[token_idx, route_idx] = projected.to(inputs.dtype)

    return expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["plain", "swigelu", "rmsnorm"])
def test_musa_fused_gemv_matches_reference(dtype: torch.dtype, mode: str):
    batch_size = 2
    hidden_size, output_size = 128, 64
    weight_rows = output_size * (2 if mode == "swigelu" else 1)
    inputs = torch.randn(batch_size, hidden_size, dtype=dtype, device=DEVICE).mul_(0.5)
    if mode == "rmsnorm":
        inputs[1].mul_(8.0)
    weights = torch.randn(weight_rows, hidden_size, dtype=dtype, device=DEVICE).mul_(
        0.5
    )
    gamma = (
        torch.randn(output_size, dtype=torch.bfloat16, device=DEVICE)
        .mul_(0.1)
        .add_(1.0)
        if mode == "rmsnorm"
        else None
    )
    inputs_before = inputs.clone()
    weights_before = weights.clone()
    gamma_before = gamma.clone() if gamma is not None else None
    expected = gemv_reference(
        inputs_before,
        weights_before,
        use_swigelu=mode == "swigelu",
        gamma=gamma_before,
    )

    result = sgl_kernel.musa_fused_gemv(
        inputs,
        weights,
        use_swigelu=mode == "swigelu",
        use_rms_norm=mode == "rmsnorm",
        gamma=gamma,
    )
    torch.musa.synchronize()

    assert torch.equal(inputs, inputs_before)
    assert torch.equal(weights, weights_before)
    if gamma is not None:
        assert torch.equal(gamma, gamma_before)
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mul_routed_weight", [False, True])
@pytest.mark.parametrize("use_swigelu", [False, True])
def test_musa_fused_moe_gemv_matches_routed_reference(
    dtype: torch.dtype,
    mul_routed_weight: bool,
    use_swigelu: bool,
):
    num_tokens, num_experts, topk = 2, 4, 2
    hidden_size, output_size = 128, 64
    weight_rows = output_size * (2 if use_swigelu else 1)
    inputs = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE).mul_(0.5)
    weights = torch.randn(
        num_experts, weight_rows, hidden_size, dtype=dtype, device=DEVICE
    ).mul_(0.5)
    topk_ids = torch.tensor([[0, 3], [2, 1]], dtype=torch.int32, device=DEVICE)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.6, 0.4]], dtype=torch.float32, device=DEVICE
    )
    output = torch.empty(num_tokens, topk, output_size, dtype=dtype, device=DEVICE)
    inputs_before = inputs.clone()
    weights_before = weights.clone()
    topk_ids_before = topk_ids.clone()
    topk_weights_before = topk_weights.clone()
    expected = moe_gemv_reference(
        inputs_before,
        weights_before,
        topk_ids_before,
        topk_weights_before,
        mul_routed_weight=mul_routed_weight,
        use_swigelu=use_swigelu,
    )

    result = sgl_kernel.musa_fused_moe_gemv(
        inputs,
        weights,
        output,
        None,
        None,
        topk_weights,
        topk_ids,
        mul_routed_weight,
        topk,
        False,
        use_swigelu,
    )
    torch.musa.synchronize()

    assert result is None
    assert torch.equal(inputs, inputs_before)
    assert torch.equal(weights, weights_before)
    assert torch.equal(topk_ids, topk_ids_before)
    assert torch.equal(topk_weights, topk_weights_before)
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("use_swigelu", [False, True])
def test_musa_fused_moe_gemv_zeros_only_invalid_experts(use_swigelu: bool):
    hidden_size, output_size = 128, 64
    weight_rows = output_size * (2 if use_swigelu else 1)
    inputs = torch.randn(1, hidden_size, dtype=torch.bfloat16, device=DEVICE).mul_(0.5)
    weights = torch.randn(
        2, weight_rows, hidden_size, dtype=torch.bfloat16, device=DEVICE
    ).mul_(0.5)
    topk_ids = torch.tensor([[0, 2]], dtype=torch.int32, device=DEVICE)
    topk_weights = torch.tensor([[0.75, 0.25]], dtype=torch.float32, device=DEVICE)
    output = torch.full(
        (1, 2, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    inputs_before = inputs.clone()
    weights_before = weights.clone()
    topk_ids_before = topk_ids.clone()
    topk_weights_before = topk_weights.clone()
    expected = moe_gemv_reference(
        inputs_before,
        weights_before,
        topk_ids_before,
        topk_weights_before,
        mul_routed_weight=False,
        use_swigelu=use_swigelu,
    )

    result = sgl_kernel.musa_fused_moe_gemv(
        inputs,
        weights,
        output,
        None,
        None,
        topk_weights,
        topk_ids,
        False,
        2,
        False,
        use_swigelu,
    )
    torch.musa.synchronize()

    assert result is None
    assert torch.equal(inputs, inputs_before)
    assert torch.equal(weights, weights_before)
    assert torch.equal(topk_ids, topk_ids_before)
    assert torch.equal(topk_weights, topk_weights_before)
    rtol, atol = tolerance(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        output[:, 1], torch.zeros_like(output[:, 1]), rtol=0, atol=0
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
