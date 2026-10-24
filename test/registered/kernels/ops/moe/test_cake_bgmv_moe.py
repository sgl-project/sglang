"""Cake BGMV MoE-LoRA shrink + expand plan through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend, that the
facade plan output is bitwise identical to a plan prepared directly through
FlashInfer (and to its own replay), and that the FP32 accumulated output
matches a pure-torch reference within BF16 tolerance. Skips (with the reason)
when FlashInfer lacks the Cake module or the GPU is not sm_90a / sm_100a /
sm_103a.
"""

import math
import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_bgmv_lora as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import cake_prepare_bgmv_moe
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

OP = "moe.prepare_bgmv_moe"


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_bgmv_lora:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks flashinfer.jit.cake_bgmv_moe")
    cc = torch.cuda.get_device_capability()
    if cc not in adapter.ARCHS:
        pytest.skip(f"Cake BGMV MoE is built for sm_90a/100a/103a, device is {cc}")
    return torch.device("cuda", torch.cuda.current_device())


def _make_inputs(hidden_size, num_tokens, dtype, device, *, rank=32, top_k=2):
    torch.manual_seed(42)
    num_experts, num_loras = 128, 2
    num_pairs = num_tokens * top_k
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    lora_a = [
        torch.randn(
            num_loras, num_experts, rank, hidden_size, dtype=dtype, device=device
        )
        * (0.5 / math.sqrt(hidden_size))
    ]
    lora_b = [
        torch.randn(
            num_loras, num_experts, hidden_size, rank, dtype=dtype, device=device
        )
        * (2.0 / math.sqrt(rank))
    ]
    sorted_token_ids = torch.arange(num_tokens, dtype=torch.int64, device=device)
    sorted_token_ids = sorted_token_ids.repeat_interleave(top_k)
    expert_ids = torch.randint(
        0, num_experts, (num_pairs,), dtype=torch.int64, device=device
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    ).reshape(-1)
    lora_indices = torch.randint(
        0, num_loras, (num_tokens,), dtype=torch.int64, device=device
    )
    if num_tokens > 1:
        lora_indices[0] = -1
    return (
        x,
        lora_a,
        lora_b,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        num_experts,
    )


def _reference(inputs):
    (
        x,
        lora_a_weights,
        lora_b_weights,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        _,
    ) = inputs
    num_tokens, hidden = x.shape
    output = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
    for pair in range(sorted_token_ids.numel()):
        token = int(sorted_token_ids[pair])
        lora = int(lora_indices[token])
        if lora < 0:
            continue
        expert = int(expert_ids[pair])
        a = lora_a_weights[0][lora, expert].float()
        b = lora_b_weights[0][lora, expert].float()
        shrink = x[token].float() @ a.T
        output[token] += (b @ shrink) * float(topk_weights[pair])
    return output


@pytest.mark.parametrize("hidden_size,num_tokens", [(3072, 16), (2048, 7)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_flashinfer_and_reference(hidden_size, num_tokens, dtype):
    device = _skip_unless_supported()
    inputs = _make_inputs(hidden_size, num_tokens, dtype, device)
    assert adapter.supports_bgmv_moe(inputs[0], inputs[1], inputs[2])

    plan = cake_prepare_bgmv_moe(*inputs)
    assert plan.backend_used == "cake"
    out = plan.run().clone()

    from flashinfer.fused_moe import prepare_bgmv_moe as fi_prepare

    plan_fi = fi_prepare(*inputs, backend="cake", fallback=False)
    out_fi = plan_fi.run().clone()
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(plan.run(), out)  # replay is bitwise reproducible

    expected = _reference(inputs)
    active = (inputs[5] >= 0).nonzero().flatten()
    assert bool((expected[active].abs().amax(dim=1) > 0.1).all())
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)


def test_unsupported_rank_is_rejected_not_fallen_back():
    device = _skip_unless_supported()
    inputs = _make_inputs(2048, 4, torch.bfloat16, device, rank=24)
    assert not adapter.supports_bgmv_moe(inputs[0], inputs[1], inputs[2])
    with pytest.raises(ValueError):
        cake_prepare_bgmv_moe(*inputs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
