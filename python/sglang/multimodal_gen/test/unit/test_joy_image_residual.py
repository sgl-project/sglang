# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.diffusion import BitExactFusionGate
from sglang.multimodal_gen.runtime.models.dits import joy_image


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)
@torch.inference_mode()
def test_image_gate_preserves_addcmul_and_replays_updated_inputs(monkeypatch):
    if not current_arch().is_sm(10, 3):
        pytest.skip("Joy image residual fast path requires SM103")
    torch.manual_seed(831)
    for batch, tokens in ((1, 8048), (2, 4024)):
        fusion = BitExactFusionGate("test Joy residual", per_signature=True)
        monkeypatch.setattr(joy_image, "_JOY_RESIDUAL_GATE", fusion)
        residual = torch.randn(batch, tokens, 4096, device="cuda", dtype=torch.bfloat16)
        update = torch.randn_like(residual)
        gate = torch.randn(batch, 6, 4096, device="cuda", dtype=residual.dtype)[:, 2]
        expected = torch.addcmul(residual, update, gate.unsqueeze(1))
        actual = joy_image.fused_add_gate(residual, update, gate)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert fusion.verified_sigs and not fusion.disabled
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = joy_image.fused_add_gate(residual, update, gate)
        residual.add_(0.25)
        update.mul_(0.75)
        gate.neg_()
        graph.replay()
        expected = torch.addcmul(residual, update, gate.unsqueeze(1))
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_gate_fallback_preserves_gradients(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    def unexpected(*args):
        pytest.fail("gradient-enabled input must retain torch.addcmul")

    monkeypatch.setattr(joy_image.diffusion_kernels, "residual_gate_fp32", unexpected)
    tokens, hidden = (4096, 4096) if device == "cuda" else (7, 64)
    residual = torch.randn(
        1, tokens, hidden, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    update = torch.randn_like(residual, requires_grad=True)
    gate = torch.randn(
        1, hidden, device=device, dtype=residual.dtype, requires_grad=True
    )
    actual = joy_image.fused_add_gate(residual, update, gate)
    expected = torch.addcmul(residual, update, gate.unsqueeze(1))
    assert torch.equal(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.sum(), (residual, update, gate)),
        torch.autograd.grad(expected.sum(), (residual, update, gate)),
        strict=True,
    ):
        assert torch.equal(actual_grad, expected_grad)
