# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers import custom_op
from sglang.multimodal_gen.runtime.layers import layernorm as layernorm_module
from sglang.multimodal_gen.runtime.models.dits.llada_image import LLaDAImageRMSNorm
from sglang.multimodal_gen.runtime.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="The HF RMSNorm kernel requires CUDA",
)
def test_llada_image_rmsnorm_dispatches_to_hf_kernel():
    torch.manual_seed(0)
    hidden_size = 128
    norm = LLaDAImageRMSNorm(hidden_size, eps=1e-5).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16))
    hidden_states = torch.randn(
        2,
        8,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )

    with patch.object(
        layernorm_module,
        "rmsnorm_hf",
        wraps=layernorm_module.rmsnorm_hf,
    ) as rmsnorm_hf:
        output = norm(hidden_states)

    rmsnorm_hf.assert_called_once()
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    expected = norm.weight * (
        hidden_states.float() * torch.rsqrt(variance + norm.variance_epsilon)
    ).to(hidden_states.dtype)
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)


def test_llada_image_rmsnorm_hip_fallback_preserves_cast_order(monkeypatch):
    monkeypatch.setattr(custom_op, "_is_cuda", False)
    monkeypatch.setattr(current_platform, "is_hip", lambda: True)
    monkeypatch.setattr(layernorm_module, "USE_AITER", False)
    torch.manual_seed(0)
    norm = LLaDAImageRMSNorm(128, eps=1e-5).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, dtype=torch.bfloat16))
    hidden_states = torch.randn(2, 8, 128, dtype=torch.bfloat16)

    with patch.object(norm, "forward_native", wraps=norm.forward_native) as native:
        output = norm(hidden_states)

    native.assert_called_once()
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    expected = norm.weight * (
        hidden_states.float() * torch.rsqrt(variance + norm.variance_epsilon)
    ).to(hidden_states.dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
