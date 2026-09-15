# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers import layernorm as layernorm_module
from sglang.multimodal_gen.runtime.models.dits.llada_image import LLaDAImageRMSNorm


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_rmsnorm_cast_order_and_cuda_dispatch(device):
    torch.manual_seed(0)
    norm = LLaDAImageRMSNorm(128, eps=1e-5).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, device=device, dtype=torch.bfloat16))
    hidden = torch.randn(2, 8, 128, device=device, dtype=torch.bfloat16)
    if device == "cuda":
        with patch.object(
            layernorm_module, "rmsnorm_hf", wraps=layernorm_module.rmsnorm_hf
        ) as kernel:
            actual = norm(hidden)
        kernel.assert_called_once()
    else:
        actual = norm.forward_native(hidden)
    variance = hidden.float().pow(2).mean(-1, keepdim=True)
    expected = norm.weight * (
        hidden.float() * torch.rsqrt(variance + norm.variance_epsilon)
    ).to(hidden.dtype)
    tolerance = 1e-2 if device == "cuda" else 0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
