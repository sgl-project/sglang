import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.diffusion import try_flux2_token_cat_fp8
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.parametrize("tokens", [1, 127, 4096])
def test_flux2_token_cat_fp8_is_bit_exact(tokens: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260831 + tokens)
    attention = torch.randn(
        (1, tokens, 6144),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    mlp = torch.randn(
        (1, tokens, 18432),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    scale = torch.tensor([0.013], dtype=torch.float32, device="cuda")

    expected, _ = static_quant_fp8(torch.cat([attention, mlp], dim=-1), scale)
    actual = try_flux2_token_cat_fp8(attention, mlp, scale)

    assert actual is not None
    assert torch.equal(actual, expected)


def test_flux2_token_cat_fp8_rejects_compile() -> None:
    attention = torch.empty((1, 1, 16), device="cuda", dtype=torch.bfloat16)
    mlp = torch.empty((1, 1, 48), device="cuda", dtype=torch.bfloat16)
    scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    with patch("torch.compiler.is_compiling", return_value=True):
        assert try_flux2_token_cat_fp8(attention, mlp, scale) is None


def test_flux2_token_cat_fp8_rejects_cuda_graph_capture() -> None:
    attention = torch.empty((1, 1, 16), device="cuda", dtype=torch.bfloat16)
    mlp = torch.empty((1, 1, 48), device="cuda", dtype=torch.bfloat16)
    scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    with patch("torch.cuda.is_current_stream_capturing", return_value=True):
        assert try_flux2_token_cat_fp8(attention, mlp, scale) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
