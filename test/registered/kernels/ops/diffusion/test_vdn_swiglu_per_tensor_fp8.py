import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_silu_mul_per_tensor_fp8,
    silu_mul_per_tensor_fp8,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    sgl_per_tensor_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.parametrize("rows,hidden", [(1, 256), (333, 512), (2048, 14336)])
def test_silu_mul_per_tensor_fp8_is_bit_exact(rows: int, hidden: int) -> None:
    """SwiGLU (bf16 rounding after the SiLU and after the product) fused with
    the per-tensor absmax equals the eager silu * up followed by the dynamic
    per-tensor quant kernel, scale and fp8 payload alike."""
    generator = torch.Generator(device="cuda").manual_seed(20260904 + rows)
    x = (torch.randn((rows, 2 * hidden), device="cuda", generator=generator) * 2).to(
        torch.bfloat16
    )
    assert can_use_silu_mul_per_tensor_fp8(x)
    expected = torch.nn.functional.silu(x[:, :hidden]) * x[:, hidden:]
    q_ref = torch.empty(rows, hidden, dtype=fp8_dtype, device="cuda")
    s_ref = torch.zeros(1, dtype=torch.float32, device="cuda")
    sgl_per_tensor_quant_fp8(expected.contiguous(), q_ref, s_ref, is_static=False)
    q, s = silu_mul_per_tensor_fp8(x)
    assert torch.equal(s, s_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


def test_silu_mul_per_tensor_fp8_rejects_unsupported_input() -> None:
    x = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    assert not can_use_silu_mul_per_tensor_fp8(x)
    with pytest.raises(ValueError):
        silu_mul_per_tensor_fp8(x)
