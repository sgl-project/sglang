import pytest
import torch

from sglang.kernels.ops.diffusion import can_use_silu_mul_mxfp8, silu_mul_mxfp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.parametrize("rows,hidden", [(1, 256), (333, 512), (2048, 14336)])
def test_silu_mul_mxfp8_is_bit_exact(rows: int, hidden: int) -> None:
    """SwiGLU fused with the MXFP8 block quant equals eager silu * up followed by
    FlashInfer's mxfp8_quantize: payload and swizzled E8M0 scales alike."""
    import flashinfer

    generator = torch.Generator(device="cuda").manual_seed(20260905 + rows)
    x = (torch.randn((rows, 2 * hidden), device="cuda", generator=generator) * 2).to(
        torch.bfloat16
    )
    assert can_use_silu_mul_mxfp8(x)
    expected = torch.nn.functional.silu(x[:, :hidden]) * x[:, hidden:]
    q_ref, s_ref = flashinfer.mxfp8_quantize(expected.contiguous(), True)
    q, s = silu_mul_mxfp8(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))


def test_silu_mul_mxfp8_rejects_unsupported_input() -> None:
    assert not can_use_silu_mul_mxfp8(
        torch.randn(4, 64, device="cuda", dtype=torch.float16)
    )
    assert not can_use_silu_mul_mxfp8(
        torch.randn(4, 96, device="cuda", dtype=torch.bfloat16)
    )
    with pytest.raises(ValueError):
        silu_mul_mxfp8(torch.randn(4, 96, device="cuda", dtype=torch.bfloat16))
