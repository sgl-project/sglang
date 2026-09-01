import math

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.elementwise import (
    _fused_q_indexer_rope_hadamard_quant_triton,
    fused_q_indexer_rope_hadamard_quant,
)
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


def _hadamard_matrix(n: int, device: torch.device) -> torch.Tensor:
    matrix = torch.ones(1, 1, dtype=torch.float32, device=device)
    while matrix.shape[0] < n:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(n)


def _reference(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
):
    q_rotated = q_input.float().clone()
    rope = q_rotated[..., -64:].reshape(*q_rotated.shape[:-1], 32, 2)
    freq = freqs_cis[positions].unsqueeze(1)
    real = rope[..., 0] * freq.real - rope[..., 1] * freq.imag
    imag = rope[..., 0] * freq.imag + rope[..., 1] * freq.real
    q_rotated[..., -64:] = torch.stack((real, imag), dim=-1).flatten(-2)
    q_rotated = q_rotated.to(q_input.dtype).float()
    q_hadamard = q_rotated @ _hadamard_matrix(128, q_input.device)

    fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    q_scale = q_hadamard.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / fp8_max
    q_fp8 = (q_hadamard / q_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    weights_out = weight.float().unsqueeze(-1) * weight_scale * q_scale
    return q_fp8, weights_out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("batch_size", [1, 7])
def test_indexer_q_triton_matches_reference(batch_size: int):
    torch.manual_seed(43)
    device = torch.device("cuda")
    num_heads = 4
    weight_scale = 0.125
    q_input = torch.randn(
        batch_size, num_heads, 128, dtype=torch.bfloat16, device=device
    )
    weight = torch.randn(batch_size, num_heads, dtype=torch.bfloat16, device=device)
    angles = torch.randn(256, 32, dtype=torch.float32, device=device)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    positions = torch.randint(0, 256, (batch_size,), dtype=torch.int32, device=device)

    actual_q, actual_weight = _fused_q_indexer_rope_hadamard_quant_triton(
        q_input, weight, weight_scale, freqs_cis, positions
    )
    expected_q, expected_weight = _reference(
        q_input, weight, weight_scale, freqs_cis, positions
    )
    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual_weight, expected_weight, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.skipif(
    not is_hip()
    or hasattr(torch.ops.sgl_kernel, "dsv4_fused_q_indexer_rope_hadamard_quant"),
    reason="requires ROCm without the native operator",
)
def test_indexer_q_public_dispatch_uses_triton_fallback():
    torch.manual_seed(8)
    device = torch.device("cuda")
    q_input = torch.randn(2, 3, 128, dtype=torch.bfloat16, device=device)
    weight = torch.randn(2, 3, dtype=torch.bfloat16, device=device)
    angles = torch.randn(32, 32, dtype=torch.float32, device=device)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    positions = torch.tensor([1, 9], dtype=torch.int32, device=device)

    expected = _fused_q_indexer_rope_hadamard_quant_triton(
        q_input, weight, 0.25, freqs_cis, positions
    )
    actual = fused_q_indexer_rope_hadamard_quant(
        q_input, weight, 0.25, freqs_cis, positions
    )

    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
