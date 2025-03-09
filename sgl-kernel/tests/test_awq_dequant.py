import itertools
from typing import Optional, Tuple

import pytest
import torch
from sgl_kernel import awq_dequantize
from vllm import _custom_ops as ops


def vllm_awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> torch.Tensor:
    return ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)


def sglang_awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> torch.Tensor:
    return awq_dequantize(qweight, scales, qzeros)


@pytest.mark.parametrize(
    "qweight_row,qweight_col",
    list(
        itertools.product(
            [3584, 18944, 128, 256, 512, 1024], [448, 576, 4736, 16, 32, 64, 128]
        )
    ),
)
def test_awq_dequant_compare_implementations(
    qweight_row: int,
    qweight_col: int,
):
    device = torch.device("cuda")

    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (qweight_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )
    group_size = qweight_row
    scales_row = qweight_row // group_size
    scales_col = qweight_col * 8
    scales = torch.rand(scales_row, scales_col, dtype=torch.float16, device=device)
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (scales_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )

    # Run both implementations
    vllm_out = vllm_awq_dequantize(qweight, scales, qzeros)
    sglang_out = sglang_awq_dequantize(qweight, scales, qzeros)

    # Compare results
    torch.testing.assert_close(
        vllm_out.to(torch.float32), sglang_out.to(torch.float32), rtol=1e-3, atol=1e-5
    )


if __name__ == "__main__":
    # Run the specific test function directly
    pytest.main([__file__])
