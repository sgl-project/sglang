import itertools

import pytest
import torch

from sglang.jit_kernel.awq_dequantize import awq_dequantize as jit_awq_dequantize

try:
    from sgl_kernel import awq_dequantize as aot_awq_dequantize

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False


def reverse_awq_order(t: torch.Tensor):
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = torch.arange(
        t.shape[-1],
        dtype=torch.int32,
        device=t.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    t = t[:, reverse_order_tensor] & 0xF
    return t


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32
def awq_dequantize_torch(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int
) -> torch.Tensor:
    if group_size == -1:
        group_size = qweight.shape[0]

    bits = 4
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8
    )

    iweights = iweights.view(iweights.shape[0], -1)

    zeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
        torch.int8
    )
    zeros = zeros.view(qzeros.shape[0], -1)
    zeros = reverse_awq_order(zeros)

    iweights = reverse_awq_order(iweights)

    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights - zeros) * scales


@pytest.mark.parametrize(
    "qweight_row,qweight_col,is_bf16_act",
    list(
        itertools.product(
            [128, 256, 512, 1024, 3584],
            [16, 32, 64, 128, 448],
            [True, False],
        )
    ),
)
def test_awq_dequantize_jit_vs_torch(
    qweight_row: int, qweight_col: int, is_bf16_act: bool
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

    if is_bf16_act:
        scales = torch.rand(scales_row, scales_col, dtype=torch.bfloat16, device=device)
    else:
        scales = torch.rand(scales_row, scales_col, dtype=torch.float16, device=device)

    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (scales_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )

    # Run both implementations
    torch_out = awq_dequantize_torch(qweight, scales, qzeros, group_size)
    jit_out = jit_awq_dequantize(qweight, scales, qzeros)

    # Compare results (approximate due to different computation paths)
    torch.testing.assert_close(
        torch_out.to(torch.float32), jit_out.to(torch.float32), rtol=1e-3, atol=1e-5
    )


@pytest.mark.parametrize(
    "qweight_row,qweight_col,is_bf16_act",
    list(
        itertools.product(
            [128, 256, 512, 1024, 3584],
            [16, 32, 64, 128, 448],
            [True, False],
        )
    ),
)
def test_awq_dequantize_jit_vs_aot(
    qweight_row: int, qweight_col: int, is_bf16_act: bool
):
    if not AOT_AVAILABLE:
        pytest.skip("sgl_kernel AOT not available")

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

    if is_bf16_act:
        scales = torch.rand(scales_row, scales_col, dtype=torch.bfloat16, device=device)
    else:
        scales = torch.rand(scales_row, scales_col, dtype=torch.float16, device=device)

    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (scales_row, qweight_col),
        dtype=torch.int32,
        device=device,
    )

    # Run both implementations
    aot_out = aot_awq_dequantize(qweight, scales, qzeros)
    jit_out = jit_awq_dequantize(qweight, scales, qzeros)

    # Bitwise equality
    torch.testing.assert_close(jit_out, aot_out, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__])
