"""Fused gather and dequantization for NVFP4 embedding tables."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_e2m1(code):
    # Bit 3 is the sign; bits 0-2 select the magnitude.
    magnitude_code = code & 0x7
    magnitude = tl.where(magnitude_code == 1, 0.5, 0.0)
    magnitude = tl.where(magnitude_code == 2, 1.0, magnitude)
    magnitude = tl.where(magnitude_code == 3, 1.5, magnitude)
    magnitude = tl.where(magnitude_code == 4, 2.0, magnitude)
    magnitude = tl.where(magnitude_code == 5, 3.0, magnitude)
    magnitude = tl.where(magnitude_code == 6, 4.0, magnitude)
    magnitude = tl.where(magnitude_code == 7, 6.0, magnitude)
    return tl.where((code & 0x8) != 0, -magnitude, magnitude)


@triton.jit
def _nvfp4_embedding_kernel(
    ids_ptr,
    codes_ptr,
    scales_ptr,
    global_scale_ptr,
    out_ptr,
    num_rows,
    embedding_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < embedding_dim

    # Large tables overflow int32 byte offsets.
    row = tl.load(ids_ptr + token).to(tl.int64)
    # Negative ids count from the end, like torch indexing in the fallback.
    row = tl.where(row < 0, row + num_rows, row)

    packed = tl.load(
        codes_ptr + row * (embedding_dim // 2) + cols // 2,
        mask=col_mask,
        other=0,
    )
    code = tl.where(cols % 2 == 0, packed & 0xF, packed >> 4)
    group_scale = tl.load(
        scales_ptr + row * (embedding_dim // group_size) + cols // group_size,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    global_scale = tl.load(global_scale_ptr)

    # Same multiplication order as the torch fallback, so results are bit-exact.
    value = _decode_e2m1(code) * (group_scale * global_scale)
    tl.store(out_ptr + token * embedding_dim + cols, value, mask=col_mask)


def nvfp4_embedding(
    input_: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Look up NVFP4 rows and dequantize them straight into the output.

    ``weight`` holds two E2M1 codes per byte (low nibble first), ``weight_scale``
    one E4M3 scale per ``group_size`` columns, and ``weight_scale_2`` the FP32
    global scale. Ids must be in ``[-num_rows, num_rows)``.
    """
    assert input_.is_cuda
    assert input_.dtype in (torch.int32, torch.int64)
    assert weight.dtype == torch.uint8
    assert weight.ndim == 2
    assert weight.is_contiguous()
    assert weight_scale.dtype == torch.float8_e4m3fn
    assert weight_scale.is_contiguous()
    assert weight_scale_2.dtype == torch.float32
    assert weight_scale_2.numel() == 1
    assert output_dtype in (torch.bfloat16, torch.float16)

    num_rows = weight.shape[0]
    embedding_dim = weight.shape[1] * 2
    assert embedding_dim % group_size == 0
    assert tuple(weight_scale.shape) == (num_rows, embedding_dim // group_size)

    flat_ids = input_.reshape(-1).contiguous()
    output = torch.empty(
        (flat_ids.numel(), embedding_dim), dtype=output_dtype, device=weight.device
    )
    if flat_ids.numel() == 0:
        return output.view(*input_.shape, embedding_dim)

    block_d = min(1024, triton.next_power_of_2(embedding_dim))
    grid = (flat_ids.numel(), triton.cdiv(embedding_dim, block_d))
    _nvfp4_embedding_kernel[grid](
        flat_ids,
        weight,
        weight_scale,
        weight_scale_2,
        output,
        num_rows,
        embedding_dim=embedding_dim,
        group_size=group_size,
        BLOCK_D=block_d,
    )
    return output.view(*input_.shape, embedding_dim)
