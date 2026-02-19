import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.multimodal_gen.runtime.platforms import current_platform


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HS_HALF": 32}, num_warps=2),
        triton.Config({"BLOCK_HS_HALF": 64}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 128}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 256}, num_warps=8),
    ],
    key=["head_size", "interleaved"],
)
@triton.jit
def _rotary_embedding_kernel(
    output_ptr,
    x_ptr,
    cos_ptr,
    sin_ptr,
    num_heads,
    head_size,
    num_tokens,
    stride_x_row,
    stride_cos_row,
    stride_sin_row,
    interleaved: tl.constexpr,
    BLOCK_HS_HALF: tl.constexpr,
):
    row_idx = tl.program_id(0)
    token_idx = (row_idx // num_heads) % num_tokens

    x_row_ptr = x_ptr + row_idx * stride_x_row
    cos_row_ptr = cos_ptr + token_idx * stride_cos_row
    sin_row_ptr = sin_ptr + token_idx * stride_sin_row
    output_row_ptr = output_ptr + row_idx * stride_x_row

    # half size for x1 and x2
    head_size_half = head_size // 2

    for block_start in range(0, head_size_half, BLOCK_HS_HALF):
        offsets_half = block_start + tl.arange(0, BLOCK_HS_HALF)
        mask = offsets_half < head_size_half

        cos_vals = tl.load(cos_row_ptr + offsets_half, mask=mask, other=0.0)
        sin_vals = tl.load(sin_row_ptr + offsets_half, mask=mask, other=0.0)

        offsets_x1 = 2 * offsets_half
        offsets_x2 = 2 * offsets_half + 1

        x1_vals = tl.load(x_row_ptr + offsets_x1, mask=mask, other=0.0)
        x2_vals = tl.load(x_row_ptr + offsets_x2, mask=mask, other=0.0)

        x1_fp32 = x1_vals.to(tl.float32)
        x2_fp32 = x2_vals.to(tl.float32)
        cos_fp32 = cos_vals.to(tl.float32)
        sin_fp32 = sin_vals.to(tl.float32)
        o1_vals = tl.fma(-x2_fp32, sin_fp32, x1_fp32 * cos_fp32)
        o2_vals = tl.fma(x1_fp32, sin_fp32, x2_fp32 * cos_fp32)

        tl.store(output_row_ptr + offsets_x1, o1_vals.to(x1_vals.dtype), mask=mask)
        tl.store(output_row_ptr + offsets_x2, o2_vals.to(x2_vals.dtype), mask=mask)


def apply_rotary_embedding(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    output = torch.empty_like(x)

    if x.dim() > 3:
        bsz, num_tokens, num_heads, head_size = x.shape
    else:
        num_tokens, num_heads, head_size = x.shape
        bsz = 1

    assert head_size % 2 == 0, "head_size must be divisible by 2"

    x_reshaped = x.view(-1, head_size)
    output_reshaped = output.view(-1, head_size)

    # num_tokens per head, 1 token per block
    grid = (bsz * num_tokens * num_heads,)

    if interleaved and cos.shape[-1] == head_size:
        cos = cos[..., ::2].contiguous()
        sin = sin[..., ::2].contiguous()
    else:
        cos = cos.contiguous()
        sin = sin.contiguous()

    _rotary_embedding_kernel[grid](
        output_reshaped,
        x_reshaped,
        cos,
        sin,
        num_heads,
        head_size,
        num_tokens,
        x_reshaped.stride(0),
        cos.stride(0),
        sin.stride(0),
        interleaved,
    )

    return output


if current_platform.is_npu():
    from .npu_fallback import apply_rotary_embedding_native

    apply_rotary_embedding = apply_rotary_embedding_native
