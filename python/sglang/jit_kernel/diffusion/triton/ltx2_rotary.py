import torch
import triton
import triton.language as tl


@triton.jit
def _ltx2_split_rotary_kernel(
    out_ptr,
    x_ptr,
    cos_ptr,
    sin_ptr,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    stride_cos_b: tl.constexpr,
    stride_cos_h: tl.constexpr,
    stride_cos_t: tl.constexpr,
    stride_sin_b: tl.constexpr,
    stride_sin_h: tl.constexpr,
    stride_sin_t: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    head_block = tl.program_id(1)
    batch = pid_bt // seq_len
    token = pid_bt - batch * seq_len
    heads = head_block * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    offsets = tl.arange(0, BLOCK_HALF)
    mask = (heads[:, None] < num_heads) & (offsets[None, :] < half_dim)

    x_base = ((batch * seq_len + token) * num_heads + heads[:, None]) * head_dim
    cos_base = (
        batch * stride_cos_b + heads[:, None] * stride_cos_h + token * stride_cos_t
    )
    sin_base = (
        batch * stride_sin_b + heads[:, None] * stride_sin_h + token * stride_sin_t
    )

    x_first = tl.load(x_ptr + x_base + offsets[None, :], mask=mask, other=0.0)
    x_second = tl.load(
        x_ptr + x_base + half_dim + offsets[None, :], mask=mask, other=0.0
    )
    cos = tl.load(cos_ptr + cos_base + offsets[None, :], mask=mask, other=0.0)
    sin = tl.load(sin_ptr + sin_base + offsets[None, :], mask=mask, other=0.0)

    # Match the original PyTorch order: x * cos is written as BF16 first, then
    # addcmul_ computes the sine product in FP32 before the final BF16 store.
    out_first = (x_first * cos).to(tl.bfloat16).to(tl.float32) + (
        -x_second.to(tl.float32) * sin.to(tl.float32)
    )
    out_second = (x_second * cos).to(tl.bfloat16).to(tl.float32) + (
        x_first.to(tl.float32) * sin.to(tl.float32)
    )

    tl.store(out_ptr + x_base + offsets[None, :], out_first, mask=mask)
    tl.store(out_ptr + x_base + half_dim + offsets[None, :], out_second, mask=mask)


def apply_ltx2_split_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    batch, seq_len, inner_dim = x.shape
    cos_batch, num_heads, cos_seq_len, half_dim = cos.shape
    head_dim = half_dim * 2
    if (
        cos_batch != batch
        or cos_seq_len != seq_len
        or inner_dim != num_heads * head_dim
        or sin.shape != cos.shape
    ):
        raise ValueError(
            "LTX2 split RoPE shape mismatch: "
            f"x={tuple(x.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
        )

    out = torch.empty_like(x)
    block_half = triton.next_power_of_2(half_dim)
    block_heads = min(16, triton.next_power_of_2(num_heads))
    num_warps = min(8, max(1, block_heads))
    grid = (batch * seq_len, triton.cdiv(num_heads, block_heads))
    _ltx2_split_rotary_kernel[grid](
        out,
        x,
        cos,
        sin,
        seq_len,
        num_heads,
        head_dim,
        half_dim,
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        BLOCK_HEADS=block_heads,
        BLOCK_HALF=block_half,
        num_warps=num_warps,
    )
    return out
