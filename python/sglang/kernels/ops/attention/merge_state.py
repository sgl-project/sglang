"""Pairwise attention-state merge, including strided and in-place CP states."""

import itertools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "L_TOKEN_STRIDE",
        "L_HEAD_STRIDE",
        "PL_TOKEN_STRIDE",
        "PL_HEAD_STRIDE",
        "SL_TOKEN_STRIDE",
        "SL_HEAD_STRIDE",
    ],
    do_not_specialize_on_alignment=["output_lse", "prefix_lse", "suffix_lse"],
)
def merge_state_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_merged
    output_lse,  # [NUM_TOKENS, NUM_HEADS] s_merged
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_a
    prefix_lse,  # [NUM_TOKENS, NUM_HEADS] s_a
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] v_b
    suffix_lse,  # [NUM_TOKENS, NUM_HEADS] s_b
    O_STRIDES: tl.constexpr,
    L_TOKEN_STRIDE,
    L_HEAD_STRIDE,
    P_STRIDES: tl.constexpr,
    PL_TOKEN_STRIDE,
    PL_HEAD_STRIDE,
    S_STRIDES: tl.constexpr,
    SL_TOKEN_STRIDE,
    SL_HEAD_STRIDE,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    p_lse = tl.load(prefix_lse + token * PL_TOKEN_STRIDE + head * PL_HEAD_STRIDE).to(
        tl.float32
    )
    s_lse = tl.load(suffix_lse + token * SL_TOKEN_STRIDE + head * SL_HEAD_STRIDE).to(
        tl.float32
    )
    # Backends use either infinity (and DCP can supply NaN) for empty rows.
    p_lse = tl.where((p_lse != p_lse) | (p_lse == float("inf")), float("-inf"), p_lse)
    s_lse = tl.where((s_lse != s_lse) | (s_lse == float("inf")), float("-inf"), s_lse)
    maximum = tl.maximum(p_lse, s_lse)
    anchor = tl.where(maximum == float("-inf"), 0.0, maximum)
    if IS_BASE_E:
        a, b = tl.exp(p_lse - anchor), tl.exp(s_lse - anchor)
    else:
        a, b = tl.exp2(p_lse - anchor), tl.exp2(s_lse - anchor)
    denominator = a + b
    denominator = tl.where(denominator == 0.0, 1.0, denominator)
    d = tl.arange(0, PADDED_HEAD_SIZE)
    p_out = tl.load(
        prefix_output + token * P_STRIDES[0] + head * P_STRIDES[1] + d * P_STRIDES[2],
        mask=d < HEAD_SIZE,
        other=0,
    ).to(tl.float32)
    s_out = tl.load(
        suffix_output + token * S_STRIDES[0] + head * S_STRIDES[1] + d * S_STRIDES[2],
        mask=d < HEAD_SIZE,
        other=0,
    ).to(tl.float32)
    # An empty partial may contain NaNs. It must contribute the identity state.
    result = tl.where(a == 0.0, 0.0, p_out * (a / denominator))
    result += tl.where(b == 0.0, 0.0, s_out * (b / denominator))
    tl.store(
        output + token * O_STRIDES[0] + head * O_STRIDES[1] + d * O_STRIDES[2],
        result,
        mask=d < HEAD_SIZE,
    )
    if IS_BASE_E:
        merged_lse = maximum + tl.log(denominator)
    else:
        merged_lse = maximum + tl.log2(denominator)
    # With in-place LSE, all warps must read the old value before it is replaced.
    tl.debug_barrier()
    tl.store(output_lse + token * L_TOKEN_STRIDE + head * L_HEAD_STRIDE, merged_lse)


def merge_state_triton(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
    *,
    is_lse_base_on_e: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Merge [T,H,D] outputs and [T,H] LSE; exact input/output aliases are allowed.

    Inputs may have different dtypes and noncontiguous strides. Accumulation
    uses FP32; the caller selects output storage dtype. Rows must not overlap.
    """
    if output is None:
        output = torch.empty_like(prefix_output)
    if output_lse is None:
        output_lse = torch.empty_like(prefix_lse)
    num_tokens, num_heads, head_size = output.shape
    if num_tokens and num_heads:
        merge_state_kernel[(num_tokens, num_heads)](
            output,
            output_lse,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output.stride(),
            output_lse.stride(0),
            output_lse.stride(1),
            prefix_output.stride(),
            prefix_lse.stride(0),
            prefix_lse.stride(1),
            suffix_output.stride(),
            suffix_lse.stride(0),
            suffix_lse.stride(1),
            head_size,
            triton.next_power_of_2(head_size),
            is_lse_base_on_e,
        )
    return output, output_lse


def warmup_merge_state(device, dtype, num_heads, head_size):
    tokens = 64
    shape = (tokens, num_heads, head_size)
    # Ring accumulation can retain FP32 while attention partials remain BF16.
    for prefix_dtype, suffix_dtype, output_dtype in itertools.product(
        (dtype, torch.float32), repeat=3
    ):
        prefix = torch.zeros(shape, device=device, dtype=prefix_dtype)
        suffix = torch.zeros(shape, device=device, dtype=suffix_dtype)
        output = torch.empty(shape, device=device, dtype=output_dtype)
        prefix_lse = torch.zeros(
            (num_heads, tokens), device=device, dtype=torch.float32
        ).T
        suffix_lse = torch.zeros_like(prefix_lse)
        output_lse = torch.empty_like(prefix_lse)
        for base_e in (True, False):
            merge_state_triton(
                prefix,
                prefix_lse,
                suffix,
                suffix_lse,
                output,
                output_lse,
                is_lse_base_on_e=base_e,
            )
