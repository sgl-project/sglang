import torch
import triton
import triton.language as tl


@triton.jit
def silu_and_mul_kernel(out_ptr, x_ptr, d: tl.constexpr, d_padded: tl.constexpr):
    token_idx = tl.program_id(0)
    idx = tl.arange(0, d_padded)

    # Load A and B from x
    mask = idx < d  # Mask for values within the original range of d
    x_a = tl.load(x_ptr + token_idx * 2 * d + idx, mask=mask, other=0.0)
    x_b = tl.load(x_ptr + token_idx * 2 * d + d + idx, mask=mask, other=0.0)

    # SiLU activation on A and element-wise multiplication with B
    x_a_fp32 = x_a.to(tl.float32)
    x_silu = x_a_fp32 / (1.0 + tl.exp(-x_a_fp32))
    result = x_silu * x_b

    # Write result to output
    tl.store(out_ptr + token_idx * d + idx, result, mask=mask)


class _SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, x):
        d = x.shape[-1] // 2
        d_padded = 1 << (d - 1).bit_length()  # Calculate d_padded in Python
        num_tokens = x.shape[0]

        # Launch the Triton kernel
        silu_and_mul_kernel[(num_tokens,)](out, x, d=d, d_padded=d_padded)


def silu_and_mul_triton(out, x):
    _SiluAndMul.apply(out, x)
