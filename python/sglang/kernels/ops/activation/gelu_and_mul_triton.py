import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_is_hip = is_hip()


# gelu on first half of vector
@triton.jit
def gelu_and_mul_kernel(
    out_hidden_states_ptr,  # (bs, hidden_dim)
    hidden_states_ptr,  # (bs, hidden_dim * 2)
    hidden_dim: tl.constexpr,  # the output hidden_dim
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    input_start = pid * hidden_dim * 2
    output_start = pid * hidden_dim

    input1_offs = tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim  # shared for input1, input3, output
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)
    output_offs = tl.arange(0, BLOCK_SIZE)

    x1 = tl.load(
        hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.load(
        hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0
    ).to(tl.float32)

    # gelu
    # cast down before mul to better match training?
    gelu_x1 = 0.5 * (1.0 + tl.erf(x1 * 0.7071067811865475)) * x1
    out = x3 * gelu_x1.to(hidden_states_ptr.dtype.element_ty)

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)


def gelu_and_mul_triton(hidden_states):
    bs, in_hidden_dim = hidden_states.shape
    hidden_dim = in_hidden_dim // 2
    out_hidden_states = torch.empty(
        (bs, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    max_warps = 16 if _is_hip else 32
    config = {
        # 8 ele per thread (not tuned)
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
        ),
    }

    gelu_and_mul_kernel[(bs,)](
        out_hidden_states,
        hidden_states,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        **config,
    )
    return out_hidden_states
