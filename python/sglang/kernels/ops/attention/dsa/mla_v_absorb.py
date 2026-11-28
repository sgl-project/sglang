"""MLA's v absorb (``attn_output @ w_vc``, a head at a time) for a decode-sized
batch: one program a (head, 16-column block) over every token, fp32 accumulate,
bf16 out into [tokens, heads, v_head_dim] -- what ``torch.bmm`` writes there,
without hipBLASLt's tile for a 6-row problem."""

import torch
import triton
import triton.language as tl

MAX_TOKENS = 64


@triton.jit
def _v_absorb_kernel(
    X, W, Y, T,
    sx_t, sx_h,
    sw_h, sw_k, sw_n,
    sy_t, sy_h,
    K: tl.constexpr, N: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):  # fmt: skip
    h = tl.program_id(0)
    nb = tl.program_id(1)
    t = tl.arange(0, BLOCK_T)
    n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            X + t[:, None] * sx_t + h * sx_h + k[None, :],
            mask=t[:, None] < T,
            other=0.0,
        )
        w = tl.load(W + h * sw_h + k[:, None] * sw_k + n[None, :] * sw_n)
        acc = tl.dot(x, w, acc)
    tl.store(
        Y + t[:, None] * sy_t + h * sy_h + n[None, :],
        acc.to(tl.bfloat16),
        mask=t[:, None] < T,
    )


def covered(attn_output: torch.Tensor, w_vc: torch.Tensor) -> bool:
    return (
        attn_output.dim() == 3
        and attn_output.dtype == torch.bfloat16
        and attn_output.stride(-1) == 1
        and 0 < attn_output.shape[0] <= MAX_TOKENS
        and w_vc.dtype == torch.bfloat16
        # a unit-stride K or N axis keeps the weight loads coalesced; DeepSeek-style
        # loaders keep w_vc as a transpose, with K unit-stride
        and (w_vc.stride(1) == 1 or w_vc.stride(2) == 1)
        and w_vc.shape[0] == attn_output.shape[1]
        and w_vc.shape[1] == attn_output.shape[2]
        and w_vc.shape[1] % 64 == 0
        and w_vc.shape[2] % 32 == 0
    )


def mla_v_absorb(
    attn_output: torch.Tensor, w_vc: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """attn_output [T, H, K] bf16 (last dim contiguous), w_vc [H, K, N] bf16 (K or N
    unit-stride), out [T, H, N] bf16 (last dim contiguous)."""
    T, H, K = attn_output.shape
    N = w_vc.shape[2]
    block_t = max(16, triton.next_power_of_2(T))
    block_k = 512 if K % 512 == 0 else 64
    _v_absorb_kernel[(H, N // 16)](
        attn_output, w_vc, out, T,
        attn_output.stride(0), attn_output.stride(1),
        w_vc.stride(0), w_vc.stride(1), w_vc.stride(2),
        out.stride(0), out.stride(1),
        K=K, N=N, BLOCK_T=block_t, BLOCK_N=16, BLOCK_K=block_k,
        num_warps=2,
    )  # fmt: skip
    return out
