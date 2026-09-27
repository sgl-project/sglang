"""Small-batch HC=4 post-mix with independent hidden-dimension CTAs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_post_split_h_kernel(X, R, P, C, Y, H: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    channel = tl.arange(0, 4)
    x = tl.load(X + token * H + h, h < H, 0).to(tl.float32)
    post = tl.load(P + token * 4 + channel)
    residual0 = tl.load(R + token * 4 * H + h, h < H, 0).to(tl.float32)
    comb0 = tl.load(C + token * 16 + channel)
    # Match NVCC's contraction in mhc_post_tilelang: round comb[0]*residual[0]
    # first, then FMA post*x into it. Reversing these two terms changes BF16
    # rounding on rare ties even though the symbolic expression is the same.
    acc = tl.fma(post[:, None], x[None, :], comb0[:, None] * residual0[None, :])
    for i in tl.static_range(1, 4):
        residual = tl.load(R + (token * 4 + i) * H + h, h < H, 0).to(tl.float32)
        comb = tl.load(C + token * 16 + i * 4 + channel)
        acc = tl.fma(comb[:, None], residual[None, :], acc)
    tl.store(Y + (token * 4 + channel[:, None]) * H + h[None, :], acc, h[None, :] < H)


def mhc_post_split_h(x, residual, post, comb):
    """Same result as the TileLang post kernel for contiguous BF16 HC=4 inputs."""
    assert x.dtype == residual.dtype == torch.bfloat16
    assert post.dtype == comb.dtype == torch.float32
    assert residual.shape == (x.shape[0], 4, x.shape[1])
    assert all(t.is_contiguous() for t in (x, residual, post, comb))
    output = torch.empty_like(residual)
    block = 128 if x.shape[0] <= 8 else 1024
    _mhc_post_split_h_kernel[(x.shape[0], triton.cdiv(x.shape[1], block))](
        x,
        residual,
        post,
        comb,
        output,
        H=x.shape[1],
        B=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
