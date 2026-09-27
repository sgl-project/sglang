"""HC=4 post-mix and pre-combine with the original BF16 intermediates."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_post_combine(X, R, P, C, A, RO, Y, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    mask = h < H
    x = tl.load(X + row * H + h, mask, 0).to(tl.float32)
    r0 = tl.load(R + (row * 4 + 0) * H + h, mask, 0).to(tl.float32)
    r1 = tl.load(R + (row * 4 + 1) * H + h, mask, 0).to(tl.float32)
    r2 = tl.load(R + (row * 4 + 2) * H + h, mask, 0).to(tl.float32)
    r3 = tl.load(R + (row * 4 + 3) * H + h, mask, 0).to(tl.float32)
    collapsed = tl.full((B,), 0, tl.float32)
    for i in tl.static_range(4):
        post = tl.load(P + row * 4 + i)
        c0 = tl.load(C + row * 16 + i)
        c1 = tl.load(C + row * 16 + 4 + i)
        c2 = tl.load(C + row * 16 + 8 + i)
        c3 = tl.load(C + row * 16 + 12 + i)
        pre = tl.load(A + row * 4 + i)
        # Match mhc_post_split_h's contraction order and BF16 store.
        mixed = tl.fma(post, x, c0 * r0)
        mixed = tl.fma(c1, r1, mixed)
        mixed = tl.fma(c2, r2, mixed)
        mixed = tl.fma(c3, r3, mixed)
        rounded = mixed.to(tl.bfloat16)
        tl.store(RO + (row * 4 + i) * H + h, rounded, mask)
        # Match hc_combine's sequential FP32 accumulation, then BF16 store.
        collapsed = tl.fma(pre, rounded.to(tl.float32), collapsed)
    tl.store(Y + row * H + h, collapsed, mask)


def mhc_post_combine(x, residual, post, comb, pre):
    """Return (updated HC streams, collapsed BF16 input) out of place."""
    assert x.dtype == residual.dtype == torch.bfloat16
    assert post.dtype == comb.dtype == pre.dtype == torch.float32
    assert x.ndim == 2 and x.shape[1] == 5120
    assert residual.shape == (x.shape[0], 4, x.shape[1])
    assert post.shape == pre.shape == (x.shape[0], 4)
    assert comb.shape == (x.shape[0], 4, 4)
    assert all(t.is_contiguous() for t in (x, residual, post, comb, pre))
    updated = torch.empty_like(residual)
    combined = torch.empty_like(x)
    if x.shape[0]:
        block = 1024 if x.shape[0] <= 192 or x.shape[0] >= 4096 else 512
        _mhc_post_combine[(x.shape[0], triton.cdiv(x.shape[1], block))](
            x,
            residual,
            post,
            comb,
            pre,
            updated,
            combined,
            H=x.shape[1],
            B=block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return updated, combined


@triton.jit
def _hc_norm_prefill(X, W, Y, EPS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, 8192)
    value = tl.load(X + row * 5120 + h, h < 5120, 0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(value * value, 0) / 5120 + EPS)
    weight = tl.load(W + h, h < 5120, 0).to(tl.float32)
    tl.store(Y + row * 5120 + h, value * inv_rms * weight, h < 5120)


def hc_norm_prefill(combined, weight, eps):
    """Keep the original fused prefill norm's arithmetic on collapsed input."""
    assert 4096 <= combined.shape[0] <= 65536 and combined.shape[1] == 5120
    assert weight.shape == (5120,)
    assert combined.dtype == weight.dtype == torch.bfloat16
    assert combined.is_contiguous() and weight.is_contiguous()
    output = torch.empty_like(combined)
    _hc_norm_prefill[(combined.shape[0],)](combined, weight, output, eps, num_warps=4)
    return output
