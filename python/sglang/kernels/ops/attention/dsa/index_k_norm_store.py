"""The DSA indexer's decode index-K write in one launch: k_norm (LayerNorm,
fp32 weights) as triton_layer_norm computes it, then aiter
indexer_k_quant_and_cache's ue8m0 FP8 quantization into the preshuffled
page_size-64 cache (16x16 token x dim tiles, one fp32 scale a token after the
page's K bytes). Slots < 0 are skipped, as aiter skips them."""

import torch
import triton
import triton.language as tl


@triton.jit
def _index_k_norm_store_kernel(
    K,
    W,
    B,
    SLOT,
    CACHE_Q,
    CACHE_S,
    k_stride,
    eps,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    TOK_STRIDE: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(SLOT + t)
    if slot < 0:
        return
    cols = tl.arange(0, D)
    # triton_layer_norm's arithmetic (layernorm_gated._layer_norm_fwd_1pass_kernel)
    x = tl.load(K + t * k_stride + cols).to(tl.float32)[None, :]
    mean = tl.sum(x, axis=1) / D
    xbar = x - mean[:, None]
    var = tl.sum(xbar * xbar, axis=1) / D
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    y = (x - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]
    y = y.to(tl.bfloat16).to(tl.float32)
    # aiter indexer_k_quant_and_cache, ue8m0
    amax = tl.max(tl.abs(y), axis=1)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, 1e-4) / 448.0)))
    q = tl.clamp(y / scale[:, None], -448.0, 448.0).to(CACHE_Q.dtype.element_ty)
    slot = slot.to(tl.int64)
    page_base = (slot // PAGE) * (PAGE * TOK_STRIDE)
    tok = slot % PAGE
    dst = (
        page_base
        + (tok // 16) * (16 * D)
        + (cols // 16) * 256
        + (tok % 16) * 16
        + cols % 16
    )
    tl.store(CACHE_Q + dst[None, :], q)
    tl.store(CACHE_S + (page_base + PAGE * D + tok * 4) // 4 + tl.arange(0, 1), scale)


def index_k_norm_store(
    key: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """key [tokens, 128] bf16 (pre-norm), the fp32 LayerNorm weight / bias, the
    index-K cache as fp8 [pages, 64, 132] (or any view of its bytes) and int64
    slots."""
    tokens, d = key.shape
    assert d == 128 and key.dtype == torch.bfloat16 and key.stride(-1) == 1
    assert norm_weight.dtype == torch.float32 and norm_bias.dtype == torch.float32
    if tokens == 0:
        return
    flat = kv_cache.reshape(-1)
    _index_k_norm_store_kernel[(tokens,)](
        key,
        norm_weight,
        norm_bias,
        slot_mapping,
        flat,
        flat.view(torch.uint8).view(torch.float32),
        key.stride(0),
        eps,
        D=128,
        PAGE=64,
        TOK_STRIDE=132,
    )
