# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HEADS": 1, "BLOCK_HALF": 64}, num_warps=2),
        triton.Config({"BLOCK_HEADS": 2, "BLOCK_HALF": 64}, num_warps=4),
        triton.Config({"BLOCK_HEADS": 4, "BLOCK_HALF": 64}, num_warps=4),
        triton.Config({"BLOCK_HEADS": 8, "BLOCK_HALF": 64}, num_warps=8),
    ],
    key=["num_heads", "head_dim"],
)
@triton.jit
def _hunyuan_qkv_rope_pack_kernel(
    img_q_ptr,
    img_k_ptr,
    img_v_ptr,
    txt_q_ptr,
    txt_k_ptr,
    txt_v_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    img_tokens,
    txt_tokens,
    num_heads,
    head_dim,
    stride_iqb,
    stride_iqs,
    stride_iqh,
    stride_ikb,
    stride_iks,
    stride_ikh,
    stride_ivb,
    stride_ivs,
    stride_ivh,
    stride_tqb,
    stride_tqs,
    stride_tqh,
    stride_tkb,
    stride_tks,
    stride_tkh,
    stride_tvb,
    stride_tvs,
    stride_tvh,
    stride_cos,
    stride_sin,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    token = tl.program_id(0)
    head_block = tl.program_id(1)
    total_tokens = img_tokens + txt_tokens
    batch = token // total_tokens
    seq = token - batch * total_tokens

    heads = head_block * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    head_mask = heads < num_heads
    half = tl.arange(0, BLOCK_HALF)
    half_mask = half < head_dim // 2
    mask = head_mask[:, None] & half_mask[None, :]
    even = 2 * half
    odd = even + 1

    output_row = (
        batch * total_tokens * num_heads * head_dim
        + seq * num_heads * head_dim
        + heads[:, None] * head_dim
    )
    plane_stride = tl.num_programs(0) * num_heads * head_dim

    if seq < img_tokens:
        q_row = (
            img_q_ptr
            + batch * stride_iqb
            + seq * stride_iqs
            + heads[:, None] * stride_iqh
        )
        k_row = (
            img_k_ptr
            + batch * stride_ikb
            + seq * stride_iks
            + heads[:, None] * stride_ikh
        )
        v_row = (
            img_v_ptr
            + batch * stride_ivb
            + seq * stride_ivs
            + heads[:, None] * stride_ivh
        )
        cos_row = cos_ptr + seq * stride_cos + half
        sin_row = sin_ptr + seq * stride_sin + half
        cos = tl.load(cos_row, mask=half_mask, other=0.0).to(tl.float32)[None, :]
        sin = tl.load(sin_row, mask=half_mask, other=0.0).to(tl.float32)[None, :]

        q0 = tl.load(q_row + even[None, :], mask=mask, other=0.0)
        q1 = tl.load(q_row + odd[None, :], mask=mask, other=0.0)
        k0 = tl.load(k_row + even[None, :], mask=mask, other=0.0)
        k1 = tl.load(k_row + odd[None, :], mask=mask, other=0.0)
        v0 = tl.load(v_row + even[None, :], mask=mask, other=0.0)
        v1 = tl.load(v_row + odd[None, :], mask=mask, other=0.0)

        q0f, q1f = q0.to(tl.float32), q1.to(tl.float32)
        k0f, k1f = k0.to(tl.float32), k1.to(tl.float32)
        oq0 = tl.fma(-q1f, sin, q0f * cos)
        oq1 = tl.fma(q0f, sin, q1f * cos)
        ok0 = tl.fma(-k1f, sin, k0f * cos)
        ok1 = tl.fma(k0f, sin, k1f * cos)
    else:
        txt_seq = seq - img_tokens
        q_row = (
            txt_q_ptr
            + batch * stride_tqb
            + txt_seq * stride_tqs
            + heads[:, None] * stride_tqh
        )
        k_row = (
            txt_k_ptr
            + batch * stride_tkb
            + txt_seq * stride_tks
            + heads[:, None] * stride_tkh
        )
        v_row = (
            txt_v_ptr
            + batch * stride_tvb
            + txt_seq * stride_tvs
            + heads[:, None] * stride_tvh
        )
        oq0 = tl.load(q_row + even[None, :], mask=mask, other=0.0).to(tl.float32)
        oq1 = tl.load(q_row + odd[None, :], mask=mask, other=0.0).to(tl.float32)
        ok0 = tl.load(k_row + even[None, :], mask=mask, other=0.0).to(tl.float32)
        ok1 = tl.load(k_row + odd[None, :], mask=mask, other=0.0).to(tl.float32)
        v0 = tl.load(v_row + even[None, :], mask=mask, other=0.0)
        v1 = tl.load(v_row + odd[None, :], mask=mask, other=0.0)

    tl.store(output_ptr + output_row + even[None, :], oq0, mask=mask)
    tl.store(output_ptr + output_row + odd[None, :], oq1, mask=mask)
    tl.store(output_ptr + plane_stride + output_row + even[None, :], ok0, mask=mask)
    tl.store(output_ptr + plane_stride + output_row + odd[None, :], ok1, mask=mask)
    tl.store(output_ptr + 2 * plane_stride + output_row + even[None, :], v0, mask=mask)
    tl.store(output_ptr + 2 * plane_stride + output_row + odd[None, :], v1, mask=mask)


def hunyuan_qkv_rope_pack(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = (img_q, img_k, img_v, txt_q, txt_k, txt_v)
    if any(x.ndim != 4 for x in tensors):
        raise ValueError("QKV tensors must have shape [B, S, H, D]")
    if any(not x.is_cuda or x.dtype != torch.bfloat16 for x in tensors):
        raise ValueError("QKV tensors must be CUDA bfloat16 tensors")
    if any(x.device != img_q.device for x in tensors):
        raise ValueError("QKV tensors must be on the same CUDA device")
    batch, img_tokens, num_heads, head_dim = img_q.shape
    txt_tokens = txt_q.shape[1]
    expected_img = (batch, img_tokens, num_heads, head_dim)
    expected_txt = (batch, txt_tokens, num_heads, head_dim)
    if any(tuple(x.shape) != expected_img for x in (img_q, img_k, img_v)):
        raise ValueError("image QKV shapes must match")
    if any(tuple(x.shape) != expected_txt for x in (txt_q, txt_k, txt_v)):
        raise ValueError("text QKV shapes must match")
    if any(x.stride(-1) != 1 for x in tensors):
        raise ValueError("QKV last dimensions must be contiguous")
    if head_dim <= 0 or head_dim > 128 or head_dim % 2:
        raise ValueError("head_dim must be positive, even, and <= 128")
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("cos and sin must have matching [S, D/2] shapes")
    if cos.shape[0] < img_tokens or cos.shape[1] != head_dim // 2:
        raise ValueError("cos/sin shape does not cover image tokens and head_dim")
    if not cos.is_cuda or not sin.is_cuda or cos.stride(-1) != 1 or sin.stride(-1) != 1:
        raise ValueError("cos and sin must be CUDA and last-dim contiguous")
    if cos.device != img_q.device or sin.device != img_q.device:
        raise ValueError("QKV and cos/sin tensors must be on the same CUDA device")

    total_tokens = img_tokens + txt_tokens
    storage = torch.empty(
        (3, batch, total_tokens, num_heads, head_dim),
        device=img_q.device,
        dtype=img_q.dtype,
    )
    args = []
    for x in tensors:
        args.extend((x.stride(0), x.stride(1), x.stride(2)))
    with torch.cuda.device(img_q.device):
        _hunyuan_qkv_rope_pack_kernel[
            lambda meta: (
                batch * total_tokens,
                triton.cdiv(num_heads, meta["BLOCK_HEADS"]),
            )
        ](
            *tensors,
            cos,
            sin,
            storage,
            img_tokens,
            txt_tokens,
            num_heads,
            head_dim,
            *args,
            cos.stride(0),
            sin.stride(0),
        )
    return tuple(storage.unbind(dim=0))
