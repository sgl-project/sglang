# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_TOKENS": 1, "BLOCK_HEADS": 1, "BLOCK_HALF": 64},
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 2, "BLOCK_HEADS": 1, "BLOCK_HALF": 64},
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 4, "BLOCK_HEADS": 1, "BLOCK_HALF": 64},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 1, "BLOCK_HEADS": 2, "BLOCK_HALF": 64},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 2, "BLOCK_HEADS": 2, "BLOCK_HALF": 64},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 4, "BLOCK_HEADS": 2, "BLOCK_HALF": 64},
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 1, "BLOCK_HEADS": 4, "BLOCK_HALF": 64},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 2, "BLOCK_HEADS": 4, "BLOCK_HALF": 64},
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 4, "BLOCK_HEADS": 4, "BLOCK_HALF": 64},
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 8, "BLOCK_HEADS": 2, "BLOCK_HALF": 64},
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 8, "BLOCK_HEADS": 4, "BLOCK_HALF": 64},
            num_warps=16,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 16, "BLOCK_HEADS": 1, "BLOCK_HALF": 64},
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 16, "BLOCK_HEADS": 2, "BLOCK_HALF": 64},
            num_warps=16,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 1, "BLOCK_HEADS": 8, "BLOCK_HALF": 64},
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 2, "BLOCK_HEADS": 8, "BLOCK_HALF": 64},
            num_warps=16,
        ),
        triton.Config(
            {"BLOCK_TOKENS": 4, "BLOCK_HEADS": 8, "BLOCK_HALF": 64},
            num_warps=16,
        ),
    ],
    key=["num_heads", "head_dim"],
)
@triton.jit
def _qkvs_small_kernel(
    img_q_ptr,
    img_k_ptr,
    img_v_ptr,
    txt_q_ptr,
    txt_k_ptr,
    txt_v_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    batch_size,
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
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    # All row-offset arithmetic stays in int64 for the >INT32_MAX production
    # video shapes (e.g. 115200 * 21504).
    total_tokens = img_tokens + txt_tokens
    img_blocks = tl.cdiv(img_tokens, BLOCK_TOKENS)

    pid = tl.program_id(0).to(tl.int64)
    batch = pid // (img_blocks + tl.cdiv(txt_tokens, BLOCK_TOKENS))
    block_in_batch = pid - batch * (img_blocks + tl.cdiv(txt_tokens, BLOCK_TOKENS))
    head_block = tl.program_id(1)

    heads = head_block * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    head_mask = heads < num_heads
    half = tl.arange(0, BLOCK_HALF)
    half_mask = half < head_dim // 2
    dims = tl.arange(0, 2 * BLOCK_HALF)
    dim_mask = dims < head_dim

    if block_in_batch < img_blocks:
        # ---- Image branch: rotate Q/K, copy V ----
        seqs = block_in_batch * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS).to(tl.int64)
        seq_mask = seqs < img_tokens

        cos_row = cos_ptr + seqs[:, None] * stride_cos + half[None, :]
        sin_row = sin_ptr + seqs[:, None] * stride_sin + half[None, :]
        cos = tl.load(
            cos_row, mask=seq_mask[:, None] & half_mask[None, :], other=0.0
        ).to(tl.float32)
        sin = tl.load(
            sin_row, mask=seq_mask[:, None] & half_mask[None, :], other=0.0
        ).to(tl.float32)

        # [T, H] combined mask
        th_mask = seq_mask[:, None] & head_mask[None, :]

        q_base = img_q_ptr + batch * stride_iqb
        k_base = img_k_ptr + batch * stride_ikb
        v_base = img_v_ptr + batch * stride_ivb

        # Contiguous full-row loads: [T, H, D]
        q_off = (
            seqs[:, None, None] * stride_iqs
            + heads[None, :, None] * stride_iqh
            + dims[None, None, :]
        )
        q = tl.load(
            q_base + q_off,
            mask=th_mask[:, :, None] & dim_mask[None, None, :],
            other=0.0,
        )
        k = tl.load(
            k_base
            + seqs[:, None, None] * stride_iks
            + heads[None, :, None] * stride_ikh
            + dims[None, None, :],
            mask=th_mask[:, :, None] & dim_mask[None, None, :],
            other=0.0,
        )
        v = tl.load(
            v_base
            + seqs[:, None, None] * stride_ivs
            + heads[None, :, None] * stride_ivh
            + dims[None, None, :],
            mask=th_mask[:, :, None] & dim_mask[None, None, :],
            other=0.0,
        )

        # Split even/odd halves: reshape [T, H, HALF, 2] then split on last dim.
        q_r = tl.reshape(q, (BLOCK_TOKENS, BLOCK_HEADS, BLOCK_HALF, 2))
        q0, q1 = tl.split(q_r)
        k_r = tl.reshape(k, (BLOCK_TOKENS, BLOCK_HEADS, BLOCK_HALF, 2))
        k0, k1 = tl.split(k_r)

        q0f = q0.to(tl.float32)
        q1f = q1.to(tl.float32)
        k0f = k0.to(tl.float32)
        k1f = k1.to(tl.float32)

        oq0 = tl.fma(-q1f, sin[:, None, :], q0f * cos[:, None, :])
        oq1 = tl.fma(q0f, sin[:, None, :], q1f * cos[:, None, :])
        ok0 = tl.fma(-k1f, sin[:, None, :], k0f * cos[:, None, :])
        ok1 = tl.fma(k0f, sin[:, None, :], k1f * cos[:, None, :])

        # Interleave back to full contiguous rows.
        oq = tl.interleave(oq0, oq1)  # [T, H, 2*HALF]
        ok = tl.interleave(ok0, ok1)

        plane_stride = batch_size * total_tokens * num_heads * head_dim
        out_row = (
            batch * total_tokens * num_heads * head_dim
            + seqs[:, None, None] * num_heads * head_dim
            + heads[None, :, None] * head_dim
            + dims[None, None, :]
        )
        out_mask = th_mask[:, :, None] & dim_mask[None, None, :]
        tl.store(output_ptr + out_row, oq, mask=out_mask)
        tl.store(output_ptr + plane_stride + out_row, ok, mask=out_mask)
        tl.store(output_ptr + 2 * plane_stride + out_row, v, mask=out_mask)
    else:
        # ---- Text branch: pure copy ----
        txt_block = block_in_batch - img_blocks
        seqs = txt_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS).to(tl.int64)
        seq_mask = seqs < txt_tokens
        th_mask = seq_mask[:, None] & head_mask[None, :]

        q_base = txt_q_ptr + batch * stride_tqb
        k_base = txt_k_ptr + batch * stride_tkb
        v_base = txt_v_ptr + batch * stride_tvb

        q = tl.load(
            q_base
            + seqs[:, None, None] * stride_tqs
            + heads[None, :, None] * stride_tqh
            + dims[None, None, :],
            mask=th_mask[:, :, None] & dim_mask[None, None, :],
            other=0.0,
        )
        k = tl.load(
            k_base
            + seqs[:, None, None] * stride_tks
            + heads[None, :, None] * stride_tkh
            + dims[None, None, :],
            mask=th_mask[:, :, None] & dim_mask[None, None, :],
            other=0.0,
        )
        v = tl.load(
            v_base
            + seqs[:, None, None] * stride_tvs
            + heads[None, :, None] * stride_tvh
            + dims[None, None, :],
            mask=th_mask[:, :, None] & dim_mask[None, None, :],
            other=0.0,
        )

        plane_stride = batch_size * total_tokens * num_heads * head_dim
        out_row = (
            batch * total_tokens * num_heads * head_dim
            + (img_tokens + seqs[:, None, None]) * num_heads * head_dim
            + heads[None, :, None] * head_dim
            + dims[None, None, :]
        )
        out_mask = th_mask[:, :, None] & dim_mask[None, None, :]
        tl.store(output_ptr + out_row, q, mask=out_mask)
        tl.store(output_ptr + plane_stride + out_row, k, mask=out_mask)
        tl.store(output_ptr + 2 * plane_stride + out_row, v, mask=out_mask)


def _grid_small(meta, batch_size, img_tokens, txt_tokens, num_heads):
    img_blocks = triton.cdiv(img_tokens, meta["BLOCK_TOKENS"])
    txt_blocks = triton.cdiv(txt_tokens, meta["BLOCK_TOKENS"])
    return (
        batch_size * (img_blocks + txt_blocks),
        triton.cdiv(num_heads, meta["BLOCK_HEADS"]),
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_TH": 8}, num_warps=2),
        triton.Config({"BLOCK_TH": 8}, num_warps=4),
        triton.Config({"BLOCK_TH": 16}, num_warps=4),
        triton.Config({"BLOCK_TH": 16}, num_warps=8),
        triton.Config({"BLOCK_TH": 32}, num_warps=8),
        triton.Config({"BLOCK_TH": 32}, num_warps=16),
        triton.Config({"BLOCK_TH": 64}, num_warps=16),
        triton.Config({"BLOCK_TH": 128}, num_warps=32),
    ],
    key=["num_heads", "head_dim"],
)
@triton.jit
def _qkvs_large_kernel(
    img_q_ptr,
    img_k_ptr,
    img_v_ptr,
    txt_q_ptr,
    txt_k_ptr,
    txt_v_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    batch_size,
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
    BLOCK_TH: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    total_tokens = img_tokens + txt_tokens
    img_rows = img_tokens * num_heads
    txt_rows = txt_tokens * num_heads
    img_row_blocks = tl.cdiv(img_rows, BLOCK_TH)
    txt_row_blocks = tl.cdiv(txt_rows, BLOCK_TH)

    pid = tl.program_id(0).to(tl.int64)
    blocks_per_batch = img_row_blocks + txt_row_blocks
    batch = pid // blocks_per_batch
    block_in_batch = pid - batch * blocks_per_batch

    dims = tl.arange(0, 2 * BLOCK_HALF)
    dim_mask = dims < head_dim
    half = tl.arange(0, BLOCK_HALF)
    half_mask = half < head_dim // 2

    if block_in_batch < img_row_blocks:
        rows = block_in_batch * BLOCK_TH + tl.arange(0, BLOCK_TH).to(tl.int64)
        row_mask = rows < img_rows
        seqs = rows // num_heads
        heads = rows - seqs * num_heads

        q_row = (
            img_q_ptr
            + batch * stride_iqb
            + seqs[:, None] * stride_iqs
            + heads[:, None] * stride_iqh
            + dims[None, :]
        )
        mask = row_mask[:, None] & dim_mask[None, :]
        q = tl.load(q_row, mask=mask, other=0.0)
        k = tl.load(
            img_k_ptr
            + batch * stride_ikb
            + seqs[:, None] * stride_iks
            + heads[:, None] * stride_ikh
            + dims[None, :],
            mask=mask,
            other=0.0,
        )
        v = tl.load(
            img_v_ptr
            + batch * stride_ivb
            + seqs[:, None] * stride_ivs
            + heads[:, None] * stride_ivh
            + dims[None, :],
            mask=mask,
            other=0.0,
        )

        cs_mask = row_mask[:, None] & half_mask[None, :]
        cos = tl.load(
            cos_ptr + seqs[:, None] * stride_cos + half[None, :],
            mask=cs_mask,
            other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + seqs[:, None] * stride_sin + half[None, :],
            mask=cs_mask,
            other=0.0,
        ).to(tl.float32)

        q0, q1 = tl.split(tl.reshape(q, (BLOCK_TH, BLOCK_HALF, 2)))
        k0, k1 = tl.split(tl.reshape(k, (BLOCK_TH, BLOCK_HALF, 2)))
        q0f = q0.to(tl.float32)
        q1f = q1.to(tl.float32)
        k0f = k0.to(tl.float32)
        k1f = k1.to(tl.float32)
        oq = tl.interleave(
            tl.fma(-q1f, sin, q0f * cos),
            tl.fma(q0f, sin, q1f * cos),
        )
        ok = tl.interleave(
            tl.fma(-k1f, sin, k0f * cos),
            tl.fma(k0f, sin, k1f * cos),
        )

        plane_stride = batch_size * total_tokens * num_heads * head_dim
        out_row = (
            batch * total_tokens * num_heads * head_dim
            + rows[:, None] * head_dim
            + dims[None, :]
        )
        tl.store(output_ptr + out_row, oq, mask=mask)
        tl.store(output_ptr + plane_stride + out_row, ok, mask=mask)
        tl.store(output_ptr + 2 * plane_stride + out_row, v, mask=mask)
    else:
        rows = (block_in_batch - img_row_blocks) * BLOCK_TH + tl.arange(0, BLOCK_TH).to(
            tl.int64
        )
        row_mask = rows < txt_rows
        seqs = rows // num_heads
        heads = rows - seqs * num_heads
        mask = row_mask[:, None] & dim_mask[None, :]

        q = tl.load(
            txt_q_ptr
            + batch * stride_tqb
            + seqs[:, None] * stride_tqs
            + heads[:, None] * stride_tqh
            + dims[None, :],
            mask=mask,
            other=0.0,
        )
        k = tl.load(
            txt_k_ptr
            + batch * stride_tkb
            + seqs[:, None] * stride_tks
            + heads[:, None] * stride_tkh
            + dims[None, :],
            mask=mask,
            other=0.0,
        )
        v = tl.load(
            txt_v_ptr
            + batch * stride_tvb
            + seqs[:, None] * stride_tvs
            + heads[:, None] * stride_tvh
            + dims[None, :],
            mask=mask,
            other=0.0,
        )

        plane_stride = batch_size * total_tokens * num_heads * head_dim
        out_row = (
            batch * total_tokens * num_heads * head_dim
            + (img_rows + rows[:, None]) * head_dim
            + dims[None, :]
        )
        tl.store(output_ptr + out_row, q, mask=mask)
        tl.store(output_ptr + plane_stride + out_row, k, mask=mask)
        tl.store(output_ptr + 2 * plane_stride + out_row, v, mask=mask)


def _grid_large(meta, batch_size, img_tokens, txt_tokens, num_heads):
    img_row_blocks = triton.cdiv(img_tokens * num_heads, meta["BLOCK_TH"])
    txt_row_blocks = triton.cdiv(txt_tokens * num_heads, meta["BLOCK_TH"])
    return (batch_size * (img_row_blocks + txt_row_blocks),)


def _validate(tensors, cos, sin):
    img_q = tensors[0]
    if any(x.ndim != 4 for x in tensors):
        raise ValueError("QKV tensors must have shape [B, S, H, D]")
    if any(not x.is_cuda or x.dtype != torch.bfloat16 for x in tensors):
        raise ValueError("QKV tensors must be CUDA bfloat16 tensors")
    if any(x.device != img_q.device for x in tensors):
        raise ValueError("QKV tensors must be on the same CUDA device")
    batch, img_tokens, num_heads, head_dim = img_q.shape
    txt_tokens = tensors[3].shape[1]
    expected_img = (batch, img_tokens, num_heads, head_dim)
    expected_txt = (batch, txt_tokens, num_heads, head_dim)
    if any(tuple(x.shape) != expected_img for x in (img_q, tensors[1], tensors[2])):
        raise ValueError("image QKV shapes must match")
    if any(
        tuple(x.shape) != expected_txt for x in (tensors[3], tensors[4], tensors[5])
    ):
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
    return batch, img_tokens, txt_tokens, num_heads, head_dim


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
    batch, img_tokens, txt_tokens, num_heads, head_dim = _validate(tensors, cos, sin)

    total_tokens = img_tokens + txt_tokens
    storage = torch.empty(
        (3, batch, total_tokens, num_heads, head_dim),
        device=img_q.device,
        dtype=img_q.dtype,
    )
    args = []
    for x in tensors:
        args.extend((x.stride(0), x.stride(1), x.stride(2)))

    total_rows = batch * total_tokens * num_heads
    with torch.cuda.device(img_q.device):
        if total_rows <= 16384:
            _qkvs_small_kernel[
                lambda meta: _grid_small(meta, batch, img_tokens, txt_tokens, num_heads)
            ](
                *tensors,
                cos,
                sin,
                storage,
                batch,
                img_tokens,
                txt_tokens,
                num_heads,
                head_dim,
                *args,
                cos.stride(0),
                sin.stride(0),
            )
        else:
            _qkvs_large_kernel[
                lambda meta: _grid_large(meta, batch, img_tokens, txt_tokens, num_heads)
            ](
                *tensors,
                cos,
                sin,
                storage,
                batch,
                img_tokens,
                txt_tokens,
                num_heads,
                head_dim,
                *args,
                cos.stride(0),
                sin.stride(0),
                BLOCK_HALF=64,
            )
    return tuple(storage.unbind(dim=0))
