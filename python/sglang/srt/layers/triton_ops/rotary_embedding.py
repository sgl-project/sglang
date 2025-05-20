from typing import Optional

import torch
import triton
import triton.language as tl


# From FlagGems
@triton.jit
def apply_rotary_pos_emb_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    s_id = tl.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM
    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    oq_ptr += s_id * oq_stride_s
    q_ptr += s_id * q_stride_s

    for off_h in range(0, NUM_Q_HEADS):
        ordered_cols = off_h * q_stride_h + (ordered_block * q_stride_d)
        rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)
        output_offs = off_h * oq_stride_h + (ordered_block * oq_stride_d)

        q = tl.load(q_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)
        y = q * cos + rotated_q * sin
        tl.store(oq_ptr + output_offs, y, mask=mask)

    ok_ptr += s_id * ok_stride_s
    k_ptr += s_id * k_stride_s

    for off_h in range(0, NUM_K_HEADS):
        ordered_cols = off_h * k_stride_h + (ordered_block * k_stride_d)
        rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)
        output_offs = off_h * ok_stride_h + (ordered_block * ok_stride_d)

        k = tl.load(k_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)
        y = k * cos + rotated_k * sin
        tl.store(ok_ptr + output_offs, y, mask=mask)


# Integrated from FlagGems
def _apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
):
    """
    Apply rotary position embedding to q and k
    Args:
        q: (*, q_heads, head_dim)
        k: (*, k_heads, head_dim)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
        position_ids: (*, ), optional, position ids for each token
        rotary_interleaved: whether the head_dim is rotated in an interleaved way
    Returns:
        q_embed: (*, q_heads, head_dim)
        k_embed: (*, k_heads, head_dim)
    """
    assert (
        k.shape[-1] == q.shape[-1]  # 128
    ), f"q and k must have the same last dimension, got {q.shape} and {k.shape}"
    assert (
        cos.shape[-1] == sin.shape[-1]  # 128 / 2 = 64
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert (
        cos.shape[-1] * 2 == q.shape[-1]  # 128
    ), f"cos/sin dim must be half of q/k dim, got {cos.shape} and {q.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

    q_shape = q.shape
    k_shape = k.shape

    assert (
        q.shape[:-2] == k.shape[:-2]
    ), f"q and k must have the same length, got {q.shape[:-2]} and {k.shape[:-2]}"
    if position_ids is None:
        assert (
            len(q.shape) == 4
        ), f"q must have 4 dimensions if position_ids is not provided, got {q.shape}"
        seq_len = q.shape[-3]
    else:
        assert (
            position_ids.shape == q.shape[:-2]  # num_head 32
        ), f"position_ids must have the same length as q, got {position_ids.shape} and {q.shape[:-2]}"

        position_ids = position_ids.view(-1)
        seq_len = None

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)

    n_tokens, q_heads, head_dim = q.shape

    # The block size must be the next power of two, sometimes we need to pad it.
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    grid = (n_tokens,)

    apply_rotary_pos_emb_kernel[grid](
        q_embed,
        k_embed,
        q,
        k,
        cos,
        sin,
        position_ids,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        q_embed.stride(0),
        q_embed.stride(1),
        q_embed.stride(2),
        k_embed.stride(0),
        k_embed.stride(1),
        k_embed.stride(2),
        position_ids.stride(0) if position_ids is not None else 0,
        cos.stride(0),
        sin.stride(0),
        seq_len,
        q.shape[-2],
        k.shape[-2],
        head_dim,
        padded_head_dim,
        rotary_interleaved,
        MAX_POSITION_EMBEDDINGS=cos.shape[0],
    )

    q_embed = q_embed.view(q_shape)
    k_embed = k_embed.view(k_shape)
    return q_embed, k_embed


def rotary_embedding_triton(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    cos_cache, sin_cache = torch.chunk(cos_sin_cache, chunks=2, dim=-1)
    reshaped_query = query.reshape(-1, query.shape[1] // head_size, head_size)
    reshaped_key = key.reshape(-1, key.shape[1] // head_size, head_size)
    reshaped_query, reshaped_key = _apply_rotary_pos_emb(
        reshaped_query, reshaped_key, cos_cache, sin_cache, positions, False
    )
    query.copy_(reshaped_query.reshape(-1, query.shape[1]))
    key.copy_(reshaped_key.reshape(-1, key.shape[1]))
