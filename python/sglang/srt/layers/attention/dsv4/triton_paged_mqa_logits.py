from __future__ import annotations

import torch
import triton
import triton.language as tl

_HEAD_DIM = 128
_NUM_HEADS = 64
_BLOCK_SIZE = 64

FP8_DTYPE = torch.float8_e4m3fn


@triton.jit
def _paged_dot_relu_kernel(
    kv_val_ptr,
    kv_srow,
    kv_sdim,
    kv_sc_ptr,
    q_ptr,
    q_sb,
    q_snh,
    q_shd,
    w_ptr,
    w_sb,
    pt_ptr,
    pt_sb,
    sl_ptr,
    out_ptr,
    out_sb,
    max_seq_len,
    NH: tl.constexpr,
    HD: tl.constexpr,
    BS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_pg = tl.program_id(1)

    seq_len = tl.load(sl_ptr + pid_b)
    kv_start = pid_pg * BS
    if kv_start >= seq_len:
        return

    page_id = tl.load(pt_ptr + pid_b * pt_sb + pid_pg)
    if page_id < 0:
        return

    p_offs = tl.arange(0, BS)
    kv_offs = kv_start + p_offs
    valid = (kv_offs < seq_len) & (kv_offs < max_seq_len)

    row_offs = page_id * BS + p_offs
    d_offs = tl.arange(0, HD)
    kv_2d = row_offs[:, None] * kv_srow + d_offs[None, :] * kv_sdim
    kv_fp8 = tl.load(kv_val_ptr + kv_2d, mask=valid[:, None], other=0.0)
    kv_f32 = kv_fp8.to(tl.float32)

    kv_scale = tl.load(kv_sc_ptr + row_offs, mask=valid, other=0.0)

    h_offs = tl.arange(0, NH)
    q_2d = h_offs[:, None] * q_snh + d_offs[None, :] * q_shd
    q = tl.load(q_ptr + pid_b * q_sb + q_2d)
    w = tl.load(w_ptr + pid_b * w_sb + h_offs)

    dot = tl.dot(
        kv_f32.to(tl.float16), tl.trans(q.to(tl.float16)), out_dtype=tl.float32
    )
    score = tl.sum(tl.maximum(dot, 0.0) * w[None, :], axis=1) * kv_scale

    tl.store(out_ptr + pid_b * out_sb + kv_offs, score, mask=valid)


def fp8_paged_mqa_logits_triton_sm89(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    _ = clean_logits

    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)

    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == _HEAD_DIM
    assert num_heads == _NUM_HEADS
    assert block_size == _BLOCK_SIZE
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size

    total_pages = kvcache_fp8.shape[0]
    total_dim = block_size * (head_dim + 4)
    scale_offset = block_size * head_dim

    kvcache_flat = kvcache_fp8.view(total_pages, total_dim)
    kv_val_u8 = kvcache_flat[:, :scale_offset].contiguous()
    kv_val_fp8 = kv_val_u8.view(dtype=FP8_DTYPE).reshape(
        total_pages * block_size, head_dim
    )
    kv_sc_u8 = kvcache_flat[:, scale_offset:].contiguous()
    kv_sc_f32 = kv_sc_u8.view(dtype=torch.float32).reshape(total_pages * block_size)

    q_f32 = q_fp8[:, 0].to(torch.float32)
    logits = torch.full(
        (batch_size, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )

    max_pages = triton.cdiv(max_seq_len, block_size)
    _paged_dot_relu_kernel[(batch_size, max_pages)](
        kv_val_fp8,
        kv_val_fp8.stride(0),
        kv_val_fp8.stride(1),
        kv_sc_f32,
        q_f32,
        q_f32.stride(0),
        q_f32.stride(1),
        q_f32.stride(2),
        weight,
        weight.stride(0),
        page_table,
        page_table.stride(0),
        seq_lens.to(torch.int32),
        logits,
        logits.stride(0),
        max_seq_len,
        NH=num_heads,
        HD=head_dim,
        BS=block_size,
        num_warps=4,
        num_stages=4,
    )
    return logits
