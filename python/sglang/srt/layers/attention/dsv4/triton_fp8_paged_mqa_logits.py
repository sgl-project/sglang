import torch
import triton
import triton.language as tl
from typing import Any, Optional
from sglang.srt.utils import is_hip

if is_hip():
    FP8_DTYPE = torch.float8_e4m3fnuz
else:
    FP8_DTYPE = torch.float8_e4m3fn

_FP8_PAGED_MQA_LOGITS_CHUNK_BYTES = 256 * 1024 * 1024


@triton.jit
def _score_relu_weight_scale_kernel(
    # score: [cb, padded_seq_len, num_heads] fp32 (input/output fused)
    # weight: [cb, num_heads] fp32
    # kv_scale: [cb, padded_seq_len] fp32
    # seq_lens: [cb] int
    # logits_out: [cb, max_seq_len] fp32
    score_ptr,
    weight_ptr,
    kv_scale_ptr,
    seq_lens_ptr,
    logits_ptr,
    cb,
    padded_seq_len,
    max_seq_len,
    num_heads: tl.constexpr,
    score_stride_b,
    score_stride_s,
    weight_stride_b,
    scale_stride_b,
    logits_stride_b,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: relu(score) * weight → sum over heads → * kv_scale → masked store.

    Grid: (cb, cdiv(padded_seq_len, BLOCK_S))
    Each program handles BLOCK_S seq positions for one batch row.
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    if pid_b >= cb:
        return

    seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int32)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < padded_seq_len

    # Load weight[batch, :] — all heads
    wt_base = weight_ptr + pid_b * weight_stride_b
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < num_heads
    w = tl.load(wt_base + h_offs, mask=h_mask, other=0.0)  # [BLOCK_H]

    # Accumulate: for each position, sum over heads of relu(score) * weight
    score_base = score_ptr + pid_b * score_stride_b
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    for h in range(num_heads):
        # score[batch, s, h]
        s_ptrs = score_base + s_offs * score_stride_s + h
        val = tl.load(s_ptrs, mask=s_mask, other=0.0)
        # ReLU
        val = tl.maximum(val, 0.0)
        # Multiply by weight[h]
        w_h = tl.load(wt_base + h).to(tl.float32)
        acc += val * w_h

    # Multiply by kv_scale[batch, s]
    scale_base = kv_scale_ptr + pid_b * scale_stride_b
    kv_s = tl.load(scale_base + s_offs, mask=s_mask, other=0.0)
    acc = acc * kv_s

    # Masked store to logits — zero out positions >= seq_len
    valid = s_mask & (s_offs < seq_len) & (s_offs < max_seq_len)
    write_mask = s_mask & (s_offs < max_seq_len)
    out_base = logits_ptr + pid_b * logits_stride_b
    tl.store(out_base + s_offs, tl.where(valid, acc, 0.0), mask=write_mask)


def fp8_paged_mqa_logits_triton(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton-accelerated fp8_paged_mqa_logits.

    Paged KV gather + fp8→fp32 upcast + einsum remain in PyTorch (already
    vectorized, no host syncs). The post-matmul pipeline (ReLU, weight multiply,
    head reduction, scale multiply, validity masking, store) is fused into a
    single Triton kernel — replacing 5 separate PyTorch ops per chunk.
    """
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128
    assert block_size == 64
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    device = q_fp8.device
    head_dim_with_sf = head_dim + 4
    SCALE_OFFSET = block_size * head_dim

    max_pages_eff = (max_seq_len + block_size - 1) // block_size
    P = min(page_table.shape[1], max_pages_eff)
    padded_seq_len = P * block_size

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)

    kv_flat = kvcache_fp8.reshape(-1, block_size * head_dim_with_sf)
    num_pages_total = kv_flat.shape[0]

    bytes_per_row = max(1, P * block_size * head_dim * 4)
    chunk_size = max(1, _FP8_PAGED_MQA_LOGITS_CHUNK_BYTES // bytes_per_row)

    pt = page_table[:, :P]
    if num_pages_total > 0:
        pt = pt.clamp_(min=0, max=num_pages_total - 1)

    BLOCK_S = 64
    BLOCK_H = triton.next_power_of_2(num_heads)

    for s in range(0, batch_size, chunk_size):
        e = min(s + chunk_size, batch_size)
        cb = e - s

        # Paged gather (vectorized, no host sync)
        kv = kv_flat[pt[s:e]]
        kv_value_b = kv[..., :SCALE_OFFSET].contiguous()
        kv_scale_b = kv[..., SCALE_OFFSET:].contiguous()

        # bytes -> fp8 -> fp32
        kv_value = (
            kv_value_b.view(dtype=FP8_DTYPE)
            .view(cb, padded_seq_len, head_dim)
            .to(torch.float32)
        )
        # bytes -> fp32 scale per token
        kv_scale = kv_scale_b.view(dtype=torch.float32).view(cb, padded_seq_len)

        # q: (cb, num_heads, head_dim) fp32
        q = q_fp8[s:e, 0].to(torch.float32)

        # Batched matmul: score[b,s,h] = sum_d(kv_value[b,s,d] * q[b,h,d])
        # shape: (cb, padded_seq_len, num_heads)
        score = torch.einsum("bsd,bhd->bsh", kv_value, q)

        # Fused Triton kernel: relu -> weight -> sum_heads -> scale -> mask -> store
        score = score.contiguous()
        kv_scale = kv_scale.contiguous()

        write_len = min(padded_seq_len, max_seq_len)
        grid = (cb, triton.cdiv(padded_seq_len, BLOCK_S))

        _score_relu_weight_scale_kernel[grid](
            score,
            weight[s:e],
            kv_scale,
            seq_lens[s:e],
            logits[s:e],
            cb,
            padded_seq_len,
            max_seq_len,
            num_heads=num_heads,
            score_stride_b=score.stride(0),
            score_stride_s=score.stride(1),
            weight_stride_b=weight.stride(0),
            scale_stride_b=kv_scale.stride(0),
            logits_stride_b=logits.stride(1) * 0 + max_seq_len,  # logits is contiguous
            BLOCK_S=BLOCK_S,
            BLOCK_H=BLOCK_H,
        )

        # Zero-fill remaining columns if padded_seq_len < max_seq_len
        if write_len < max_seq_len:
            logits[s:e, write_len:] = 0

    return logits
