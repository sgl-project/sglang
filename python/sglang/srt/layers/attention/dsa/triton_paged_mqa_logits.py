# SPDX-License-Identifier: Apache-2.0
"""Triton-fused paged-MQA-logits kernel: a fast path for the DSA indexer's
`dsa_paged_mqa_logits_backend="torch"` fallback (see `torch_paged_mqa_logits.py`).

Under CUDA graph capture, the page-table scan width used by the paged-MQA
logits computation is fixed at the full page-table capacity (the CUDA-graph
capture width), not the request's true (much shorter) sequence length --
shapes must be static for a captured graph. `fp8_paged_mqa_logits_torch_dsa`
pays that full width unconditionally on every step (gather + fp8-dequant +
matmul over the whole capture width). This kernel keeps the same static
launch grid (so it stays CUDA-graph-safe) but gives each program an early
exit, read from `seq_lens` at REPLAY time, once its 64-token KV block lies
entirely past the request's true sequence length -- so cost tracks the true
context length instead of the capture width.

Numerically bit-exact with `fp8_paged_mqa_logits_torch_dsa` for a given input
(same fp8-load + inline-scale-dequant + q.k dot + relu + weighted head-sum;
no fp32 HBM intermediates). Requires `num_heads >= 16` (`tl.dot`'s minimum
inner dimension) -- the caller falls back to the torch kernel below that.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _dsa_indexer_logits_kernel(
    q_ptr,  # fp8 [B, H, D], contiguous
    kv_ptr,  # fp8 flat: per 64-token block, 64*D values then 64 fp32 scales
    scale_ptr,  # same storage viewed as fp32, element-indexed
    w_ptr,  # fp32 [B, H]
    seq_ptr,  # int32 [B]
    pt_ptr,  # int32 [B, W], page ids (page size 64)
    out_ptr,  # fp32 [B, MAX_SEQ]
    W: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    pblk = tl.program_id(1)
    base = pblk * BLOCK
    rows = base + tl.arange(0, BLOCK)
    out_off = b * MAX_SEQ + rows
    out_mask = rows < MAX_SEQ

    seq = tl.load(seq_ptr + b)
    if base >= seq:
        # Entire block lies past this request's true sequence length: the
        # static launch grid still schedules this program (grid width is
        # capture-time constant), but there is nothing valid to score here.
        # -inf matches the "invalid" fill the pure-torch path (and the
        # downstream topk_transform's masking convention) expects.
        tl.store(
            out_ptr + out_off,
            tl.full((BLOCK,), float("-inf"), tl.float32),
            mask=out_mask,
        )
        return

    page = tl.load(pt_ptr + b * W + pblk).to(tl.int64)
    blk_base = page * (BLOCK * (D + 4))
    offs = blk_base + tl.arange(0, BLOCK)[:, None] * D + tl.arange(0, D)[None, :]
    kv = tl.load(kv_ptr + offs).to(tl.float32)  # [BLOCK, D]
    s_base = page * (BLOCK * (D + 4) // 4) + (BLOCK * D // 4)
    scale = tl.load(scale_ptr + s_base + tl.arange(0, BLOCK))  # [BLOCK]

    q = tl.load(
        q_ptr + b * H * D + tl.arange(0, H)[:, None] * D + tl.arange(0, D)[None, :]
    ).to(
        tl.float32
    )  # [H, D]
    w = tl.load(w_ptr + b * H + tl.arange(0, H))  # [H]

    score = tl.dot(kv, tl.trans(q))  # [BLOCK, H]
    score = tl.maximum(score, 0.0)
    score = tl.sum(score * w[None, :], axis=1)  # [BLOCK]
    score = score * scale

    # Sub-block masking: a block can straddle the true seq_len boundary even
    # though it passed the whole-block early-exit check above (base < seq but
    # base + BLOCK > seq); rows past `seq` within this block must still be
    # -inf.
    valid = rows < seq
    score = tl.where(valid, score, float("-inf"))
    tl.store(out_ptr + out_off, score, mask=out_mask)


def fp8_paged_mqa_logits_triton_dsa(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Same signature/semantics as `fp8_paged_mqa_logits_torch_dsa` (bit-exact).

    Callers are expected to have already validated shapes (this is dispatched
    from inside `fp8_paged_mqa_logits_torch_dsa`, after its asserts).
    """
    _ = deep_gemm_metadata
    _ = clean_logits
    batch_size, _one, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == 128 and block_size == 64
    assert num_heads >= 16, "Triton kernel requires num_heads >= 16 (tl.dot minimum)"
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    table_width = page_table.shape[1]
    max_pages = (max_seq_len + block_size - 1) // block_size
    grid_w = min(table_width, max_pages)

    # page_table entries beyond a request's allocated page count are, in
    # SGLang's current DSA req_to_token pool, always zero-initialized (never
    # negative) -- but page ids are read directly (no clamp) inside the
    # kernel above. This is safe here specifically because every program
    # whose block lies past `seq_lens[b]` takes the early-exit branch BEFORE
    # loading a page id at all (see the `base >= seq` check), so an
    # out-of-range or otherwise garbage page id in a padding column is never
    # dereferenced.
    kv_flat = kvcache_fp8.reshape(-1)
    scale_view = kvcache_fp8.view(torch.uint8).view(torch.float32).reshape(-1)
    q = q_fp8.reshape(batch_size, num_heads, head_dim).contiguous()
    out = torch.empty(batch_size, max_seq_len, dtype=torch.float32, device=q_fp8.device)
    if grid_w * block_size < max_seq_len:
        out.fill_(float("-inf"))

    _dsa_indexer_logits_kernel[(batch_size, grid_w)](
        q,
        kv_flat,
        scale_view,
        weight.to(torch.float32).contiguous(),
        seq_lens.to(torch.int32).contiguous(),
        page_table.contiguous(),
        out,
        W=table_width,
        MAX_SEQ=max_seq_len,
        H=num_heads,
        D=head_dim,
        BLOCK=block_size,
        num_warps=4,
    )
    return out
