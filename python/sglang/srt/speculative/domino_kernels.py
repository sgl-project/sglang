"""Triton kernels for the DFLASH Domino sequential rollout.

The per-step body of Domino rollout is dominated by tiny ops at bs=1. Even with a
CUDA graph capturing the loop, the per-step work is fundamentally limited by
fc2 ([B,256] @ [256, V=152K]) bandwidth and a handful of elementwise/reduce
launches. These kernels collapse the small ops:

  1. ``fused_silu_fc2_argmax`` — fuses ``SiLU(z+s) @ fc2.T + fc2_bias +
     base_logits`` into a single per-step matmul that emits per-block
     (val, idx) pairs, followed by a reduction kernel that produces the global
     argmax token. Avoids materializing the full [B, V] bias tensor.
  2. ``fused_gru_cell_from_table`` — fuses GRU input-table lookup and the
     sigmoid/tanh/gate update into one kernel.

Both kernels are CUDA-graph friendly (no host-driven sync, fixed shapes).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _domino_fused_silu_fc2_argmax_kernel(
    z_ptr,
    s_ptr,
    fc2_ptr,
    base_ptr,
    fc2_bias_ptr,
    out_val_ptr,
    out_idx_ptr,
    M,
    V,
    stride_z_b,
    stride_z_m,
    stride_s_b,
    stride_s_m,
    stride_f2_v,
    stride_f2_m,
    stride_base_b,
    stride_base_v,
    stride_outv_b,
    stride_outv_n,
    stride_outi_b,
    stride_outi_n,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_FC2_BIAS: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    acc = tl.zeros([BLOCK_V], dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        z_block = tl.load(
            z_ptr + pid_b * stride_z_b + offs_m * stride_z_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        s_block = tl.load(
            s_ptr + pid_b * stride_s_b + offs_m * stride_s_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        preact = z_block + s_block
        mid_block = preact * tl.sigmoid(preact)

        fc2_ptrs = (
            fc2_ptr + offs_v[:, None] * stride_f2_v + offs_m[None, :] * stride_f2_m
        )
        fc2_block = tl.load(
            fc2_ptrs,
            mask=mask_v[:, None] & mask_m[None, :],
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)

        acc += tl.sum(fc2_block * mid_block[None, :], axis=1)

    if HAS_FC2_BIAS:
        bias_block = tl.load(fc2_bias_ptr + offs_v, mask=mask_v, other=0.0).to(
            tl.float32
        )
        acc += bias_block

    base_block = tl.load(
        base_ptr + pid_b * stride_base_b + offs_v * stride_base_v,
        mask=mask_v,
        other=-float("inf"),
        cache_modifier=".cg",
    ).to(tl.float32)
    acc += base_block

    acc = tl.where(mask_v, acc, -float("inf"))

    local_max_val = tl.max(acc, axis=0)
    local_max_idx = tl.argmax(acc, axis=0)
    global_max_idx = pid_v * BLOCK_V + local_max_idx

    out_val_offs = pid_b * stride_outv_b + pid_v * stride_outv_n
    out_idx_offs = pid_b * stride_outi_b + pid_v * stride_outi_n
    tl.store(out_val_ptr + out_val_offs, local_max_val)
    tl.store(out_idx_ptr + out_idx_offs, global_max_idx)


@triton.jit
def _domino_reduce_argmax_kernel(
    out_val_ptr,
    out_idx_ptr,
    final_token_ptr,
    N,
    stride_val_b,
    stride_val_n,
    stride_idx_b,
    stride_idx_n,
    stride_final_b,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    best_val = -float("inf")
    best_idx = 0

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        vals = tl.load(
            out_val_ptr + pid_b * stride_val_b + offs_n * stride_val_n,
            mask=mask_n,
            other=-float("inf"),
        )
        local_pos = tl.argmax(vals, axis=0)
        local_val = tl.max(vals, axis=0)
        local_idx = tl.load(
            out_idx_ptr + pid_b * stride_idx_b + (start_n + local_pos) * stride_idx_n
        )

        take = local_val > best_val
        best_val = tl.where(take, local_val, best_val)
        best_idx = tl.where(take, local_idx, best_idx)

    tl.store(final_token_ptr + pid_b * stride_final_b, best_idx)


def fused_silu_fc2_argmax(
    *,
    z_proj: torch.Tensor,  # [B, M]
    s_proj: torch.Tensor,  # [B, M]
    fc2_weight: torch.Tensor,  # [V, M]
    fc2_bias: torch.Tensor | None,  # [V] or None
    base_logits: torch.Tensor,  # [B, V]
    out_val: torch.Tensor,  # [B, num_v_blocks] fp32 scratch
    out_idx: torch.Tensor,  # [B, num_v_blocks] int32 scratch
    final_token: torch.Tensor,  # [B] int64 destination
    block_v: int = 512,
    block_m: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Single-step fused: SiLU(z+s) @ fc2.T + bias + base_logits -> argmax.

    Writes the per-batch global argmax into ``final_token`` in-place.
    """
    B, M = z_proj.shape
    V = base_logits.shape[1]
    num_v_blocks = (V + block_v - 1) // block_v
    has_fc2_bias = fc2_bias is not None
    fc2_bias_arg = fc2_bias if has_fc2_bias else out_val  # placeholder ptr

    grid_main = (num_v_blocks, B)
    _domino_fused_silu_fc2_argmax_kernel[grid_main](
        z_proj,
        s_proj,
        fc2_weight,
        base_logits,
        fc2_bias_arg,
        out_val,
        out_idx,
        M,
        V,
        z_proj.stride(0),
        z_proj.stride(1),
        s_proj.stride(0),
        s_proj.stride(1),
        fc2_weight.stride(0),
        fc2_weight.stride(1),
        base_logits.stride(0),
        base_logits.stride(1),
        out_val.stride(0),
        out_val.stride(1),
        out_idx.stride(0),
        out_idx.stride(1),
        BLOCK_V=block_v,
        BLOCK_M=block_m,
        HAS_FC2_BIAS=has_fc2_bias,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _domino_reduce_argmax_kernel[(B,)](
        out_val,
        out_idx,
        final_token,
        num_v_blocks,
        out_val.stride(0),
        out_val.stride(1),
        out_idx.stride(0),
        out_idx.stride(1),
        final_token.stride(0),
        BLOCK_N=512,
        num_warps=1,
    )


@triton.jit
def _domino_fused_gru_cell_from_table_kernel(
    tok_ptr,
    table_ptr,
    gh_ptr,
    gh_bias_ptr,
    h_ptr,
    h_out_ptr,
    G,
    V,
    stride_table_v,
    stride_table_g,
    stride_gh_b,
    stride_gh_g,
    stride_gh_bias_g,
    stride_h_b,
    stride_h_g,
    stride_hout_b,
    stride_hout_g,
    BLOCK_G: tl.constexpr,
    HAS_GH_BIAS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    mask_g = offs_g < G

    tok_id_raw = tl.load(tok_ptr + pid_b)
    tok_in_range = (tok_id_raw >= 0) & (tok_id_raw < V)
    tok_id = tl.where(tok_in_range, tok_id_raw, 0)
    mask_tok = mask_g & tok_in_range

    h_state = tl.load(
        h_ptr + pid_b * stride_h_b + offs_g * stride_h_g,
        mask=mask_g,
        other=0.0,
    ).to(tl.float32)

    row_base = tok_id * stride_table_v
    gi_r = tl.load(
        table_ptr + row_base + offs_g * stride_table_g,
        mask=mask_tok,
        other=0.0,
    ).to(tl.float32)
    gi_z = tl.load(
        table_ptr + row_base + (G + offs_g) * stride_table_g,
        mask=mask_tok,
        other=0.0,
    ).to(tl.float32)
    gi_n = tl.load(
        table_ptr + row_base + (2 * G + offs_g) * stride_table_g,
        mask=mask_tok,
        other=0.0,
    ).to(tl.float32)

    gh_r = tl.load(
        gh_ptr + pid_b * stride_gh_b + offs_g * stride_gh_g,
        mask=mask_g,
        other=0.0,
    ).to(tl.float32)
    gh_z = tl.load(
        gh_ptr + pid_b * stride_gh_b + (G + offs_g) * stride_gh_g,
        mask=mask_g,
        other=0.0,
    ).to(tl.float32)
    gh_n = tl.load(
        gh_ptr + pid_b * stride_gh_b + (2 * G + offs_g) * stride_gh_g,
        mask=mask_g,
        other=0.0,
    ).to(tl.float32)

    if HAS_GH_BIAS:
        gh_r = (
            (
                gh_r
                + tl.load(
                    gh_bias_ptr + offs_g * stride_gh_bias_g,
                    mask=mask_g,
                    other=0.0,
                ).to(tl.float32)
            )
            .to(gh_ptr.dtype.element_ty)
            .to(tl.float32)
        )
        gh_z = (
            (
                gh_z
                + tl.load(
                    gh_bias_ptr + (G + offs_g) * stride_gh_bias_g,
                    mask=mask_g,
                    other=0.0,
                ).to(tl.float32)
            )
            .to(gh_ptr.dtype.element_ty)
            .to(tl.float32)
        )
        gh_n = (
            (
                gh_n
                + tl.load(
                    gh_bias_ptr + (2 * G + offs_g) * stride_gh_bias_g,
                    mask=mask_g,
                    other=0.0,
                ).to(tl.float32)
            )
            .to(gh_ptr.dtype.element_ty)
            .to(tl.float32)
        )

    r = tl.sigmoid(gi_r + gh_r)
    z = tl.sigmoid(gi_z + gh_z)
    n = 2.0 * tl.sigmoid(2.0 * (gi_n + r * gh_n)) - 1.0
    h_new = (1.0 - z) * n + z * h_state

    tl.store(
        h_out_ptr + pid_b * stride_hout_b + offs_g * stride_hout_g,
        h_new.to(h_ptr.dtype.element_ty),
        mask=mask_g,
    )


def fused_gru_cell_from_table(
    *,
    tok_full: torch.Tensor,  # [B] int64 token ids (already shifted by org_vocab_start)
    gru_input_table: torch.Tensor,  # [V, 3*G] precomputed embed @ W_ih.T + b_ih
    gh: torch.Tensor,  # [B, 3*G] h_state @ W_hh.T (+ b_hh)
    gh_bias: torch.Tensor | None = None,  # [3*G] or None
    h_state: torch.Tensor,  # [B, G]
    h_out: torch.Tensor,  # [B, G] destination
    block_g: int = 256,
) -> None:
    """Fused: gi = gru_input_table[tok_full]; then GRU cell update.

    Replaces the index_select + fused_gru_cell pair with a single kernel that
    reads gi directly from the table by tok id, skipping the gi_buf materialize.
    """
    B = tok_full.shape[0]
    G = h_state.shape[1]
    V = gru_input_table.shape[0]
    has_gh_bias = gh_bias is not None
    gh_bias_arg = gh_bias if has_gh_bias else gh  # placeholder ptr
    gh_bias_stride = gh_bias_arg.stride(0) if has_gh_bias else 0
    grid = (B, (G + block_g - 1) // block_g)
    _domino_fused_gru_cell_from_table_kernel[grid](
        tok_full,
        gru_input_table,
        gh,
        gh_bias_arg,
        h_state,
        h_out,
        G,
        V,
        gru_input_table.stride(0),
        gru_input_table.stride(1),
        gh.stride(0),
        gh.stride(1),
        gh_bias_stride,
        h_state.stride(0),
        h_state.stride(1),
        h_out.stride(0),
        h_out.stride(1),
        BLOCK_G=block_g,
        HAS_GH_BIAS=has_gh_bias,
    )
