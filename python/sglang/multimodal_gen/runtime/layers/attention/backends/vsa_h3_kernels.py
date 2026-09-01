# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# (fastvideo-kernel triton_kernels/block_sparse_attn_triton.py).
# Inference-only subset: the block-sparse forward. Backward stays upstream;
# SGLang serves no-grad forwards. The tile pack/unpack kernels are SGLang's.

# SPDX-License-Identifier: Apache-2.0
"""Triton 64-token block-sparse attention for MiniMax-H3 VSA.

The attention kernel consumes an explicit per-query-block index list
(``q2k_index`` / ``q2k_num``) plus per-key-block valid token counts
(``variable_block_sizes``), so ragged interior tiles - segment-pure prefix
chunks and 3D video tiles whose dimensions do not divide the tile shape - mask
their pad columns exactly.

``vsa_h3_pack_tiles`` gathers packed ``[T, H, D]`` rows into the head-major
padded tile layout the attention kernel reads and pools each tile in the same
pass; ``vsa_h3_untile`` scatters the attention output back to packed rows and
folds in the gated compression branch. Both replace chains of index copies
and transposes that otherwise cost more than the attention kernel itself.
"""

import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# BLOCK_M / BLOCK_N are structural, not tunable: the kernel indexes the top-k
# list per BLOCK_M q-tile and addresses keys as kv_idx * BLOCK_N, so both must
# match the granularity q2k_index and variable_block_sizes were built at.
VSA_H3_KERNEL_BLOCK = 64

# Pinned instead of autotuned: on B300 num_warps=4 wins at every sequence
# length and num_stages=5 is within 0.5% of the best (7 spills); FastVideo
# reports the same optimum for Blackwell.
_ATTN_NUM_WARPS = 4
_ATTN_NUM_STAGES = 5


@triton.jit
def _attn_fwd_sparse(
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    sm_scale,
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    H,
    N_CTX_Q,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_blk = tl.program_id(0)
    off_hz = tl.program_id(1)
    b = off_hz // H
    h = off_hz % H
    q_tiles = N_CTX_Q // BLOCK_M
    meta_base = off_hz * q_tiles + q_blk

    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base.to(tl.int64) * max_kv_blks

    q = desc_q.load([b, h, q_blk * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/ln2

    for i in range(0, kv_blocks):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)
        k = desc_k.load([b, h, kv_idx * BLOCK_N, 0]).reshape([BLOCK_N, HEAD_DIM])
        qk = tl.dot(q, tl.trans(k))
        mask = tl.arange(0, BLOCK_N) < block_size
        qk = tl.where(mask[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = desc_v.load([b, h, kv_idx * BLOCK_N, 0]).reshape([BLOCK_N, HEAD_DIM])
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    desc_o.store(
        [b, h, q_blk * BLOCK_M, 0],
        acc.to(desc_o.dtype).reshape([1, 1, BLOCK_M, HEAD_DIM]),
    )


def vsa_h3_block_sparse_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """q/k/v: contiguous [B, H, S_pad, D] bf16 with S_pad = n_tiles * 64; pad
    rows zero. q2k_index/q2k_num: contiguous [B, H, n_tiles, max_kv] /
    [B, H, n_tiles] int32."""
    batch, heads, seq_q, head_dim = q.shape
    seq_kv = k.shape[2]
    if seq_q % VSA_H3_KERNEL_BLOCK or seq_kv % VSA_H3_KERNEL_BLOCK:
        raise ValueError(
            f"VSA-H3 kernel needs 64-multiple sequence lengths, got q={seq_q}, "
            f"kv={seq_kv}"
        )
    if variable_block_sizes.numel() != seq_kv // VSA_H3_KERNEL_BLOCK:
        raise ValueError(
            "variable_block_sizes must have one entry per 64-token key block: "
            f"{variable_block_sizes.numel()} vs {seq_kv // VSA_H3_KERNEL_BLOCK}"
        )
    if out is None:
        out = torch.empty_like(q)
    block = [1, 1, VSA_H3_KERNEL_BLOCK, head_dim]
    desc_q, desc_k, desc_v, desc_o = (
        TensorDescriptor.from_tensor(t, block_shape=block) for t in (q, k, v, out)
    )
    grid = (seq_q // VSA_H3_KERNEL_BLOCK, batch * heads, 1)
    _attn_fwd_sparse[grid](
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        1.0 / math.sqrt(head_dim),
        q2k_index,
        q2k_num,
        q2k_index.shape[-1],
        variable_block_sizes,
        heads,
        seq_q,
        HEAD_DIM=head_dim,
        BLOCK_M=VSA_H3_KERNEL_BLOCK,
        BLOCK_N=VSA_H3_KERNEL_BLOCK,
        num_warps=_ATTN_NUM_WARPS,
        num_stages=_ATTN_NUM_STAGES,
    )
    return out


@triton.jit
def _pack_tiles_kernel(
    Q,
    K,
    V,
    G,
    src_index,
    variable_block_sizes,
    Tiled,
    Pooled,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_v_row,
    stride_v_head,
    stride_g_row,
    stride_g_head,
    H,
    S_PAD,
    N_TILES,
    HAS_GATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    h = tl.program_id(1)
    rows = tile * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, HEAD_DIM)
    src = tl.load(src_index + rows)
    valid = src >= 0
    src = tl.where(valid, src, 0).to(tl.int64)
    mask = valid[:, None]
    size = tl.load(variable_block_sizes + tile).to(tl.float32)

    tensor_stride = H.to(tl.int64) * S_PAD * HEAD_DIM
    out_off = (
        h.to(tl.int64) * S_PAD * HEAD_DIM + rows[:, None] * HEAD_DIM + cols[None, :]
    )
    pool_off = (h * N_TILES + tile).to(tl.int64) * HEAD_DIM + cols

    x = tl.load(
        Q + src[:, None] * stride_q_row + h * stride_q_head + cols[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(Tiled + out_off, x)
    tl.store(Pooled + pool_off, tl.sum(x.to(tl.float32), 0) / size)

    x = tl.load(
        K + src[:, None] * stride_k_row + h * stride_k_head + cols[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(Tiled + tensor_stride + out_off, x)
    tl.store(
        Pooled + H * N_TILES * HEAD_DIM + pool_off,
        tl.sum(x.to(tl.float32), 0) / size,
    )

    x = tl.load(
        V + src[:, None] * stride_v_row + h * stride_v_head + cols[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(Tiled + 2 * tensor_stride + out_off, x)
    tl.store(
        Pooled + 2 * H * N_TILES * HEAD_DIM + pool_off,
        tl.sum(x.to(tl.float32), 0) / size,
    )

    if HAS_GATE:
        x = tl.load(
            G + src[:, None] * stride_g_row + h * stride_g_head + cols[None, :],
            mask=mask,
            other=0.0,
        )
        tl.store(Tiled + 3 * tensor_stride + out_off, x)


def vsa_h3_pack_tiles(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor | None,
    src_index: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    tiled: torch.Tensor,
    pooled: torch.Tensor,
) -> None:
    """Gather packed [T, H, D] rows into ``tiled`` [3|4, H, S_pad, D] and write
    fp32 per-tile means of q/k/v into ``pooled`` [3, H, n_tiles, D].
    ``src_index`` maps each padded position to its packed row, or -1 (pad -> 0).
    """
    _, heads, seq_pad, head_dim = tiled.shape
    n_tiles = seq_pad // VSA_H3_KERNEL_BLOCK
    has_gate = gate is not None
    g = gate if has_gate else q
    assert all(t.stride(-1) == 1 for t in (q, k, v, g))
    _pack_tiles_kernel[(n_tiles, heads)](
        q,
        k,
        v,
        g,
        src_index,
        variable_block_sizes,
        tiled,
        pooled,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        g.stride(0),
        g.stride(1),
        heads,
        seq_pad,
        n_tiles,
        HAS_GATE=has_gate,
        HEAD_DIM=head_dim,
        BLOCK=VSA_H3_KERNEL_BLOCK,
    )


@triton.jit
def _untile_kernel(
    OutTiled,
    Gate,
    OutC,
    dst_index,
    Res,
    used,
    total,
    S_PAD,
    N_TILES,
    HAS_GATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_block = tl.program_id(0)
    h = tl.program_id(1)
    H = tl.num_programs(1)
    rows = row_block * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, HEAD_DIM)
    in_used = rows < used
    pos = tl.load(dst_index + rows, mask=in_used, other=0).to(tl.int64)
    head_base = h.to(tl.int64) * S_PAD * HEAD_DIM
    off = head_base + pos[:, None] * HEAD_DIM + cols[None, :]
    o = tl.load(OutTiled + off, mask=in_used[:, None], other=0.0).to(tl.float32)
    if HAS_GATE:
        g = tl.load(Gate + off, mask=in_used[:, None], other=0.0).to(tl.float32)
        tile = pos // BLOCK
        c = tl.load(
            OutC + (h * N_TILES + tile)[:, None] * HEAD_DIM + cols[None, :],
            mask=in_used[:, None],
            other=0.0,
        )
        o = o + c * g
    res_off = (rows[:, None] * H + h).to(tl.int64) * HEAD_DIM + cols[None, :]
    tl.store(Res + res_off, o.to(Res.type.element_ty), mask=(rows < total)[:, None])


def vsa_h3_untile(
    out_tiled: torch.Tensor,
    gate_tiled: torch.Tensor | None,
    out_compress: torch.Tensor | None,
    dst_index: torch.Tensor,
    used: int,
    result: torch.Tensor,
) -> None:
    """Scatter ``out_tiled`` [H, S_pad, D] to packed rows of ``result`` [T, H, D]
    (rows past ``used`` are zero), adding ``out_compress`` [H, n_tiles, D] fp32
    scaled by ``gate_tiled`` when given."""
    heads, seq_pad, head_dim = out_tiled.shape
    total = result.shape[0]
    has_gate = gate_tiled is not None
    _untile_kernel[(triton.cdiv(total, VSA_H3_KERNEL_BLOCK), heads)](
        out_tiled,
        gate_tiled if has_gate else out_tiled,
        out_compress if has_gate else out_tiled,
        dst_index,
        result,
        used,
        total,
        seq_pad,
        seq_pad // VSA_H3_KERNEL_BLOCK,
        HAS_GATE=has_gate,
        HEAD_DIM=head_dim,
        BLOCK=VSA_H3_KERNEL_BLOCK,
    )
