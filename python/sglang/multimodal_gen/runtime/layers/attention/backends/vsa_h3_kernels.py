# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# (fastvideo-kernel triton_kernels/block_sparse_attn_triton.py and index.py).
# Inference-only subset: the block-sparse forward and the mask -> index
# conversion. Backward stays upstream; SGLang serves no-grad forwards.

# SPDX-License-Identifier: Apache-2.0
"""Triton 64-token block-sparse attention for MiniMax-H3 VSA.

The kernel consumes an explicit per-query-block index list (``q2k_index`` /
``q2k_num``) plus per-key-block valid token counts (``variable_block_sizes``),
so ragged interior tiles - segment-pure prefix chunks and 3D video tiles whose
dimensions do not divide the tile shape - mask their pad columns exactly.
"""

import math

import torch
import triton
import triton.language as tl

# BLOCK_M / BLOCK_N are structural, not tunable: the kernel indexes the top-k
# list per BLOCK_M q-tile and addresses keys as kv_idx * BLOCK_N, so both must
# match the granularity q2k_index and variable_block_sizes were built at.
VSA_H3_KERNEL_BLOCK = 64

_configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=s, num_warps=w)
    for s in (2, 3, 4, 5, 6, 7)
    for w in (4, 8)
]


@triton.autotune(_configs, key=["N_CTX_Q", "HEAD_DIM"])
@triton.jit
def _attn_fwd_sparse(
    Q,
    K,
    V,
    sm_scale,
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    M,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX_Q,
    N_CTX_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_blk = tl.program_id(0)
    off_hz = tl.program_id(1)
    b = off_hz // H
    h = off_hz % H
    q_tiles = N_CTX_Q // BLOCK_M
    meta_base = (b * H + h) * q_tiles + q_blk

    kv_blocks = tl.load(q2k_num + meta_base)
    kv_ptr = q2k_index + meta_base * max_kv_blks

    q_off = b.to(tl.int64) * stride_qz + h.to(tl.int64) * stride_qh
    k_off = b.to(tl.int64) * stride_kz + h.to(tl.int64) * stride_kh
    v_off = b.to(tl.int64) * stride_vz + h.to(tl.int64) * stride_vh
    o_off = b.to(tl.int64) * stride_oz + h.to(tl.int64) * stride_oh

    Q_ptr = tl.make_block_ptr(
        base=Q + q_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_base = tl.make_block_ptr(
        base=K + k_off,
        shape=(HEAD_DIM, N_CTX_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_base = tl.make_block_ptr(
        base=V + v_off,
        shape=(N_CTX_KV, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_ptr = tl.make_block_ptr(
        base=Out + o_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/ln2
    q = tl.load(Q_ptr)

    for i in range(0, kv_blocks):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)
        K_ptr = tl.advance(K_base, (0, kv_idx * BLOCK_N))
        V_ptr = tl.advance(V_base, (kv_idx * BLOCK_N, 0))

        k = tl.load(K_ptr)
        qk = tl.dot(q, k)
        mask = tl.arange(0, BLOCK_N) < block_size
        qk = tl.where(mask[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_ptr)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(M + off_hz * N_CTX_Q + offs_m, m_i)
    tl.store(O_ptr, acc.to(Out.type.element_ty))


@triton.jit
def _map_to_index_kernel(
    map_ptr,
    index_ptr,
    index_num_ptr,
    map_bs_stride,
    map_h_stride,
    map_q_stride,
    map_kv_stride,
    index_bs_stride,
    index_h_stride,
    index_q_stride,
    index_kv_stride,
    index_num_bs_stride,
    index_num_h_stride,
    index_num_q_stride,
    num_kv_blocks,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_ptr_base = (
        index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    )
    map_ptr_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    num = 0
    for i in tl.range(num_kv_blocks):
        map_entry = tl.load(map_ptr_base + i * map_kv_stride)
        if map_entry:
            tl.store(index_ptr_base + num * index_kv_stride, i)
            num += 1

    tl.store(
        index_num_ptr
        + b * index_num_bs_stride
        + h * index_num_h_stride
        + q * index_num_q_stride,
        num,
    )


def vsa_h3_map_to_index(block_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, H, Gq, Gk] bool map -> ([B, H, Gq, Gk] int32 ascending index list
    padded with -1, [B, H, Gq] int32 per-row counts)."""
    batch, heads, num_q_blocks, num_kv_blocks = block_map.shape
    index = torch.full(block_map.shape, -1, dtype=torch.int32, device=block_map.device)
    index_num = torch.empty(
        (batch, heads, num_q_blocks), dtype=torch.int32, device=block_map.device
    )
    grid = (batch, heads, num_q_blocks)
    _map_to_index_kernel[grid](
        block_map,
        index,
        index_num,
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        index_num.stride(0),
        index_num.stride(1),
        index_num.stride(2),
        num_kv_blocks=num_kv_blocks,
    )
    return index, index_num


def vsa_h3_block_sparse_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_index: torch.Tensor,
    q2k_num: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """q/k/v: [B, H, S_pad, D] bf16 with S_pad = n_tiles * 64; pad rows zero."""
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
    sm_scale = 1.0 / math.sqrt(head_dim)
    max_kv_blks = q2k_index.shape[-1]
    out = torch.empty_like(q)
    # Row-max running stats in Triton's base-2 M format; inference discards it.
    row_max = torch.empty((batch, heads, seq_q), dtype=torch.float32, device=q.device)
    grid = lambda _: (triton.cdiv(seq_q, VSA_H3_KERNEL_BLOCK), batch * heads, 1)
    _attn_fwd_sparse[grid](
        q,
        k,
        v,
        sm_scale,
        q2k_index,
        q2k_num,
        max_kv_blks,
        variable_block_sizes,
        row_max,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        batch,
        heads,
        seq_q,
        seq_kv,
        HEAD_DIM=head_dim,
    )
    return out
