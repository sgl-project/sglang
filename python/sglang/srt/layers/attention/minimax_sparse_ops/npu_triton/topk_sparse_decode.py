from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# =============================================================================
# Utilities
# =============================================================================


def _floor_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x).bit_length() - 1)


# ---------------------------------------------------------------------------
# Tunable launch configs for the served decode kernels.
#
# Defaults are the validated (num_warps=4, num_stages=2). The bench/tuning
# scripts override these module globals before the first launch to sweep configs
# and land the winners here. They are NOT triton.autotune so that each shape
# bucket compiles to a single deterministic artifact (cuda-graph capture safe).
# ---------------------------------------------------------------------------
# NOTE: swept num_warps∈{2,4,8} × num_stages∈{2,3} via sweep_cfg.py on the real
# captured decode workload (mostly short seqs -> launch-bound). No config beat the
# validated (4,2) outside run-to-run noise, so 4/2 is kept. The hooks remain for
# re-tuning if the served batch/context-length profile shifts to compute-bound.
_SPARSE_DECODE_NW = 4
_SPARSE_DECODE_NS = 2
_MERGE_NW = 4
_MERGE_NS = 2


def _get_vectorcore_num_safe() -> int:
    """Return the Ascend NPU vector-core count (sglang-native).

    Read ``num_vectorcore`` from triton's active-driver device properties for the
    current NPU. Falls back to 32 off-NPU or if the property is unavailable.
    """
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        n = int(props.get("num_vectorcore", -1))
    except Exception:
        # Conservative fallback.
        return 32
    return max(1, n) if n > 0 else 32


def _choose_num_topk_chunks(
    batch_size: int,
    num_kv_heads: int,
    max_topk: int,
    max_num_topk_chunks: int = 8,
) -> int:
    """Choose split-topk chunks in an SGLang-like but Ascend-conservative way."""
    if max_topk <= 1:
        return 1

    num_vectorcore = _get_vectorcore_num_safe()
    # SGLang CUDA uses TARGET_GRID=256 for this sparse decode kernel.
    # Use a vectorcore-based target on Ascend and cap conservatively.
    target_grid = num_vectorcore * 4
    target = max(1, target_grid // max(1, batch_size * num_kv_heads))
    target = min(max_topk, max_num_topk_chunks, target)
    return _floor_power_of_2(target)


def _normalize_topk_idx_for_gqa(
    topk_idx: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    gqa_group_size: int,
) -> torch.Tensor:
    """Ensure topk_idx has shape [num_kv_heads, batch_size, topk].

    The sparse GQA-share decode kernel uses one topk list per KV head, shared by
    all query heads in the corresponding GQA group.

    If a per-query-head topk tensor [num_q_heads, batch_size, topk] is provided,
    we take the first q-head from each GQA group.
    """
    if topk_idx.shape[0] == num_kv_heads:
        return topk_idx.contiguous()

    if topk_idx.shape[0] == num_q_heads:
        batch_size = topk_idx.shape[1]
        max_topk = topk_idx.shape[2]
        return topk_idx.view(num_kv_heads, gqa_group_size, batch_size, max_topk)[
            :, 0, :, :
        ].contiguous()

    raise AssertionError(
        "topk_idx first dimension must be either num_kv_heads or num_q_heads, "
        f"got {topk_idx.shape[0]}, num_kv_heads={num_kv_heads}, "
        f"num_q_heads={num_q_heads}"
    )


# =============================================================================
# Sparse BNSD Decode Kernel
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit
def _gqa_share_sparse_decode_bnsd_kernel(
    q_ptr,  # [B, QH, D]
    sink_ptr,  # optional [QH, D]
    k_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    v_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    block_table_ptr,  # [B, max_num_blocks]
    idx_ptr,  # [KVH, B, max_topk]
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    seq_lens,  # [B]
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    max_topk,
    max_kv_len,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    # strides
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_sink_h,
    stride_sink_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    stride_bt_b,
    stride_bt_n,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    # meta
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    CHUNK_SIZE_T: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.minimum(tl.load(seq_lens + pid_b).to(tl.int32), max_kv_len)

    # TopK list base for this KV head and request.
    #
    # Do NOT compute real_topk with tl.sum(topk_vals >= 0) here. On Ascend this
    # pattern can compile but appear to behave like only one block is active,
    # producing an output magnitude similar to attending a single block. Instead
    # iterate over the fixed per-chunk topk range and mask out -1 entries.
    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_b * stride_ti_b

    chunk_start_topk = pid_c * CHUNK_SIZE_T

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    dim_mask = off_d < head_dim

    # Q: [H, D]
    q_offsets = (
        pid_b * stride_q_b
        + (pid_h + off_h[:, None]) * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
        other=0.0,
    )

    # Sink belongs only to chunk 0 so it is counted once across split-topk chunks.
    if HAS_SINK and pid_c == 0:
        sink_offsets = (pid_h + off_h[:, None]) * stride_sink_h + off_d[
            None, :
        ] * stride_sink_d
        sink = tl.load(
            sink_ptr + sink_offsets,
            mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale
        m_i = qsink
        lse_i = qsink
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)

    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0.0, dtype=tl.float32)

    # Iterate over the fixed topk slice assigned to this chunk. The actual valid
    # length is encoded by -1 sentinels in topk_idx.
    for step in tl.range(CHUNK_SIZE_T):
        topk_pos = chunk_start_topk + step
        in_topk_range = topk_pos < max_topk

        logical_block = tl.load(
            idx_base + topk_pos * stride_ti_t,
            mask=in_topk_range,
            other=-1,
        ).to(tl.int32)
        valid_block = logical_block >= 0

        physical_block = tl.load(
            block_table_ptr + pid_b * stride_bt_b + logical_block * stride_bt_n,
            mask=valid_block,
            other=0,
        ).to(tl.int64)

        pos = logical_block * block_size + off_n
        pos_mask = valid_block & (pos < seq_len)

        # K: [D, N]
        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )

        # V: [N, D]
        v_offsets = (
            physical_block * stride_v_block
            + off_n[:, None] * stride_v_offset
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_offsets,
            mask=pos_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        # [H, D] @ [D, N] -> [H, N]
        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.where(
            valid_block,
            tl.exp(qk - m_ij[:, None]),
            tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32),
        )
        l_ij = tl.sum(p, axis=1)

        acc_o_scale = tl.where(
            valid_block,
            tl.exp(m_i - m_ij),
            tl.full((BLOCK_SIZE_H,), 1.0, dtype=tl.float32),
        )
        acc_o_new = acc_o * acc_o_scale[:, None] + tl.dot(p.to(v.dtype), v)
        lse_i_new = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

        acc_o = tl.where(valid_block, acc_o_new, acc_o)
        m_i = tl.where(valid_block, m_ij, m_i)
        lse_i = tl.where(valid_block, lse_i_new, lse_i)

    # Final scale.
    # Empty chunks keep lse_i=-inf and should output clean zeros.
    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]

    # Store partial output: [C, B, QH, D]
    o_offsets = (
        pid_c * stride_o_c
        + pid_b * stride_o_b
        + (pid_h + off_h[:, None]) * stride_o_h
        + off_d[None, :] * stride_o_d
    )
    tl.store(
        o_ptr + o_offsets,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
    )

    l_offsets = pid_c * stride_l_c + pid_b * stride_l_b + (pid_h + off_h) * stride_l_h
    tl.store(
        lse_ptr + l_offsets,
        lse_i.to(lse_ptr.dtype.element_ty),
        mask=off_h < gqa_group_size,
    )


# =============================================================================
# Merge split-topk sparse attention output
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _merge_topk_attn_out_bnsd_kernel(
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    out_ptr,  # [B, QH, D]
    head_dim,
    # strides
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    # meta
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_d = tl.arange(0, BLOCK_SIZE_D)

    m = tl.full((), float("-inf"), dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.full((BLOCK_SIZE_D,), 0.0, dtype=tl.float32)

    for c in tl.static_range(0, NUM_TOPK_CHUNKS):
        lse_c = tl.load(
            lse_ptr + c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h
        )

        o_c = tl.load(
            o_ptr
            + c * stride_o_c
            + pid_b * stride_o_b
            + pid_h * stride_o_h
            + off_d * stride_o_d,
            mask=off_d < head_dim,
            other=0.0,
        ).to(tl.float32)

        # Avoid -inf - -inf -> NaN for all-empty chunks.
        valid = lse_c > float("-inf")
        m_new = tl.maximum(m, lse_c)

        scale_old = tl.where(
            m > float("-inf"),
            tl.exp(m - m_new),
            tl.zeros_like(m),
        )
        scale_new = tl.where(
            valid,
            tl.exp(lse_c - m_new),
            tl.zeros_like(lse_c),
        )

        acc = acc * scale_old + o_c * scale_new
        l = l * scale_old + scale_new
        m = m_new

    out = tl.where(l > 0.0, acc / l, acc)

    tl.store(
        out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + off_d * stride_out_d,
        out.to(out_ptr.dtype.element_ty),
        mask=off_d < head_dim,
    )


# =============================================================================
# Python Wrapper
# =============================================================================


@torch.no_grad()
def flash_decode_bnsd_with_gqa_share_sparse(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],  # optional [num_q_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_bnsd: torch.Tensor,  # same shape
    block_table: torch.Tensor,  # [batch_size, max_num_blocks]
    seq_lens: torch.Tensor,  # [batch_size]
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads or num_q_heads, batch_size, topk]
    sm_scale: Optional[float] = None,
    num_topk_chunks: Optional[int] = None,
    max_num_topk_chunks: int = 8,
) -> torch.Tensor:
    """Sparse decode attention using BNSD KV cache and precomputed topk blocks.

    This is the BNSD/Ascend-friendly counterpart of SGLang's
    flash_decode_with_gqa_share_sparse.

    Args:
        q:
            [batch_size, num_q_heads, head_dim].
        sink:
            Optional [num_q_heads, head_dim].
        k_cache_bnsd / v_cache_bnsd:
            [num_blocks, block_size, num_kv_heads, head_dim].
        block_table:
            [batch_size, max_num_blocks].
        seq_lens:
            [batch_size].
        block_size:
            KV block size.
        topk_idx:
            Prefer [num_kv_heads, batch_size, topk]. If [num_q_heads, batch_size,
            topk] is provided, the first q-head of each GQA group is used.
        num_topk_chunks:
            If None, choose dynamically. Otherwise must be power-of-two.

    Returns:
        o:
            [batch_size, num_q_heads, head_dim].
    """
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert v_cache_bnsd.dtype == q.dtype
    assert k_cache_bnsd.shape == v_cache_bnsd.shape

    batch_size, num_q_heads, head_dim = q.shape
    _, block_size_from_cache, num_kv_heads, cache_head_dim = k_cache_bnsd.shape

    assert block_size_from_cache == block_size
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    assert block_table.shape[0] == batch_size
    assert seq_lens.shape[0] == batch_size
    assert topk_idx.shape[1] == batch_size

    gqa_group_size = num_q_heads // num_kv_heads

    topk_idx = _normalize_topk_idx_for_gqa(
        topk_idx,
        num_q_heads,
        num_kv_heads,
        gqa_group_size,
    )

    max_topk = topk_idx.shape[2]
    max_kv_len = block_table.shape[1] * block_size

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if num_topk_chunks is None:
        num_topk_chunks = _choose_num_topk_chunks(
            batch_size,
            num_kv_heads,
            max_topk,
            max_num_topk_chunks=max_num_topk_chunks,
        )
    else:
        num_topk_chunks = int(num_topk_chunks)

    assert num_topk_chunks >= 1
    assert (num_topk_chunks & (num_topk_chunks - 1)) == 0
    assert num_topk_chunks <= max(1, max_topk)

    chunk_size_topk = (max_topk + num_topk_chunks - 1) // num_topk_chunks
    # Ascend BiSheng can crash at ConvertLinalgRToBinary when CHUNK_SIZE_T=1
    # in this sparse-decode kernel:
    #   LLVM ERROR: operation destroyed but still has uses
    # Use a minimum static topk loop width of 2. Extra iterations are safely
    # masked by ``topk_pos < max_topk`` and ``logical_block >= 0``.
    # This keeps correctness unchanged while avoiding the backend corner case.
    chunk_size_topk = max(2, chunk_size_topk)

    # Single-chunk fast path: with NUM_TOPK_CHUNKS==1 the decode kernel writes the
    # already-final-normalized output (it applies the final exp(m_i - lse_i) scale
    # before storing), so the merge kernel would be a no-op copy. Alias o_partial
    # to the final output buffer (pid_c is always 0 -> pid_c*stride_o_c == 0) and
    # skip the merge launch + the [C,B,QH,D] temp allocation.
    single_chunk = num_topk_chunks == 1
    out = torch.empty_like(q)
    if single_chunk:
        o_partial = out.view(1, batch_size, num_q_heads, head_dim)
    else:
        o_partial = torch.empty(
            (num_topk_chunks, batch_size, num_q_heads, head_dim),
            dtype=q.dtype,
            device=q.device,
        )
    # lse_partial is always written by the kernel; unused on the single-chunk path
    # but required as a store target (small, [C,B,QH]).
    lse_partial = torch.empty(
        (num_topk_chunks, batch_size, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )

    # Triton still type-checks pointer arguments in constexpr-dead branches on
    # some Ascend builds. Do not pass Python None as sink_ptr. Instead pass any
    # typed tensor pointer and control the real behavior with HAS_SINK.
    sink_arg = sink if sink is not None else q

    grid = (batch_size * num_topk_chunks, num_kv_heads)
    _gqa_share_sparse_decode_bnsd_kernel[grid](
        q,
        sink_arg,
        k_cache_bnsd,
        v_cache_bnsd,
        block_table,
        topk_idx,
        o_partial,
        lse_partial,
        seq_lens,
        batch_size,
        gqa_group_size,
        head_dim,
        max_topk,
        max_kv_len,
        block_size,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink_arg.stride(0) if sink is not None else 0,
        sink_arg.stride(1) if sink is not None else 0,
        k_cache_bnsd.stride(0),
        k_cache_bnsd.stride(1),
        k_cache_bnsd.stride(2),
        k_cache_bnsd.stride(3),
        v_cache_bnsd.stride(0),
        v_cache_bnsd.stride(1),
        v_cache_bnsd.stride(2),
        v_cache_bnsd.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        BLOCK_SIZE_N=block_size,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        CHUNK_SIZE_T=chunk_size_topk,
        HAS_SINK=sink is not None,
        num_warps=_SPARSE_DECODE_NW,
        num_stages=_SPARSE_DECODE_NS,
    )

    if not single_chunk:
        merge_grid = (batch_size, num_q_heads)
        _merge_topk_attn_out_bnsd_kernel[merge_grid](
            o_partial,
            lse_partial,
            out,
            head_dim,
            o_partial.stride(0),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0),
            lse_partial.stride(1),
            lse_partial.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks,
            num_warps=_MERGE_NW,
            num_stages=_MERGE_NS,
        )

    return out
