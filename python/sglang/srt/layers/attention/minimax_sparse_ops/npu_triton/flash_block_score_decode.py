from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


# ---------------------------------------------------------------------------
# Tunable launch configs for the served decode kernels.
#
# Defaults are the validated (num_warps=4, num_stages=2). The bench/tuning
# scripts override these module globals before the first launch to sweep configs
# and land the winners here. They are NOT triton.autotune so that each shape
# bucket compiles to a single deterministic artifact (cuda-graph capture safe).
# ---------------------------------------------------------------------------
_SCORE_CHUNK_NW = 4
_SCORE_CHUNK_NS = 2


def _get_vectorcore_num_safe() -> int:
    """Return the Ascend NPU vector-core count (sglang-native).

    Read ``num_vectorcore`` from triton's active-driver device properties for the
    current NPU. Falls back to 32 off-NPU or if the property is unavailable.
    """
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        vc_num = int(props.get("num_vectorcore", -1))
    except Exception:
        return 32
    return max(1, vc_num) if vc_num > 0 else 32


def _choose_num_kv_chunks(
    batch_size: int,
    num_kv_heads: int,
    max_num_kv_chunks: int = 8,
) -> int:
    """Choose NUM_KV_CHUNKS using the SGLang-style CTA target rule.

    SGLang CUDA decode derives NUM_KV_CHUNKS from a target grid size and
    rounds down to a power of two. For Ascend we use vector-core count instead
    of CUDA's hardcoded TARGET_GRID=4096.

    This first dynamic version is intentionally conservative: the default cap
    is 8 because the BNSD chunk/merge path has already been validated for
    1/2/4/8 in tests. Increase the cap after benchmarking long-context cases.
    """
    max_num_kv_chunks = max(1, int(max_num_kv_chunks))
    # Make the cap a power of two, because kernels/merge paths specialize on
    # NUM_KV_CHUNKS and power-of-two values are easier to reason about.
    max_num_kv_chunks = 1 << (max_num_kv_chunks.bit_length() - 1)

    vectorcore_num = _get_vectorcore_num_safe()
    # Similar spirit to the earlier Ascend migration: oversubscribe vector cores
    # modestly to hide latency, but keep the cap conservative.
    target_grid = max(1, vectorcore_num * 8)
    denom = max(1, int(batch_size) * int(num_kv_heads))

    target = max(1, min(max_num_kv_chunks, target_grid // denom))
    return 1 << (target.bit_length() - 1)


def _choose_num_score_chunks(
    max_seqblock: int,
    blocks_per_chunk: int = 16,
    max_chunks: int = 32,
    all_seqblock_q: int = 1,
    num_kv_heads: int = 1,
    program_cap: int = 32768,
) -> int:
    """Pick the number of block-tiles for the chunked score-only kernel.

    Each program iterates over a tile of ~``blocks_per_chunk`` blocks, so the
    program count no longer scales with context length. We pick enough chunks to
    keep the per-program serial loop near ``blocks_per_chunk`` while bounding the
    total (power-of-two to keep the kernel-specialization set small).

    Grid = ``all_seqblock_q * num_score_chunks * num_kv_heads``. Ascend caps the
    total program count of a single kernel launch (~``program_cap``, 32768). At
    ``block_size_q=1`` (MiniMax-M3 per-token topk) ``all_seqblock_q`` equals the
    extend length (e.g. 3072 at dp4), and once KV crosses 256 blocks the balance
    pushes ``num_score_chunks`` to 16 -> grid 49152 > cap, and the launch HANGS
    (aicore execution timeout 507014). Cap ``num_score_chunks`` so the grid stays
    under ``program_cap``; each program then scans more KV blocks serially (slower
    per program, but the launch completes). Defaults leave existing callers
    (small batch / decode) unchanged.
    """
    if max_seqblock <= 0:
        return 1
    balance = (max_seqblock + max(1, blocks_per_chunk) - 1) // max(1, blocks_per_chunk)
    n = max(1, min(balance, max(1, max_chunks)))
    per_chunk = max(1, all_seqblock_q * max(1, num_kv_heads))
    n = min(n, max(1, program_cap // per_chunk))
    return 1 << (n.bit_length() - 1)


def _torch_topk_from_score(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
) -> torch.Tensor:
    """Compute topk block indices from score using PyTorch.

    Args:
        score: [num_q_heads, batch_size, max_seqblock], float32.
        seq_lens: [batch_size], int32.
        block_size: KV cache block size.
        topk: requested top-k block count.

    Returns:
        topk_idx: [num_q_heads, batch_size, topk], int32.
            Invalid positions are filled with -1.
    """
    num_q_heads, batch_size, max_seqblock = score.shape
    device = score.device

    if topk <= 0:
        return torch.empty(
            (num_q_heads, batch_size, 0),
            dtype=torch.int32,
            device=device,
        )

    num_blocks = torch.div(
        seq_lens.to(torch.int64) + block_size - 1,
        block_size,
        rounding_mode="floor",
    )

    block_ids = torch.arange(max_seqblock, device=device, dtype=torch.int64)
    valid_score_mask = block_ids[None, None, :] < num_blocks[None, :, None]
    score_masked = score.masked_fill(~valid_score_mask, -float("inf"))

    k_eff = min(topk, max_seqblock)
    _, idx_eff = torch.topk(score_masked, k=k_eff, dim=-1)

    topk_idx = torch.full(
        (num_q_heads, batch_size, topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_idx[:, :, :k_eff] = idx_eff.to(torch.int32)

    rank = torch.arange(topk, device=device, dtype=torch.int64)
    # Clamp `num_blocks` against `topk` via `.clamp(max=topk)` rather than
    # `torch.minimum(..., torch.tensor(topk, device=...))`. Building a device
    # tensor from a host int with `torch.tensor` triggers a synchronized
    # aclrtMemcpy (host->device), which is illegal inside a captured stream and
    # aborts CUDA-graph capture with NPU error 107030 ("the current capture mode
    # does not support this operation"). `clamp` takes the scalar as a kernel
    # arg with no host->device copy, so this path stays capture-safe. This
    # fallback is exactly the path hit during decode-graph capture when
    # max_seqblock > 256 (see the `use_triton_topk` gate).
    valid_topk_mask = rank[None, None, :] < num_blocks[None, :, None].clamp(max=topk)

    topk_idx = torch.where(
        valid_topk_mask,
        topk_idx,
        torch.full_like(topk_idx, -1),
    )
    return topk_idx


# =============================================================================
# Ascend-friendly Streaming TopK Kernel
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
    }
)
@triton.jit
def _topk_index_streaming_bnsd_kernel(
    score_ptr,  # [QH, B, max_seqblock]
    topk_idx_ptr,  # [QH, B, topk]
    seq_lens,  # [B]
    block_size: tl.constexpr,
    topk: tl.constexpr,
    max_seqblock: tl.constexpr,
    # strides
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    # meta
    BLOCK_SIZE_T: tl.constexpr,
):
    """Streaming top-k over score blocks.

    One program handles one (batch, q_head) pair. It keeps an unsorted top-k
    buffer in registers and replaces the current minimum when a larger score is
    found. This avoids the bitonic split-K implementation that can produce very
    large inline-asm immediates on Ascend.

    The output order is not guaranteed. Existing correctness tests compare the
    selected set of block indices, which is the important semantic requirement
    for downstream sparse attention.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    off_t = tl.arange(0, BLOCK_SIZE_T)

    # Trivial batch: every valid block fits in the top-k budget. Emit
    # [0..num_blocks) + -1 padding directly, WITHOUT reading score. This is the
    # safety counterpart to the score-kernel trivial-skip: when num_blocks <= topk
    # the score columns may be uninitialized (the score kernel early-
    # returns for this batch), so the score read below must be bypassed.
    # Per-(batch,head) program -> mixed batches handled naturally, each program
    # decides from its own seq_len.
    if num_blocks <= topk:
        out_idx = tl.where(
            off_t < num_blocks,
            off_t,
            tl.full((BLOCK_SIZE_T,), -1, dtype=tl.int32),
        )
        tl.store(
            topk_idx_ptr
            + pid_h * stride_tif_h
            + pid_b * stride_tif_b
            + off_t * stride_tif_t,
            out_idx.to(topk_idx_ptr.dtype.element_ty),
            mask=off_t < topk,
        )
        return

    valid_topk_lane = off_t < topk

    # Invalid lanes are +inf so they never become the replacement target.
    top_scores = tl.where(
        valid_topk_lane,
        tl.full((BLOCK_SIZE_T,), -1.0e30, dtype=tl.float32),
        tl.full((BLOCK_SIZE_T,), 1.0e30, dtype=tl.float32),
    )
    top_indices = tl.full((BLOCK_SIZE_T,), -1, dtype=tl.int32)

    # Scan only the actual valid blocks, not the static padded max_seqblock.
    # (Runtime loop bound is the same pattern as the score chunk kernels'
    # `tl.range(num_steps)`; cuda-graph safe -- grid is fixed, only per-program
    # trip count varies.)
    for block_idx in tl.range(0, num_blocks):
        valid_block = block_idx < num_blocks
        score = tl.load(
            score_ptr
            + pid_h * stride_s_h
            + pid_b * stride_s_b
            + block_idx * stride_s_k,
            mask=valid_block,
            other=-1.0e30,
        ).to(tl.float32)
        score = tl.where(score != score, -1.0e30, score)

        min_score = tl.min(top_scores, axis=0)

        # Pick the first lane whose score equals the current minimum. This keeps
        # replacement deterministic without needing an argmin primitive.
        candidate_pos = tl.where(
            (top_scores == min_score) & valid_topk_lane,
            off_t,
            tl.full((BLOCK_SIZE_T,), BLOCK_SIZE_T, dtype=tl.int32),
        )
        min_pos = tl.min(candidate_pos, axis=0)

        do_replace = valid_block & (score > min_score)
        replace_mask = off_t == min_pos

        top_scores = tl.where(replace_mask & do_replace, score, top_scores)
        top_indices = tl.where(
            replace_mask & do_replace,
            block_idx,
            top_indices,
        )

    tl.store(
        topk_idx_ptr
        + pid_h * stride_tif_h
        + pid_b * stride_tif_b
        + off_t * stride_tif_t,
        top_indices.to(topk_idx_ptr.dtype.element_ty),
        mask=off_t < topk,
    )


def _streaming_topk_from_score(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
) -> torch.Tensor:
    """Compute topk block indices with a simple Triton streaming kernel."""
    num_q_heads, batch_size, max_seqblock = score.shape
    device = score.device

    if topk <= 0:
        return torch.empty(
            (num_q_heads, batch_size, 0),
            dtype=torch.int32,
            device=device,
        )

    topk_idx = torch.empty(
        (num_q_heads, batch_size, topk),
        dtype=torch.int32,
        device=device,
    )

    grid = (batch_size, num_q_heads)
    _topk_index_streaming_bnsd_kernel[grid](
        score,
        topk_idx,
        seq_lens,
        block_size,
        topk,
        max_seqblock,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        num_warps=1,
        num_stages=1,
    )
    return topk_idx


@triton.heuristics(
    {
        "BLOCK_SIZE_CANDIDATES": lambda args: triton.next_power_of_2(
            args["NUM_SCORE_CHUNKS"] * args["topk"]
        ),
    }
)
@triton.jit
def _merge_bnsd_score_topk_candidates_kernel(
    candidate_scores_ptr,  # [C, QH, B, topk]
    candidate_indices_ptr,  # [C, QH, B, topk]
    topk_indices_ptr,  # [QH, B, topk]
    # strides
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    # meta
    NUM_SCORE_CHUNKS: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_CANDIDATES: tl.constexpr,
):
    """Merge fixed-size chunk-local candidates into global TopK indices."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_candidates = tl.arange(0, BLOCK_SIZE_CANDIDATES)
    candidate_chunk = off_candidates // topk
    candidate_rank = off_candidates - candidate_chunk * topk
    valid_candidate = off_candidates < NUM_SCORE_CHUNKS * topk

    score_offsets = (
        candidate_chunk * stride_cs_c
        + pid_h * stride_cs_h
        + pid_b * stride_cs_b
        + candidate_rank * stride_cs_t
    )
    index_offsets = (
        candidate_chunk * stride_ci_c
        + pid_h * stride_ci_h
        + pid_b * stride_ci_b
        + candidate_rank * stride_ci_t
    )
    candidate_scores = tl.load(
        candidate_scores_ptr + score_offsets,
        mask=valid_candidate,
        other=float("-inf"),
    ).to(tl.float32)
    candidate_indices = tl.load(
        candidate_indices_ptr + index_offsets,
        mask=valid_candidate,
        other=-1,
    ).to(tl.int32)
    candidate_scores = tl.where(
        candidate_indices >= 0,
        candidate_scores,
        float("-inf"),
    )

    for rank in tl.static_range(0, topk):
        best_score = tl.max(candidate_scores, axis=0)
        best_positions = tl.where(
            candidate_scores == best_score,
            off_candidates,
            tl.full((BLOCK_SIZE_CANDIDATES,), BLOCK_SIZE_CANDIDATES, tl.int32),
        )
        best_position = tl.min(best_positions, axis=0)
        selected_index = tl.max(
            tl.where(
                off_candidates == best_position,
                candidate_indices,
                tl.full((BLOCK_SIZE_CANDIDATES,), -1, tl.int32),
            ),
            axis=0,
        )
        tl.store(
            topk_indices_ptr
            + pid_h * stride_ti_h
            + pid_b * stride_ti_b
            + rank * stride_ti_t,
            selected_index,
        )
        candidate_scores = tl.where(
            off_candidates == best_position,
            float("-inf"),
            candidate_scores,
        )


def _merge_bnsd_score_topk_candidates(
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Return global TopK indices from chunk-local score candidates."""
    num_score_chunks, num_q_heads, batch_size, _ = candidate_scores.shape
    topk_indices = torch.empty(
        (num_q_heads, batch_size, topk),
        dtype=torch.int32,
        device=candidate_scores.device,
    )
    _merge_bnsd_score_topk_candidates_kernel[(batch_size, num_q_heads)](
        candidate_scores,
        candidate_indices,
        topk_indices,
        candidate_scores.stride(0),
        candidate_scores.stride(1),
        candidate_scores.stride(2),
        candidate_scores.stride(3),
        candidate_indices.stride(0),
        candidate_indices.stride(1),
        candidate_indices.stride(2),
        candidate_indices.stride(3),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        NUM_SCORE_CHUNKS=num_score_chunks,
        topk=topk,
        num_warps=1,
        num_stages=1,
    )
    return topk_indices


# =============================================================================
# BNSD Decode Score Kernel
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _decode_bnsd_score_kernel(
    q_ptr,  # [B, QH, D]
    k_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    block_table_ptr,  # [B, max_num_blocks]
    score_ptr,  # [QH, B, max_seqblock]
    seq_lens,  # [B]
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    max_seqblock: tl.constexpr,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    init_blocks,
    local_blocks,
    # strides
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_bt_b,
    stride_bt_n,
    stride_s_h,
    stride_s_b,
    stride_s_n,
    # meta
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    topk: tl.constexpr,
    SKIP_TRIVIAL_TOPK_SCORE: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_b = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_blk = tl.program_id(2)

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    # Trivial-skip: when the sequence has <= topk blocks, every block is
    # selected regardless of score, so the QK + scoring work is wasted.
    # Mirrors the GPU _decode_score_kernel trivial-skip; paired with the
    # streaming topk short-circuit so the uninitialized score is never read.
    if SKIP_TRIVIAL_TOPK_SCORE:
        if num_blocks <= topk:
            return

    if pid_blk >= num_blocks:
        return

    pid_h = pid_kh * gqa_group_size

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

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

    physical_block = tl.load(
        block_table_ptr + pid_b * stride_bt_b + pid_blk * stride_bt_n
    ).to(tl.int64)

    pos = pid_blk * block_size + off_n
    pos_mask = pos < seq_len

    # K: [D, N]
    k_offsets = (
        physical_block * stride_k_block
        + off_n[None, :] * stride_k_offset
        + pid_kh * stride_k_h
        + off_d[:, None] * stride_k_d
    )
    k = tl.load(
        k_cache_ptr + k_offsets,
        mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
        other=0.0,
    )

    sm_scale_log2e = sm_scale * 1.4426950409
    qk = tl.dot(q, k) * sm_scale_log2e
    qk = tl.where(pos_mask[None, :], qk, float("-inf"))

    sub_max = tl.max(qk, axis=1)

    if SCORE_TYPE == "max":
        score = sub_max
    else:
        score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
        score = tl.where(score != score, float("-inf"), score)

    # Match the original/reference behavior:
    # init blocks are strongly boosted, local blocks are also forced in.
    # If a block is both init and local, this order gives local = 1e29.
    local_start = tl.maximum(0, num_blocks - local_blocks)
    is_init = pid_blk < init_blocks
    is_local = (pid_blk >= local_start) & (pid_blk < num_blocks)

    score = tl.where(is_init, 1e30, score)
    score = tl.where(is_local, 1e29, score)

    s_offsets = (pid_h + off_h) * stride_s_h + pid_b * stride_s_b + pid_blk * stride_s_n

    tl.store(
        score_ptr + s_offsets,
        score.to(score_ptr.dtype.element_ty),
        mask=off_h < gqa_group_size,
    )


# =============================================================================
# BNSD Decode Score Chunk Kernel (score-only, one program per block-tile)
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _decode_bnsd_score_chunk_kernel(
    q_ptr,  # [B, QH, D]
    k_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    block_table_ptr,  # [B, max_num_blocks]
    score_ptr,  # [QH, B, max_seqblock]
    seq_lens,  # [B]
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    init_blocks,
    local_blocks,
    num_kv_chunks,
    # strides
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_bt_b,
    stride_bt_n,
    stride_s_h,
    stride_s_b,
    stride_s_n,
    # meta
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    topk: tl.constexpr,
    SKIP_TRIVIAL_TOPK_SCORE: tl.constexpr,
):
    """Score-only chunked kernel.

    Same per-block score math as ``_decode_bnsd_score_kernel``, but one program
    iterates over a tile of consecutive logical blocks (its chunk), loading Q
    once and reusing it for every block in the tile. Grid is
    ``(batch_size * num_kv_chunks, num_kv_heads)`` -- independent of context
    length -- so total score time no longer scales with ``max_seqblock``.
    Output score tensor is bit-identical to the 1-block-per-program kernel.

    NOTE: a multi-block K-tile (BLOCKS_PER_K_BLOCK > 1, one tl.dot over several
    consecutive blocks) was attempted to cut dot/loop count for the full-context
    scan, but it is NOT viable on this Ascend TBE backend: the 3D tl.reshape of
    the dot result miscompiles (~400x slowdown at BPK=1), the lane-arithmetic
    fallback is similarly pathological (~1500x), and BPK>1 hard-crashes the
    compiler (BLOCK_SIZE_N=256/512 and/or the vectorized block_table gather).
    Kept as 1-block-per-iteration. A fused/tiled version would need a non-Triton
    (Ascend C++/TBE) kernel.
    """
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    # Trivial-skip: when the sequence has <= topk blocks, every block is
    # selected regardless of score, so the QK + scoring work is wasted. Early-
    # return leaves score[(h,b),:] uninitialized; the streaming topk kernel
    # short-circuits the same (num_blocks <= topk) case without reading score,
    # so this is safe. Mirrors the GPU _decode_score_kernel trivial-skip.
    if SKIP_TRIVIAL_TOPK_SCORE:
        if num_blocks <= topk:
            return

    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, num_kv_chunks))
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)

    if chunk_start_block >= chunk_end_block:
        return

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    # Q: [H, D] -- loaded once, reused across the whole chunk.
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

    sm_scale_log2e = sm_scale * 1.4426950409
    local_start = tl.maximum(0, num_blocks - local_blocks)

    num_steps = chunk_end_block - chunk_start_block
    for step in tl.range(num_steps):
        logical_block = chunk_start_block + step
        physical_block = tl.load(
            block_table_ptr + pid_b * stride_bt_b + logical_block * stride_bt_n
        ).to(tl.int64)

        pos = logical_block * block_size + off_n
        pos_mask = pos < seq_len

        # K: [D, N]
        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))

        sub_max = tl.max(qk, axis=1)
        if SCORE_TYPE == "max":
            score = sub_max
        else:
            score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
            score = tl.where(score != score, float("-inf"), score)

        is_init = logical_block < init_blocks
        is_local = (logical_block >= local_start) & (logical_block < num_blocks)
        score = tl.where(is_init, 1e30, score)
        score = tl.where(is_local, 1e29, score)

        s_offsets = (
            (pid_h + off_h) * stride_s_h
            + pid_b * stride_s_b
            + logical_block * stride_s_n
        )
        tl.store(
            score_ptr + s_offsets,
            score.to(score_ptr.dtype.element_ty),
            mask=off_h < gqa_group_size,
        )


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
    }
)
@triton.jit
def _decode_bnsd_score_topk_chunk_kernel(
    q_ptr,  # [B, QH, D]
    k_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    block_table_ptr,  # [B, max_num_blocks] or typed direct-map placeholder
    req_to_token_ptr,  # [num_requests, max_context] in direct-map mode
    req_pool_indices_ptr,  # [B] in direct-map mode
    candidate_scores_ptr,  # [C, QH, B, topk]
    candidate_indices_ptr,  # [C, QH, B, topk]
    seq_lens,  # [B]
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    init_blocks,
    local_blocks,
    num_score_chunks,
    # strides
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_bt_b,
    stride_bt_n,
    stride_rtt_r,
    stride_rtt_t,
    max_req_to_token_cols,
    num_pages,
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    # meta
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_DIRECT_PAGE_LOOKUP: tl.constexpr,
    SANITIZE_PAGE_IDS: tl.constexpr,
):
    """Fuse block score computation with one register-resident TopK per chunk."""
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    candidate_mask = (off_h[:, None] < gqa_group_size) & (off_t[None, :] < topk)
    candidate_offsets = (
        pid_c * stride_cs_c
        + (pid_h + off_h[:, None]) * stride_cs_h
        + pid_b * stride_cs_b
        + off_t[None, :] * stride_cs_t
    )
    candidate_index_offsets = (
        pid_c * stride_ci_c
        + (pid_h + off_h[:, None]) * stride_ci_h
        + pid_b * stride_ci_b
        + off_t[None, :] * stride_ci_t
    )

    top_scores = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), float("-inf"), tl.float32)
    top_indices = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), -1, tl.int32)
    if num_blocks <= topk:
        top_indices = tl.where(
            (pid_c == 0) & (off_t[None, :] < num_blocks),
            off_t[None, :] + tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_T), tl.int32),
            top_indices,
        )
        top_scores = tl.where(
            top_indices >= 0,
            tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_T), tl.float32),
            top_scores,
        )
        tl.store(candidate_scores_ptr + candidate_offsets, top_scores, mask=candidate_mask)
        tl.store(
            candidate_indices_ptr + candidate_index_offsets,
            top_indices,
            mask=candidate_mask,
        )
        return

    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, num_score_chunks))
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        tl.store(candidate_scores_ptr + candidate_offsets, top_scores, mask=candidate_mask)
        tl.store(
            candidate_indices_ptr + candidate_index_offsets,
            top_indices,
            mask=candidate_mask,
        )
        return

    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)
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

    sm_scale_log2e = sm_scale * 1.4426950409
    local_start = tl.maximum(0, num_blocks - local_blocks)
    num_steps = chunk_end_block - chunk_start_block
    for step in tl.range(num_steps):
        logical_block = chunk_start_block + step
        if USE_DIRECT_PAGE_LOOKUP:
            req_idx = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
            token_col = tl.minimum(
                logical_block * block_size, max_req_to_token_cols - 1
            )
            token_slot = tl.load(
                req_to_token_ptr
                + req_idx * stride_rtt_r
                + token_col * stride_rtt_t
            ).to(tl.int64)
            physical_block = token_slot // block_size
            if SANITIZE_PAGE_IDS:
                physical_block = tl.minimum(
                    tl.maximum(physical_block, 0), num_pages - 1
                )
        else:
            physical_block = tl.load(
                block_table_ptr + pid_b * stride_bt_b + logical_block * stride_bt_n
            ).to(tl.int64)
        pos = logical_block * block_size + off_n
        pos_mask = pos < seq_len
        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))
        sub_max = tl.max(qk, axis=1)
        if SCORE_TYPE == "max":
            score = sub_max
        else:
            score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
            score = tl.where(score != score, float("-inf"), score)
        is_init = logical_block < init_blocks
        is_local = (logical_block >= local_start) & (logical_block < num_blocks)
        score = tl.where(is_init, 1e30, score)
        score = tl.where(is_local, 1e29, score)

        valid_topk_lane = off_t[None, :] < topk
        current_min = tl.min(top_scores, axis=1)
        min_positions = tl.where(
            (top_scores == current_min[:, None]) & valid_topk_lane,
            off_t[None, :],
            tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), BLOCK_SIZE_T, tl.int32),
        )
        min_position = tl.min(min_positions, axis=1)
        replace = (
            (off_t[None, :] == min_position[:, None])
            & valid_topk_lane
            & (score[:, None] > current_min[:, None])
        )
        top_scores = tl.where(replace, score[:, None], top_scores)
        top_indices = tl.where(replace, logical_block, top_indices)

    tl.store(candidate_scores_ptr + candidate_offsets, top_scores, mask=candidate_mask)
    tl.store(
        candidate_indices_ptr + candidate_index_offsets,
        top_indices,
        mask=candidate_mask,
    )


# =============================================================================
# BNSD Decode Fused Score + Attention Chunk Kernel
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.jit
def _decode_bnsd_score_attn_chunk_kernel(
    q_ptr,  # [B, QH, D]
    sink_ptr,  # optional [QH, D]
    k_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    v_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    block_table_ptr,  # [B, max_num_blocks]
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    candidate_scores_ptr,  # [C, QH, B, topk]
    candidate_indices_ptr,  # [C, QH, B, topk]
    seq_lens,  # [B]
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    init_blocks,
    local_blocks,
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
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_cs_c,
    stride_cs_h,
    stride_cs_b,
    stride_cs_t,
    stride_ci_c,
    stride_ci_h,
    stride_ci_b,
    stride_ci_t,
    # meta
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_KV_CHUNKS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Fused full-decode kernel.

    Compared with the v3 path, this computes block scores and attention output
    in the same pass over BNSD KV cache, so full decode no longer computes QK
    twice. The Ascend-friendly constraints are preserved:
      - no tl.make_block_ptr
      - no qk 3D reshape
      - each chunk keeps its block TopK in registers
      - direct BNSD + block_table addressing
    """
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    candidate_mask = (off_h[:, None] < gqa_group_size) & (off_t[None, :] < topk)
    candidate_offsets = (
        pid_c * stride_cs_c
        + (pid_h + off_h[:, None]) * stride_cs_h
        + pid_b * stride_cs_b
        + off_t[None, :] * stride_cs_t
    )
    candidate_index_offsets = (
        pid_c * stride_ci_c
        + (pid_h + off_h[:, None]) * stride_ci_h
        + pid_b * stride_ci_b
        + off_t[None, :] * stride_ci_t
    )
    top_scores = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), float("-inf"), tl.float32)
    top_indices = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), -1, tl.int32)

    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, NUM_KV_CHUNKS))
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)

    if chunk_start_block >= chunk_end_block:
        tl.store(candidate_scores_ptr + candidate_offsets, top_scores, mask=candidate_mask)
        tl.store(
            candidate_indices_ptr + candidate_index_offsets,
            top_indices,
            mask=candidate_mask,
        )
        return

    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

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

    sm_scale_log2e = sm_scale * 1.4426950409

    if HAS_SINK:
        if pid_c == 0:
            sink_offsets = (pid_h + off_h[:, None]) * stride_sink_h + off_d[
                None, :
            ] * stride_sink_d
            sink = tl.load(
                sink_ptr + sink_offsets,
                mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale_log2e
            m_i = qsink
            l_i = tl.full((BLOCK_SIZE_H,), 1.0, dtype=tl.float32)
        else:
            m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)

    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0.0, dtype=tl.float32)
    local_start = tl.maximum(0, num_blocks - local_blocks)

    num_steps = chunk_end_block - chunk_start_block
    for step in tl.range(num_steps):
        logical_block = chunk_start_block + step
        physical_block = tl.load(
            block_table_ptr + pid_b * stride_bt_b + logical_block * stride_bt_n
        ).to(tl.int64)

        pos = logical_block * block_size + off_n
        pos_mask = pos < seq_len

        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
            other=0.0,
        )

        v_offsets = (
            physical_block * stride_v_block
            + off_n[:, None] * stride_v_offset
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_offsets,
            mask=pos_mask[:, None] & (off_d[None, :] < head_dim),
            other=0.0,
        )

        qk = tl.dot(q, k) * sm_scale_log2e
        qk = tl.where(pos_mask[None, :], qk, float("-inf"))

        # ---- block score path ----
        sub_max = tl.max(qk, axis=1)
        if SCORE_TYPE == "max":
            score = sub_max
        else:
            score = sub_max + tl.log2(tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1))
            score = tl.where(score != score, float("-inf"), score)

        is_init = logical_block < init_blocks
        is_local = (logical_block >= local_start) & (logical_block < num_blocks)
        score = tl.where(is_init, 1e30, score)
        score = tl.where(is_local, 1e29, score)

        valid_topk_lane = off_t[None, :] < topk
        current_min = tl.min(top_scores, axis=1)
        min_positions = tl.where(
            (top_scores == current_min[:, None]) & valid_topk_lane,
            off_t[None, :],
            tl.full((BLOCK_SIZE_H, BLOCK_SIZE_T), BLOCK_SIZE_T, tl.int32),
        )
        min_position = tl.min(min_positions, axis=1)
        replace = (
            (off_t[None, :] == min_position[:, None])
            & valid_topk_lane
            & (score[:, None] > current_min[:, None])
        )
        top_scores = tl.where(replace, score[:, None], top_scores)
        top_indices = tl.where(replace, logical_block, top_indices)

        # ---- attention path ----
        m_new = tl.maximum(m_i, sub_max)
        p = tl.exp2(qk - m_new[:, None])
        l_new = tl.sum(p, axis=1)

        acc_scale = tl.exp2(m_i - m_new)
        acc_o = acc_o * acc_scale[:, None]
        acc_o += tl.dot(p.to(v.dtype), v)

        l_i = l_i * acc_scale + l_new
        m_i = m_new

    acc_o = acc_o / l_i[:, None]
    lse_i = m_i + tl.log2(l_i)

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
    tl.store(candidate_scores_ptr + candidate_offsets, top_scores, mask=candidate_mask)
    tl.store(
        candidate_indices_ptr + candidate_index_offsets,
        top_indices,
        mask=candidate_mask,
    )


# =============================================================================
# BNSD Decode Attention Merge Kernel
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _merge_bnsd_attn_out_kernel(
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    seq_lens,  # [B]
    out_ptr,  # [B, QH, D]
    # shape
    head_dim,
    block_size: tl.constexpr,
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
    NUM_KV_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_d = tl.arange(0, BLOCK_SIZE_D)

    seq_len = tl.load(seq_lens + pid_b).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    chunk_size_blocks = tl.maximum(1, tl.cdiv(num_blocks, NUM_KV_CHUNKS))
    valid_chunks = tl.cdiv(num_blocks, chunk_size_blocks)

    m = tl.full((), float("-inf"), dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.full((BLOCK_SIZE_D,), 0.0, dtype=tl.float32)

    for c in tl.static_range(0, NUM_KV_CHUNKS):
        valid = c < valid_chunks

        lse_c = tl.load(
            lse_ptr + c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
            mask=valid,
            other=float("-inf"),
        )

        o_c = tl.load(
            o_ptr
            + c * stride_o_c
            + pid_b * stride_o_b
            + pid_h * stride_o_h
            + off_d * stride_o_d,
            mask=valid & (off_d < head_dim),
            other=0.0,
        ).to(tl.float32)

        m_new = tl.maximum(m, lse_c)
        scale_old = tl.exp2(m - m_new)
        scale_new = tl.exp2(lse_c - m_new)

        acc = acc * scale_old + o_c * scale_new
        l = l * scale_old + scale_new
        m = m_new

    out = acc / l

    tl.store(
        out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + off_d * stride_out_d,
        out.to(out_ptr.dtype.element_ty),
        mask=off_d < head_dim,
    )


# =============================================================================
# Python Wrapper
# =============================================================================


@torch.no_grad()
def flash_decode_bnsd_with_topk_idx(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],  # optional [num_q_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_bnsd: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],  # [batch_size, max_num_blocks]
    seq_lens: torch.Tensor,  # [batch_size]
    max_seqlen: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    num_kv_chunks: Optional[int] = None,
    max_num_kv_chunks: int = 8,
    # Retained for call-site compatibility. Both short and long contexts use
    # the same chunk-candidate + merge Triton route.
    use_triton_topk: Optional[bool] = None,
    # Kept for API compatibility with earlier split-K implementations.
    num_topk_chunks: Optional[int] = None,
    # Retained for call-site compatibility. Score-only always uses the fused
    # chunk-candidate kernel.
    use_chunked_score: bool = True,
    # Target block tiles per program for the chunked score kernel. Larger ->
    # fewer programs (less scheduling) but longer serial loop; smaller -> more
    # parallelism. Tuned via bench_sparse_decode / bench_scale.
    score_blocks_per_chunk: int = 16,
    # Direct request-map page source. This is intentionally an alternative to a
    # materialized block table so graph replay cannot reuse a stale layer buffer.
    req_to_token: Optional[torch.Tensor] = None,
    req_pool_indices: Optional[torch.Tensor] = None,
    max_num_blocks: Optional[int] = None,
    num_pages: Optional[int] = None,
    sanitize_page_ids: bool = False,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Decode attention with BNSD KV cache and block-level topk indices.

    v5 optimization:
        full decode can run a fused score+attention kernel, avoiding the v3
        double QK computation. The fallback unfused path is kept for debugging.
        When num_kv_chunks is None, choose NUM_KV_CHUNKS dynamically using a
        SGLang-style target-grid rule adapted to Ascend vector cores.

    Returns:
        o:
            [batch_size, num_q_heads, head_dim], or None when
            disable_index_value=True.
        topk_idx:
            [num_q_heads, batch_size, topk], int32.
    """
    assert score_type in ("max", "lse")
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert seq_lens.dtype in (torch.int32, torch.int64)

    use_direct_page_lookup = req_to_token is not None
    assert (req_pool_indices is not None) == use_direct_page_lookup
    if use_direct_page_lookup:
        assert block_table is None
        assert disable_index_value
        assert req_to_token.ndim == 2
        assert req_to_token.dtype in (torch.int32, torch.int64)
        assert req_pool_indices.ndim == 1
        assert req_pool_indices.dtype in (torch.int32, torch.int64)
        assert max_num_blocks is not None and max_num_blocks > 0
        assert num_pages is not None and num_pages > 0
    else:
        assert block_table is not None
        assert block_table.dtype in (torch.int32, torch.int64)

    if not disable_index_value:
        assert v_cache_bnsd is not None
        assert v_cache_bnsd.dtype == q.dtype
        assert v_cache_bnsd.shape == k_cache_bnsd.shape

    batch_size, num_q_heads, head_dim = q.shape
    _, block_size_from_cache, num_kv_heads, cache_head_dim = k_cache_bnsd.shape

    assert block_size_from_cache == block_size
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    assert seq_lens.shape[0] == batch_size
    if use_direct_page_lookup:
        assert req_pool_indices.shape[0] == batch_size
        assert max_num_blocks * block_size <= req_to_token.shape[1]
        page_source = req_to_token
        page_source_rows = req_pool_indices
        direct_num_pages = int(num_pages)
    else:
        assert block_table.shape[0] == batch_size
        page_source = block_table
        # Triton requires a typed pointer even for constexpr-dead direct-mode
        # arguments. seq_lens is never read as an index in this legacy branch.
        page_source_rows = seq_lens
        direct_num_pages = 1

    gqa_group_size = num_q_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    max_seqblock = (max_seqlen + block_size - 1) // block_size
    block_size_n = _next_power_of_2(block_size)

    if disable_index_value:
        if topk <= 0:
            return None, torch.empty(
                (num_q_heads, batch_size, 0), dtype=torch.int32, device=q.device
            )
        num_score_chunks = _choose_num_score_chunks(
            max_seqblock,
            blocks_per_chunk=score_blocks_per_chunk,
            all_seqblock_q=batch_size,
            num_kv_heads=num_kv_heads,
        )
        candidate_scores = torch.empty(
            (num_score_chunks, num_q_heads, batch_size, topk),
            dtype=torch.float32,
            device=q.device,
        )
        candidate_indices = torch.empty(
            (num_score_chunks, num_q_heads, batch_size, topk),
            dtype=torch.int32,
            device=q.device,
        )
        _decode_bnsd_score_topk_chunk_kernel[
            (batch_size * num_score_chunks, num_kv_heads)
        ](
            q,
            k_cache_bnsd,
            page_source,
            page_source,
            page_source_rows,
            candidate_scores,
            candidate_indices,
            seq_lens,
            batch_size,
            gqa_group_size,
            head_dim,
            block_size,
            sm_scale,
            init_blocks,
            local_blocks,
            num_score_chunks,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache_bnsd.stride(0),
            k_cache_bnsd.stride(1),
            k_cache_bnsd.stride(2),
            k_cache_bnsd.stride(3),
            page_source.stride(0),
            page_source.stride(1),
            req_to_token.stride(0) if use_direct_page_lookup else 0,
            req_to_token.stride(1) if use_direct_page_lookup else 0,
            req_to_token.shape[1] if use_direct_page_lookup else 1,
            direct_num_pages,
            candidate_scores.stride(0),
            candidate_scores.stride(1),
            candidate_scores.stride(2),
            candidate_scores.stride(3),
            candidate_indices.stride(0),
            candidate_indices.stride(1),
            candidate_indices.stride(2),
            candidate_indices.stride(3),
            BLOCK_SIZE_N=block_size_n,
            SCORE_TYPE=score_type,
            topk=topk,
            USE_DIRECT_PAGE_LOOKUP=use_direct_page_lookup,
            SANITIZE_PAGE_IDS=sanitize_page_ids,
            num_warps=_SCORE_CHUNK_NW,
            num_stages=_SCORE_CHUNK_NS,
        )
        return None, _merge_bnsd_score_topk_candidates(
            candidate_scores, candidate_indices, topk
        )

    if num_kv_chunks is None:
        num_kv_chunks = _choose_num_kv_chunks(
            batch_size,
            num_kv_heads,
            max_num_kv_chunks=max_num_kv_chunks,
        )
    else:
        num_kv_chunks = int(num_kv_chunks)

    assert num_kv_chunks >= 1
    assert (num_kv_chunks & (num_kv_chunks - 1)) == 0

    o_chunks = torch.empty(
        (num_kv_chunks, batch_size, num_q_heads, head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    lse_chunks = torch.empty(
        (num_kv_chunks, batch_size, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )
    candidate_scores = torch.empty(
        (num_kv_chunks, num_q_heads, batch_size, topk),
        dtype=torch.float32,
        device=q.device,
    )
    candidate_indices = torch.empty(
        (num_kv_chunks, num_q_heads, batch_size, topk),
        dtype=torch.int32,
        device=q.device,
    )

    grid_attn = (batch_size * num_kv_chunks, num_kv_heads)
    _decode_bnsd_score_attn_chunk_kernel[grid_attn](
        q,
        sink,
        k_cache_bnsd,
        v_cache_bnsd,
        block_table,
        o_chunks,
        lse_chunks,
        candidate_scores,
        candidate_indices,
        seq_lens,
        batch_size,
        gqa_group_size,
        head_dim,
        block_size,
        sm_scale,
        init_blocks,
        local_blocks,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
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
        o_chunks.stride(0),
        o_chunks.stride(1),
        o_chunks.stride(2),
        o_chunks.stride(3),
        lse_chunks.stride(0),
        lse_chunks.stride(1),
        lse_chunks.stride(2),
        candidate_scores.stride(0),
        candidate_scores.stride(1),
        candidate_scores.stride(2),
        candidate_scores.stride(3),
        candidate_indices.stride(0),
        candidate_indices.stride(1),
        candidate_indices.stride(2),
        candidate_indices.stride(3),
        BLOCK_SIZE_N=block_size_n,
        NUM_KV_CHUNKS=num_kv_chunks,
        SCORE_TYPE=score_type,
        topk=topk,
        num_warps=4,
        num_stages=2,
    )

    topk_idx = _merge_bnsd_score_topk_candidates(
        candidate_scores, candidate_indices, topk
    )

    o = torch.empty_like(q)

    grid_merge = (batch_size, num_q_heads)
    _merge_bnsd_attn_out_kernel[grid_merge](
        o_chunks,
        lse_chunks,
        seq_lens,
        o,
        head_dim,
        block_size,
        o_chunks.stride(0),
        o_chunks.stride(1),
        o_chunks.stride(2),
        o_chunks.stride(3),
        lse_chunks.stride(0),
        lse_chunks.stride(1),
        lse_chunks.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        NUM_KV_CHUNKS=num_kv_chunks,
        num_warps=4,
        num_stages=2,
    )

    return o, topk_idx
