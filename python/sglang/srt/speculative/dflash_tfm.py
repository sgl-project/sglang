from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_correct_drafts_and_bonus,
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dflash_sampling_verify_available,
    sample_dflash_proposal_from_logits,
    top_k_renorm_prob,
    top_p_renorm_prob,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func
from sglang.srt.speculative.triton_ops.dflash import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.utils import is_cuda, is_musa

logger = logging.getLogger(__name__)

WEAVER_TREE_EXPAND_WIDTH = 8
WEAVER_TREE_BATCH_EXPAND_WIDTH = 8
WEAVER_TREE_BATCH_EXPAND_BUDGET_UNIT = 16


def weaver_fused_frontier_materialize_enabled() -> bool:
    return os.environ.get("SGLANG_WEAVER_FUSED_FRONTIER_MATERIALIZE", "1") == "1"


@triton.jit
def _weaver_candidate_frontier_kernel(
    logits_ptr,
    candidate_ids_ptr,
    prefix_score_ptr,
    node_depth_ptr,
    active_ptr,
    frontier_tokens_ptr,
    frontier_parents_ptr,
    frontier_depths_ptr,
    frontier_scores_ptr,
    frontier_logprobs_ptr,
    frontier_active_ptr,
    slot_start,
    WIDTH: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    EXPAND_WIDTH: tl.constexpr,
    DEPTH: tl.constexpr,
    FRONTIER_SLOTS: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_POOL)
    pool_mask = offsets < POOL_SIZE
    row_base = row * POOL_SIZE + offsets
    token_ids = tl.load(candidate_ids_ptr + row_base, mask=pool_mask, other=-1)
    scores = tl.load(logits_ptr + row_base, mask=pool_mask, other=-float("inf")).to(
        tl.float32
    )
    scores = tl.where((token_ids >= 0) & pool_mask, scores, -float("inf"))
    parent_score = tl.load(prefix_score_ptr + row)
    parent_depth = tl.load(node_depth_ptr + row)
    parent_active = (tl.load(active_ptr + row) != 0) & (parent_depth < DEPTH)
    batch = row // WIDTH
    row_in_width = row - batch * WIDTH
    max_score = tl.max(scores, axis=0)
    exp_scores = tl.where(scores == -float("inf"), 0.0, tl.exp(scores - max_score))
    log_denom = tl.log(tl.sum(exp_scores, axis=0)) + max_score
    child_base = batch * FRONTIER_SLOTS + (slot_start + row_in_width) * EXPAND_WIDTH
    child_depth = parent_depth + 1
    for child in tl.static_range(0, EXPAND_WIDTH):
        top_value, top_index = tl.max(
            scores,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        child_token = tl.load(candidate_ids_ptr + row * POOL_SIZE + top_index)
        child_valid = parent_active & (child_token >= 0) & (top_value != -float("inf"))
        out_index = child_base + child
        tl.store(frontier_tokens_ptr + out_index, tl.where(child_valid, child_token, 0))
        tl.store(
            frontier_parents_ptr + out_index,
            tl.where(child_valid, slot_start + row_in_width, 0),
        )
        tl.store(frontier_depths_ptr + out_index, tl.where(child_valid, child_depth, 0))
        tl.store(
            frontier_scores_ptr + out_index,
            tl.where(child_valid, parent_score + top_value - log_denom, -float("inf")),
        )
        tl.store(
            frontier_logprobs_ptr + out_index,
            tl.where(child_valid, top_value - log_denom, -float("inf")),
        )
        tl.store(frontier_active_ptr + out_index, child_valid)
        scores = tl.where(offsets == top_index, -float("inf"), scores)

@triton.jit
def _weaver_bitonic_step(
    keys,
    values,
    valid,
    lanes,
    SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    FINAL: tl.constexpr,
):
    low_lanes = lanes - (lanes & STRIDE)
    high_lanes = low_lanes + STRIDE
    low_keys = tl.gather(keys, low_lanes, axis=0)
    high_keys = tl.gather(keys, high_lanes, axis=0)
    low_values = tl.gather(values, low_lanes, axis=0)
    high_values = tl.gather(values, high_lanes, axis=0)
    low_valid = tl.gather(valid, low_lanes, axis=0)
    high_valid = tl.gather(valid, high_lanes, axis=0)

    swap = ((low_keys > high_keys) & low_valid) | (~high_valid)
    if FINAL:
        direction = tl.full((32,), False, tl.int1)
    else:
        thread = (low_lanes // (2 * STRIDE)) * STRIDE + low_lanes % STRIDE
        direction = (thread & (SIZE // 2)) != 0
    swap = swap == direction
    is_low = (lanes & STRIDE) == 0
    keys = tl.where(
        is_low,
        tl.where(swap, high_keys, low_keys),
        tl.where(swap, low_keys, high_keys),
    )
    values = tl.where(
        is_low,
        tl.where(swap, high_values, low_values),
        tl.where(swap, low_values, high_values),
    )
    valid = tl.where(
        is_low,
        tl.where(swap, high_valid, low_valid),
        tl.where(swap, low_valid, high_valid),
    )
    return keys, values, valid


@triton.jit
def _weaver_materialize_frontier_kernel(
    selected_indices_ptr,
    frontier_tokens_ptr,
    frontier_parents_ptr,
    frontier_depths_ptr,
    frontier_scores_ptr,
    frontier_logprobs_ptr,
    frontier_active_ptr,
    slot_ancestors_ptr,
    tokens_ptr,
    parents_ptr,
    depths_ptr,
    node_mask_ptr,
    draft_logprobs_ptr,
    selected_tokens_ptr,
    selected_depths_ptr,
    selected_position_ids_ptr,
    selected_candidate_rows_ptr,
    selected_batch_indices_ptr,
    selected_scores_ptr,
    selected_active_ptr,
    selected_parent_ancestors_ptr,
    slot_start,
    NUM_NODES: tl.constexpr,
    DEPTH: tl.constexpr,
    FRONTIER_SLOTS: tl.constexpr,
    SELECT_WIDTH: tl.constexpr,
    SCRATCH_WIDTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
    WRITE_ANCESTORS: tl.constexpr,
):
    batch = tl.program_id(0)
    selected_offset = tl.arange(0, 32)
    selected_mask = selected_offset < SELECT_WIDTH
    frontier_index = tl.load(
        selected_indices_ptr + batch * SELECT_WIDTH + selected_offset,
        mask=selected_mask,
        other=0,
    )
    selected_index = batch * FRONTIER_SLOTS + frontier_index
    score = tl.load(
        frontier_scores_ptr + selected_index,
        mask=selected_mask,
        other=0.0,
    ).to(tl.float32)
    sort_valid = selected_mask
    for stage in tl.static_range(1, 5):
        size = 1 << stage
        for pass_index in tl.static_range(0, stage):
            stride = size >> (pass_index + 1)
            score, frontier_index, sort_valid = _weaver_bitonic_step(
                score,
                frontier_index,
                sort_valid,
                selected_offset,
                SIZE=size,
                STRIDE=stride,
                FINAL=False,
            )
    for pass_index in tl.static_range(0, 5):
        stride = 16 >> pass_index
        score, frontier_index, sort_valid = _weaver_bitonic_step(
            score,
            frontier_index,
            sort_valid,
            selected_offset,
            SIZE=32,
            STRIDE=stride,
            FINAL=True,
        )

    selected_index = batch * FRONTIER_SLOTS + frontier_index
    valid = (
        tl.load(frontier_active_ptr + selected_index, mask=selected_mask, other=0)
        != 0
    )
    token = tl.load(frontier_tokens_ptr + selected_index, mask=selected_mask, other=0)
    parent = tl.load(frontier_parents_ptr + selected_index, mask=selected_mask, other=0)
    depth = tl.load(frontier_depths_ptr + selected_index, mask=selected_mask, other=0)
    logprob = tl.load(
        frontier_logprobs_ptr + selected_index,
        mask=selected_mask,
        other=-float("inf"),
    )

    output_index = batch * NUM_NODES + slot_start + selected_offset
    tl.store(
        tokens_ptr + output_index,
        tl.where(valid, token, 0),
        mask=selected_mask,
    )
    tl.store(
        parents_ptr + output_index,
        tl.where(valid, parent, -1),
        mask=selected_mask,
    )
    tl.store(
        depths_ptr + output_index,
        tl.where(valid, depth, 0),
        mask=selected_mask,
    )
    tl.store(node_mask_ptr + output_index, valid, mask=selected_mask)
    tl.store(
        draft_logprobs_ptr + output_index,
        tl.where(valid, logprob, -float("inf")),
        mask=selected_mask,
    )

    scratch_index = batch * SCRATCH_WIDTH + selected_offset
    tl.store(
        selected_tokens_ptr + scratch_index,
        tl.where(valid, token, 0),
        mask=selected_mask,
    )
    tl.store(
        selected_depths_ptr + scratch_index,
        tl.where(valid, depth, 0),
        mask=selected_mask,
    )
    position = tl.minimum(depth, DEPTH - 1)
    tl.store(
        selected_position_ids_ptr + scratch_index,
        tl.where(valid, position, 0),
        mask=selected_mask,
    )
    tl.store(
        selected_candidate_rows_ptr + scratch_index,
        tl.where(valid, batch * DEPTH + position, 0),
        mask=selected_mask,
    )
    tl.store(
        selected_batch_indices_ptr + scratch_index,
        batch,
        mask=selected_mask,
    )
    tl.store(
        selected_scores_ptr + scratch_index,
        tl.where(valid, score, -float("inf")),
        mask=selected_mask,
    )
    tl.store(selected_active_ptr + scratch_index, valid, mask=selected_mask)
    tl.store(
        frontier_active_ptr + selected_index,
        False,
        mask=selected_mask,
    )
    tl.store(
        frontier_scores_ptr + selected_index,
        -float("inf"),
        mask=selected_mask,
    )

    if WRITE_ANCESTORS:
        ancestor_offsets = tl.arange(0, BLOCK_DEPTH)[None, :]
        ancestor_mask = (ancestor_offsets < DEPTH) & selected_mask[:, None]
        parent_safe = tl.minimum(tl.maximum(parent, 0), NUM_NODES - 1)[:, None]
        ancestors = tl.load(
            slot_ancestors_ptr
            + (batch * NUM_NODES + parent_safe) * DEPTH
            + ancestor_offsets,
            mask=ancestor_mask & valid[:, None],
            other=-1,
        )
        tl.store(
            selected_parent_ancestors_ptr
            + scratch_index[:, None] * DEPTH
            + ancestor_offsets,
            ancestors,
            mask=ancestor_mask,
        )


@triton.jit
def _weaver_indexed_attention_kernel(
    q_ptr,
    current_keys_ptr,
    current_values_ptr,
    external_keys_ptr,
    external_values_ptr,
    external_mask_ptr,
    node_keys_ptr,
    node_values_ptr,
    parent_ancestors_ptr,
    row_batch_indices_ptr,
    position_ids_ptr,
    out_ptr,
    LAYER: tl.constexpr,
    PREFIX: tl.constexpr,
    DEPTH: tl.constexpr,
    NUM_NODES: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_CTX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    ctx = tl.arange(0, BLOCK_CTX)
    ctx_offsets = ctx[:, None]
    dim_offsets = tl.arange(0, BLOCK_D)[None, :]
    dim_mask = dim_offsets < HEAD_DIM
    batch = tl.load(row_batch_indices_ptr + row)
    pos = tl.load(position_ids_ptr + row)
    pos = tl.minimum(tl.maximum(pos, 0), DEPTH - 1)

    q = tl.load(
        q_ptr + (row * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    ext_pos = ctx_offsets
    tok_pos = ctx_offsets - PREFIX
    tok_pos_1d = ctx - PREFIX
    ext_pos_safe = tl.minimum(ctx_offsets, PREFIX - 1)
    ext_pos_1d_safe = tl.minimum(ctx, PREFIX - 1)
    tok_pos_1d_safe = tl.maximum(tok_pos_1d, 0)
    ext_ctx = ctx_offsets < PREFIX
    tok_ctx_1d = (ctx >= PREFIX) & (tok_pos_1d <= pos)
    tok_current_1d = tok_ctx_1d & (tok_pos_1d == pos)
    tok_ancestor_1d = tok_ctx_1d & (tok_pos_1d < pos)
    ancestor_slot = tl.load(
        parent_ancestors_ptr + row * DEPTH + tok_pos_1d_safe,
        mask=tok_ancestor_1d,
        other=-1,
    )
    ancestor_valid_1d = tok_ancestor_1d & (ancestor_slot >= 0)
    ancestor_slot = tl.maximum(ancestor_slot, 0)
    tok_current = tok_current_1d[:, None]
    ancestor_valid = ancestor_valid_1d[:, None]
    ancestor_slot = ancestor_slot[:, None]

    ext_key = tl.load(
        external_keys_ptr
        + (
            ((batch * PREFIX + ext_pos_safe) * NUM_HEADS + head)
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ext_ctx & dim_mask,
        other=0.0,
    )
    current_key = tl.load(
        current_keys_ptr + (row * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    ancestor_key = tl.load(
        node_keys_ptr
        + (
            (
                ((batch * NUM_NODES + ancestor_slot) * NUM_LAYERS + LAYER)
                * NUM_HEADS
                + head
            )
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ancestor_valid & dim_mask,
        other=0.0,
    )
    tok_key = tl.where(tok_current, current_key, ancestor_key)
    key = tl.where(ext_ctx, ext_key, tok_key).to(tl.float32)
    scores = tl.dot(key.to(tl.bfloat16), tl.trans(q.to(tl.bfloat16)))
    scores = (
        scores.to(tl.bfloat16) * tl.rsqrt(HEAD_DIM + 0.0)
    ).to(tl.bfloat16).to(tl.float32)
    ext_mask = (
        tl.load(
        external_mask_ptr + batch * PREFIX + ext_pos_1d_safe,
        mask=ctx < PREFIX,
        other=0,
        )
        != 0
    )
    ctx_valid = (
        ((ctx < PREFIX) & ext_mask)
        | ((ctx >= PREFIX) & ((ctx - PREFIX) == pos))
        | ancestor_valid_1d
    )
    scores = tl.where(ctx_valid[:, None], scores, -float("inf"))
    max_score = tl.max(scores, axis=0)
    probs = tl.exp(scores - max_score)
    probs = probs / tl.sum(probs, axis=0)
    probs = probs.to(tl.bfloat16)

    ext_value = tl.load(
        external_values_ptr
        + (
            ((batch * PREFIX + ext_pos_safe) * NUM_HEADS + head)
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ext_ctx & dim_mask,
        other=0.0,
    )
    current_value = tl.load(
        current_values_ptr + (row * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    ancestor_value = tl.load(
        node_values_ptr
        + (
            (
                ((batch * NUM_NODES + ancestor_slot) * NUM_LAYERS + LAYER)
                * NUM_HEADS
                + head
            )
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ancestor_valid & dim_mask,
        other=0.0,
    )
    tok_value = tl.where(tok_current, current_value, ancestor_value)
    ext_probs = tl.where(ext_ctx, probs, 0.0).to(tl.bfloat16)
    tok_probs = tl.where(ctx_offsets >= PREFIX, probs, 0.0).to(tl.bfloat16)
    ext_out = tl.dot(tl.trans(ext_probs), ext_value).to(tl.bfloat16)
    tok_out = tl.dot(tl.trans(tok_probs), tok_value).to(tl.bfloat16)
    out = (ext_out + tok_out).to(tl.bfloat16)
    tl.store(
        out_ptr + (row * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        out,
        mask=dim_mask,
    )


@triton.jit
def _weaver_chain_attention_kernel(
    q_ptr,
    current_keys_ptr,
    current_values_ptr,
    external_keys_ptr,
    external_values_ptr,
    external_mask_ptr,
    chain_keys_ptr,
    chain_values_ptr,
    position_ids_ptr,
    out_ptr,
    LAYER: tl.constexpr,
    PREFIX: tl.constexpr,
    DEPTH: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_CTX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    ctx = tl.arange(0, BLOCK_CTX)
    ctx_offsets = ctx[:, None]
    dim_offsets = tl.arange(0, BLOCK_D)[None, :]
    dim_mask = dim_offsets < HEAD_DIM
    pos = tl.load(position_ids_ptr + batch)
    pos = tl.minimum(tl.maximum(pos, 0), DEPTH - 1)

    q = tl.load(
        q_ptr + (batch * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    ext_pos = ctx_offsets
    tok_pos = ctx_offsets - PREFIX
    tok_pos_1d = ctx - PREFIX
    ext_pos_safe = tl.minimum(ctx_offsets, PREFIX - 1)
    ext_pos_1d_safe = tl.minimum(ctx, PREFIX - 1)
    tok_pos_safe = tl.maximum(tl.minimum(tok_pos, DEPTH - 1), 0)
    tok_ctx_1d = (ctx >= PREFIX) & (tok_pos_1d <= pos)
    tok_current = (tok_ctx_1d & (tok_pos_1d == pos))[:, None]
    tok_history = (tok_ctx_1d & (tok_pos_1d < pos))[:, None]
    ext_ctx = ctx_offsets < PREFIX

    ext_key = tl.load(
        external_keys_ptr
        + (
            ((batch * PREFIX + ext_pos_safe) * NUM_HEADS + head)
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ext_ctx & dim_mask,
        other=0.0,
    )
    current_key = tl.load(
        current_keys_ptr + (batch * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    history_key = tl.load(
        chain_keys_ptr
        + (
            (
                ((batch * DEPTH + tok_pos_safe) * NUM_LAYERS + LAYER) * NUM_HEADS
                + head
            )
            * HEAD_DIM
            + dim_offsets
        ),
        mask=tok_history & dim_mask,
        other=0.0,
    )
    tok_key = tl.where(tok_current, current_key, history_key)
    key = tl.where(ext_ctx, ext_key, tok_key).to(tl.float32)
    scores = tl.dot(key.to(tl.bfloat16), tl.trans(q.to(tl.bfloat16)))
    scores = (
        scores.to(tl.bfloat16) * tl.rsqrt(HEAD_DIM + 0.0)
    ).to(tl.bfloat16).to(tl.float32)
    ext_mask = (
        tl.load(
            external_mask_ptr + batch * PREFIX + ext_pos_1d_safe,
            mask=ctx < PREFIX,
            other=0,
        )
        != 0
    )
    ctx_valid = ((ctx < PREFIX) & ext_mask) | tok_ctx_1d
    scores = tl.where(ctx_valid[:, None], scores, -float("inf"))
    max_score = tl.max(scores, axis=0)
    probs = tl.exp(scores - max_score)
    probs = probs / tl.sum(probs, axis=0)
    probs = probs.to(tl.bfloat16)

    ext_value = tl.load(
        external_values_ptr
        + (
            ((batch * PREFIX + ext_pos_safe) * NUM_HEADS + head)
            * HEAD_DIM
            + dim_offsets
        ),
        mask=ext_ctx & dim_mask,
        other=0.0,
    )
    current_value = tl.load(
        current_values_ptr + (batch * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    history_value = tl.load(
        chain_values_ptr
        + (
            (
                ((batch * DEPTH + tok_pos_safe) * NUM_LAYERS + LAYER) * NUM_HEADS
                + head
            )
            * HEAD_DIM
            + dim_offsets
        ),
        mask=tok_history & dim_mask,
        other=0.0,
    )
    tok_value = tl.where(tok_current, current_value, history_value)
    ext_probs = tl.where(ext_ctx, probs, 0.0).to(tl.bfloat16)
    tok_probs = tl.where(ctx_offsets >= PREFIX, probs, 0.0).to(tl.bfloat16)
    ext_out = tl.dot(tl.trans(ext_probs), ext_value).to(tl.bfloat16)
    tok_out = tl.dot(tl.trans(tok_probs), tok_value).to(tl.bfloat16)
    out = (ext_out + tok_out).to(tl.bfloat16)
    tl.store(
        out_ptr + (batch * NUM_HEADS + head) * HEAD_DIM + dim_offsets,
        out,
        mask=dim_mask,
    )


@triton.jit
def _tree_metadata_parent_chain_kernel(
    parent_indices_ptr,
    node_mask_ptr,
    prefix_lens_ptr,
    mask_offsets_ptr,
    custom_mask_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    NUM_NODES: tl.constexpr,
    CHAIN_STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    row = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < NUM_NODES
    row_base = batch * NUM_NODES
    row_valid = tl.load(node_mask_ptr + row_base + row) != 0
    prefix_len = tl.load(prefix_lens_ptr + batch)
    width = prefix_len + NUM_NODES

    ancestor = tl.full((BLOCK_N,), False, dtype=tl.int1)
    current = row
    current_valid = row_valid
    for _ in tl.static_range(0, CHAIN_STEPS):
        ancestor |= current_valid & col_mask & (cols == current)
        parent = tl.load(
            parent_indices_ptr + row_base + current,
            mask=current_valid,
            other=-1,
        )
        current_valid = current_valid & (parent >= 0) & (parent < NUM_NODES)
        current = tl.maximum(tl.minimum(parent, NUM_NODES - 1), 0)

    mask_offset = tl.load(mask_offsets_ptr + batch) + row * width + prefix_len + cols
    tl.store(custom_mask_ptr + mask_offset, ancestor, mask=col_mask)

    upper = col_mask & (cols > row)
    rows = tl.full((BLOCK_N,), row, dtype=tl.int64)
    descendant = tl.full((BLOCK_N,), False, dtype=tl.int1)
    current = cols
    col_valid = tl.load(node_mask_ptr + row_base + cols, mask=col_mask, other=0) != 0
    current_valid = upper & col_valid & row_valid
    for _ in tl.static_range(0, CHAIN_STEPS):
        descendant |= current_valid & (current == rows)
        parent = tl.load(
            parent_indices_ptr + row_base + current,
            mask=current_valid,
            other=-1,
        )
        current_valid = current_valid & (parent >= 0) & (parent < NUM_NODES)
        current = tl.maximum(tl.minimum(parent, NUM_NODES - 1), 0)
    next_token_values = tl.where(descendant, cols, NUM_NODES)
    next_token = tl.min(next_token_values, axis=0)
    next_token = tl.where(next_token == NUM_NODES, -1, next_token)

    row_parent = tl.load(parent_indices_ptr + row_base + row)
    col_parent = tl.load(parent_indices_ptr + row_base + cols, mask=col_mask, other=-2)
    sibling = (
        upper
        & row_valid
        & (row_parent >= 0)
        & (col_parent == row_parent)
        & col_valid
    )
    next_sibling_values = tl.where(sibling, cols, NUM_NODES)
    next_sibling = tl.min(next_sibling_values, axis=0)
    next_sibling = tl.where(next_sibling == NUM_NODES, -1, next_sibling)

    tl.store(retrieve_next_token_ptr + row_base + row, next_token)
    tl.store(retrieve_next_sibling_ptr + row_base + row, next_sibling)


@triton.jit
def _weaver_traversal_verify_kernel(
    candidates_ptr,
    parent_indices_ptr,
    depths_ptr,
    node_mask_ptr,
    draft_logprobs_ptr,
    target_probs_ptr,
    uniform_samples_ptr,
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    accept_leaf_ptr,
    NUM_NODES: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    col_mask = offsets < NUM_NODES
    row_base = batch * NUM_NODES

    tl.store(
        predicts_ptr + row_base + offsets,
        tl.full((BLOCK_N,), -1, dtype=tl.int32),
        mask=col_mask,
    )
    tl.store(
        accept_index_ptr + row_base + offsets,
        tl.full((BLOCK_N,), -1, dtype=tl.int32),
        mask=col_mask,
    )

    parents = tl.load(parent_indices_ptr + row_base + offsets, mask=col_mask, other=-1)
    depths = tl.load(depths_ptr + row_base + offsets, mask=col_mask, other=0)
    tokens = tl.load(candidates_ptr + row_base + offsets, mask=col_mask, other=0)
    active = (tl.load(node_mask_ptr + row_base + offsets, mask=col_mask, other=0) != 0) & col_mask
    active = active | (offsets == 0)

    local_logprobs = tl.load(
        draft_logprobs_ptr + row_base + offsets,
        mask=col_mask,
        other=-float("inf"),
    ).to(tl.float32)
    local_weights = tl.where((offsets > 0) & active, tl.exp(local_logprobs), 0.0)
    draft_probs = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for node in tl.range(1, NUM_NODES, loop_unroll_factor=1):
        node_parent = tl.load(parent_indices_ptr + row_base + node)
        node_weight = tl.load(draft_logprobs_ptr + row_base + node).to(tl.float32)
        node_weight = tl.exp(node_weight)
        sibling_weight = tl.sum(
            tl.where((parents == node_parent) & active & (offsets > 0), local_weights, 0.0),
            axis=0,
        )
        node_prob = node_weight / tl.maximum(sibling_weight, 1.0e-20)
        node_is_active = (tl.load(node_mask_ptr + row_base + node) != 0) & (sibling_weight > 0.0)
        draft_probs = tl.where((offsets == node) & node_is_active, node_prob, draft_probs)

    node_p = tl.where(offsets == 0, 1.0, 0.0).to(tl.float32)
    node_p_valid = offsets == 0
    accept_leaf = tl.full((), 0, dtype=tl.int64)
    done = tl.full((), False, dtype=tl.int1)

    verify_step = tl.full((), 0, dtype=tl.int64)
    while (verify_step < NUM_NODES) & (~done):
        cur = tl.full((), 0, dtype=tl.int64)
        cur_p = tl.full((), 1.0, dtype=tl.float32)
        parent_for_leaf = tl.full((), 0, dtype=tl.int64)
        p_parent_for_leaf = tl.full((), 1.0, dtype=tl.float32)
        leaf = tl.full((), 0, dtype=tl.int64)
        leaf_p = tl.full((), 1.0, dtype=tl.float32)
        descending = ~done

        descend_step = tl.full((), 0, dtype=tl.int64)
        while (descend_step < NUM_NODES) & descending:
            child_values = tl.where(active & (parents == cur), offsets, NUM_NODES)
            child = tl.min(child_values, axis=0)
            has_child = child < NUM_NODES
            take_child = descending & has_child
            take_leaf = descending & (~has_child)

            child_safe = tl.minimum(tl.maximum(child, 0), NUM_NODES - 1)
            child_token = tl.load(candidates_ptr + row_base + child_safe)
            child_token_safe = tl.minimum(tl.maximum(child_token, 0), VOCAB_SIZE - 1)
            child_q = tl.load(
                target_probs_ptr + (row_base + cur) * VOCAB_SIZE + child_token_safe,
                mask=take_child,
                other=0.0,
            ).to(tl.float32)
            child_s = tl.sum(tl.where(offsets == child, draft_probs, 0.0), axis=0)
            computed_child_p = tl.minimum(
                cur_p * child_q / tl.maximum(child_s, 1.0e-20),
                1.0,
            )
            stored_child_p = tl.sum(tl.where(offsets == child, node_p, 0.0), axis=0)
            stored_child_valid = tl.sum(
                tl.where(offsets == child, node_p_valid.to(tl.int32), 0),
                axis=0,
            ) != 0
            next_child_p = tl.where(stored_child_valid, stored_child_p, computed_child_p)

            leaf = tl.where(take_leaf, cur, leaf)
            leaf_p = tl.where(take_leaf, cur_p, leaf_p)
            parent_for_leaf = tl.where(take_child, cur, parent_for_leaf)
            p_parent_for_leaf = tl.where(take_child, cur_p, p_parent_for_leaf)
            cur = tl.where(take_child, child, cur)
            cur_p = tl.where(take_child, next_child_p, cur_p)
            descending = descending & has_child
            descend_step += 1

        eta = tl.load(uniform_samples_ptr + row_base + verify_step, mask=~done, other=0.0)
        accept_now = (~done) & ((leaf == 0) | (eta < leaf_p))
        reject_now = (~done) & (~accept_now)
        accept_leaf = tl.where(accept_now, leaf, accept_leaf)

        leaf_safe = tl.minimum(tl.maximum(leaf, 0), NUM_NODES - 1)
        reject_parent = tl.load(parent_indices_ptr + row_base + leaf_safe, mask=reject_now, other=0)
        reject_parent = tl.minimum(tl.maximum(reject_parent, 0), NUM_NODES - 1)

        child_mask = active & (parents == reject_parent)
        child_tokens = tl.minimum(tl.maximum(tokens, 0), VOCAB_SIZE - 1)
        q_children = tl.load(
            target_probs_ptr + (row_base + reject_parent) * VOCAB_SIZE + child_tokens,
            mask=child_mask & reject_now,
            other=0.0,
        ).to(tl.float32)
        q_sum = tl.sum(tl.where(child_mask, q_children, 0.0), axis=0)
        positive = tl.maximum(p_parent_for_leaf * q_children - draft_probs, 0.0)
        positive_sum = tl.sum(tl.where(child_mask, positive, 0.0), axis=0)
        target_tail = tl.maximum(p_parent_for_leaf * (1.0 - q_sum), 0.0)
        residual_mass = positive_sum + target_tail
        new_parent_p = residual_mass / tl.maximum(
            residual_mass + 1.0 - p_parent_for_leaf,
            1.0e-20,
        )

        rejected_s = tl.sum(tl.where(offsets == leaf, draft_probs, 0.0), axis=0)
        renorm = 1.0 / tl.maximum(1.0 - rejected_s, 1.0e-20)
        draft_probs = tl.where(
            reject_now & child_mask & (offsets != leaf),
            draft_probs * renorm,
            draft_probs,
        )
        draft_probs = tl.where(reject_now & (offsets == leaf), 0.0, draft_probs)
        active = tl.where(reject_now & (offsets == leaf), False, active)
        node_p = tl.where(reject_now & (offsets == reject_parent), new_parent_p, node_p)
        node_p_valid = node_p_valid | (reject_now & (offsets == reject_parent))
        done = done | accept_now
        verify_step += 1

    accept_leaf = tl.minimum(tl.maximum(accept_leaf, 0), NUM_NODES - 1)
    tl.store(accept_leaf_ptr + batch, accept_leaf)
    leaf_depth = tl.load(depths_ptr + row_base + accept_leaf).to(tl.int32)
    tl.store(accept_token_num_ptr + batch, leaf_depth)

    chain_node = accept_leaf
    chain_step = tl.full((), 0, dtype=tl.int64)
    while (chain_step < NUM_NODES) & (chain_node >= 0):
        chain_valid = chain_node >= 0
        chain_safe = tl.minimum(tl.maximum(chain_node, 0), NUM_NODES - 1)
        chain_depth = tl.load(depths_ptr + row_base + chain_safe, mask=chain_valid, other=0)
        tl.store(
            accept_index_ptr + row_base + chain_depth,
            (row_base + chain_safe).to(tl.int32),
            mask=chain_valid & (chain_depth < NUM_NODES),
        )
        parent = tl.load(parent_indices_ptr + row_base + chain_safe, mask=chain_valid, other=-1)
        parent_safe = tl.minimum(tl.maximum(parent, 0), NUM_NODES - 1)
        token = tl.load(candidates_ptr + row_base + chain_safe, mask=chain_valid, other=0)
        tl.store(
            predicts_ptr + row_base + parent_safe,
            token.to(tl.int32),
            mask=chain_valid & (parent >= 0),
        )
        chain_node = tl.where(chain_valid, parent, chain_node)
        chain_step += 1


@triton.jit
def _weaver_current_cache_write_kernel(
    current_keys_ptr,
    current_values_ptr,
    node_keys_ptr,
    node_values_ptr,
    parent_ancestors_ptr,
    slot_ancestors_ptr,
    valid_ptr,
    node_depth_ptr,
    slot_start,
    BS: tl.constexpr,
    WIDTH: tl.constexpr,
    DEPTH: tl.constexpr,
    NUM_NODES: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOTAL_KV: tl.constexpr,
    TOTAL_ANCESTORS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    kv_mask = offsets < TOTAL_KV
    hd = offsets % HEAD_DIM
    head = (offsets // HEAD_DIM) % NUM_HEADS
    layer = (offsets // (HEAD_DIM * NUM_HEADS)) % NUM_LAYERS
    row_in_width = (offsets // (HEAD_DIM * NUM_HEADS * NUM_LAYERS)) % WIDTH
    batch = (offsets // (HEAD_DIM * NUM_HEADS * NUM_LAYERS * WIDTH)) % BS
    row = batch * WIDTH + row_in_width
    valid = tl.load(valid_ptr + row, mask=kv_mask, other=0) != 0
    current_index = (((layer * BS * WIDTH + row) * NUM_HEADS + head) * HEAD_DIM + hd)
    slot = slot_start + row_in_width
    node_index = (
        ((((batch * NUM_NODES + slot) * NUM_LAYERS + layer) * NUM_HEADS + head)
        * HEAD_DIM
        + hd)
    )
    key_value = tl.load(current_keys_ptr + current_index, mask=kv_mask & valid, other=0.0)
    value_value = tl.load(current_values_ptr + current_index, mask=kv_mask & valid, other=0.0)
    tl.store(node_keys_ptr + node_index, key_value, mask=kv_mask)
    tl.store(node_values_ptr + node_index, value_value, mask=kv_mask)

    ancestor_mask = offsets < TOTAL_ANCESTORS
    ancestor_depth = offsets % DEPTH
    ancestor_row = (offsets // DEPTH) % WIDTH
    ancestor_batch = (offsets // (DEPTH * WIDTH)) % BS
    ancestor_flat_row = ancestor_batch * WIDTH + ancestor_row
    ancestor_valid = (
        tl.load(
        valid_ptr + ancestor_flat_row,
        mask=ancestor_mask,
        other=0,
        )
        != 0
    )
    current_pos = tl.load(
        node_depth_ptr + ancestor_flat_row,
        mask=ancestor_mask,
        other=0,
    )
    current_pos = tl.minimum(current_pos, DEPTH - 1)
    parent_value = tl.load(parent_ancestors_ptr + offsets, mask=ancestor_mask, other=-1)
    ancestor_slot = slot_start + ancestor_row
    ancestor_value = tl.where(ancestor_depth == current_pos, ancestor_slot, parent_value)
    ancestor_value = tl.where(ancestor_valid, ancestor_value, -1)
    out_index = (ancestor_batch * NUM_NODES + ancestor_slot) * DEPTH + ancestor_depth
    tl.store(slot_ancestors_ptr + out_index, ancestor_value, mask=ancestor_mask)


@triton.jit
def _weaver_publish_frontier_kernel(
    current_keys_ptr,
    current_values_ptr,
    node_keys_ptr,
    node_values_ptr,
    parent_ancestors_ptr,
    slot_ancestors_ptr,
    logits_ptr,
    candidate_ids_ptr,
    prefix_score_ptr,
    valid_ptr,
    node_depth_ptr,
    frontier_tokens_ptr,
    frontier_parents_ptr,
    frontier_depths_ptr,
    frontier_scores_ptr,
    frontier_logprobs_ptr,
    frontier_active_ptr,
    slot_start,
    BS: tl.constexpr,
    WIDTH: tl.constexpr,
    DEPTH: tl.constexpr,
    NUM_NODES: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOTAL_KV: tl.constexpr,
    TOTAL_ANCESTORS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    EXPAND_WIDTH: tl.constexpr,
    FRONTIER_SLOTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
):
    program = tl.program_id(0)
    offsets = program * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    kv_mask = offsets < TOTAL_KV
    hd = offsets % HEAD_DIM
    head = (offsets // HEAD_DIM) % NUM_HEADS
    layer = (offsets // (HEAD_DIM * NUM_HEADS)) % NUM_LAYERS
    row_in_width = (offsets // (HEAD_DIM * NUM_HEADS * NUM_LAYERS)) % WIDTH
    batch = (offsets // (HEAD_DIM * NUM_HEADS * NUM_LAYERS * WIDTH)) % BS
    row = batch * WIDTH + row_in_width
    valid = tl.load(valid_ptr + row, mask=kv_mask, other=0) != 0
    current_index = (((layer * BS * WIDTH + row) * NUM_HEADS + head) * HEAD_DIM + hd)
    slot = slot_start + row_in_width
    node_index = (
        ((((batch * NUM_NODES + slot) * NUM_LAYERS + layer) * NUM_HEADS + head)
        * HEAD_DIM
        + hd)
    )
    key_value = tl.load(current_keys_ptr + current_index, mask=kv_mask & valid, other=0.0)
    value_value = tl.load(
        current_values_ptr + current_index, mask=kv_mask & valid, other=0.0
    )
    tl.store(node_keys_ptr + node_index, key_value, mask=kv_mask)
    tl.store(node_values_ptr + node_index, value_value, mask=kv_mask)

    ancestor_mask = offsets < TOTAL_ANCESTORS
    ancestor_depth = offsets % DEPTH
    ancestor_row = (offsets // DEPTH) % WIDTH
    ancestor_batch = (offsets // (DEPTH * WIDTH)) % BS
    ancestor_flat_row = ancestor_batch * WIDTH + ancestor_row
    ancestor_valid = (
        tl.load(valid_ptr + ancestor_flat_row, mask=ancestor_mask, other=0) != 0
    )
    current_pos = tl.load(
        node_depth_ptr + ancestor_flat_row,
        mask=ancestor_mask,
        other=0,
    )
    current_pos = tl.minimum(current_pos, DEPTH - 1)
    parent_value = tl.load(parent_ancestors_ptr + offsets, mask=ancestor_mask, other=-1)
    ancestor_slot = slot_start + ancestor_row
    ancestor_value = tl.where(ancestor_depth == current_pos, ancestor_slot, parent_value)
    ancestor_value = tl.where(ancestor_valid, ancestor_value, -1)
    out_index = (ancestor_batch * NUM_NODES + ancestor_slot) * DEPTH + ancestor_depth
    tl.store(slot_ancestors_ptr + out_index, ancestor_value, mask=ancestor_mask)

    if program < BS * WIDTH:
        candidate_offsets = tl.arange(0, BLOCK_POOL)
        pool_mask = candidate_offsets < POOL_SIZE
        candidate_base = program * POOL_SIZE + candidate_offsets
        token_ids = tl.load(
            candidate_ids_ptr + candidate_base, mask=pool_mask, other=-1
        )
        scores = tl.load(
            logits_ptr + candidate_base,
            mask=pool_mask,
            other=-float("inf"),
        ).to(tl.float32)
        scores = tl.where((token_ids >= 0) & pool_mask, scores, -float("inf"))
        parent_score = tl.load(prefix_score_ptr + program)
        parent_depth = tl.load(node_depth_ptr + program)
        parent_active = (tl.load(valid_ptr + program) != 0) & (parent_depth < DEPTH)
        frontier_batch = program // WIDTH
        frontier_row = program - frontier_batch * WIDTH
        max_score = tl.max(scores, axis=0)
        exp_scores = tl.where(
            scores == -float("inf"), 0.0, tl.exp(scores - max_score)
        )
        log_denom = tl.log(tl.sum(exp_scores, axis=0)) + max_score
        child_base = (
            frontier_batch * FRONTIER_SLOTS
            + (slot_start + frontier_row) * EXPAND_WIDTH
        )
        child_depth = parent_depth + 1
        for child in tl.static_range(0, EXPAND_WIDTH):
            top_value, top_index = tl.max(
                scores,
                axis=0,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            child_token = tl.load(candidate_ids_ptr + program * POOL_SIZE + top_index)
            child_valid = (
                parent_active & (child_token >= 0) & (top_value != -float("inf"))
            )
            child_index = child_base + child
            tl.store(
                frontier_tokens_ptr + child_index,
                tl.where(child_valid, child_token, 0),
            )
            tl.store(
                frontier_parents_ptr + child_index,
                tl.where(child_valid, slot_start + frontier_row, 0),
            )
            tl.store(
                frontier_depths_ptr + child_index,
                tl.where(child_valid, child_depth, 0),
            )
            tl.store(
                frontier_scores_ptr + child_index,
                tl.where(
                    child_valid,
                    parent_score + top_value - log_denom,
                    -float("inf"),
                ),
            )
            tl.store(
                frontier_logprobs_ptr + child_index,
                tl.where(child_valid, top_value - log_denom, -float("inf")),
            )
            tl.store(frontier_active_ptr + child_index, child_valid)
            scores = tl.where(candidate_offsets == top_index, -float("inf"), scores)


def weaver_tree_batch_expand_width(tree_budget: Optional[int] = None) -> int:
    """Weaver expansion batch width: one Weaver call expands this many nodes.

    Scales with the tree budget so a tree of B nodes takes ~B/16 batched calls.
    """
    if tree_budget is None:
        return WEAVER_TREE_BATCH_EXPAND_WIDTH
    budget = int(tree_budget)
    if budget <= 0:
        return 1
    return max(
        1,
        (budget + WEAVER_TREE_BATCH_EXPAND_BUDGET_UNIT - 1)
        // WEAVER_TREE_BATCH_EXPAND_BUDGET_UNIT,
    )


def _weaver_indexed_attention(
    q: torch.Tensor,
    current_keys: torch.Tensor,
    current_values: torch.Tensor,
    external_keys: torch.Tensor,
    external_values: torch.Tensor,
    external_mask: torch.Tensor,
    node_keys: torch.Tensor,
    node_values: torch.Tensor,
    parent_ancestors: torch.Tensor,
    row_batch_indices: torch.Tensor,
    position_ids: torch.Tensor,
    layer_index: int,
) -> torch.Tensor:
    if triton is None or q.device.type != "cuda":
        raise RuntimeError("indexed weaver attention requires Triton on CUDA.")
    rows, num_heads, head_dim = q.shape
    prefix = external_keys.shape[1]
    depth = parent_ancestors.shape[1]
    num_nodes = node_keys.shape[1]
    num_layers = node_keys.shape[2]
    out = torch.empty_like(q)
    block_ctx = triton.next_power_of_2(int(prefix + depth))
    block_d = triton.next_power_of_2(int(head_dim))
    _weaver_indexed_attention_kernel[(int(rows), int(num_heads))](
        q,
        current_keys,
        current_values,
        external_keys,
        external_values,
        external_mask,
        node_keys,
        node_values,
        parent_ancestors,
        row_batch_indices,
        position_ids,
        out,
        LAYER=int(layer_index),
        PREFIX=int(prefix),
        DEPTH=int(depth),
        NUM_NODES=int(num_nodes),
        NUM_LAYERS=int(num_layers),
        NUM_HEADS=int(num_heads),
        HEAD_DIM=int(head_dim),
        BLOCK_CTX=int(block_ctx),
        BLOCK_D=int(block_d),
    )
    return out


def _weaver_chain_attention(
    q: torch.Tensor,
    current_keys: torch.Tensor,
    current_values: torch.Tensor,
    external_keys: torch.Tensor,
    external_values: torch.Tensor,
    external_mask: torch.Tensor,
    chain_keys: torch.Tensor,
    chain_values: torch.Tensor,
    position_ids: torch.Tensor,
    layer_index: int,
) -> torch.Tensor:
    if triton is None or q.device.type != "cuda":
        raise RuntimeError("chain weaver attention requires Triton on CUDA.")
    rows, num_heads, head_dim = q.shape
    prefix = external_keys.shape[1]
    depth = chain_keys.shape[1]
    num_layers = chain_keys.shape[2]
    out = torch.empty_like(q)
    block_ctx = triton.next_power_of_2(int(prefix + depth))
    block_d = triton.next_power_of_2(int(head_dim))
    _weaver_chain_attention_kernel[(int(rows), int(num_heads))](
        q,
        current_keys,
        current_values,
        external_keys,
        external_values,
        external_mask,
        chain_keys,
        chain_values,
        position_ids,
        out,
        LAYER=int(layer_index),
        PREFIX=int(prefix),
        DEPTH=int(depth),
        NUM_LAYERS=int(num_layers),
        NUM_HEADS=int(num_heads),
        HEAD_DIM=int(head_dim),
        BLOCK_CTX=int(block_ctx),
        BLOCK_D=int(block_d),
    )
    return out


TREE_ATTENTION_BACKENDS = frozenset(
    {
        "AiterAttnBackend",
        "FlashAttentionBackend",
        "FlashInferAttnBackend",
        "FlashInferMLAAttnBackend",
        "TritonAttnBackend",
        "WaveAttnBackend",
        "XPUAttentionBackend",
    }
)


def _tree_attention_backend(attn_backend):
    backend = attn_backend
    for _ in range(8):
        select_backend = getattr(backend, "_select_backend", None)
        if select_backend is not None:
            selected_backend = select_backend(ForwardMode.TARGET_VERIFY)
            if selected_backend is not backend:
                backend = selected_backend
                continue

        full_backend = getattr(backend, "full_attn_backend", None)
        if full_backend is None:
            break
        backend = full_backend
    return backend


def _tree_attention_backend_name(attn_backend) -> str:
    backend = _tree_attention_backend(attn_backend)
    return type(backend).__name__


def require_tree_attention_support(attn_backend) -> None:
    backend_name = _tree_attention_backend_name(attn_backend)
    if backend_name not in TREE_ATTENTION_BACKENDS:
        raise RuntimeError(
            "DFLASH_TFM requires TreeAttention custom-mask support, "
            f"but the selected target-verify attention backend is {backend_name}. "
            "Use a backend with speculative tree custom_mask support, such as "
            "triton, flashinfer, fa3/flashattention, aiter, or wave. "
            "For trtllm_mha decode, use a split backend with "
            "--prefill-attention-backend flashinfer and "
            "--speculative-attention-mode prefill."
        )


class SplitHiddenStates(msgspec.Struct):
    target_hidden: torch.Tensor
    output_norm: torch.Tensor


def split_dflash_tfm_hidden(
    hidden_states: torch.Tensor, hidden_size: int
) -> SplitHiddenStates:
    if hidden_states is None:
        raise RuntimeError("DFlash+Weaver requires captured target hidden states.")
    hidden_size = int(hidden_size)
    if hidden_states.shape[-1] <= hidden_size:
        raise RuntimeError(
            "DFlash+Weaver expected concatenated DFlash aux hidden and final hidden, "
            f"got feature_dim={hidden_states.shape[-1]}, hidden_size={hidden_size}."
        )
    return SplitHiddenStates(
        target_hidden=hidden_states[..., :-hidden_size].contiguous(),
        output_norm=hidden_states[..., -hidden_size:].contiguous(),
    )


def _last_extend_indices(
    extend_lens: torch.Tensor | List[int], device: torch.device
) -> torch.Tensor:
    if not isinstance(extend_lens, torch.Tensor):
        extend_lens = torch.tensor(extend_lens, dtype=torch.int64, device=device)
    else:
        extend_lens = extend_lens.to(device=device, dtype=torch.int64)
    return torch.cumsum(extend_lens, dim=0) - 1


class DFlashTfmDraftInput(DFlashDraftInputV2):
    output_norm: torch.Tensor

    def __init__(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        output_norm: torch.Tensor,
        committed_seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        bs = int(new_seq_lens.numel())
        device = bonus_tokens.device
        super().__init__(
            topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64),
            hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
        )
        self.output_norm = output_norm
        self.committed_seq_lens_cpu = committed_seq_lens_cpu

    @classmethod
    def create_idle_input(
        cls, device: torch.device, output_norm_dim: int
    ) -> "DFlashTfmDraftInput":
        return cls(
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            output_norm=torch.empty(
                (0, int(output_norm_dim)), device=device, dtype=torch.float16
            ),
        )

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        super().filter_batch(new_indices, has_been_filtered=has_been_filtered)
        self.output_norm = self.output_norm[new_indices]
        if self.committed_seq_lens_cpu is not None:
            self.committed_seq_lens_cpu = self.committed_seq_lens_cpu[
                new_indices.cpu()
            ]

    def merge_batch(self, spec_info: "DFlashTfmDraftInput"):
        super().merge_batch(spec_info)
        self.output_norm = torch.cat([self.output_norm, spec_info.output_norm], dim=0)
        if self.committed_seq_lens_cpu is not None:
            assert spec_info.committed_seq_lens_cpu is not None
            self.committed_seq_lens_cpu = torch.cat(
                [self.committed_seq_lens_cpu, spec_info.committed_seq_lens_cpu]
            )
        elif spec_info.committed_seq_lens_cpu is not None:
            self.committed_seq_lens_cpu = spec_info.committed_seq_lens_cpu


class WeaverRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (y * self.weight + self.bias).to(dtype=x.dtype)


class WeaverBlock(nn.Module):
    def __init__(self, d_rank: int, num_heads: int, mlp_dim: int):
        super().__init__()
        if d_rank % num_heads != 0:
            raise ValueError("d_rank must be divisible by num_heads")
        self.d_rank = int(d_rank)
        self.num_heads = int(num_heads)
        self.head_dim = int(d_rank // num_heads)
        self.norm_attn = WeaverRMSNorm(d_rank)
        self.qkv_proj = nn.Linear(d_rank, 3 * d_rank, bias=False)
        self.o_proj = nn.Linear(d_rank, d_rank, bias=False)
        self.norm_mlp = WeaverRMSNorm(d_rank)
        self.fc1 = nn.Linear(d_rank, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, d_rank)

    def forward(
        self,
        x: torch.Tensor,
        token_attention_mask: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, steps, _ = x.shape
        h = self.norm_attn(x)
        qkv = self.qkv_proj(h).view(
            rows, steps, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        scale = self.head_dim**-0.5
        ext_scores = torch.einsum("rshd,rsphd->rhsp", q, external_keys) * scale
        tok_scores = torch.einsum("rshd,rthd->rhst", q, k) * scale
        ext_scores = ext_scores.masked_fill(~external_mask[:, None], -torch.inf)
        tok_scores = tok_scores.masked_fill(~token_attention_mask[:, None], -torch.inf)
        scores = torch.cat([ext_scores, tok_scores], dim=-1)
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        prefix = external_keys.shape[2]
        ext_y = torch.einsum(
            "rhsp,rsphd->rshd", attn[:, :, :, :prefix], external_values
        )
        tok_y = torch.einsum("rhst,rthd->rshd", attn[:, :, :, prefix:], v)
        x = x + self.o_proj((ext_y + tok_y).reshape(rows, steps, self.d_rank))
        x = x + self.fc2(F.gelu(self.fc1(self.norm_mlp(x))))
        return x, k, v

    def forward_indexed(
        self,
        x: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        node_keys: torch.Tensor,
        node_values: torch.Tensor,
        parent_ancestors: torch.Tensor,
        row_batch_indices: torch.Tensor,
        position_ids: torch.Tensor,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, steps, _ = x.shape
        if steps != 1:
            raise RuntimeError("indexed weaver step requires a single token step.")
        h = self.norm_attn(x)
        qkv = self.qkv_proj(h).view(
            rows, steps, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.squeeze(1).contiguous()
        k = k.squeeze(1).contiguous()
        v = v.squeeze(1).contiguous()
        y = _weaver_indexed_attention(
            q,
            k,
            v,
            external_keys,
            external_values,
            external_mask,
            node_keys,
            node_values,
            parent_ancestors,
            row_batch_indices,
            position_ids,
            layer_index,
        )
        x = x + self.o_proj(y.reshape(rows, steps, self.d_rank))
        x = x + self.fc2(F.gelu(self.fc1(self.norm_mlp(x))))
        return x, k, v

    def forward_chain(
        self,
        x: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        chain_keys: torch.Tensor,
        chain_values: torch.Tensor,
        position_ids: torch.Tensor,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, steps, _ = x.shape
        if steps != 1:
            raise RuntimeError("chain weaver step requires a single token step.")
        h = self.norm_attn(x)
        qkv = self.qkv_proj(h).view(
            rows, steps, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.squeeze(1).contiguous()
        k = k.squeeze(1).contiguous()
        v = v.squeeze(1).contiguous()
        y = _weaver_chain_attention(
            q,
            k,
            v,
            external_keys,
            external_values,
            external_mask,
            chain_keys,
            chain_values,
            position_ids,
            layer_index,
        )
        x = x + self.o_proj(y.reshape(rows, steps, self.d_rank))
        x = x + self.fc2(F.gelu(self.fc1(self.norm_mlp(x))))
        return x, k, v


class Weaver(nn.Module):
    ENCODER_GLOBAL_PROMPT = 3
    SCORE_SIMPLE = 4

    def __init__(
        self,
        *,
        d_model: int,
        d_embed: int,
        d_rank: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        K: int,
        candidate_pool_size: int,
        encoder_mode: int = ENCODER_GLOBAL_PROMPT,
        score_head: int = SCORE_SIMPLE,
    ):
        super().__init__()
        if int(encoder_mode) != self.ENCODER_GLOBAL_PROMPT:
            raise ValueError(
                "DFlash+Weaver MVP supports encoder_mode=global_prompt only."
            )
        if int(score_head) != self.SCORE_SIMPLE:
            raise ValueError(
                "DFlash+Weaver MVP supports score_head=simple_score only."
            )
        self.d_model = int(d_model)
        self.d_embed = int(d_embed)
        self.d_rank = int(d_rank)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)
        self.K = int(K)
        self.candidate_pool_size = int(candidate_pool_size)
        self.output_norm = WeaverRMSNorm(d_model)
        self.embed_norm = WeaverRMSNorm(d_embed)
        self.token_in = nn.Linear(d_embed, d_rank)
        self.proposal_in = nn.Linear(d_model, d_rank)
        self.blocks = nn.ModuleList(
            [WeaverBlock(d_rank, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.out_norm = WeaverRMSNorm(d_rank)
        self.lm_head_query_in = nn.Linear(d_rank, d_model, bias=False)
        self.pos_emb = nn.Parameter(torch.zeros(K, d_rank))

    @staticmethod
    def _migrate_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        migrated = dict(state_dict)
        q_suffix = "q_proj.weight"
        prefixes = [
            key[: -len(q_suffix)] for key in migrated.keys() if key.endswith(q_suffix)
        ]
        for prefix in prefixes:
            q_key = f"{prefix}q_proj.weight"
            k_key = f"{prefix}k_proj.weight"
            v_key = f"{prefix}v_proj.weight"
            qkv_key = f"{prefix}qkv_proj.weight"
            if qkv_key not in migrated:
                migrated[qkv_key] = torch.cat(
                    [migrated[q_key], migrated[k_key], migrated[v_key]], dim=0
                )
            migrated.pop(q_key, None)
            migrated.pop(k_key, None)
            migrated.pop(v_key, None)
        return migrated

    @classmethod
    def load(
        cls, path: str, *, device: torch.device, dtype: torch.dtype
    ) -> "Weaver":
        payload = torch.load(path, map_location=device)
        if (
            not isinstance(payload, dict)
            or "config" not in payload
            or "state_dict" not in payload
        ):
            raise ValueError(
                "Weaver checkpoint must be a torch file "
                "containing {'config': ..., 'state_dict': ...}. "
                "JAX/Equinox conversion is intentionally a separate final step."
            )
        model = cls(**payload["config"]).to(device=device, dtype=dtype)
        model.load_state_dict(
            cls._migrate_state_dict(payload["state_dict"]), strict=True
        )
        model.eval()
        return model

    def _token_project(
        self, token_ids: torch.Tensor, token_embed: torch.Tensor
    ) -> torch.Tensor:
        token_ids = token_ids.clamp(min=0, max=token_embed.shape[0] - 1)
        return torch.index_select(token_embed, 0, token_ids.reshape(-1)).view(
            *token_ids.shape, token_embed.shape[-1]
        )

    def _prompt_tokens(
        self,
        output_norm_features: torch.Tensor,
        proposal_features: torch.Tensor,
    ) -> torch.Tensor:
        rows, steps, _ = proposal_features.shape
        first_output = self.output_norm(output_norm_features[:, :1].float()).to(
            dtype=proposal_features.dtype
        )
        output_token = self.proposal_in(first_output).reshape(rows, 1, self.d_rank)
        proposal = self.output_norm(proposal_features.float()).to(
            dtype=proposal_features.dtype
        )
        proposal_tokens = self.proposal_in(proposal)
        proposal_tokens = (
            proposal_tokens + self.pos_emb[:steps].to(dtype=proposal_tokens.dtype)[None]
        )
        return torch.cat([output_token, proposal_tokens], dim=1)

    def prompt_external_kv(
        self,
        output_norm_features: torch.Tensor,
        proposal_features: torch.Tensor,
        steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._prompt_tokens(output_norm_features, proposal_features)
        rows, prefix, _ = x.shape
        attention_mask = torch.ones(
            (rows, prefix, prefix), dtype=torch.bool, device=x.device
        ).tril()
        empty_keys = torch.empty(
            (rows, prefix, 0, self.num_heads, self.d_rank // self.num_heads),
            dtype=x.dtype,
            device=x.device,
        )
        empty_mask = torch.empty((rows, prefix, 0), dtype=torch.bool, device=x.device)
        key_layers = []
        value_layers = []
        for block in self.blocks:
            x, layer_keys, layer_values = block(
                x,
                attention_mask,
                empty_keys,
                empty_keys,
                empty_mask,
            )
            key_layers.append(layer_keys)
            value_layers.append(layer_values)
        keys = torch.stack(key_layers)
        values = torch.stack(value_layers)
        return (
            keys,
            values,
            torch.ones((rows, prefix), dtype=torch.bool, device=x.device),
        )

    def step_indexed(
        self,
        *,
        token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        position_ids: torch.Tensor,
        node_keys: torch.Tensor,
        node_values: torch.Tensor,
        parent_ancestors: torch.Tensor,
        row_batch_indices: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = parent_ancestors.shape[1]
        x = self._token_project(token_ids[:, None], token_embed)
        position_ids = position_ids.clamp(min=0, max=depth - 1)
        pos_emb_ids = position_ids.clamp(max=self.K - 1)
        pos_emb = torch.index_select(self.pos_emb, 0, pos_emb_ids.reshape(-1)).view(
            pos_emb_ids.shape[0], self.d_rank
        )
        x = x + pos_emb[:, None].to(dtype=x.dtype)
        current_key_layers = []
        current_value_layers = []
        for layer_index, block in enumerate(self.blocks):
            x, layer_keys, layer_values = block.forward_indexed(
                x,
                external_keys[layer_index],
                external_values[layer_index],
                external_mask,
                node_keys,
                node_values,
                parent_ancestors,
                row_batch_indices,
                position_ids,
                layer_index,
            )
            current_key_layers.append(layer_keys)
            current_value_layers.append(layer_values)
        query = self.out_norm(x).to(dtype=candidate_weights.dtype).squeeze(1)
        residual = (
            torch.matmul(candidate_weights, query[:, :, None]).squeeze(-1).float()
        )
        logits = candidate_scores.float() + residual
        logits = logits.masked_fill(candidate_ids < 0, -torch.inf)
        return logits, torch.stack(current_key_layers), torch.stack(current_value_layers)

    def step_chain(
        self,
        *,
        token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        position_ids: torch.Tensor,
        chain_keys: torch.Tensor,
        chain_values: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = chain_keys.shape[1]
        x = self._token_project(token_ids[:, None], token_embed)
        position_ids = position_ids.clamp(min=0, max=depth - 1)
        pos_emb_ids = position_ids.clamp(max=self.K - 1)
        pos_emb = torch.index_select(self.pos_emb, 0, pos_emb_ids.reshape(-1)).view(
            pos_emb_ids.shape[0], self.d_rank
        )
        x = x + pos_emb[:, None].to(dtype=x.dtype)
        current_key_layers = []
        current_value_layers = []
        for layer_index, block in enumerate(self.blocks):
            x, layer_keys, layer_values = block.forward_chain(
                x,
                external_keys[layer_index],
                external_values[layer_index],
                external_mask,
                chain_keys,
                chain_values,
                position_ids,
                layer_index,
            )
            current_key_layers.append(layer_keys)
            current_value_layers.append(layer_values)
        query = self.out_norm(x).to(dtype=candidate_weights.dtype).squeeze(1)
        residual = (
            torch.matmul(candidate_weights, query[:, :, None]).squeeze(-1).float()
        )
        logits = candidate_scores.float() + residual
        logits = logits.masked_fill(candidate_ids < 0, -torch.inf)
        return logits, torch.stack(current_key_layers), torch.stack(current_value_layers)


class WeaverTree(msgspec.Struct):
    draft_tokens: torch.Tensor
    parent_indices: torch.Tensor
    depths: torch.Tensor
    node_mask: torch.Tensor
    draft_logprobs: torch.Tensor


class WeaverTreeCudaGraph(msgspec.Struct):
    graph: torch.cuda.CUDAGraph
    root_ids: torch.Tensor
    output_norm: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_weights: torch.Tensor
    candidate_scores: torch.Tensor
    proposal_features: torch.Tensor
    tree: WeaverTree


class WeaverChainGraphSamplingInfo(msgspec.Struct):
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    is_all_greedy: bool
    need_top_p_sampling: bool
    need_top_k_sampling: bool


class WeaverChainCudaGraph(msgspec.Struct):
    graph: torch.cuda.CUDAGraph
    root_ids: torch.Tensor
    output_norm: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_weights: torch.Tensor
    candidate_scores: torch.Tensor
    proposal_features: torch.Tensor
    draft_tokens: torch.Tensor
    proposal_uniforms: Optional[torch.Tensor] = None
    proposal_tokens: Optional[torch.Tensor] = None
    proposal_probs: Optional[torch.Tensor] = None
    sampling_info: Optional[WeaverChainGraphSamplingInfo] = None


class WeaverChain(msgspec.Struct):
    draft_tokens: torch.Tensor
    proposal_tokens: Optional[torch.Tensor] = None
    proposal_probs: Optional[torch.Tensor] = None


def build_tree_metadata(
    *,
    draft_tokens: torch.Tensor,
    parent_indices: torch.Tensor,
    depths: torch.Tensor,
    node_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    max_depth: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, num_nodes = draft_tokens.shape
    device = draft_tokens.device
    if device.type != "cuda":
        raise RuntimeError("tree metadata construction requires CUDA.")
    node_mask = node_mask.to(dtype=torch.bool, device=device)
    parent_indices = parent_indices.to(dtype=torch.long, device=device)

    retrieve_index = torch.arange(
        bs * num_nodes, dtype=torch.int64, device=device
    ).view(bs, num_nodes)
    retrieve_next_token = torch.empty((bs, num_nodes), dtype=torch.int64, device=device)
    retrieve_next_sibling = torch.empty_like(retrieve_next_token)
    positions = (seq_lens[:, None].to(torch.int64) + depths.to(torch.int64)).reshape(-1)

    prefix_lens = seq_lens.to(device=device, dtype=torch.int64)
    mask_sizes = num_nodes * (prefix_lens + num_nodes)
    mask_offsets = torch.empty((bs,), dtype=torch.int64, device=device)
    mask_offsets[0] = 0
    mask_offsets[1:] = torch.cumsum(mask_sizes[:-1], dim=0)

    prefix_lens_cpu = seq_lens_cpu.to(dtype=torch.int64)
    total_mask_size = int((num_nodes * (prefix_lens_cpu + num_nodes)).sum().item())
    custom_mask = torch.empty(total_mask_size, dtype=torch.bool, device=device)
    custom_mask.fill_(True)

    chain_steps = num_nodes if max_depth is None else min(num_nodes, int(max_depth) + 1)
    block_n = triton.next_power_of_2(int(num_nodes))
    _tree_metadata_parent_chain_kernel[(bs, num_nodes)](
        parent_indices,
        node_mask,
        prefix_lens,
        mask_offsets,
        custom_mask,
        retrieve_next_token,
        retrieve_next_sibling,
        NUM_NODES=int(num_nodes),
        CHAIN_STEPS=int(chain_steps),
        BLOCK_N=int(block_n),
    )
    return (
        custom_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    )


def _traversal_verify_target_probs(
    *,
    candidates: torch.Tensor,
    parent_indices: torch.Tensor,
    depths: torch.Tensor,
    node_mask: torch.Tensor,
    draft_logprobs: torch.Tensor,
    target_probs: torch.Tensor,
    uniform_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not candidates.is_cuda:
        raise RuntimeError("DFLASH_TFM traversal verification requires CUDA.")
    if candidates.dim() != 2:
        raise RuntimeError(
            f"traversal candidates must be rank-2, got {candidates.dim()}."
        )
    bs, num_nodes = candidates.shape
    if target_probs.shape[:2] != (bs, num_nodes):
        raise RuntimeError(
            "target_probs shape must start with candidates.shape, "
            f"got target_probs={tuple(target_probs.shape)}, candidates={tuple(candidates.shape)}."
        )
    target_probs = target_probs.contiguous()
    parent_indices = parent_indices.to(device=candidates.device, dtype=torch.int64)
    depths = depths.to(device=candidates.device, dtype=torch.int64)
    node_mask = node_mask.to(device=candidates.device, dtype=torch.bool)
    draft_logprobs = draft_logprobs.to(device=candidates.device, dtype=torch.float32)
    uniform_samples = uniform_samples.to(device=candidates.device, dtype=torch.float32)
    if uniform_samples.shape != (bs, num_nodes):
        raise RuntimeError(
            "uniform_samples shape mismatch for traversal verification: "
            f"expected {(bs, num_nodes)}, got {tuple(uniform_samples.shape)}."
        )

    predict = torch.empty((bs * num_nodes,), dtype=torch.int32, device=candidates.device)
    accept_index = torch.empty(
        (bs, num_nodes), dtype=torch.int32, device=candidates.device
    )
    num_correct = torch.empty((bs,), dtype=torch.int32, device=candidates.device)
    accept_leaf = torch.empty((bs,), dtype=torch.int64, device=candidates.device)
    block_n = triton.next_power_of_2(int(num_nodes))
    _weaver_traversal_verify_kernel[(int(bs),)](
        candidates.to(torch.int64),
        parent_indices,
        depths,
        node_mask,
        draft_logprobs,
        target_probs,
        uniform_samples,
        predict,
        accept_index,
        num_correct,
        accept_leaf,
        NUM_NODES=int(num_nodes),
        VOCAB_SIZE=int(target_probs.shape[-1]),
        BLOCK_N=int(block_n),
        num_warps=8,
    )
    row_ids = torch.arange(bs, dtype=torch.long, device=candidates.device)
    bonus = torch.multinomial(target_probs[row_ids, accept_leaf], 1).squeeze(1)
    predict[row_ids * num_nodes + accept_leaf] = bonus.to(torch.int32)
    return predict, accept_index, num_correct, accept_leaf


class DFlashTfmVerifyInput(DFlashVerifyInput):
    def __init__(
        self,
        *,
        draft_token: torch.Tensor,
        positions: torch.Tensor,
        draft_token_num: int,
        custom_mask: torch.Tensor,
        mask_seq_lens_cpu: Optional[torch.Tensor] = None,
        retrieve_index: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        depths: torch.Tensor,
        parent_indices: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        draft_logprobs: Optional[torch.Tensor] = None,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL,
    ):
        super().__init__(
            draft_token=draft_token,
            positions=positions,
            draft_token_num=int(draft_token_num),
            topk=2,
            custom_mask=custom_mask,
            capture_hidden_mode=capture_hidden_mode,
        )
        self.retrieve_index = retrieve_index
        self.retrieve_next_token = retrieve_next_token
        self.retrieve_next_sibling = retrieve_next_sibling
        self.mask_seq_lens_cpu = mask_seq_lens_cpu
        self.mask_seq_lens_sum = (
            int(mask_seq_lens_cpu.sum().item())
            if mask_seq_lens_cpu is not None
            else None
        )
        self.depths = depths
        self.parent_indices = parent_indices
        self.node_mask = node_mask
        self.draft_logprobs = draft_logprobs
        # Tree-local slot of each request's last accepted node; populated by
        # verify() and consumed by the post-verify Mamba/GDN state commit.
        self.accept_leaf_slots: Optional[torch.Tensor] = None

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ) -> tuple[ForwardBatch, bool]:
        if not build_custom_mask or self.custom_mask is None:
            raise RuntimeError(
                "DFLASH_TFM requires TreeAttention custom_mask support; "
                "disabling or omitting the tree mask would change verification semantics."
            )
        batch.input_ids = self.draft_token
        batch.spec_info = self
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = self.capture_hidden_mode
        if not batch.forward_mode.is_idle():
            end_offset = batch.seq_lens + int(self.draft_token_num)
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=batch.req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=end_offset,
                batch_size=batch.batch_size(),
                draft_token_num=int(self.draft_token_num),
                device=batch.device,
            )

        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)
        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
                verify_forward_batch
            )
        )
        if can_run_cuda_graph:
            target_worker.model_runner.decode_cuda_graph_runner.load_batch(
                verify_forward_batch
            )
        elif not batch.forward_mode.is_idle():
            target_worker.model_runner.attn_backend.init_forward_metadata(
                verify_forward_batch
            )
        return verify_forward_batch, can_run_cuda_graph

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
        kv_start_idx: Optional[torch.Tensor] = None,
    ):
        # Weaver tree masks are laid out against the logical committed prefix
        # length. The spec-v2 page allocator may pass a larger host-side
        # planning/reserved sum for buffer sizing; using that value here would
        # make FlashInfer pad the tree mask as if the reserved KV tail belonged
        # to the attention prefix.
        if self.mask_seq_lens_sum is not None:
            paged_kernel_lens_sum = self.mask_seq_lens_sum
        return super().generate_attn_arg_prefill(
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            req_to_token,
            kv_start_idx,
        )

    def _verify_from_target_predict(self, target_predict: torch.Tensor, bs: int):
        candidates = self.draft_token.view(bs, self.draft_token_num)
        predict = torch.full(
            (bs * self.draft_token_num,),
            -1,
            dtype=torch.int32,
            device=candidates.device,
        )
        accept_index = torch.full(
            (bs, self.draft_token_num), -1, dtype=torch.int32, device=candidates.device
        )
        num_correct = torch.empty((bs,), dtype=torch.int32, device=candidates.device)
        if not (is_cuda() or is_musa()):
            for b in range(bs):
                last = int(self.retrieve_index[b, 0].item())
                accept_index[b, 0] = last
                num_correct_drafts = 0
                cur = 0
                for _ in range(1, self.draft_token_num):
                    cur = int(self.retrieve_next_token[b, cur].item())
                    while cur != -1:
                        draft_index = int(self.retrieve_index[b, cur].item())
                        draft_token = int(candidates[b, cur].item())
                        target_token = int(target_predict.view(-1)[last].item())
                        if draft_token == target_token:
                            predict[last] = target_token
                            num_correct_drafts += 1
                            accept_index[b, num_correct_drafts] = draft_index
                            last = draft_index
                            break
                        cur = int(self.retrieve_next_sibling[b, cur].item())
                    if cur == -1:
                        break
                num_correct[b] = num_correct_drafts
                predict[last] = int(target_predict.view(-1)[last].item())
            return predict, accept_index, num_correct
        from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func

        verify_tree_greedy_func(
            predicts=predict,
            accept_index=accept_index,
            accept_token_num=num_correct,
            candidates=candidates,
            retrieve_index=self.retrieve_index,
            retrieve_next_token=self.retrieve_next_token,
            retrieve_next_sibling=self.retrieve_next_sibling,
            target_predict=target_predict,
        )
        return predict, accept_index, num_correct

    def _greedy_verify(self, logits_output: LogitsProcessorOutput, bs: int):
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
            bs, self.draft_token_num
        )
        return self._verify_from_target_predict(target_predict, bs)

    def _sampling_verify(
        self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput, sampling_info
    ):
        bs = batch.batch_size()
        candidates = self.draft_token.view(bs, self.draft_token_num)
        if (
            self.parent_indices is None
            or self.node_mask is None
            or self.draft_logprobs is None
        ):
            raise RuntimeError(
                "DFLASH_TFM traversal verification requires tree parents, "
                "node mask, and draft log-probabilities."
            )
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, self.draft_token_num, dim=0
        )
        target_probs = F.softmax(
            logits_output.next_token_logits / expanded_temperature, dim=-1
        )
        if getattr(sampling_info, "need_top_k_sampling", True):
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )
        if sampling_info.need_top_p_sampling:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
        target_probs = target_probs.view(bs, self.draft_token_num, -1)
        predict, accept_index, num_correct, _ = _traversal_verify_target_probs(
            candidates=candidates.to(torch.int64),
            parent_indices=self.parent_indices,
            depths=self.depths,
            node_mask=self.node_mask,
            draft_logprobs=self.draft_logprobs,
            target_probs=target_probs,
            uniform_samples=torch.rand_like(candidates, dtype=torch.float32),
        )
        return predict, accept_index, num_correct

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
        hidden_size: int,
        token_to_kv_pool_allocator=None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
    ]:
        """Verify the Weaver tree and return spec-v2 style commit data.

        This method intentionally does not append to req.output_ids or update
        request-level speculative counters. The spec-v2 result processor owns
        output mutation from ``next_token_ids`` and ``accept_lens``. We still
        assign the accepted tree slots into the committed prefix and advance
        the batch-local sequence lengths so the draft KV materialization below
        can use the accepted target slots. Unaccepted slots stay in the DFlashV2
        over-allocation window and may be reused by the next decode step.
        """
        bs = batch.batch_size()
        sampling_info = batch.sampling_info
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits_output.next_token_logits,
            sampling_info=sampling_info,
            draft_token_num=self.draft_token_num,
        )
        if sampling_info is None or sampling_info.is_all_greedy:
            predict, accept_index, num_correct = self._greedy_verify(
                logits_output, bs
            )
        else:
            predict, accept_index, num_correct = self._sampling_verify(
                batch, logits_output, sampling_info
            )

        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        commit_lens_cpu: List[int] = []
        num_correct_cpu: List[int] = []
        out_tokens_cpu: List[List[int]] = []
        for row in accept_index_cpu:
            row_tokens: List[int] = []
            for idx in row:
                if idx == -1:
                    break
                row_tokens.append(int(predict_cpu[int(idx)]))
            if not row_tokens:
                raise RuntimeError(
                    "DFlash+Weaver verify produced an empty accept path."
                )
            commit_lens_cpu.append(len(row_tokens))
            num_correct_cpu.append(max(0, len(row_tokens) - 1))
            out_tokens_cpu.append(row_tokens)

        commit_lens = torch.tensor(
            commit_lens_cpu, dtype=torch.int32, device=batch.device
        )
        row_ids = torch.arange(bs, device=batch.device, dtype=torch.long)
        self.accept_leaf_slots = (
            accept_index[row_ids, commit_lens.to(torch.long) - 1].to(torch.long)
            - row_ids * self.draft_token_num
        )
        out_tokens = torch.zeros(
            (bs, self.draft_token_num), dtype=torch.int64, device=batch.device
        )
        for i, row_tokens in enumerate(out_tokens_cpu):
            out_tokens[i, : len(row_tokens)] = torch.tensor(
                row_tokens, dtype=torch.int64, device=batch.device
            )

        out_cache_loc = batch.out_cache_loc
        out_cache_loc_2d = out_cache_loc.view(bs, self.draft_token_num)
        if bs == 1:
            flat_accept = accept_index[0, : commit_lens_cpu[0]].to(torch.long)
        else:
            flat_accept = torch.cat(
                [
                    accept_index[i, :commit_len]
                    for i, commit_len in enumerate(commit_lens_cpu)
                ]
            ).to(torch.long)

        if page_size > 1:
            if token_to_kv_pool_allocator is None:
                raise RuntimeError(
                    "DFLASH_TFM page_size>1 commit requires target KV cache access."
                )
            dst_parts = []
            for i, commit_len in enumerate(commit_lens_cpu):
                if commit_len > 0:
                    dst_parts.append(out_cache_loc_2d[i, :commit_len])
                if commit_len < self.draft_token_num:
                    req_idx = batch.req_pool_indices[i].to(torch.long)
                    seq_len = int(batch.seq_lens_cpu[i].item())
                    batch.req_to_token_pool.req_to_token[
                        req_idx,
                        seq_len + commit_len : seq_len + self.draft_token_num,
                    ] = out_cache_loc_2d[i, commit_len : self.draft_token_num]
            compact_cache_loc = (
                torch.cat(dst_parts) if dst_parts else out_cache_loc.new_empty((0,))
            )
            accept_cache_loc = out_cache_loc[flat_accept]
            token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                compact_cache_loc, accept_cache_loc
            )
            batch.out_cache_loc = compact_cache_loc

        else:
            for i, row in enumerate(accept_index_cpu):
                accept_local = {
                    int(idx) - i * self.draft_token_num for idx in row if idx != -1
                }
                commit_len = commit_lens_cpu[i]
                if commit_len >= self.draft_token_num:
                    continue
                remaining_local = [
                    j
                    for j in range(self.draft_token_num)
                    if j not in accept_local
                ]
                req_idx = batch.req_pool_indices[i].to(torch.long)
                seq_len = int(batch.seq_lens_cpu[i].item())
                remaining_slots = out_cache_loc[
                    i * self.draft_token_num
                    + torch.tensor(
                        remaining_local, dtype=torch.long, device=batch.device
                    )
                ]
                batch.req_to_token_pool.req_to_token[
                    req_idx,
                    seq_len + commit_len : seq_len + self.draft_token_num,
                ] = remaining_slots
            batch.out_cache_loc = out_cache_loc[flat_accept]

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + commit_lens.to(batch.seq_lens.dtype),
            batch.out_cache_loc,
            bs,
        )
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        batch.seq_lens_sum += sum(commit_lens_cpu)

        split = split_dflash_tfm_hidden(
            logits_output.hidden_states, hidden_size
        )
        target_hidden = split.target_hidden[flat_accept]
        target_positions = self.positions[flat_accept]
        output_norm = split.output_norm[flat_accept]
        terminal_offsets = torch.cumsum(commit_lens.to(torch.long), dim=0) - 1
        next_output_norm = output_norm[terminal_offsets]
        logits_output.hidden_states = None
        return (
            out_tokens,
            commit_lens,
            target_hidden,
            target_positions,
            next_output_norm,
            num_correct_cpu,
        )


class DFlashTfmWorker(DFlashWorkerV2):
    def _maybe_build_draft_sampler(self):
        # Weaver selects a top-k candidate pool from the draft hidden states.
        # The inherited greedy sampler computes a second, unused LM-head pass.
        return None

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: List[int], batch_size: int = 0
    ) -> None:
        """Spec-v2 result processor hook; Weaver is not adaptive yet."""
        pass

    def __init__(self, *args, **kwargs):
        server_args = args[0] if args else kwargs["server_args"]
        target_verify_tokens = server_args.speculative_num_draft_tokens
        if target_verify_tokens is None:
            target_verify_tokens = (
                int(server_args.speculative_dflash_tfm_tree_budget or 128) + 1
            )
        dflash_block_size_value = (
            server_args.speculative_dflash_block_size or target_verify_tokens
        )
        if dflash_block_size_value is None:
            raise ValueError(
                "DFLASH_TFM requires a DFlash block size. "
                "Run the speculative arg hook or set --speculative-dflash-block-size."
            )
        dflash_block_size = int(dflash_block_size_value)
        server_args.speculative_num_draft_tokens = dflash_block_size
        try:
            super().__init__(*args, **kwargs)
        finally:
            server_args.speculative_num_draft_tokens = target_verify_tokens
        path = self.server_args.speculative_dflash_tfm_path
        if path is None:
            raise ValueError(
                "DFLASH_TFM requires --speculative-dflash-tfm-path."
            )
        dtype = getattr(
            self.target_worker.model_runner.model_config, "dtype", torch.bfloat16
        )
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16
        self.weaver = Weaver.load(
            path,
            device=self.device,
            dtype=dtype,
        )
        self.tree_budget = int(self.server_args.speculative_dflash_tfm_tree_budget or 128)
        requested_pool_size = int(
            self.server_args.speculative_dflash_tfm_candidate_pool_size
            or self.weaver.candidate_pool_size
        )
        if requested_pool_size <= 0:
            raise ValueError(
                "DFLASH_TFM candidate pool size must be positive, "
                f"got {requested_pool_size}."
            )
        self.candidate_pool_size = min(
            requested_pool_size, int(self.weaver.candidate_pool_size)
        )
        if get_tp_group().world_size != 1:
            raise NotImplementedError(
                "DFLASH_TFM MVP supports tensor_parallel_size=1 only."
            )
        self.hidden_size = int(self.target_worker.model_runner.model_config.hidden_size)
        self._weaver_residual_lm_head_cache: Optional[torch.Tensor] = None
        self._weaver_residual_lm_head_cache_key: Optional[tuple[object, ...]] = None
        self._weaver_token_embed_cache: Optional[torch.Tensor] = None
        self._weaver_token_embed_cache_key: Optional[tuple[object, ...]] = None
        self._weaver_tree_cuda_graphs: dict[
            tuple[object, ...], WeaverTreeCudaGraph
        ] = {}
        self._weaver_chain_cuda_graphs: dict[
            tuple[object, ...], WeaverChainCudaGraph
        ] = {}
        self.target_verify_tokens = int(
            self.server_args.speculative_num_draft_tokens or self.block_size
        )
        self.use_chain_verify = self.target_verify_tokens <= int(self.block_size)

    def init_attention_backends(self):
        if self.target_verify_tokens <= int(self.block_size):
            backend_name = _tree_attention_backend_name(
                self.target_worker.model_runner.attn_backend
            )
            if backend_name not in TREE_ATTENTION_BACKENDS:
                self.use_chain_verify = True
        if not self.use_chain_verify:
            require_tree_attention_support(self.target_worker.model_runner.attn_backend)
        super().init_attention_backends()

    def _target_embedding_and_lm_head(self):
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH_TFM requires vocab-parallel target lm_head."
            )
        if not hasattr(embed_module, "weight"):
            raise RuntimeError(
                "DFLASH_TFM requires target input embedding weight."
            )
        return embed_module, lm_head

    def _weaver_token_embed(self, embed_module) -> torch.Tensor:
        weight = embed_module.weight
        norm_weight = self.weaver.embed_norm.weight
        norm_bias = self.weaver.embed_norm.bias
        projection_weight = self.weaver.token_in.weight
        projection_bias = self.weaver.token_in.bias
        key = (
            weight.data_ptr(),
            norm_weight.data_ptr(),
            norm_bias.data_ptr(),
            projection_weight.data_ptr(),
            projection_bias.data_ptr(),
            tuple(weight.shape),
            weight.dtype,
            projection_weight.dtype,
            weight.device,
            projection_weight.device,
            getattr(weight, "_version", 0),
            getattr(norm_weight, "_version", 0),
            getattr(norm_bias, "_version", 0),
            getattr(projection_weight, "_version", 0),
            getattr(projection_bias, "_version", 0),
        )
        if self._weaver_token_embed_cache_key != key:
            with torch.inference_mode():
                normalized = self.weaver.embed_norm(weight.float()).to(
                    dtype=weight.dtype
                )
                self._weaver_token_embed_cache = self.weaver.token_in(
                    normalized
                ).contiguous()
            self._weaver_token_embed_cache_key = key
        assert self._weaver_token_embed_cache is not None
        return self._weaver_token_embed_cache

    def _weaver_residual_lm_head(self, lm_head) -> torch.Tensor:
        weight = lm_head.weight
        projection = self.weaver.lm_head_query_in.weight
        key = (
            weight.data_ptr(),
            projection.data_ptr(),
            tuple(weight.shape),
            tuple(projection.shape),
            weight.dtype,
            projection.dtype,
            weight.device,
            projection.device,
            getattr(weight, "_version", 0),
            getattr(projection, "_version", 0),
        )
        if self._weaver_residual_lm_head_cache_key != key:
            with torch.inference_mode():
                self._weaver_residual_lm_head_cache = torch.matmul(
                    weight.to(dtype=projection.dtype), projection
                ).contiguous()
            self._weaver_residual_lm_head_cache_key = key
        assert self._weaver_residual_lm_head_cache is not None
        return self._weaver_residual_lm_head_cache

    def _topk_from_lm_head(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shard = lm_head.shard_indices
        weight = lm_head.weight
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)
        hs = hidden_states.to(dtype=weight.dtype)
        if num_org > 0 and num_added == 0:
            logits = torch.matmul(hs, weight[:num_org].T).float()
            values, indices = torch.topk(logits, min(int(k), logits.shape[-1]), dim=-1)
            return values, indices.to(torch.long) + org_vocab_start

        logits_parts = []
        ids_parts = []
        if num_org > 0:
            logits_parts.append(torch.matmul(hs, weight[:num_org].T).float())
            ids_parts.append(
                torch.arange(
                    org_vocab_start,
                    org_vocab_start + num_org,
                    dtype=torch.long,
                    device=hs.device,
                )
            )
        if num_added > 0:
            added = weight[num_org_padded : num_org_padded + num_added]
            logits_parts.append(torch.matmul(hs, added.T).float())
            ids_parts.append(
                torch.arange(
                    added_vocab_start,
                    added_vocab_start + num_added,
                    dtype=torch.long,
                    device=hs.device,
                )
            )
        logits = torch.cat(logits_parts, dim=-1)
        ids = torch.cat(ids_parts, dim=0)
        _, indices = torch.topk(logits, min(int(k), logits.shape[-1]), dim=-1)
        values = torch.gather(logits, 1, indices)
        return values, ids[indices]

    def _weaver_indexed_step_compiled(
        self,
        *,
        token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_row_index: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        position_ids: torch.Tensor,
        node_keys: torch.Tensor,
        node_values: torch.Tensor,
        parent_ancestors: torch.Tensor,
        row_batch_indices: torch.Tensor,
        token_embed: torch.Tensor,
        gather_candidates: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            tuple(token_ids.shape),
            token_ids.dtype,
            tuple(candidate_ids.shape),
            candidate_ids.dtype,
            tuple(candidate_weights.shape),
            candidate_weights.dtype,
            tuple(candidate_scores.shape),
            candidate_scores.dtype,
            tuple(candidate_row_index.shape),
            candidate_row_index.dtype,
            gather_candidates,
            tuple(external_keys.shape),
            external_keys.dtype,
            tuple(external_values.shape),
            external_values.dtype,
            tuple(external_mask.shape),
            tuple(position_ids.shape),
            tuple(node_keys.shape),
            node_keys.dtype,
            tuple(node_values.shape),
            node_values.dtype,
            tuple(parent_ancestors.shape),
            tuple(row_batch_indices.shape),
            tuple(token_embed.shape),
            token_embed.dtype,
            token_embed.device,
            int(self.weaver.num_layers),
            int(self.weaver.num_heads),
            int(self.weaver.d_rank),
            int(self.weaver.K),
        )
        compiled_steps = getattr(self, "_weaver_compiled_indexed_step_fns", None)
        if compiled_steps is None:
            compiled_steps = {}
            self._weaver_compiled_indexed_step_fns = compiled_steps
        compiled_step = compiled_steps.get(key)
        if compiled_step is None:
            def step_fn(
                token_ids,
                candidate_ids,
                candidate_weights,
                candidate_scores,
                candidate_row_index,
                external_keys,
                external_values,
                external_mask,
                position_ids,
                node_keys,
                node_values,
                parent_ancestors,
                row_batch_indices,
                token_embed,
            ):
                if gather_candidates:
                    candidate_ids = candidate_ids[candidate_row_index]
                    candidate_weights = candidate_weights[candidate_row_index]
                    candidate_scores = candidate_scores[candidate_row_index]
                outputs = self.weaver.step_indexed(
                    token_ids=token_ids,
                    candidate_ids=candidate_ids,
                    candidate_weights=candidate_weights,
                    candidate_scores=candidate_scores,
                    external_keys=external_keys,
                    external_values=external_values,
                    external_mask=external_mask,
                    position_ids=position_ids,
                    node_keys=node_keys,
                    node_values=node_values,
                    parent_ancestors=parent_ancestors,
                    row_batch_indices=row_batch_indices,
                    token_embed=token_embed,
                )
                return (*outputs, candidate_ids)

            compiled_step = torch.compile(
                step_fn,
                fullgraph=True,
                dynamic=False,
                options={
                    "triton.cudagraphs": False,
                    "emulate_precision_casts": True,
                },
            )
            compiled_steps[key] = compiled_step
        return compiled_step(
            token_ids,
            candidate_ids,
            candidate_weights,
            candidate_scores,
            candidate_row_index,
            external_keys,
            external_values,
            external_mask,
            position_ids,
            node_keys,
            node_values,
            parent_ancestors,
            row_batch_indices,
            token_embed,
        )

    def _weaver_chain_step_compiled(
        self,
        *,
        token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        external_keys: torch.Tensor,
        external_values: torch.Tensor,
        external_mask: torch.Tensor,
        position_ids: torch.Tensor,
        chain_keys: torch.Tensor,
        chain_values: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            tuple(token_ids.shape),
            token_ids.dtype,
            tuple(candidate_ids.shape),
            candidate_ids.dtype,
            tuple(candidate_weights.shape),
            candidate_weights.dtype,
            tuple(candidate_scores.shape),
            candidate_scores.dtype,
            tuple(external_keys.shape),
            external_keys.dtype,
            tuple(external_values.shape),
            external_values.dtype,
            tuple(external_mask.shape),
            tuple(position_ids.shape),
            tuple(chain_keys.shape),
            chain_keys.dtype,
            tuple(chain_values.shape),
            chain_values.dtype,
            token_embed.data_ptr(),
            tuple(token_embed.shape),
            token_embed.dtype,
            token_embed.device,
            int(self.weaver.num_layers),
            int(self.weaver.num_heads),
            int(self.weaver.d_rank),
            int(self.weaver.K),
        )
        compiled_steps = getattr(self, "_weaver_compiled_chain_step_fns", None)
        if compiled_steps is None:
            compiled_steps = {}
            self._weaver_compiled_chain_step_fns = compiled_steps
        compiled_step = compiled_steps.get(key)
        if compiled_step is None:
            def step_fn(
                token_ids,
                candidate_ids,
                candidate_weights,
                candidate_scores,
                external_keys,
                external_values,
                external_mask,
                position_ids,
                chain_keys,
                chain_values,
                token_embed,
            ):
                return self.weaver.step_chain(
                    token_ids=token_ids,
                    candidate_ids=candidate_ids,
                    candidate_weights=candidate_weights,
                    candidate_scores=candidate_scores,
                    external_keys=external_keys,
                    external_values=external_values,
                    external_mask=external_mask,
                    position_ids=position_ids,
                    chain_keys=chain_keys,
                    chain_values=chain_values,
                    token_embed=token_embed,
                )

            compiled_step = torch.compile(
                step_fn,
                fullgraph=True,
                dynamic=False,
                options={
                    "triton.cudagraphs": False,
                    "emulate_precision_casts": True,
                },
            )
            compiled_steps[key] = compiled_step
        return compiled_step(
            token_ids,
            candidate_ids,
            candidate_weights,
            candidate_scores,
            external_keys,
            external_values,
            external_mask,
            position_ids,
            chain_keys,
            chain_values,
            token_embed,
        )

    def _build_tree_impl(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> WeaverTree:
        bs, depth, pool_size = candidate_ids.shape
        node_budget = int(self.tree_budget)
        num_nodes = node_budget + 1
        device = root_ids.device
        tokens = torch.zeros((bs, num_nodes), dtype=torch.long, device=device)
        parents = torch.full((bs, num_nodes), -1, dtype=torch.long, device=device)
        depths = torch.zeros((bs, num_nodes), dtype=torch.long, device=device)
        node_mask = torch.zeros((bs, num_nodes), dtype=torch.bool, device=device)
        draft_logprobs = torch.full(
            (bs, num_nodes), -torch.inf, dtype=torch.float32, device=device
        )
        tokens[:, 0] = root_ids
        node_mask[:, 0] = True
        draft_logprobs[:, 0] = 0.0
        if node_budget <= 0 or depth <= 0:
            return WeaverTree(tokens, parents, depths, node_mask, draft_logprobs)

        expand_width = min(WEAVER_TREE_EXPAND_WIDTH, int(pool_size))
        frontier_slots = (node_budget + 1) * expand_width
        batch_indices = torch.arange(bs, dtype=torch.long, device=device)
        num_layers = self.weaver.num_layers
        num_heads = self.weaver.num_heads
        head_dim = self.weaver.d_rank // self.weaver.num_heads
        batch_expand_width = min(
            weaver_tree_batch_expand_width(node_budget), node_budget
        )
        use_fused_frontier_materialize = weaver_fused_frontier_materialize_enabled()
        if use_fused_frontier_materialize and batch_expand_width > 32:
            raise RuntimeError(
                "Fused Weaver frontier materialization supports at most "
                f"32 selected nodes per expansion, got {batch_expand_width}."
            )
        external_keys, external_values, external_mask = (
            self.weaver.prompt_external_kv(output_norm[:, None], proposal_features)
        )
        candidate_ids_rows = candidate_ids.reshape(bs * depth, pool_size)
        candidate_weights_rows = candidate_weights.reshape(
            bs * depth, pool_size, candidate_weights.shape[-1]
        )
        candidate_scores_rows = candidate_scores.reshape(bs * depth, pool_size)
        node_keys = torch.empty(
            (bs, num_nodes, num_layers, num_heads, head_dim),
            dtype=proposal_features.dtype,
            device=device,
        )
        node_values = torch.empty_like(node_keys)
        slot_ancestors = torch.empty(
            (bs, num_nodes, depth), dtype=torch.long, device=device
        )

        frontier_tokens = torch.empty(
            (bs, frontier_slots), dtype=torch.long, device=device
        )
        frontier_parents = torch.empty_like(frontier_tokens)
        frontier_depths = torch.empty_like(frontier_tokens)
        frontier_scores = torch.empty(
            (bs, frontier_slots), dtype=torch.float32, device=device
        )
        frontier_logprobs = torch.empty_like(frontier_scores)
        frontier_active = torch.empty(
            (bs, frontier_slots), dtype=torch.bool, device=device
        )
        if not use_fused_frontier_materialize:
            node_keys.zero_()
            node_values.zero_()
            slot_ancestors.fill_(-1)
            slot_ancestors[:, 0, 0] = 0
            frontier_tokens.zero_()
            frontier_parents.zero_()
            frontier_depths.zero_()
            frontier_scores.fill_(-torch.inf)
            frontier_logprobs.fill_(-torch.inf)
            frontier_active.zero_()
        elif batch_expand_width > expand_width:
            padding = slice(expand_width, batch_expand_width)
            frontier_scores[:, padding].fill_(-torch.inf)
            frontier_active[:, padding].zero_()
        if use_fused_frontier_materialize:
            selected_tokens = torch.empty(
                (bs, batch_expand_width), dtype=torch.long, device=device
            )
            selected_depths = torch.empty_like(selected_tokens)
            selected_position_ids = torch.empty_like(selected_tokens)
            selected_candidate_rows = torch.empty_like(selected_tokens)
            selected_batch_indices = torch.empty_like(selected_tokens)
            selected_scores = torch.empty(
                (bs, batch_expand_width), dtype=torch.float32, device=device
            )
            selected_active = torch.empty(
                (bs, batch_expand_width), dtype=torch.bool, device=device
            )
            selected_parent_ancestors = torch.empty(
                (bs, batch_expand_width, depth),
                dtype=torch.long,
                device=device,
            )
        else:
            selected_tokens = None
            selected_depths = None
            selected_position_ids = None
            selected_candidate_rows = None
            selected_batch_indices = None
            selected_scores = None
            selected_active = None
            selected_parent_ancestors = None
        if device.type != "cuda" or expand_width != 8:
            raise RuntimeError(
                "Weaver tree construction requires Triton on CUDA with "
                "expand_width=8."
            )

        def write_candidate_frontier(
            logits: torch.Tensor,
            row_candidate_ids: torch.Tensor,
            prefix_score: torch.Tensor,
            node_depth: torch.Tensor,
            active: torch.Tensor,
            slot_start: int,
            width: int,
        ) -> None:
            block_pool = triton.next_power_of_2(int(pool_size))
            _weaver_candidate_frontier_kernel[(logits.shape[0],)](
                logits,
                row_candidate_ids,
                prefix_score,
                node_depth,
                active,
                frontier_tokens,
                frontier_parents,
                frontier_depths,
                frontier_scores,
                frontier_logprobs,
                frontier_active,
                int(slot_start),
                WIDTH=int(width),
                POOL_SIZE=int(pool_size),
                EXPAND_WIDTH=int(expand_width),
                DEPTH=int(depth),
                FRONTIER_SLOTS=int(frontier_slots),
                BLOCK_POOL=int(block_pool),
            )

        def write_current_slot_cache(
            current_keys: torch.Tensor,
            current_values: torch.Tensor,
            parent_ancestors: torch.Tensor,
            valid: torch.Tensor,
            node_depth: torch.Tensor,
            slot_start: int,
            width: int,
        ) -> None:
            total_kv = bs * width * num_layers * num_heads * head_dim
            total_ancestors = bs * width * depth
            block_size = 256
            grid = (triton.cdiv(max(total_kv, total_ancestors), block_size),)
            _weaver_current_cache_write_kernel[grid](
                current_keys,
                current_values,
                node_keys,
                node_values,
                parent_ancestors,
                slot_ancestors,
                valid,
                node_depth,
                int(slot_start),
                BS=int(bs),
                WIDTH=int(width),
                DEPTH=int(depth),
                NUM_NODES=int(num_nodes),
                NUM_LAYERS=int(num_layers),
                NUM_HEADS=int(num_heads),
                HEAD_DIM=int(head_dim),
                TOTAL_KV=int(total_kv),
                TOTAL_ANCESTORS=int(total_ancestors),
                BLOCK_SIZE=int(block_size),
            )

        def publish_frontier(
            logits: torch.Tensor,
            row_candidate_ids: torch.Tensor,
            current_keys: torch.Tensor,
            current_values: torch.Tensor,
            parent_ancestors: torch.Tensor,
            prefix_score: torch.Tensor,
            valid: torch.Tensor,
            node_depth: torch.Tensor,
            slot_start: int,
            width: int,
        ) -> None:
            total_kv = bs * width * num_layers * num_heads * head_dim
            total_ancestors = bs * width * depth
            block_size = 256
            grid = (
                max(
                    triton.cdiv(total_kv, block_size),
                    triton.cdiv(total_ancestors, block_size),
                    bs * width,
                ),
            )
            _weaver_publish_frontier_kernel[grid](
                current_keys,
                current_values,
                node_keys,
                node_values,
                parent_ancestors,
                slot_ancestors,
                logits,
                row_candidate_ids,
                prefix_score,
                valid,
                node_depth,
                frontier_tokens,
                frontier_parents,
                frontier_depths,
                frontier_scores,
                frontier_logprobs,
                frontier_active,
                int(slot_start),
                BS=int(bs),
                WIDTH=int(width),
                DEPTH=int(depth),
                NUM_NODES=int(num_nodes),
                NUM_LAYERS=int(num_layers),
                NUM_HEADS=int(num_heads),
                HEAD_DIM=int(head_dim),
                TOTAL_KV=int(total_kv),
                TOTAL_ANCESTORS=int(total_ancestors),
                POOL_SIZE=int(pool_size),
                EXPAND_WIDTH=int(expand_width),
                FRONTIER_SLOTS=int(frontier_slots),
                BLOCK_SIZE=int(block_size),
                BLOCK_POOL=int(triton.next_power_of_2(pool_size)),
            )

        def materialize_frontier(
            frontier_index: torch.Tensor, slot_start: int, width: int
        ) -> None:
            assert selected_tokens is not None
            assert selected_depths is not None
            assert selected_position_ids is not None
            assert selected_candidate_rows is not None
            assert selected_batch_indices is not None
            assert selected_scores is not None
            assert selected_active is not None
            assert selected_parent_ancestors is not None
            _weaver_materialize_frontier_kernel[(bs,)](
                frontier_index,
                frontier_tokens,
                frontier_parents,
                frontier_depths,
                frontier_scores,
                frontier_logprobs,
                frontier_active,
                slot_ancestors,
                tokens,
                parents,
                depths,
                node_mask,
                draft_logprobs,
                selected_tokens,
                selected_depths,
                selected_position_ids,
                selected_candidate_rows,
                selected_batch_indices,
                selected_scores,
                selected_active,
                selected_parent_ancestors,
                int(slot_start),
                NUM_NODES=int(num_nodes),
                DEPTH=int(depth),
                FRONTIER_SLOTS=int(frontier_slots),
                SELECT_WIDTH=int(width),
                SCRATCH_WIDTH=int(batch_expand_width),
                BLOCK_DEPTH=int(triton.next_power_of_2(depth)),
                WRITE_ANCESTORS=slot_start + width <= node_budget,
                num_warps=1,
            )

        def expand_node_indexed(
            token_ids: torch.Tensor,
            position_ids: torch.Tensor,
            candidate_row_index: torch.Tensor,
            parent_ancestors: torch.Tensor,
            row_batch_indices: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if use_fused_frontier_materialize:
                selected_candidate_ids = candidate_ids_rows
                selected_candidate_weights = candidate_weights_rows
                selected_candidate_scores = candidate_scores_rows
            else:
                selected_candidate_ids = candidate_ids_rows[candidate_row_index]
                selected_candidate_weights = candidate_weights_rows[
                    candidate_row_index
                ]
                selected_candidate_scores = candidate_scores_rows[
                    candidate_row_index
                ]
            step_kwargs = dict(
                token_ids=token_ids,
                candidate_ids=selected_candidate_ids,
                candidate_weights=selected_candidate_weights,
                candidate_scores=selected_candidate_scores,
                candidate_row_index=candidate_row_index,
                external_keys=external_keys,
                external_values=external_values,
                external_mask=external_mask,
                position_ids=position_ids,
                node_keys=node_keys,
                node_values=node_values,
                parent_ancestors=parent_ancestors.reshape(
                    bs * parent_ancestors.shape[1], depth
                ).contiguous(),
                row_batch_indices=row_batch_indices,
                token_embed=token_embed,
                gather_candidates=use_fused_frontier_materialize,
            )
            logits, current_keys, current_values, row_candidate_ids = (
                self._weaver_indexed_step_compiled(
                    **step_kwargs,
                )
            )
            return logits.float(), row_candidate_ids, current_keys, current_values

        root_parent_ancestors = torch.full(
            (bs, 1, depth), -1, dtype=torch.long, device=device
        )
        root_prefix_score = torch.zeros((bs,), dtype=torch.float32, device=device)
        root_depth = torch.zeros((bs,), dtype=torch.long, device=device)
        root_active = torch.ones((bs,), dtype=torch.bool, device=device)
        root_logits, root_candidate_ids, root_keys, root_values = expand_node_indexed(
            root_ids,
            root_depth,
            batch_indices * depth,
            root_parent_ancestors,
            batch_indices,
        )
        if use_fused_frontier_materialize:
            publish_frontier(
                root_logits,
                root_candidate_ids,
                root_keys,
                root_values,
                root_parent_ancestors,
                root_prefix_score,
                root_active[:, None],
                root_depth[:, None],
                0,
                1,
            )
        else:
            write_current_slot_cache(
                root_keys,
                root_values,
                root_parent_ancestors,
                root_active[:, None],
                root_depth[:, None],
                0,
                1,
            )
            write_candidate_frontier(
                root_logits,
                root_candidate_ids,
                root_prefix_score,
                root_depth,
                root_active,
                0,
                1,
            )

        def gather_parent_ancestors(parent: torch.Tensor) -> torch.Tensor:
            width = parent.shape[1]
            gather_index = parent.clamp(min=0, max=num_nodes - 1)[:, :, None].expand(
                bs, width, depth
            )
            return torch.gather(slot_ancestors, 1, gather_index)

        row_base = batch_indices[:, None]
        slot_start = 1
        while slot_start <= node_budget:
            width = min(batch_expand_width, node_budget - slot_start + 1)
            slot_stop = slot_start + width
            slot_slice = slice(slot_start, slot_stop)
            if use_fused_frontier_materialize:
                _, frontier_index = torch.topk(
                    frontier_scores[
                        :, : max(width, slot_start * expand_width)
                    ],
                    width,
                    dim=1,
                    sorted=False,
                )
                materialize_frontier(frontier_index, slot_start, width)
                valid = selected_active[:, :width]
                token = selected_tokens[:, :width]
                node_depth = selected_depths[:, :width]
                node_score = selected_scores[:, :width]
                parent_ancestors = selected_parent_ancestors[:, :width]
                position_ids = selected_position_ids[:, :width]
                candidate_row_index = selected_candidate_rows[:, :width]
                row_batch_indices = selected_batch_indices[:, :width]
            else:
                masked_priorities = frontier_scores.masked_fill(
                    ~frontier_active, -torch.inf
                )
                _, frontier_index = torch.topk(masked_priorities, width, dim=1)
                valid = frontier_active.gather(1, frontier_index)
                token = frontier_tokens.gather(1, frontier_index)
                parent = frontier_parents.gather(1, frontier_index)
                node_depth = frontier_depths.gather(1, frontier_index)
                node_score = frontier_scores.gather(1, frontier_index)
                node_logprob = frontier_logprobs.gather(1, frontier_index)

                tokens[:, slot_slice] = torch.where(
                    valid, token, torch.zeros_like(token)
                )
                parents[:, slot_slice] = torch.where(
                    valid, parent, torch.full_like(parent, -1)
                )
                depths[:, slot_slice] = torch.where(
                    valid, node_depth, torch.zeros_like(node_depth)
                )
                node_mask[:, slot_slice] = valid
                draft_logprobs[:, slot_slice] = torch.where(
                    valid, node_logprob, torch.full_like(node_logprob, -torch.inf)
                )
                frontier_active.scatter_(1, frontier_index, False)

            if slot_stop > node_budget:
                break

            if not use_fused_frontier_materialize:
                parent_ancestors = gather_parent_ancestors(parent)
            token_flat = token.reshape(bs * width)
            node_score_flat = node_score.reshape(bs * width)
            node_depth_flat = node_depth.reshape(bs * width)
            valid_flat = valid.reshape(bs * width)
            if use_fused_frontier_materialize:
                position_ids_flat = position_ids.reshape(bs * width)
                candidate_row_index_flat = candidate_row_index.reshape(bs * width)
                row_batch_indices_flat = row_batch_indices.reshape(bs * width)
            else:
                token_flat = torch.where(
                    valid_flat,
                    token_flat,
                    torch.zeros_like(token_flat),
                )
                position_ids_flat = node_depth_flat.clamp(max=depth - 1)
                row_batch_indices_flat = (
                    row_base.expand(bs, width).reshape(bs * width).contiguous()
                )
                candidate_row_index_flat = (
                    row_batch_indices_flat * depth + position_ids_flat
                )

            logits, row_candidate_ids, current_keys, current_values = expand_node_indexed(
                token_flat,
                position_ids_flat,
                candidate_row_index_flat,
                parent_ancestors,
                row_batch_indices_flat,
            )
            if use_fused_frontier_materialize:
                publish_frontier(
                    logits,
                    row_candidate_ids,
                    current_keys,
                    current_values,
                    parent_ancestors,
                    node_score_flat,
                    valid_flat,
                    node_depth_flat,
                    slot_start,
                    width,
                )
            else:
                write_current_slot_cache(
                    current_keys,
                    current_values,
                    parent_ancestors,
                    valid,
                    node_depth,
                    slot_start,
                    width,
                )
                write_candidate_frontier(
                    logits,
                    row_candidate_ids,
                    node_score_flat,
                    node_depth_flat,
                    valid_flat,
                    slot_start,
                    width,
                )
            slot_start = slot_stop
        return WeaverTree(tokens, parents, depths, node_mask, draft_logprobs)

    def _build_chain_impl(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        draft_token_num: int,
        sampling_info=None,
        proposal_uniforms: Optional[torch.Tensor] = None,
        draft_tokens_out: Optional[torch.Tensor] = None,
        proposal_tokens_out: Optional[torch.Tensor] = None,
        proposal_probs_out: Optional[torch.Tensor] = None,
    ) -> WeaverChain:
        bs, candidate_depth, pool_size = candidate_ids.shape
        draft_token_num = int(draft_token_num)
        chain_depth = draft_token_num - 1
        block_size = int(self.block_size)
        if candidate_depth + 1 != block_size:
            raise RuntimeError(
                "DFLASH_TFM chain requires candidate depth to match "
                f"block_size - 1, got depth={candidate_depth}, block_size={block_size}."
            )
        if draft_token_num < 1 or draft_token_num > block_size:
            raise RuntimeError(
                "DFLASH_TFM chain draft_token_num must be in [1, block_size], "
                f"got draft_token_num={draft_token_num}, block_size={block_size}."
            )
        device = root_ids.device
        batch_indices = torch.arange(bs, dtype=torch.long, device=device)
        if draft_tokens_out is None:
            draft_tokens = torch.empty(
                (bs, draft_token_num), dtype=torch.long, device=device
            )
        else:
            draft_tokens = draft_tokens_out
        draft_tokens[:, 0] = root_ids.to(torch.long)
        num_layers = self.weaver.num_layers
        num_heads = self.weaver.num_heads
        head_dim = self.weaver.d_rank // self.weaver.num_heads

        external_keys, external_values, external_mask = (
            self.weaver.prompt_external_kv(
                output_norm[:, None],
                proposal_features,
            )
        )
        candidate_ids_rows = candidate_ids.reshape(bs * candidate_depth, pool_size)
        candidate_weights_rows = candidate_weights.reshape(
            bs * candidate_depth, pool_size, candidate_weights.shape[-1]
        )
        candidate_scores_rows = candidate_scores.reshape(bs * candidate_depth, pool_size)
        chain_keys = torch.empty(
            (bs, chain_depth, num_layers, num_heads, head_dim),
            dtype=proposal_features.dtype,
            device=device,
        )
        chain_values = torch.empty_like(chain_keys)
        token = draft_tokens[:, 0]
        do_sample = sampling_info is not None and not sampling_info.is_all_greedy
        if do_sample:
            if proposal_tokens_out is None:
                proposal_tokens = torch.empty(
                    (bs, chain_depth, pool_size), dtype=torch.long, device=device
                )
            else:
                proposal_tokens = proposal_tokens_out
            if proposal_probs_out is None:
                proposal_probs = torch.empty(
                    (bs, chain_depth, pool_size), dtype=torch.float32, device=device
                )
            else:
                proposal_probs = proposal_probs_out
            proposal_tokens.fill_(-1)
            proposal_probs.zero_()
            if proposal_uniforms is None:
                proposal_uniforms = torch.rand(
                    (chain_depth, bs), dtype=torch.float32, device=device
                )
            elif proposal_uniforms.shape != (chain_depth, bs):
                raise ValueError(
                    "proposal_uniforms shape mismatch for DFLASH_TFM chain, "
                    f"got {tuple(proposal_uniforms.shape)}, expected {(chain_depth, bs)}."
                )
        else:
            proposal_tokens = None
            proposal_probs = None
            proposal_uniforms = None

        for step in range(chain_depth):
            row_index = batch_indices * candidate_depth + step
            token_position_ids = torch.full(
                (bs,), step, dtype=torch.long, device=device
            )
            logits, current_keys, current_values = self._weaver_chain_step_compiled(
                token_ids=token,
                candidate_ids=candidate_ids_rows[row_index],
                candidate_weights=candidate_weights_rows[row_index],
                candidate_scores=candidate_scores_rows[row_index],
                external_keys=external_keys,
                external_values=external_values,
                external_mask=external_mask,
                position_ids=token_position_ids,
                chain_keys=chain_keys,
                chain_values=chain_values,
                token_embed=token_embed,
            )
            chain_keys[:, step].copy_(current_keys.permute(1, 0, 2, 3))
            chain_values[:, step].copy_(current_values.permute(1, 0, 2, 3))
            if do_sample:
                token, step_proposal_tokens, step_proposal_probs = (
                    sample_dflash_proposal_from_logits(
                        logits=logits,
                        sampling_info=sampling_info,
                        steps_per_batch=1,
                        token_ids=candidate_ids_rows[row_index],
                        uniform_samples=proposal_uniforms[step],
                    )
                )
                support_width = step_proposal_tokens.shape[1]
                proposal_tokens[:, step, :support_width].copy_(step_proposal_tokens)
                proposal_probs[:, step, :support_width].copy_(step_proposal_probs)
            else:
                next_index = torch.argmax(logits, dim=-1)
                token = candidate_ids_rows[row_index].gather(
                    1, next_index[:, None]
                ).squeeze(1)
            draft_tokens[:, step + 1] = token

        return WeaverChain(draft_tokens, proposal_tokens, proposal_probs)

    def _capture_weaver_chain_cuda_graph(
        self,
        *,
        key: tuple[object, ...],
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        draft_token_num: int,
        sampling_info=None,
    ) -> WeaverChainCudaGraph:
        device = root_ids.device
        if device.type != "cuda":
            raise RuntimeError("_build_chain CUDA Graph requires a CUDA device.")
        do_sample = sampling_info is not None and not sampling_info.is_all_greedy
        bs, candidate_depth, pool_size = candidate_ids.shape
        chain_depth = int(draft_token_num) - 1
        root_ids_buffer = torch.empty_like(root_ids)
        output_norm_buffer = torch.empty_like(output_norm)
        candidate_ids_buffer = torch.empty_like(candidate_ids)
        candidate_weights_buffer = torch.empty_like(candidate_weights)
        candidate_scores_buffer = torch.empty_like(candidate_scores)
        proposal_features_buffer = torch.empty_like(proposal_features)
        draft_tokens_buffer = torch.empty(
            (bs, int(draft_token_num)), dtype=torch.long, device=device
        )
        proposal_tokens_buffer = (
            torch.empty((bs, chain_depth, pool_size), dtype=torch.long, device=device)
            if do_sample
            else None
        )
        proposal_probs_buffer = (
            torch.empty((bs, chain_depth, pool_size), dtype=torch.float32, device=device)
            if do_sample
            else None
        )
        graph_sampling_info = None
        if do_sample:
            graph_sampling_info = WeaverChainGraphSamplingInfo(
                temperatures=torch.empty_like(sampling_info.temperatures),
                top_ps=torch.empty_like(sampling_info.top_ps),
                top_ks=torch.empty_like(sampling_info.top_ks),
                is_all_greedy=False,
                need_top_p_sampling=bool(
                    getattr(sampling_info, "need_top_p_sampling", False)
                ),
                need_top_k_sampling=bool(
                    getattr(sampling_info, "need_top_k_sampling", True)
                ),
            )
        proposal_uniforms_buffer = (
            torch.empty((chain_depth, bs), dtype=torch.float32, device=device)
            if do_sample
            else None
        )
        root_ids_buffer.copy_(root_ids)
        output_norm_buffer.copy_(output_norm)
        candidate_ids_buffer.copy_(candidate_ids)
        candidate_weights_buffer.copy_(candidate_weights)
        candidate_scores_buffer.copy_(candidate_scores)
        proposal_features_buffer.copy_(proposal_features)
        if graph_sampling_info is not None:
            graph_sampling_info.temperatures.copy_(sampling_info.temperatures)
            graph_sampling_info.top_ps.copy_(sampling_info.top_ps)
            graph_sampling_info.top_ks.copy_(sampling_info.top_ks)
        if proposal_uniforms_buffer is not None:
            proposal_uniforms_buffer.uniform_()

        with torch.inference_mode():
            self._build_chain_impl(
                root_ids=root_ids_buffer,
                output_norm=output_norm_buffer,
                candidate_ids=candidate_ids_buffer,
                candidate_weights=candidate_weights_buffer,
                candidate_scores=candidate_scores_buffer,
                proposal_features=proposal_features_buffer,
                token_embed=token_embed,
                draft_token_num=draft_token_num,
                sampling_info=graph_sampling_info if do_sample else sampling_info,
                proposal_uniforms=proposal_uniforms_buffer,
                draft_tokens_out=draft_tokens_buffer,
                proposal_tokens_out=proposal_tokens_buffer,
                proposal_probs_out=proposal_probs_buffer,
            )
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain = self._build_chain_impl(
                    root_ids=root_ids_buffer,
                    output_norm=output_norm_buffer,
                    candidate_ids=candidate_ids_buffer,
                    candidate_weights=candidate_weights_buffer,
                    candidate_scores=candidate_scores_buffer,
                    proposal_features=proposal_features_buffer,
                    token_embed=token_embed,
                    draft_token_num=draft_token_num,
                    sampling_info=graph_sampling_info if do_sample else sampling_info,
                    proposal_uniforms=proposal_uniforms_buffer,
                    draft_tokens_out=draft_tokens_buffer,
                    proposal_tokens_out=proposal_tokens_buffer,
                    proposal_probs_out=proposal_probs_buffer,
                )
            torch.cuda.synchronize(device)

        graph_state = WeaverChainCudaGraph(
            graph=graph,
            root_ids=root_ids_buffer,
            output_norm=output_norm_buffer,
            candidate_ids=candidate_ids_buffer,
            candidate_weights=candidate_weights_buffer,
            candidate_scores=candidate_scores_buffer,
            proposal_features=proposal_features_buffer,
            draft_tokens=draft_tokens_buffer,
            proposal_uniforms=proposal_uniforms_buffer,
            proposal_tokens=proposal_tokens_buffer,
            proposal_probs=proposal_probs_buffer,
            sampling_info=graph_sampling_info,
        )
        chain_graphs = getattr(self, "_weaver_chain_cuda_graphs", None)
        if chain_graphs is None:
            chain_graphs = {}
            self._weaver_chain_cuda_graphs = chain_graphs
        chain_graphs[key] = graph_state
        return graph_state

    def _build_chain_with_cuda_graph(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        draft_token_num: int,
        sampling_info=None,
    ) -> WeaverChain:
        do_sample = sampling_info is not None and not sampling_info.is_all_greedy
        key = (
            int(self.block_size),
            int(draft_token_num),
            bool(do_sample),
            bool(getattr(sampling_info, "need_top_k_sampling", False)) if do_sample else False,
            bool(getattr(sampling_info, "need_top_p_sampling", False)) if do_sample else False,
            root_ids.device.index,
            tuple(root_ids.shape),
            root_ids.dtype,
            tuple(output_norm.shape),
            output_norm.dtype,
            tuple(candidate_ids.shape),
            candidate_ids.dtype,
            tuple(candidate_weights.shape),
            candidate_weights.dtype,
            tuple(candidate_scores.shape),
            candidate_scores.dtype,
            tuple(proposal_features.shape),
            proposal_features.dtype,
            token_embed.data_ptr(),
            tuple(token_embed.shape),
            token_embed.dtype,
            token_embed.device,
            int(self.weaver.num_layers),
            int(self.weaver.num_heads),
            int(self.weaver.d_rank),
            int(self.weaver.K),
        )
        chain_graphs = getattr(self, "_weaver_chain_cuda_graphs", None)
        graph_state = None if chain_graphs is None else chain_graphs.get(key)
        if graph_state is None:
            graph_state = self._capture_weaver_chain_cuda_graph(
                key=key,
                root_ids=root_ids,
                output_norm=output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
                draft_token_num=draft_token_num,
                sampling_info=sampling_info,
            )
        graph_state.root_ids.copy_(root_ids)
        graph_state.output_norm.copy_(output_norm)
        graph_state.candidate_ids.copy_(candidate_ids)
        graph_state.candidate_weights.copy_(candidate_weights)
        graph_state.candidate_scores.copy_(candidate_scores)
        graph_state.proposal_features.copy_(proposal_features)
        if graph_state.sampling_info is not None:
            graph_state.sampling_info.temperatures.copy_(sampling_info.temperatures)
            graph_state.sampling_info.top_ps.copy_(sampling_info.top_ps)
            graph_state.sampling_info.top_ks.copy_(sampling_info.top_ks)
        if graph_state.proposal_uniforms is not None:
            graph_state.proposal_uniforms.uniform_()
        graph_state.graph.replay()
        if graph_state.proposal_probs is not None:
            probs = graph_state.proposal_probs
            with torch.inference_mode():
                probs.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                probs.clamp_(min=0.0)
                probs.div_(probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-20))
        return WeaverChain(
            graph_state.draft_tokens,
            graph_state.proposal_tokens,
            graph_state.proposal_probs,
        )

    def _build_chain(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
        draft_token_num: int,
        sampling_info=None,
    ) -> WeaverChain:
        can_graph_sample = (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and not bool(getattr(sampling_info, "need_top_k_sampling", True))
            and not bool(getattr(sampling_info, "need_top_p_sampling", False))
        )
        if (
            root_ids.device.type == "cuda"
            and (
                sampling_info is None
                or sampling_info.is_all_greedy
                or can_graph_sample
            )
        ):
            return self._build_chain_with_cuda_graph(
                root_ids=root_ids,
                output_norm=output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
                draft_token_num=draft_token_num,
                sampling_info=sampling_info,
            )
        return self._build_chain_impl(
            root_ids=root_ids,
            output_norm=output_norm,
            candidate_ids=candidate_ids,
            candidate_weights=candidate_weights,
            candidate_scores=candidate_scores,
            proposal_features=proposal_features,
            token_embed=token_embed,
            draft_token_num=draft_token_num,
            sampling_info=sampling_info,
        )

    def _capture_weaver_tree_cuda_graph(
        self,
        *,
        key: tuple[object, ...],
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> WeaverTreeCudaGraph:
        device = root_ids.device
        if device.type != "cuda":
            raise RuntimeError("_build_tree CUDA Graph requires a CUDA device.")
        root_ids_buffer = torch.empty_like(root_ids)
        output_norm_buffer = torch.empty_like(output_norm)
        candidate_ids_buffer = torch.empty_like(candidate_ids)
        candidate_weights_buffer = torch.empty_like(candidate_weights)
        candidate_scores_buffer = torch.empty_like(candidate_scores)
        proposal_features_buffer = torch.empty_like(proposal_features)
        root_ids_buffer.copy_(root_ids)
        output_norm_buffer.copy_(output_norm)
        candidate_ids_buffer.copy_(candidate_ids)
        candidate_weights_buffer.copy_(candidate_weights)
        candidate_scores_buffer.copy_(candidate_scores)
        proposal_features_buffer.copy_(proposal_features)

        with torch.inference_mode():
            self._build_tree_impl(
                root_ids=root_ids_buffer,
                output_norm=output_norm_buffer,
                candidate_ids=candidate_ids_buffer,
                candidate_weights=candidate_weights_buffer,
                candidate_scores=candidate_scores_buffer,
                proposal_features=proposal_features_buffer,
                token_embed=token_embed,
            )
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                tree = self._build_tree_impl(
                    root_ids=root_ids_buffer,
                    output_norm=output_norm_buffer,
                    candidate_ids=candidate_ids_buffer,
                    candidate_weights=candidate_weights_buffer,
                    candidate_scores=candidate_scores_buffer,
                    proposal_features=proposal_features_buffer,
                    token_embed=token_embed,
                )
            torch.cuda.synchronize(device)

        graph_state = WeaverTreeCudaGraph(
            graph=graph,
            root_ids=root_ids_buffer,
            output_norm=output_norm_buffer,
            candidate_ids=candidate_ids_buffer,
            candidate_weights=candidate_weights_buffer,
            candidate_scores=candidate_scores_buffer,
            proposal_features=proposal_features_buffer,
            tree=tree,
        )
        tree_graphs = getattr(self, "_weaver_tree_cuda_graphs", None)
        if tree_graphs is None:
            tree_graphs = {}
            self._weaver_tree_cuda_graphs = tree_graphs
        tree_graphs[key] = graph_state
        return graph_state

    def _build_tree_with_cuda_graph(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> WeaverTree:
        key = (
            int(self.tree_budget),
            int(weaver_tree_batch_expand_width(self.tree_budget)),
            weaver_fused_frontier_materialize_enabled(),
            root_ids.device.index,
            tuple(root_ids.shape),
            root_ids.dtype,
            tuple(output_norm.shape),
            output_norm.dtype,
            tuple(candidate_ids.shape),
            candidate_ids.dtype,
            tuple(candidate_weights.shape),
            candidate_weights.dtype,
            tuple(candidate_scores.shape),
            candidate_scores.dtype,
            tuple(proposal_features.shape),
            proposal_features.dtype,
            token_embed.data_ptr(),
            tuple(token_embed.shape),
            token_embed.dtype,
            token_embed.device,
            int(self.weaver.num_layers),
            int(self.weaver.num_heads),
            int(self.weaver.d_rank),
            int(self.weaver.K),
        )
        tree_graphs = getattr(self, "_weaver_tree_cuda_graphs", None)
        graph_state = None if tree_graphs is None else tree_graphs.get(key)
        if graph_state is None:
            graph_state = self._capture_weaver_tree_cuda_graph(
                key=key,
                root_ids=root_ids,
                output_norm=output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
            )
        graph_state.root_ids.copy_(root_ids)
        graph_state.output_norm.copy_(output_norm)
        graph_state.candidate_ids.copy_(candidate_ids)
        graph_state.candidate_weights.copy_(candidate_weights)
        graph_state.candidate_scores.copy_(candidate_scores)
        graph_state.proposal_features.copy_(proposal_features)
        graph_state.graph.replay()
        return graph_state.tree

    def _build_tree(
        self,
        *,
        root_ids: torch.Tensor,
        output_norm: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_weights: torch.Tensor,
        candidate_scores: torch.Tensor,
        proposal_features: torch.Tensor,
        token_embed: torch.Tensor,
    ) -> WeaverTree:
        if root_ids.device.type == "cuda":
            return self._build_tree_with_cuda_graph(
                root_ids=root_ids,
                output_norm=output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
            )
        return self._build_tree_impl(
            root_ids=root_ids,
            output_norm=output_norm,
            candidate_ids=candidate_ids,
            candidate_weights=candidate_weights,
            candidate_scores=candidate_scores,
            proposal_features=proposal_features,
            token_embed=token_embed,
        )

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashTfmDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return
        if not isinstance(draft_input, DFlashTfmDraftInput):
            raise RuntimeError(
                "DFLASH_TFM decode requires DFlashTfmDraftInput state."
            )
        if batch.has_grammar:
            raise RuntimeError(
                "DFLASH_TFM does not support grammar constraints in the MVP."
            )
        bs = batch.batch_size()
        embed_module, lm_head = self._target_embedding_and_lm_head()
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_size = int(self.block_size)
        block_ids = self._draft_block_ids_buf[:bs]
        prefix_lens = batch.seq_lens
        positions_2d = self._draft_block_positions_buf[:bs]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=draft_input.bonus_tokens.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=batch.req_pool_indices.view(-1),
                    req_to_token=batch.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DFLASH_TFM Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )
                block_ids.fill_(int(self._mask_token_id))
                block_ids[:, 0].copy_(draft_input.bonus_tokens.to(torch.long))
                torch.add(
                    prefix_lens.unsqueeze(1),
                    self._block_pos_offsets,
                    out=positions_2d,
                )
                end_offset = prefix_lens + block_size
                verify_out_cache_loc = assign_extend_cache_locs_func(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=batch.req_to_token_pool.req_to_token,
                    start_offset=prefix_lens,
                    end_offset=end_offset,
                    batch_size=bs,
                    draft_token_num=block_size,
                    device=batch.device,
                )
                verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))
        else:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.bonus_tokens.to(torch.long))
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets,
                out=positions_2d,
            )
            end_offset = prefix_lens + block_size
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=batch.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=end_offset,
                batch_size=bs,
                draft_token_num=block_size,
                device=batch.device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        input_embeds_2d = embed_module(block_ids)
        input_embeds = input_embeds_2d.view(-1, input_embeds_2d.shape[-1])
        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)
        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]

        if self.use_compact_draft_cache:
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            seq_lens_cpu.copy_(
                draft_prefix_lens.to(device="cpu", dtype=torch.int32)
            )

            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=batch.req_to_token_pool.req_to_token,
                req_pool_indices=batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )

            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            draft_seq_lens = prefix_lens
            if draft_input.reserved_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            elif batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(batch.seq_lens_cpu)
                draft_seq_lens_sum = (
                    int(batch.seq_lens_sum)
                    if batch.seq_lens_sum is not None
                    else int(batch.seq_lens_cpu.sum())
                )
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH_TFM,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        with torch.inference_mode():
            draft_logits_output = self.draft_model_runner.forward(
                forward_batch
            ).logits_output
        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError(
                "DFLASH_TFM draft model returned no hidden states."
            )
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        depth = min(self.block_size - 1, self.weaver.K)
        proposal_features = draft_hidden[:, 1 : 1 + depth].contiguous()
        scores, ids = self._topk_from_lm_head(
            proposal_features.reshape(bs * depth, proposal_features.shape[-1]),
            lm_head,
            self.candidate_pool_size,
        )
        candidate_scores = scores.view(bs, depth, -1)
        candidate_ids = ids.view(bs, depth, -1)
        residual_lm_head = self._weaver_residual_lm_head(lm_head)
        candidate_weights = residual_lm_head[candidate_ids.clamp_min(0)]
        token_embed = self._weaver_token_embed(embed_module)
        if self.use_chain_verify:
            draft_token_num = min(int(self.target_verify_tokens), block_size)
            if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
                if not is_dflash_sampling_verify_available():
                    raise RuntimeError(
                        "DFLASH_TFM chain non-greedy proposal sampling "
                        "requires DFlash sampling verify."
                    )
            chain = self._build_chain(
                root_ids=draft_input.bonus_tokens.to(torch.long),
                output_norm=draft_input.output_norm,
                candidate_ids=candidate_ids,
                candidate_weights=candidate_weights,
                candidate_scores=candidate_scores,
                proposal_features=proposal_features,
                token_embed=token_embed,
                draft_token_num=draft_token_num,
                sampling_info=batch.sampling_info,
            )
            draft_tokens = chain.draft_tokens
            verify_positions = positions_2d[:, :draft_token_num].reshape(-1)
            verify_out_cache_loc = verify_out_cache_loc_2d[:, :draft_token_num].reshape(-1)
            verify_input = DFlashVerifyInput(
                draft_token=draft_tokens.reshape(-1),
                positions=verify_positions,
                draft_token_num=draft_token_num,
                custom_mask=None,
                proposal_tokens=chain.proposal_tokens,
                proposal_probs=chain.proposal_probs,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            batch.out_cache_loc = verify_out_cache_loc
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            batch.spec_info = verify_input
            batch.return_hidden_states = False
            return
        tree = self._build_tree(
            root_ids=draft_input.bonus_tokens.to(torch.long),
            output_norm=draft_input.output_norm,
            candidate_ids=candidate_ids,
            candidate_weights=candidate_weights,
            candidate_scores=candidate_scores,
            proposal_features=proposal_features,
            token_embed=token_embed,
        )
        mask_seq_lens_cpu = batch.seq_lens_cpu
        if mask_seq_lens_cpu is None:
            mask_seq_lens_cpu = batch.seq_lens.to("cpu", dtype=torch.int32)
        (
            custom_mask,
            verify_positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = build_tree_metadata(
            draft_tokens=tree.draft_tokens,
            parent_indices=tree.parent_indices,
            depths=tree.depths,
            node_mask=tree.node_mask,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=mask_seq_lens_cpu,
            max_depth=depth,
        )
        verify_input = DFlashTfmVerifyInput(
            draft_token=tree.draft_tokens.reshape(-1),
            positions=verify_positions,
            draft_token_num=tree.draft_tokens.shape[1],
            custom_mask=custom_mask,
            mask_seq_lens_cpu=mask_seq_lens_cpu,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            depths=tree.depths,
            parent_indices=tree.parent_indices,
            node_mask=tree.node_mask,
            draft_logprobs=tree.draft_logprobs,
        )
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "DFLASH_TFM does not support return_logprob yet."
            )
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_result = self.target_worker.forward_batch_generation(batch, **kwargs)
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            batch_result.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_result.new_seq_lens)
            split = split_dflash_tfm_hidden(
                logits_output.hidden_states, self.hidden_size
            )
            if batch.extend_lens is None or batch.prefix_lens is None:
                raise RuntimeError(
                    "DFLASH_TFM expected extend_lens / prefix_lens in extend mode."
                )
            if batch.out_cache_loc is None:
                raise RuntimeError(
                    "DFLASH_TFM prefill expected out_cache_loc, but got None."
                )
            device = next_token_ids.device

            def _to_int32_device_tensor(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device=device, dtype=torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(batch.extend_lens)
            prefix_lens = _to_int32_device_tensor(batch.prefix_lens)
            positions, _ = compute_position(
                self.model_runner.server_args.attention_backend,
                prefix_lens,
                extend_seq_lens,
                int(sum(batch.extend_lens)),
            )
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=split.target_hidden,
                cache_loc=batch.out_cache_loc,
                positions=positions,
            )
            logits_output.hidden_states = None

            draft_input = DFlashTfmDraftInput(
                bonus_tokens=next_token_ids.to(torch.int64),
                new_seq_lens=batch.seq_lens,
                output_norm=split.output_norm[
                    _last_extend_indices(batch.extend_lens, device)
                ],
                committed_seq_lens_cpu=(
                    batch.seq_lens_cpu.clone()
                    if batch.seq_lens_cpu is not None
                    else None
                ),
            )
            batch.spec_info = draft_input
            batch_result.next_draft_input = draft_input
            batch_result.speculative_num_draft_tokens = int(
                self.server_args.speculative_num_draft_tokens
            )
            batch_result.num_correct_drafts = 0
            return batch_result

        if batch.spec_info is None:
            batch.spec_info = DFlashTfmDraftInput.create_idle_input(
                self.device, int(self.weaver.d_model)
            )
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashTfmDraftInput):
            raise RuntimeError(
                "DFLASH_TFM decode requires DFlashTfmDraftInput state."
            )
        if batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_draft_input = DFlashTfmDraftInput.create_idle_input(
                self.device, int(self.weaver.d_model)
            )
            if on_publish is not None:
                on_publish(next_draft_input.new_seq_lens)
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=next_draft_input,
                can_run_cuda_graph=False,
                speculative_num_draft_tokens=int(
                    self.server_args.speculative_num_draft_tokens
                ),
                new_seq_lens=next_draft_input.new_seq_lens,
            )

        # `seq_lens` may have been produced on another stream in the spec-v2 path.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = batch.batch_size()
        self._prepare_for_speculative_decoding(batch, draft_input)
        assert batch.forward_mode.is_target_verify()
        verify_input = batch.spec_info
        if isinstance(verify_input, DFlashVerifyInput) and not isinstance(
            verify_input, DFlashTfmVerifyInput
        ):
            need_mamba_verify_commit = hasattr(
                self.target_worker.model_runner.attn_backend,
                "update_mamba_state_after_mtp_verify",
            )
            seq_lens_pre_verify = (
                batch.seq_lens.clone() if need_mamba_verify_commit else None
            )
            seq_lens_cpu_backup = batch.seq_lens_cpu
            seq_lens_sum_backup = batch.seq_lens_sum
            if draft_input.reserved_seq_lens_cpu is not None:
                batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
                batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            try:
                verify_forward_batch, can_run_cuda_graph = (
                    verify_input.prepare_for_verify(batch, self.target_worker)
                )
                batch_result = self.target_worker.forward_batch_generation(
                    batch=None,
                    forward_batch=verify_forward_batch,
                    is_verify=True,
                    skip_attn_backend_init=True,
                    **kwargs,
                )
            finally:
                batch.seq_lens_cpu = seq_lens_cpu_backup
                batch.seq_lens_sum = seq_lens_sum_backup

            logits_output = batch_result.logits_output
            sampling_info = batch.sampling_info
            draft_token_num = int(verify_input.draft_token_num)
            if sampling_info is not None:
                apply_dflash_verify_logits_adjustments(
                    next_token_logits=logits_output.next_token_logits,
                    sampling_info=sampling_info,
                    draft_token_num=draft_token_num,
                )
            candidates = verify_input.draft_token.view(bs, draft_token_num)
            new_seq_lens = None
            if sampling_info is not None and not sampling_info.is_all_greedy:
                accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                    candidates=candidates,
                    next_token_logits=logits_output.next_token_logits,
                    sampling_info=sampling_info,
                    proposal_tokens=verify_input.proposal_tokens,
                    proposal_probs=verify_input.proposal_probs,
                )
                commit_lens = accept_len.to(torch.int32) + 1
                out_tokens = torch.empty(
                    (bs, draft_token_num), dtype=torch.int64, device=batch.device
                )
                if draft_token_num > 1:
                    out_tokens[:, : draft_token_num - 1].copy_(candidates[:, 1:])
                out_tokens[:, draft_token_num - 1].fill_(0)
                out_tokens.scatter_(
                    1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                )
            else:
                target_predict = torch.argmax(
                    logits_output.next_token_logits, dim=-1
                ).view(bs, draft_token_num)
                accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
                commit_lens = accept_len.to(torch.int32) + 1
                out_tokens = torch.empty(
                    (bs, draft_token_num),
                    dtype=torch.int64,
                    device=batch.device,
                )
                if draft_token_num > 1:
                    out_tokens[:, : draft_token_num - 1].copy_(
                        candidates[:, 1:]
                    )
                out_tokens[:, draft_token_num - 1].fill_(0)
                out_tokens.scatter_(
                    1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                )
            if new_seq_lens is None:
                new_seq_lens = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)

            if need_mamba_verify_commit:
                assert seq_lens_pre_verify is not None
                self._update_target_mamba_state_after_verify(
                    batch=batch,
                    seq_lens_pre_verify=seq_lens_pre_verify,
                    commit_lens=commit_lens,
                )
            if on_publish is not None:
                on_publish(new_seq_lens)
            split = split_dflash_tfm_hidden(
                logits_output.hidden_states, self.hidden_size
            )
            cache_loc_2d = batch.out_cache_loc.view(bs, draft_token_num)
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=split.target_hidden.reshape(
                    -1, split.target_hidden.shape[-1]
                ),
                cache_loc=batch.out_cache_loc,
                positions=verify_input.positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )
            terminal = (
                torch.arange(bs, device=batch.device, dtype=torch.long)
                * draft_token_num
                + accept_len.to(torch.long)
            )
            next_output_norm = split.output_norm[terminal]
            logits_output.hidden_states = None

            committed_seq_lens_cpu = (
                new_seq_lens.to("cpu", dtype=batch.seq_lens_cpu.dtype)
                if batch.seq_lens_cpu is not None
                else None
            )
            next_draft_input = DFlashTfmDraftInput(
                bonus_tokens=bonus,
                new_seq_lens=new_seq_lens,
                output_norm=next_output_norm,
                committed_seq_lens_cpu=committed_seq_lens_cpu,
            )
            batch.spec_info = next_draft_input
            batch.forward_mode = ForwardMode.DECODE
            num_correct_cpu = [int(x) for x in accept_len.to("cpu").tolist()]
            num_correct_drafts = sum(num_correct_cpu)
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=out_tokens.reshape(-1),
                accept_lens=commit_lens,
                next_draft_input=next_draft_input,
                speculative_num_draft_tokens=draft_token_num,
                new_seq_lens=new_seq_lens,
                num_correct_drafts=num_correct_drafts,
                num_correct_drafts_per_req_cpu=num_correct_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
                extra_keep_alive_refs=[verify_forward_batch],
            )

        assert isinstance(verify_input, DFlashTfmVerifyInput)

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_verify(
            batch,
            self.target_worker,
            self.page_size,
        )
        batch_result = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
            **kwargs,
        )
        logits_output = batch_result.logits_output
        (
            out_tokens,
            commit_lens,
            next_target_hidden,
            next_target_positions,
            next_output_norm,
            num_correct_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
            hidden_size=self.hidden_size,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
        )
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
                accept_leaf_slots=verify_input.accept_leaf_slots,
            )
        new_bonus_tokens = out_tokens[
            torch.arange(bs, device=batch.device),
            commit_lens.to(torch.long) - 1,
        ]
        new_seq_lens = batch.seq_lens.clone()
        if on_publish is not None:
            on_publish(new_seq_lens)
        append_cache_loc_2d = None
        append_commit_lens = None
        if (
            self.page_size > 1
            and int(batch.out_cache_loc.numel())
            == bs * int(verify_input.draft_token_num)
        ):
            append_cache_loc_2d = batch.out_cache_loc.view(
                bs, int(verify_input.draft_token_num)
            )
            append_commit_lens = commit_lens
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=next_target_hidden,
            cache_loc=batch.out_cache_loc,
            positions=next_target_positions,
            cache_loc_2d=append_cache_loc_2d,
            commit_lens=append_commit_lens,
        )

        next_draft_input = DFlashTfmDraftInput(
            bonus_tokens=new_bonus_tokens,
            new_seq_lens=new_seq_lens,
            output_norm=next_output_norm,
            committed_seq_lens_cpu=(
                batch.seq_lens_cpu.clone() if batch.seq_lens_cpu is not None else None
            ),
        )
        batch.spec_info = next_draft_input
        batch.forward_mode = ForwardMode.DECODE
        num_correct_drafts = sum(num_correct_cpu)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(
                getattr(
                    verify_input,
                    "draft_token_num",
                    self.server_args.speculative_num_draft_tokens,
                )
            ),
            new_seq_lens=new_seq_lens,
            num_correct_drafts=num_correct_drafts,
            num_correct_drafts_per_req_cpu=num_correct_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            extra_keep_alive_refs=[verify_forward_batch],
        )
