from enum import IntEnum
from typing import Tuple
import os

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2

def _get_num_vectorcore() -> int:
    try:
        from sgl_kernel_npu.utils.triton_utils import get_device_properties

        _, num_vectorcore = get_device_properties()
        return int(num_vectorcore)
    except Exception:
        device = torch.npu.current_device()
        device_properties = triton.runtime.driver.active.utils.get_device_properties(
            device
        )
        num_vectorcore = int(device_properties.get("num_vectorcore", -1))
        if num_vectorcore <= 0:
            raise RuntimeError("Failed to detect Ascend vector core count.")
        return num_vectorcore

NUM_VECTOR_CORES = _get_num_vectorcore()

@triton.jit(do_not_specialize=["batch_size", "topk", "parent_stride"])
def _build_tree_efficient_kernel(
    parent_list_ptr,
    selected_index_ptr,
    verified_seq_len_ptr,
    tree_mask_ptr,
    positions_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
    batch_size,
    topk,
    parent_stride,
    TREE_MASK_MODE: tl.constexpr,
    DRAFT_TOKEN_NUM: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_DRAFT)
    offsets_i64 = offsets.to(tl.int64)
    row_mask = offsets < DRAFT_TOKEN_NUM
    selected_stride = DRAFT_TOKEN_NUM - 1

    for bs in tl.range(pid, batch_size, num_programs):
        bs_i64 = bs.to(tl.int64)
        batch_token_base = bs_i64 * DRAFT_TOKEN_NUM
        batch_parent_base = bs_i64 * parent_stride
        batch_selected_base = bs_i64 * selected_stride
        seq_len = tl.load(verified_seq_len_ptr + bs_i64)

        seq_tree_idx = batch_token_base * DRAFT_TOKEN_NUM
        for prev_bs in tl.range(0, bs, 1):
            prev_bs_i64 = prev_bs.to(tl.int64)
            prev_seq_len = tl.load(verified_seq_len_ptr + prev_bs_i64)
            seq_tree_idx += prev_seq_len * DRAFT_TOKEN_NUM

        tl.store(
            retrive_index_ptr + batch_token_base + offsets_i64,
            batch_token_base + offsets_i64,
            mask=row_mask,
        )
        tl.store(
            retrive_next_token_ptr + batch_token_base + offsets_i64,
            -1,
            mask=row_mask,
        )
        tl.store(
            retrive_next_sibling_ptr + batch_token_base + offsets_i64,
            -1,
            mask=row_mask,
        )
        tl.store(positions_ptr + batch_token_base, seq_len)

        for i in range(DRAFT_TOKEN_NUM - 1, 0, -1):
            current_selected_offset = batch_selected_base + (i - 1)
            parent_tb_idx = tl.load(selected_index_ptr + current_selected_offset) // topk
            parent_position = 0

            if parent_tb_idx > 0:
                parent_token_idx = tl.load(parent_list_ptr + batch_parent_base + parent_tb_idx)
                parent_position = DRAFT_TOKEN_NUM
                for candidate_pos in range(DRAFT_TOKEN_NUM - 1):
                    candidate_token_idx = tl.load(
                        selected_index_ptr + batch_selected_base + candidate_pos
                    )
                    if parent_position == DRAFT_TOKEN_NUM and candidate_token_idx == parent_token_idx:
                        parent_position = candidate_pos + 1

            if parent_position != DRAFT_TOKEN_NUM:
                existing_next = tl.load(
                    retrive_next_token_ptr + batch_token_base + parent_position
                )
                if existing_next == -1:
                    tl.store(
                        retrive_next_token_ptr + batch_token_base + parent_position, i
                    )
                else:
                    tl.store(
                        retrive_next_token_ptr + batch_token_base + parent_position, i
                    )
                    tl.store(
                        retrive_next_sibling_ptr + batch_token_base + i, existing_next
                    )

        for tid in range(DRAFT_TOKEN_NUM):
            if TREE_MASK_MODE == 0:
                token_tree_idx = seq_tree_idx + (seq_len + DRAFT_TOKEN_NUM) * tid + seq_len
            else:
                token_tree_idx = batch_token_base * DRAFT_TOKEN_NUM + batch_token_base

            tl.store(
                tree_mask_ptr + token_tree_idx + offsets_i64,
                offsets == 0,
                mask=row_mask,
            )

            if tid > 0:
                position = 0
                cur_position = tid - 1
                active = 1

                for _ in range(DRAFT_TOKEN_NUM):
                    if active == 1:
                        position += 1
                        tl.store(
                            tree_mask_ptr + token_tree_idx + 1 + cur_position,
                            True,
                        )
                        parent_tb_idx = (
                            tl.load(selected_index_ptr + batch_selected_base + cur_position)
                            // topk
                        )
                        if parent_tb_idx == 0:
                            active = 0
                        else:
                            token_idx = tl.load(
                                parent_list_ptr + batch_parent_base + parent_tb_idx
                            )
                            next_position = DRAFT_TOKEN_NUM - 1
                            for candidate_pos in range(DRAFT_TOKEN_NUM - 1):
                                candidate_token_idx = tl.load(
                                    selected_index_ptr + batch_selected_base + candidate_pos
                                )
                                if (
                                    next_position == DRAFT_TOKEN_NUM - 1
                                    and candidate_token_idx == token_idx
                                ):
                                    next_position = candidate_pos
                            cur_position = next_position

                tl.store(positions_ptr + batch_token_base + tid, seq_len + position)


def build_tree_kernel_efficient_triton(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
) -> None:
    batch_size = int(verified_seq_len.numel())
    parent_stride = topk * (depth - 1) + 1
    num_cores = NUM_VECTOR_CORES
    block_draft = triton.next_power_of_2(draft_token_num)

    _build_tree_efficient_kernel[(num_cores,)](
        parent_list_ptr=parent_list.reshape(-1),
        selected_index_ptr=selected_index.reshape(-1),
        verified_seq_len_ptr=verified_seq_len.reshape(-1),
        tree_mask_ptr=tree_mask.reshape(-1),
        positions_ptr=positions.reshape(-1),
        retrive_index_ptr=retrive_index.reshape(-1),
        retrive_next_token_ptr=retrive_next_token.reshape(-1),
        retrive_next_sibling_ptr=retrive_next_sibling.reshape(-1),
        batch_size=batch_size,
        topk=topk,
        parent_stride=parent_stride,
        TREE_MASK_MODE=int(tree_mask_mode),
        DRAFT_TOKEN_NUM=draft_token_num,
        BLOCK_DRAFT=block_draft,
    )


ASSIGN_TO_POOL = 0
RETRIEVE_FROM_POOL = 1
MAX_STEP = 6

@triton.jit(do_not_specialize=["batch_size", "pool_len"])
def _cache_location_assigns_kernel(
    req_pool_indices_ptr,
    token_pool_ptr,
    start_offset_ptr,
    end_offset_ptr,
    out_cache_loc_ptr,
    batch_size,
    pool_len,
    ASSIGN_MODE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BS_UPPER: tl.constexpr,
    MAX_STEP_CONST: tl.constexpr,
):
    pid = tl.program_id(0)
    for row_idx in tl.range(pid, batch_size, NUM_CORES):
        req_idx = tl.load(req_pool_indices_ptr + row_idx)
        kv_start = tl.load(start_offset_ptr + row_idx)
        kv_end = tl.load(end_offset_ptr + row_idx)
        step = kv_end - kv_start

        prefix_idx = tl.arange(0, BS_UPPER)
        prefix_start = tl.load(start_offset_ptr + prefix_idx, mask=prefix_idx < row_idx, other=0)
        prefix_end = tl.load(end_offset_ptr + prefix_idx, mask=prefix_idx < row_idx, other=0)
        cache_idx_start = tl.sum(prefix_end - prefix_start, axis=0)

        token_ptr = token_pool_ptr + req_idx * pool_len + kv_start
        cache_ptr = out_cache_loc_ptr + cache_idx_start
        elem = tl.arange(0, MAX_STEP_CONST)
        mask = elem < step

        if ASSIGN_MODE == 0:
            data = tl.load(cache_ptr + elem, mask=mask, other=0)
            tl.store(token_ptr + elem, data, mask=mask)
        else:
            data = tl.load(token_ptr + elem, mask=mask, other=0)
            tl.store(cache_ptr + elem, data, mask=mask)


def _cache_location_assigns_impl(
    req_pool_indices: torch.Tensor,
    token_pool: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    assign_mode: int = ASSIGN_TO_POOL,
    num_cores: int | None = None,
) -> torch.Tensor:
    if assign_mode not in (ASSIGN_TO_POOL, RETRIEVE_FROM_POOL):
        raise ValueError("assign_mode must be 0 or 1.")
    batch_size = int(req_pool_indices.shape[0])
    if batch_size == 0:
        return token_pool if assign_mode == ASSIGN_TO_POOL else out_cache_loc
    if num_cores is None:
        num_cores = NUM_VECTOR_CORES
    num_cores = int(max(1, num_cores))
    bs_upper = int(triton.next_power_of_2(batch_size))
    _cache_location_assigns_kernel[(num_cores,)](
        req_pool_indices,
        token_pool,
        start_offset,
        end_offset,
        out_cache_loc,
        batch_size,
        int(token_pool.shape[1]),
        ASSIGN_MODE=assign_mode,
        NUM_CORES=num_cores,
        BS_UPPER=bs_upper,
        MAX_STEP_CONST=MAX_STEP,
    )
    return token_pool if assign_mode == ASSIGN_TO_POOL else out_cache_loc


def cache_loc_assign(
    req_pool_indices: torch.Tensor,
    token_pool: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> torch.Tensor:
    return _cache_location_assigns_impl(
        req_pool_indices,
        token_pool,
        start_offset,
        end_offset,
        out_cache_loc,
        ASSIGN_TO_POOL,
    )


def cache_loc_update(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc_copy: torch.Tensor,
) -> torch.Tensor:
    return _cache_location_assigns_impl(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc_copy,
        RETRIEVE_FROM_POOL,
    )

@triton.jit
def verify_tree_greedy_kernel(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
    accept_index_stride,
    num_draft_tokens: tl.constexpr,
):
    req_idx = tl.program_id(0)
    base = req_idx * num_draft_tokens

    last_accepted_idx = tl.load(retrive_index + base).to(tl.int32)
    tl.store(accept_index + req_idx * accept_index_stride, last_accepted_idx)

    num_accepted = 0
    rejected = False

    for i in range(1, num_draft_tokens):
        if not rejected:
            draft_token = tl.load(candidates + base + i).to(tl.int32)
            target_token = tl.load(target_predict + base + i - 1).to(tl.int32)

            if draft_token == target_token:
                draft_idx = tl.load(retrive_index + base + i).to(tl.int32)
                tl.store(predicts + last_accepted_idx, target_token)
                num_accepted += 1
                tl.store(
                    accept_index + req_idx * accept_index_stride + num_accepted,
                    draft_idx,
                )
                last_accepted_idx = draft_idx
            else:
                rejected = True

    final_pos = last_accepted_idx - base
    final_token = tl.load(target_predict + base + final_pos).to(tl.int32)

    tl.store(accept_token_num + req_idx, num_accepted)
    tl.store(predicts + last_accepted_idx, final_token)


def verify_tree_greedy_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    target_predict,
):
    bs, num_draft_tokens = candidates.shape

    verify_tree_greedy_kernel[(bs,)](
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
        accept_index_stride=accept_index.shape[1],
        num_draft_tokens=num_draft_tokens,
    )

@triton.jit
def alloc_extend_kernel_triton(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
    max_num_extend_tokens,
    BLOCK_SIZE: tl.constexpr = 2048,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid).to(tl.int64)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )

    num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
    blk_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(num_loop):
        offset_many_page = blk_offset + i * BLOCK_SIZE
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset_many_page // page_size,
            mask=offset_many_page < num_part2,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset_many_page,
            page_start * page_size + offset_many_page % page_size,
            mask=offset_many_page < num_part2,
        )

    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )