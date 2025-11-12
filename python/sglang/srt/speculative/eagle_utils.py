import math
from enum import IntEnum
from typing import List, Optional

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if _is_cuda or _is_hip:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def build_tree_efficient_native(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    draft_token_num: int,
    tree_mask_mode: int,
    bs: int,
):
    # Generate batch and token index ranges
    bs_range = torch.arange(bs, device=tree_mask.device).view(-1, 1)
    draft_token_num_range = torch.arange(draft_token_num, device=tree_mask.device)

    # Optimized common case for performance.
    if draft_token_num == 2 and topk == 1 and tree_mask_mode == TreeMaskMode.FULL_MASK:
        positions = verified_seq_len.repeat_interleave(draft_token_num)
        positions = (positions.view(bs, -1) + draft_token_num_range).view(-1)

        retrive_index[:] = bs_range * draft_token_num + draft_token_num_range
        retrive_next_token[:, 0] = 1
        retrive_next_token[:, 1] = -1
        return (
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            tree_mask,
        )

    # Precompute sequence tree indices
    draft_token_num_range1 = torch.arange(draft_token_num - 1, device=tree_mask.device)
    cum_seq_len = torch.cumsum(verified_seq_len * draft_token_num, dim=0)
    cum_seq_len = torch.cat((torch.tensor([0], device=tree_mask.device), cum_seq_len))
    cum_seq_len = cum_seq_len[:-1]
    seq_tree_idx = (
        draft_token_num * draft_token_num * torch.arange(bs, device=tree_mask.device)
        + cum_seq_len
    )

    # Batch processing for tree mask
    if tree_mask_mode == TreeMaskMode.FULL_MASK:
        token_tree_base = (
            seq_tree_idx.view(-1, 1)
            + (verified_seq_len.view(-1, 1) + draft_token_num) * draft_token_num_range
        )
        token_tree_indices = token_tree_base + verified_seq_len.view(-1, 1) + 1
    else:
        token_tree_indices = (
            bs_range * draft_token_num**2 + draft_token_num_range * draft_token_num + 1
        )

    tree_mask[token_tree_indices.flatten() - 1] = True
    indices = token_tree_indices.unsqueeze(-1) + draft_token_num_range1.view(1, 1, -1)
    tree_mask[indices.view(-1)] = False

    positions = verified_seq_len.repeat_interleave(draft_token_num)
    parent_tb_indices = selected_index // topk
    retrive_index[:] = bs_range * draft_token_num + draft_token_num_range
    tree_mask[token_tree_indices.view(-1, 1) + draft_token_num_range1] = True

    for bid in range(bs):
        for tid in range(draft_token_num):
            position = 0
            if tid == 0:
                # Process root node
                for i in range(draft_token_num - 1, 0, -1):
                    parent_position = 0
                    parent_tb_idx = parent_tb_indices[bid][i - 1]
                    if parent_tb_idx > 0:
                        parent_token_idx = parent_list[bid][parent_tb_idx]
                        loop_num = draft_token_num - parent_position
                        for _ in range(loop_num):
                            if selected_index[bid][parent_position] == parent_token_idx:
                                parent_position += 1
                                break
                            parent_position += 1
                    if parent_position == draft_token_num:
                        continue

                    if retrive_next_token[bid][parent_position] != -1:
                        retrive_next_sibling[bid][i] = retrive_next_token[bid][
                            parent_position
                        ]
                    retrive_next_token[bid][parent_position] = i
            else:
                # Process no-root nodes
                cur_position = tid - 1
                while True:
                    position += 1
                    if cur_position >= draft_token_num:
                        tree_mask[token_tree_indices + cur_position] = True
                        parent_tb_idx = selected_index[bid][cur_position] // topk
                    else:
                        parent_tb_idx = parent_tb_indices[bid][cur_position]
                    if parent_tb_idx == 0:
                        break
                    token_idx = parent_list[bid][parent_tb_idx]
                    cur_position = 0
                    for _ in range(draft_token_num):
                        if selected_index[bid][cur_position] == token_idx:
                            break
                        cur_position += 1
                positions[bid * draft_token_num + tid] += position
    return positions, retrive_index, retrive_next_token, retrive_next_sibling, tree_mask


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_npu:
        (
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            tree_mask,
        ) = build_tree_efficient_native(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            num_verify_tokens,
            tree_mask_mode,
            bs,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_native(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    batch_size, num_draft_tokens = candidates.shape

    # Optimized common case for performance.
    if num_draft_tokens == 2 and accept_index.shape[1] == 2 and topk == 1:
        comparison_result = candidates[:, 1] == target_predict[:, 0]

        predicts = target_predict.flatten()

        accept_index = torch.arange(
            0, num_draft_tokens * batch_size, device=candidates.device, dtype=torch.long
        ).reshape(batch_size, num_draft_tokens)
        comparison_result = comparison_result.to(torch.int64)
        accept_index_mask = accept_index[:, 1] * comparison_result
        accept_index[:, 1] = accept_index_mask - (1 - comparison_result)

        accept_token_num = comparison_result.int()
        return predicts, accept_index, accept_token_num

    # BFS
    for bx in range(batch_size):
        cur_candidates = candidates[bx]
        cur_retrive_index = retrive_index[bx]
        cur_next_token = retrive_next_token[bx]
        cur_next_sibling = retrive_next_sibling[bx]
        cur_target = target_predict[bx]

        last_accepted_idx = cur_retrive_index[0]
        accept_index[bx, 0] = last_accepted_idx
        num_accepted = 0
        cur_node = 0

        for _ in range(1, num_draft_tokens):
            cur_node = cur_next_token[cur_node]
            found = False
            while cur_node != -1:
                draft_idx = cur_retrive_index[cur_node]
                draft_token = cur_candidates[cur_node]
                target_token = cur_target[last_accepted_idx - num_draft_tokens * bx]

                if draft_token == target_token:
                    predicts[last_accepted_idx] = target_token
                    num_accepted += 1
                    accept_index[bx, num_accepted] = draft_idx
                    last_accepted_idx = draft_idx
                    found = True
                    break
                else:
                    cur_node = cur_next_sibling[cur_node]
            if not found:
                break

        accept_token_num[bx] = num_accepted
        predicts[last_accepted_idx] = cur_target[
            last_accepted_idx - num_draft_tokens * bx
        ]
    return predicts, accept_index, accept_token_num


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        predicts, accept_index, accept_token_num = verify_tree_greedy_native(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
            topk=topk,
        )

    return predicts, accept_index, accept_token_num
