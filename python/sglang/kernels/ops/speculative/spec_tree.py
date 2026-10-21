# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from enum import IntEnum

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_musa, is_npu, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_musa = is_musa()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_cpu = is_cpu()

if _is_cuda or _is_hip:
    from sglang.kernels.ops.speculative.tree import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )
elif _is_musa:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )
elif _is_cpu:
    from sgl_kernel import (
        build_tree_kernel_efficient_cpu as sgl_build_tree_kernel_efficient_cpu,
    )
    from sgl_kernel import verify_tree_greedy_cpu as sgl_verify_tree_greedy_cpu


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


@triton.jit
def sgl_build_tree_kernel_efficient_triton(
    parent_list_ptr,
    selected_index_ptr,
    verified_seq_len_ptr,
    seq_len_prefix_sum_ptr,
    tree_mask_ptr,
    positions_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    topk: tl.constexpr,
    depth: tl.constexpr,
    draft_token_num: tl.constexpr,
    tree_mask_mode: tl.constexpr,
    batch_size: tl.constexpr,
    parent_list_stride: tl.constexpr,
    selected_index_stride: tl.constexpr,
):
    """
    Triton kernel for building EAGLE tree structure.
    Each program handles one batch item (batch_idx).
    """
    batch_idx = tl.program_id(0)

    # Calculate seq_tree_idx
    seq_len = tl.load(verified_seq_len_ptr + batch_idx)
    seq_len_prefix_sum = tl.load(seq_len_prefix_sum_ptr + batch_idx)

    # Cast initial value to match the dtype of loaded tensors to avoid type inconsistency
    seq_tree_idx = (
        tl.cast(draft_token_num * draft_token_num * batch_idx, seq_len.dtype)
        + seq_len_prefix_sum * draft_token_num
    )

    positions_offset = batch_idx * draft_token_num
    tl.store(positions_ptr + positions_offset, seq_len)

    retrieve_index_offset = batch_idx * draft_token_num

    # Build retrieval index structure (reverse loop from draft_token_num-1 to 1)
    for i in range(draft_token_num - 1, 0, -1):
        current_token_idx = retrieve_index_offset + i
        tl.store(
            retrieve_index_ptr + batch_idx * draft_token_num + i,
            current_token_idx,
        )

        parent_tb_idx = (
            tl.load(selected_index_ptr + batch_idx * selected_index_stride + (i - 1))
            // topk
        )
        parent_position = 0
        found = 0

        if parent_tb_idx == 0:
            found = 1
        else:
            parent_token_idx = tl.load(
                parent_list_ptr + batch_idx * parent_list_stride + parent_tb_idx
            )

            # Find parent position
            for pp in range(draft_token_num - 1):
                if found == 0:
                    sel_idx = tl.load(
                        selected_index_ptr + batch_idx * selected_index_stride + pp
                    )
                    if sel_idx == parent_token_idx:
                        parent_position = pp + 1
                        found = 1

        if found == 1:
            # Update next token links
            next_tok_addr = (
                retrieve_next_token_ptr + batch_idx * draft_token_num + parent_position
            )
            next_tok = tl.load(next_tok_addr)

            if next_tok == -1:
                tl.store(next_tok_addr, i)
            else:
                tl.store(next_tok_addr, i)
                tl.store(
                    retrieve_next_sibling_ptr + batch_idx * draft_token_num + i,
                    next_tok,
                )

    tl.store(retrieve_index_ptr + batch_idx * draft_token_num, retrieve_index_offset)

    # Process all draft token indices for tree mask
    for draft_tokenx in range(draft_token_num):
        if tree_mask_mode == 0:  # FULL_MASK
            token_tree_idx = (
                seq_tree_idx + (seq_len + draft_token_num) * draft_tokenx + seq_len + 1
            )
        else:
            token_tree_idx = (
                draft_token_num * draft_token_num * batch_idx
                + draft_token_num * draft_tokenx
                + 1
            )

        tl.store(tree_mask_ptr + token_tree_idx - 1, 1)
        for i in range(draft_token_num - 1):
            tl.store(tree_mask_ptr + token_tree_idx + i, 0)

        if draft_tokenx > 0:
            # Build tree path for draft_tokenx > 0
            cur_position = draft_tokenx - 1
            position = 0
            should_continue = 1

            for _ in range(depth):
                if should_continue:
                    position += 1
                    tl.store(tree_mask_ptr + token_tree_idx + cur_position, 1)

                    parent_tb_idx = (
                        tl.load(
                            selected_index_ptr
                            + batch_idx * selected_index_stride
                            + cur_position
                        )
                        // topk
                    )
                    if parent_tb_idx == 0:
                        should_continue = 0
                    else:
                        parent_token_idx = tl.load(
                            parent_list_ptr
                            + batch_idx * parent_list_stride
                            + parent_tb_idx
                        )

                        # Find cur_position for next iteration
                        found = 0
                        for cp in range(draft_token_num - 1):
                            if found == 0:
                                if (
                                    tl.load(
                                        selected_index_ptr
                                        + batch_idx * selected_index_stride
                                        + cp
                                    )
                                    == parent_token_idx
                                ):
                                    cur_position = cp
                                    found = 1
                        if found == 0:
                            should_continue = 0

            tl.store(
                positions_ptr + batch_idx * draft_token_num + draft_tokenx,
                position + seq_len,
            )


@triton.jit
def verify_tree_greedy_kernel_triton(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    target_predict_ptr,
    batch_size: tl.constexpr,
    num_speculative_tokens: tl.constexpr,
    num_draft_tokens: tl.constexpr,
):
    """
    Triton kernel for verifying EAGLE tree in greedy mode.
    Each program handles one batch item.
    """
    bx = tl.program_id(0)

    # Initialize
    last_accept_retrieve_idx = tl.load(retrieve_index_ptr + bx * num_draft_tokens)
    tl.store(accept_index_ptr + bx * num_speculative_tokens, last_accept_retrieve_idx)
    # Cast to match dtype of loaded tensors to avoid type inconsistency
    num_accept_tokens = tl.cast(0, last_accept_retrieve_idx.dtype)
    cur_index = tl.cast(0, last_accept_retrieve_idx.dtype)

    # Tree traversal loop
    should_continue = 1
    for j in range(1, num_speculative_tokens):
        if should_continue:  # Early exit guard
            cur_index = tl.load(
                retrieve_next_token_ptr + bx * num_draft_tokens + cur_index
            )

            # Load target token once per level (before sibling search)
            # last_accept_retrieve_idx is constant during sibling traversal
            target_row = last_accept_retrieve_idx // num_draft_tokens
            target_col = last_accept_retrieve_idx % num_draft_tokens
            target_token = tl.load(
                target_predict_ptr + target_row * num_draft_tokens + target_col
            )

            # Traverse siblings
            found_match = 0
            for _ in range(num_draft_tokens):  # Max iterations = num_draft_tokens
                if found_match == 0:  # Early exit guard
                    # Check if we've reached end of sibling list
                    is_valid = cur_index != -1

                    # Use masked loads with safe address (0 when invalid)
                    safe_cur_index = (
                        cur_index * is_valid
                    )  # 0 if invalid, cur_index if valid
                    safe_index = bx * num_draft_tokens + safe_cur_index

                    # Load draft token info (loads from index 0 when invalid, but we won't use it)
                    draft_index = tl.load(retrieve_index_ptr + safe_index)
                    draft_token = tl.load(candidates_ptr + safe_index)

                    # Check for token match (only valid when is_valid is True)
                    token_match = is_valid & (draft_token == target_token)

                    # Accept token using predicated stores (only write if matched)
                    tl.store(
                        predicts_ptr + last_accept_retrieve_idx,
                        target_token,
                        mask=token_match,
                    )
                    next_num_accept_tokens = num_accept_tokens + 1
                    tl.store(
                        accept_index_ptr
                        + bx * num_speculative_tokens
                        + next_num_accept_tokens,
                        draft_index,
                        mask=token_match,
                    )

                    num_accept_tokens = num_accept_tokens + token_match
                    last_accept_retrieve_idx = (
                        token_match * draft_index
                        + (~token_match) * last_accept_retrieve_idx
                    )
                    found_match = token_match * 1 + (~is_valid) * (-1)

                    # Masked load: only load next sibling when no match (hardware predication)
                    # When matched: returns cur_index (other); when not matched: loads sibling
                    cur_index = tl.load(
                        retrieve_next_sibling_ptr + safe_index,
                        mask=~token_match
                        & is_valid,  # Only load when valid and NOT matched
                        other=cur_index,  # Keep cur_index when matched or invalid
                    )

            if found_match != 1:
                should_continue = 0

    # Store final results
    tl.store(accept_token_num_ptr + bx, num_accept_tokens)

    target_row = last_accept_retrieve_idx // num_draft_tokens
    target_col = last_accept_retrieve_idx % num_draft_tokens
    final_target = tl.load(
        target_predict_ptr + target_row * num_draft_tokens + target_col
    )
    tl.store(predicts_ptr + last_accept_retrieve_idx, final_target)


def sgl_build_tree_kernel_triton(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
):
    """Triton-based implementation."""
    # TODO: Add support for QLEN_ONLY_BITPACKING mode
    if tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        raise NotImplementedError(
            "QLEN_ONLY_BITPACKING is not supported in Triton implementation"
        )

    batch_size = verified_seq_len.shape[0]
    seq_len_prefix_sum = torch.cumsum(verified_seq_len, dim=0) - verified_seq_len

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    sgl_build_tree_kernel_efficient_triton[grid](
        parent_list,
        selected_index,
        verified_seq_len,
        seq_len_prefix_sum,
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=int(tree_mask_mode),
        batch_size=batch_size,
        parent_list_stride=(
            parent_list.stride(0) if parent_list.dim() > 1 else parent_list.shape[0]
        ),
        selected_index_stride=selected_index.stride(0),
    )


def build_tree_kernel_efficient_func(
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    seq_lens: torch.Tensor,
    tree_mask: torch.Tensor,  # mutable
    positions: torch.Tensor,  # mutable
    retrieve_index: torch.Tensor,  # mutable
    retrieve_next_token: torch.Tensor,  # mutable
    retrieve_next_sibling: torch.Tensor,  # mutable
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode,
) -> None:
    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    elif _is_xpu:
        sgl_build_tree_kernel_triton(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    elif _is_cpu:
        sgl_build_tree_kernel_efficient_cpu(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )


def verify_tree_greedy_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """Triton-based implementation."""
    batch_size = candidates.shape[0]
    num_speculative_tokens = accept_index.shape[1]
    num_draft_tokens = candidates.shape[1]

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    verify_tree_greedy_kernel_triton[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        num_draft_tokens=num_draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip or _is_musa:
        if _is_cuda or _is_hip:
            from sglang.kernels.ops.speculative.tree import verify_tree_greedy
        else:
            from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_cpu:
        sgl_verify_tree_greedy_cpu(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match the CUDA op schema, so
            # the CPU/CUDA call sites stay grep-symmetric.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    elif _is_xpu:
        verify_tree_greedy_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num
