import triton
import triton.language as tl


@triton.jit
def sgl_build_tree_kernel_efficient_triton(
    parent_list_ptr,
    selected_index_ptr,
    verified_seq_len_ptr,
    seq_len_prefix_sum_ptr,
    tree_mask_ptr,
    positions_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
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

    if batch_idx >= batch_size:
        return

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

    retrive_index_offset = batch_idx * draft_token_num

    # Build retrieval index structure (reverse loop from draft_token_num-1 to 1)
    for i in range(draft_token_num - 1, 0, -1):
        current_token_idx = retrive_index_offset + i
        tl.store(
            retrive_index_ptr + batch_idx * draft_token_num + i,
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
                retrive_next_token_ptr + batch_idx * draft_token_num + parent_position
            )
            next_tok = tl.load(next_tok_addr)

            if next_tok == -1:
                tl.store(next_tok_addr, i)
            else:
                tl.store(next_tok_addr, i)
                tl.store(
                    retrive_next_sibling_ptr + batch_idx * draft_token_num + i,
                    next_tok,
                )

    tl.store(retrive_index_ptr + batch_idx * draft_token_num, retrive_index_offset)

    # Process all draft token indices for tree mask
    for draft_token_idx in range(draft_token_num):
        if tree_mask_mode == 0:  # FULL_MASK
            token_tree_idx = (
                seq_tree_idx
                + (seq_len + draft_token_num) * draft_token_idx
                + seq_len
                + 1
            )
        else:
            token_tree_idx = (
                draft_token_num * draft_token_num * batch_idx
                + draft_token_num * draft_token_idx
                + 1
            )

        tl.store(tree_mask_ptr + token_tree_idx - 1, 1)
        for i in range(draft_token_num - 1):
            tl.store(tree_mask_ptr + token_tree_idx + i, 0)

        if draft_token_idx > 0:
            # Build tree path for draft_token_idx > 0
            cur_position = draft_token_idx - 1
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

            tl.store(
                positions_ptr + batch_idx * draft_token_num + draft_token_idx,
                position + seq_len,
            )


@triton.jit
def verify_tree_greedy_kernel_triton(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
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

    if bx >= batch_size:
        return

    # Initialize
    last_accepted_retrive_idx = tl.load(retrive_index_ptr + bx * num_draft_tokens)
    tl.store(accept_index_ptr + bx * num_speculative_tokens, last_accepted_retrive_idx)
    # Cast to match dtype of loaded tensors to avoid type inconsistency
    num_accepted_tokens = tl.cast(0, last_accepted_retrive_idx.dtype)
    cur_index = tl.cast(0, last_accepted_retrive_idx.dtype)

    # Tree traversal loop
    should_continue = 1
    for j in range(1, num_speculative_tokens):
        if should_continue:  # Early exit guard
            cur_index = tl.load(
                retrive_next_token_ptr + bx * num_draft_tokens + cur_index
            )

            # Load target token once per level (before sibling search)
            # last_accepted_retrive_idx is constant during sibling traversal
            target_row = last_accepted_retrive_idx // num_draft_tokens
            target_col = last_accepted_retrive_idx % num_draft_tokens
            target_token_id = tl.load(
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
                    draft_index = tl.load(retrive_index_ptr + safe_index)
                    draft_token_id = tl.load(candidates_ptr + safe_index)

                    # Check for token match (only valid when is_valid is True)
                    token_match = is_valid & (draft_token_id == target_token_id)

                    # Accept token using predicated stores (only write if matched)
                    tl.store(
                        predicts_ptr + last_accepted_retrive_idx,
                        target_token_id,
                        mask=token_match,
                    )
                    next_num_accepted_tokens = num_accepted_tokens + 1
                    tl.store(
                        accept_index_ptr
                        + bx * num_speculative_tokens
                        + next_num_accepted_tokens,
                        draft_index,
                        mask=token_match,
                    )

                    num_accepted_tokens = num_accepted_tokens + token_match
                    last_accepted_retrive_idx = (
                        token_match * draft_index
                        + (~token_match) * last_accepted_retrive_idx
                    )
                    found_match = token_match * 1 + (~is_valid) * (-1)

                    # Masked load: only load next sibling when no match (hardware predication)
                    # When matched: returns cur_index (other); when not matched: loads sibling
                    cur_index = tl.load(
                        retrive_next_sibling_ptr + safe_index,
                        mask=~token_match
                        & is_valid,  # Only load when valid and NOT matched
                        other=cur_index,  # Keep cur_index when matched or invalid
                    )

            if found_match != 1:
                should_continue = 0

    # Store final results
    tl.store(accept_token_num_ptr + bx, num_accepted_tokens)

    target_row = last_accepted_retrive_idx // num_draft_tokens
    target_col = last_accepted_retrive_idx % num_draft_tokens
    final_target = tl.load(
        target_predict_ptr + target_row * num_draft_tokens + target_col
    )
    tl.store(predicts_ptr + last_accepted_retrive_idx, final_target)
