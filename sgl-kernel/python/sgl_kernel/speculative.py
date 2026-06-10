import torch


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    torch.ops.sgl_kernel.tree_speculative_sampling_target_only.default(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
    )


def verify_tree_greedy(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.verify_tree_greedy.default(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )


def build_tree_kernel_efficient(
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
    torch.ops.sgl_kernel.build_tree_kernel_efficient.default(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )


def reconstruct_indices_from_tree_mask(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:
    torch.ops.sgl_kernel.reconstruct_indices_from_tree_mask.default(
        tree_mask,
        verified_seq_len,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        batch_size,
        draft_token_num,
    )


def segment_packbits(
    x: torch.Tensor,
    input_indptr: torch.Tensor,
    output_indptr: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> None:
    torch.ops.sgl_kernel.segment_packbits.default(
        x,
        input_indptr,
        output_indptr,
        y,
        batch_size,
        torch.cuda.current_stream().cuda_stream,
    )


def verify_tree_greedy_cpu(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.verify_tree_greedy_cpu(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )


def build_tree_kernel_efficient_cpu(
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
    torch.ops.sgl_kernel.build_tree_kernel_efficient_cpu(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )


def create_extend_after_decode_spec_info_cpu(
    verified_id: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    positions: torch.Tensor,
    new_verified_id: torch.Tensor,
    bs_upper: int,
) -> None:
    torch.ops.sgl_kernel.create_extend_after_decode_spec_info_cpu(
        verified_id,
        seq_lens,
        accept_lens,
        positions,
        new_verified_id,
        bs_upper,
    )


def align_evict_mask_to_page_size_cpu(
    seq_lens: torch.Tensor,
    evict_mask: torch.Tensor,
    page_size: int,
    num_draft_tokens: int,
) -> None:
    torch.ops.sgl_kernel.align_evict_mask_to_page_size_cpu(
        seq_lens,
        evict_mask,
        page_size,
        num_draft_tokens,
    )


def get_target_cache_loc_cpu(
    tgt_cache_loc: torch.Tensor,
    to_free_slots: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    to_free_num_slots: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_verify_tokens: int,
) -> None:
    torch.ops.sgl_kernel.get_target_cache_loc_cpu(
        tgt_cache_loc,
        to_free_slots,
        num_correct_drafts.to(torch.int64),
        to_free_num_slots.to(torch.int64),
        out_cache_loc,
        num_verify_tokens,
    )


def assign_req_to_token_pool_cpu(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
) -> None:
    torch.ops.sgl_kernel.assign_req_to_token_pool_cpu(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        pool_len,
    )


def create_flashinfer_kv_indices_cpu(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_kernel_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_start_idx,
    kv_indices: torch.Tensor,
    req_to_token_stride: int,
) -> None:
    torch.ops.sgl_kernel.create_flashinfer_kv_indices_cpu(
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
        req_to_token_stride,
    )


def build_draft_decode_metadata_cpu(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    num_steps: int,
    pool_len: int,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.build_draft_decode_metadata_cpu(
        req_to_token,
        req_pool_indices,
        seq_lens,
        topk,
        num_steps,
        pool_len,
    )


def fill_bonus_tokens_cpu(
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    accept_stride: int,
) -> None:
    torch.ops.sgl_kernel.fill_bonus_tokens_cpu(
        accept_tokens,
        accept_lens,
        bonus_tokens,
        accept_stride,
    )


def fill_accept_out_cache_loc_cpu(
    accept_index: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_out_cache_loc: torch.Tensor,
    size: int,
) -> None:
    torch.ops.sgl_kernel.fill_accept_out_cache_loc_cpu(
        accept_index,
        out_cache_loc,
        accept_out_cache_loc,
        size,
    )


def assign_draft_cache_locs_contiguous_cpu(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    topk: int,
    num_steps: int,
) -> None:
    torch.ops.sgl_kernel.assign_draft_cache_locs_contiguous_cpu(
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        pool_len,
        topk,
        num_steps,
    )


def assign_extend_cache_locs_cpu(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
) -> None:
    torch.ops.sgl_kernel.assign_extend_cache_locs_cpu(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        pool_len,
    )


def rotate_input_ids_cpu(
    input_ids: torch.Tensor,
    extend_start_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    topk_index: torch.Tensor,
    select_index=None,
) -> None:
    torch.ops.sgl_kernel.rotate_input_ids_cpu(
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        topk_index,
        select_index,
    )


def assign_new_state_cpu(
    next_token_ids: torch.Tensor,
    old_input_ids: torch.Tensor,
    old_positions: torch.Tensor,
    old_out_cache_loc: torch.Tensor,
    old_extend_seq_lens: torch.Tensor,
    old_extend_start_loc: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    padding_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    num_seqs: int,
    step: int,
    hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    req_to_hidden_states_pool: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.assign_new_state_cpu(
        next_token_ids,
        old_input_ids,
        old_positions,
        old_out_cache_loc,
        old_extend_seq_lens,
        old_extend_start_loc,
        input_ids,
        positions,
        out_cache_loc,
        extend_seq_lens,
        extend_start_loc,
        seq_lens,
        padding_lens,
        req_pool_indices,
        req_to_token,
        num_seqs,
        step,
        hidden_states,
        old_hidden_states,
        req_to_hidden_states_pool,
    )
