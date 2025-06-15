import torch
from sgl_kernel.utils import get_cuda_stream


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
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
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
        get_cuda_stream(),
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
        get_cuda_stream(),
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
    )


def segment_packbits(
    x: torch.Tensor,
    input_indptr: torch.Tensor,
    output_indptr: torch.Tensor,
    y: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.segment_packbits.default(
        x,
        input_indptr,
        output_indptr,
        y,
        torch.cuda.current_stream().cuda_stream,
    )


def copy_cuda_graph_replay_inputs(
    input_ids_dst: torch.Tensor,  # mutable
    seq_lens_dst: torch.Tensor,  # mutable
    extend_seq_lens_dst: torch.Tensor,  # mutable
    out_cache_loc_dst: torch.Tensor,  # mutable
    positions_dst: torch.Tensor,  # mutable
    req_pool_indices_dst: torch.Tensor,  # mutable
    accept_length_dst: torch.Tensor,  # mutable
    hidden_states_dst: torch.Tensor,  # mutable
    input_ids_src: torch.Tensor,
    seq_lens_src: torch.Tensor,
    extend_seq_lens_src: torch.Tensor,
    out_cache_loc_src: torch.Tensor,
    positions_src: torch.Tensor,
    req_pool_indices_src: torch.Tensor,
    accept_length_src: torch.Tensor,
    hidden_states_src: torch.Tensor,
    num_tokens: int,
    raw_bs: int,
    hidden_size: int,
) -> None:
    torch.ops.sgl_kernel.copy_cuda_graph_replay_inputs.default(
        input_ids_dst,
        seq_lens_dst,
        extend_seq_lens_dst,
        out_cache_loc_dst,
        positions_dst,
        req_pool_indices_dst,
        accept_length_dst,
        hidden_states_dst,
        input_ids_src,
        seq_lens_src,
        extend_seq_lens_src,
        out_cache_loc_src,
        positions_src,
        req_pool_indices_src,
        accept_length_src,
        hidden_states_src,
        num_tokens,
        raw_bs,
        hidden_size,
    )
