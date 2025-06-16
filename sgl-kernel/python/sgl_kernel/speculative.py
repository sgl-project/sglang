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


def copy_cuda_graph_replay_inputs(
    seq_lens_dst: torch.Tensor,
    seq_lens_src: torch.Tensor,
    out_cache_loc_dst: torch.Tensor,
    out_cache_loc_src: torch.Tensor,
    positions_dst: torch.Tensor,
    positions_src: torch.Tensor,
    req_pool_indices_dst: torch.Tensor,
    req_pool_indices_src: torch.Tensor,
    input_ids_dst: torch.Tensor = None,
    input_ids_src: torch.Tensor = None,
    extend_seq_lens_dst: torch.Tensor = None,
    extend_seq_lens_src: torch.Tensor = None,
    accept_length_dst: torch.Tensor = None,
    accept_length_src: torch.Tensor = None,
    hidden_states_dst: torch.Tensor = None,
    hidden_states_src: torch.Tensor = None,
    topk_p_dst: torch.Tensor = None,
    topk_p_src: torch.Tensor = None,
    topk_index_dst: torch.Tensor = None,
    topk_index_src: torch.Tensor = None,
    num_tokens: int = 0,
    raw_bs: int = 0,
    num_hidden_states: int = 0,
    hidden_size: int = 0,
    num_speculative_steps: int = 0,
    speculative_topk: int = 0,
) -> None:
    torch.ops.sgl_kernel.copy_cuda_graph_replay_inputs.default(
        seq_lens_dst,
        seq_lens_src,
        out_cache_loc_dst,
        out_cache_loc_src,
        positions_dst,
        positions_src,
        req_pool_indices_dst,
        req_pool_indices_src,
        input_ids_dst,
        input_ids_src,
        extend_seq_lens_dst,
        extend_seq_lens_src,
        accept_length_dst,
        accept_length_src,
        hidden_states_dst,
        hidden_states_src,
        topk_p_dst,
        topk_p_src,
        topk_index_dst,
        topk_index_src,
        num_tokens,
        raw_bs,
        num_hidden_states,
        hidden_size,
        num_speculative_steps,
        speculative_topk,
    )
