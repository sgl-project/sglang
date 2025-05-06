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
    r"""
    Perform speculative decoding using a tree-based acceptance mechanism, focusing on the target model only.

    This function implements a tree-structured speculative sampling strategy, where candidate tokens are proposed
    and then selectively accepted or rejected based on probabilities from both the draft and target model outputs.
    Only the target model's probabilities are ultimately considered for acceptance, which can enhance efficiency
    while maintaining accuracy, especially in large language model decoding tasks.

    Parameters
    ----------
    predicts : torch.Tensor
        Mutable tensor. Stores the accepted tokens after speculative sampling. Will be updated in-place.
    accept_index : torch.Tensor
        Mutable tensor. Stores the indices of accepted tokens in the tree. Updated in-place.
    accept_token_num : torch.Tensor
        Mutable tensor. Stores the number of accepted tokens. Updated in-place.
    candidates : torch.Tensor
        Proposed candidate tokens for speculative decoding.
    retrive_index : torch.Tensor
        Index tensor for retrieving nodes in the tree structure.
    retrive_next_token : torch.Tensor
        Tensor for retrieving the next token in the tree expansion.
    retrive_next_sibling : torch.Tensor
        Tensor for retrieving the next sibling node in the tree.
    uniform_samples : torch.Tensor
        Pre-sampled uniform random values for acceptance/rejection decisions.
    target_probs : torch.Tensor
        Tensor of probabilities for each candidate token from the target model.
    draft_probs : torch.Tensor
        Tensor of probabilities for each candidate token from the draft model (used for reference).
    threshold_single : float, optional
        Threshold for single-step acceptance. Default: 1.0 (accept if target prob ≥ draft prob).
    threshold_acc : float, optional
        Accumulated acceptance threshold for multi-step acceptance. Default: 1.0.
    deterministic : bool, optional
        Whether to use deterministic acceptance logic (recommended for reproducibility). Default: True.

    Returns
    -------
    None

    """
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
    r"""
    Perform greedy verification on a tree-structured set of candidate tokens using the target model prediction.

    This function traverses a candidate tree (as used in speculative or tree-based decoding)
    and greedily selects the path that matches the target model’s predictions.
    The accepted tokens, their indices, and the total number of accepted tokens are updated in-place.

    Parameters
    ----------
    predicts : torch.Tensor
        Mutable tensor. Stores the final sequence of accepted tokens after greedy verification. Updated in-place.
    accept_index : torch.Tensor
        Mutable tensor. Stores the indices of accepted tokens in the tree. Updated in-place.
    accept_token_num : torch.Tensor
        Mutable tensor. Stores the number of accepted tokens. Updated in-place.
    candidates : torch.Tensor
        Tensor of candidate tokens organized in a tree structure for speculative decoding.
    retrive_index : torch.Tensor
        Index tensor for retrieving nodes in the candidate tree.
    retrive_next_token : torch.Tensor
        Tensor for retrieving the next token node in the candidate tree.
    retrive_next_sibling : torch.Tensor
        Tensor for retrieving the next sibling node in the candidate tree.
    target_predict : torch.Tensor
        Tensor containing the target model's predicted tokens; used for greedy matching.

    Returns
    -------
    None

    """
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
    r"""
    Efficiently build the candidate tree structure for tree-based speculative decoding.

    This function constructs an efficient tree representation for a batch of speculative decoding paths,
    allowing subsequent tree-based operations (such as acceptance, verification, or path sampling) to
    traverse and manipulate candidate sequences efficiently. The tree is built based on parent-child relationships,
    selection indices, and position information, and is designed for use in high-throughput LLM inference.

    Parameters
    ----------
    parent_list : torch.Tensor
        Tensor specifying, for each node, the index of its parent node in the tree.
    selected_index : torch.Tensor
        Tensor specifying the selected (e.g., top-k) candidate indices for each expansion in the tree.
    verified_seq_len : torch.Tensor
        Tensor indicating the verified sequence length for each path.
    tree_mask : torch.Tensor
        Mask tensor indicating valid nodes/positions in the tree (used for pruning and efficient traversal).
    positions : torch.Tensor
        Tensor specifying position indices for nodes within each tree level.
    retrive_index : torch.Tensor
        Output tensor to be filled with tree node indices for retrieval operations.
    retrive_next_token : torch.Tensor
        Output tensor to be filled with next-token pointers for tree traversal.
    retrive_next_sibling : torch.Tensor
        Output tensor to be filled with sibling pointers for tree traversal.
    topk : int
        The number of candidates (branches) to expand at each tree level (beam width).
    depth : int
        The maximum tree depth to construct (number of decoding steps).
    draft_token_num : int
        Number of draft (proposed) tokens to include in the tree structure.

    Returns
    -------
    None

    """
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
    r"""
    Perform segment-wise bit-packing (packbits) on the input tensor.

    For each segment defined by `input_indptr`, this function applies a bit-packing operation
    to the corresponding slice of `x`, and writes the result to the corresponding segment
    in `y` as determined by `output_indptr`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, typically of boolean or 0/1 integer type, organized in segments.
    input_indptr : torch.Tensor
        1D tensor of shape (num_segments + 1,). Defines the start and end indices of each input segment.
        The i-th segment is `x[input_indptr[i]:input_indptr[i+1]]`.
    output_indptr : torch.Tensor
        1D tensor of shape (num_segments + 1,). Defines the start and end indices for each output segment in `y`.
        The i-th output segment is `y[output_indptr[i]:output_indptr[i+1]]`, which receives the packed bits.
    y : torch.Tensor
        Output tensor. Each segment receives the packed bit representation of the corresponding segment in `x`.

    Returns
    -------
    None

    """
    torch.ops.sgl_kernel.segment_packbits.default(
        x,
        input_indptr,
        output_indptr,
        y,
        torch.cuda.current_stream().cuda_stream,
    )
