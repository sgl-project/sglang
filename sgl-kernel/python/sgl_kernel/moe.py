import torch


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    token_cnts_buffer,
    cumsum_buffer,
):
    r"""
    Align tokens to experts in blocks for MoE (Mixture-of-Experts) computation.

    This function takes the routing results of a MoE (Mixture-of-Experts) layer and
    aligns token assignments to experts in block sizes suitable for efficient batching
    and parallel processing. It pads the number of tokens for each expert to the
    nearest multiple of `block_size`, sorts and outputs the aligned token indices,
    and prepares auxiliary buffers for fast indexing.

    Parameters
    ----------
    topk_ids : torch.Tensor or np.ndarray
        Tensor of shape (num_tokens, top_k), containing for each token the indices of the selected experts.
    num_experts : int
        Total number of experts.
    block_size : int
        Size of each block for alignment. Tokens are padded so that the count for each expert
        is a multiple of `block_size`.
    sorted_token_ids : torch.Tensor or np.ndarray
        Output tensor to hold sorted token indices after block alignment. Should be preallocated.
    experts_ids : torch.Tensor or np.ndarray
        Output tensor holding expert assignment per token, sorted and padded. Should be preallocated.
    num_tokens_post_pad : torch.Tensor or np.ndarray
        Output tensor for each expert, indicating the total number of tokens assigned after padding.
        Shape: (num_experts,)
    token_cnts_buffer : torch.Tensor or np.ndarray
        Temporary buffer storing the count of tokens assigned to each expert before padding.
        Shape: (num_experts,)
    cumsum_buffer : torch.Tensor or np.ndarray
        Temporary buffer for cumulative sum calculation, used for sorting and indexing.
        Shape: (num_experts + 1,)

    Returns
    -------
    None

    Notes
    -----
    - The function typically performs the following steps:
        1. Count how many tokens are routed to each expert.
        2. Pad the count for each expert to a multiple of `block_size`.
        3. Compute a cumulative sum for indexing.
        4. Sort the token indices and expert ids accordingly.
    - This is commonly used in MoE implementations to maximize hardware utilization and enable fast, batched
      expert computation.
    - Output tensors must be preallocated and will be filled in-place.
    - Proper alignment and padding are essential for efficient GPU execution.

    """
    torch.ops.sgl_kernel.moe_align_block_size.default(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: float,
) -> None:
    r"""
    Apply softmax normalization to top-k gating weights for sparse MoE routing.

    This function takes the top-k gating weights (before softmax) and, for each token, computes the softmax
    over its selected experts. The results are written in-place to `topk_weights`. This is typically used
    in Mixture-of-Experts (MoE) models, where each token is routed to a small subset of experts and their
    contributions are weighted by a softmax-normalized gating value.

    Parameters
    ----------
    topk_weights : torch.Tensor
        Tensor of shape (num_tokens, k), containing the pre-softmax gating weights for the top-k selected experts
        for each token. The result of the softmax is written in-place to this tensor.
    topk_ids : torch.Tensor
        Tensor of shape (num_tokens, k), containing the expert indices selected by routing for each token.
        Used to identify which expert each weight corresponds to.
    token_expert_indices : torch.Tensor
        Tensor of shape (num_tokens, k), representing the mapping between tokens and their selected experts.
        Typically used as an index or mask for further computation.
    gating_output : float
        Optional scaling factor or placeholder; can be used to combine with the softmax output (e.g., for
        temperature scaling or normalization). If not used, set as 1.0.

    Returns
    -------
    None

    Notes
    -----
    - For each token, this function applies a numerically-stable softmax across the k experts assigned to it:
        softmax(x_i) = exp(x_i) / sum_j exp(x_j)
    - The result is written in-place to `topk_weights`.
    - This is a key step in MoE gating, ensuring expert contributions sum to one for each token.
    - If `gating_output` is used, the softmax output can be further scaled by this value.
    - All output tensors must be preallocated and will be filled in-place.

    Example
    -------
    >>> topk_weights = torch.randn(num_tokens, k)
    >>> topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output=1.0)
    # topk_weights now contains the softmax probabilities for each token's top-k experts

    """
    torch.ops.sgl_kernel.topk_softmax.default(
        topk_weights, topk_ids, token_expert_indices, gating_output
    )


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    n_share_experts_fusion=0,
    routed_scaling_factor=0,
):
    r"""
    Fused hierarchical top-k gating for Mixture-of-Experts (MoE) routing.

    This fused kernel function selects the top-k experts for each token in a 2-level, hierarchical fashion:
    it splits all experts into `num_expert_group` groups, computes the sum of the top-2 expert weights in each group,
    and uses this as the group score to select the top expert groups. Then, within the selected groups,
    it selects the top-k experts per token.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor of shape (num_tokens, num_experts), representing the logits or pre-gating scores for each token.
    bias : torch.Tensor or None
        Optional bias tensor to be added to `input_tensor` before gating.
    num_expert_group : int
        Number of expert groups to split all experts into for hierarchical selection.
        Must divide `num_experts` evenly.
    topk_group : int
        Number of expert groups to select in the first gating stage (per token).
    topk : int
        Number of experts to select per token in the second gating stage (within selected groups).
    n_share_experts_fusion : int, optional
        If > 0, the last expert will be replaced with a round-robin shared expert.
        Default: 0.
    routed_scaling_factor : float, optional
        If > 0, the last expert will be scaled by this factor.
        Default: 0.

    Returns
    -------
    topk_weights : torch.Tensor
        Tensor with the softmax-normalized weights of the selected top-k experts per token.
    topk_ids : torch.Tensor
        Tensor with the indices of the selected top-k experts per token.

    Notes
    -----
    - This fused kernel function is used to select top-k experts in a hierarchical 2-layer fashion.
    - It splits the group of experts into `num_expert_group`, and uses the sum of the top-2 expert weights in each group
      as the group weight to select expert groups, then selects top-k experts within the selected groups.
    - The number of experts is decided by the input tensor shape; currently only power-of-2 expert counts are supported,
      and the total number of experts should be divisible by `num_expert_group`.
    - For now, `num_experts / num_expert_group` <= 32 is required.
    - For non-supported cases, use `biased_grouped_topk` in `sglang.srt.layers.moe.topk`.
    - `n_share_experts_fusion`: if > 0, the last expert will be replaced with a round-robin shared expert.
    - `routed_scaling_factor`: if > 0, the last expert will be scaled by this factor.
    """
    return torch.ops.sgl_kernel.moe_fused_gate.default(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        n_share_experts_fusion,
        routed_scaling_factor,
    )


def fp8_blockwise_scaled_grouped_mm(
    output,
    a,
    b,
    scales_a,
    scales_b,
    stride_a,
    stride_b,
    stride_c,
    layout_sfa,
    layout_sfb,
    problem_sizes,
    expert_offsets,
):
    torch.ops.sgl_kernel.fp8_blockwise_scaled_grouped_mm.default(
        output,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_c,
        layout_sfa,
        layout_sfb,
        problem_sizes,
        expert_offsets,
    )
