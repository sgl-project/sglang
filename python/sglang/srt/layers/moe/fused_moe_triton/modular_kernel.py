import torch


def _moe_problem_size(
    a1: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    """
    Extract the MoE problem size from the given tensor arguments:
    - a: The hidden states, input to the MoE layer.
    - w1: The first set of expert weights.
    - w2: The second set of expert weights.
    - topk_ids: The topk ids.

    Note: extracting the problem shape from the weight and activation tensors is
    not obvious.  It needs to be done this way specifically due to subtle issues
    with particular kernels, e.g. the int4 kernels divide the trailing dimension
    by two, so it's not "correct" to extract N or K from the trailing dimension
    of w1 or w2.  Similarly, some kernels transpose the weights, so this needs
    to be kept in mind.
    """
    assert w1.dim() == 3 and w2.dim() == 3
    E, N, _ = w1.size()
    K = w2.size(1)

    if a1.dim() == 2:
        # Make sure we are using the correct a1 (pre-permute).
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)
    else:
        assert a1.dim() == 3
        assert a1.size(0) == E, f"{a1.size(0)} == {E}"
        M = a1.size(1)  # This is max_num_tokens

    assert topk_ids.dim() == 2
    topk = topk_ids.size(1)

    return E, M, N, K, topk
