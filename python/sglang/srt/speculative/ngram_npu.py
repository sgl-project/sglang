"""Device-native NGRAM tree metadata for Ascend.

Only fixed tensor shapes are used: no CUDA JIT, host length reads, or
per-request data-dependent Python control flow. Attention verification remains
responsible for the tree mask and retained reference prompt.
"""

import torch


def reconstruct_indices_from_tree_mask_npu(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:
    """Write NGRAM positions and tree links using the existing op contract.

    Node order is topological. The mask contains each node's ancestors and
    itself; its strict lower triangle determines parent/depth/links. A missing
    child or sibling is -1, and retrieve indices are flattened batch offsets.
    """
    if batch_size == 0:
        return
    nodes = torch.arange(draft_token_num, dtype=torch.int64, device=tree_mask.device)
    rows, cols = nodes[:, None], nodes[None, :]
    tree = tree_mask.view(batch_size, draft_token_num, draft_token_num).bool()
    ancestors = tree & (cols < rows)
    parents = torch.where(ancestors, cols, -1).amax(dim=-1)
    children = torch.where(ancestors.transpose(1, 2), cols, draft_token_num).amin(
        dim=-1
    )
    siblings = (
        (parents[:, :, None] == parents[:, None, :])
        & (cols > rows)
        & (parents[:, :, None] >= 0)
    )
    siblings = torch.where(siblings, cols, draft_token_num).amin(dim=-1)
    positions.view(batch_size, draft_token_num).copy_(
        ancestors.sum(dim=-1).to(torch.int64) + verified_seq_len[:, None]
    )
    retrieve_index.copy_(
        torch.arange(
            batch_size * draft_token_num, dtype=torch.int64, device=tree_mask.device
        ).view(batch_size, draft_token_num)
    )
    retrieve_next_token.copy_(torch.where(children < draft_token_num, children, -1))
    retrieve_next_sibling.copy_(torch.where(siblings < draft_token_num, siblings, -1))
