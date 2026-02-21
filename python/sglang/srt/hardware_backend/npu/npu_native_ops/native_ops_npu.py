from typing import Tuple


import torch


def reconstruct_indices_from_tree_mask_torch(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    batch_size: int,
    draft_token_num: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    D = draft_token_num
    bs = batch_size

    tree_mask = tree_mask.view(bs, D, D)  # [bs, D, D]  → [batch, token(i), token(t)]
    device = tree_mask.device
    dtype = torch.int64

    retrive_index = torch.arange(bs * D, device=device, dtype=dtype).view(bs, D)  # [bs, D]

    t_indices = torch.arange(D, device=device).unsqueeze(0).expand(D, D)

    i_lt_t_mask = t_indices < torch.arange(D, device=device).unsqueeze(1)  # [D, D] → (i_lt_t_mask[t,i] = i < t)

    i_lt_t_mask = i_lt_t_mask.unsqueeze(0).expand(bs, D, D)

    # sum(tree_mask[b,t,i] for i < t)
    depth = (tree_mask & i_lt_t_mask).sum(dim=-1)  # [bs, D]

    reversed_tree_mask = tree_mask.flip(dims=[-1]) & i_lt_t_mask.flip(dims=[-1])
    parent_idx = torch.argmax(reversed_tree_mask.int(), dim=-1)
    has_parent = (reversed_tree_mask.sum(dim=-1) > 0)
    parent_idx = (torch.arange(D, device=device).unsqueeze(0).expand(bs, D) - 1) - parent_idx
    parent_idx = torch.where(has_parent, parent_idx, -torch.ones_like(parent_idx))

    positions = depth + verified_seq_len.unsqueeze(1)  # [bs, D]

    i_indices = torch.arange(D, device=device).reshape(1, D, 1).expand(bs, D, D)
    t_indices = torch.arange(D, device=device).reshape(1, 1, D).expand(bs, D, D)
    i_gt_t_mask = i_indices > t_indices  # [bs, D, D]

    next_token_candidates = tree_mask & i_gt_t_mask
    next_token_idx = torch.argmax(next_token_candidates.int(), dim=1)  # [bs, D]
    has_next_token = (next_token_candidates.sum(dim=1) > 0)  # [bs, D]
    retrive_next_token = torch.where(has_next_token, next_token_idx, -torch.ones_like(next_token_idx))

    parent_expanded = parent_idx.unsqueeze(1).expand(bs, D, D)  # [bs, D, D]
    cond_a = torch.gather(tree_mask, dim=2, index=parent_expanded) & i_gt_t_mask
    j_indices = torch.arange(D, device=device).reshape(1, 1, D).expand(bs, D, D)  # [bs, D, D]
    cond_b_mask = (j_indices > parent_expanded) & (j_indices < i_indices)  # [bs, D, D]
    cond_b = (tree_mask & cond_b_mask).sum(dim=2) == 0  # [bs, D]
    cond_b = cond_b.unsqueeze(2).expand(bs, D, D)
    sibling_candidates = cond_a & cond_b
    next_sibling_idx = torch.argmax(sibling_candidates.int(), dim=1)
    has_next_sibling = (sibling_candidates.sum(dim=1) > 0) & (parent_idx != -1)
    retrive_next_sibling = torch.where(has_next_sibling, next_sibling_idx, -torch.ones_like(next_sibling_idx))

    retrive_index = retrive_index.reshape(-1)
    positions = positions.reshape(-1)
    retrive_next_token = retrive_next_token.reshape(-1)
    retrive_next_sibling = retrive_next_sibling.reshape(-1)

    return retrive_index, positions, retrive_next_token, retrive_next_sibling


def reconstruct_indices_from_tree_mask(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int
) -> None:
    origin_retrive_index_shape = retrive_index.shape
    origin_positions_shape = positions.shape
    origin_retrive_next_token = retrive_next_token.shape
    origin_retrive_next_sibling = retrive_next_sibling.shape

    ri, pos, ntk, nsb = reconstruct_indices_from_tree_mask_torch(
        tree_mask, verified_seq_len, batch_size, draft_token_num
    )
    ri = ri.contiguous().view(*origin_retrive_index_shape)
    pos = pos.contiguous().view(*origin_positions_shape)
    ntk = ntk.contiguous().view(*origin_retrive_next_token)
    nsb = nsb.contiguous().view(*origin_retrive_next_sibling)

    retrive_index.copy_(ri)
    positions.copy_(pos)
    retrive_next_token.copy_(ntk)
    retrive_next_sibling.copy_(nsb)
