"""Build target-verify metadata for a DFLASH beam tree.

Two masks are involved:

- the **QLEN** mask, `[bs, N, N]` bool, is the ancestor closure over draft nodes
  alone. `reconstruct_indices_from_tree_mask` consumes exactly this; the committed
  prefix reaches it only through `prefix_lens`, which the kernel adds to the
  per-node depth to produce absolute positions.
- the **FULL_MASK**, a flat bool buffer, is what the attention backends consume.
  Its trailing `N x N` block per request is the QLEN mask, preceded by the
  request's committed-prefix columns.

The sgl_kernel op schema spells its link arguments `retrive_*`; values passed to
that op keep the spelling while `DFlashVerifyInput` uses `retrieve_*`.
"""

from __future__ import annotations

import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask


def build_ancestor_mask(*, node_parents: torch.Tensor, max_depth: int) -> torch.Tensor:
    """Return `[bs, N, N]` ancestor closure for BFS-ordered parent links."""
    batch_size, num_nodes = node_parents.shape
    device = node_parents.device

    mask = torch.zeros(
        (batch_size, num_nodes, num_nodes), dtype=torch.bool, device=device
    )
    cursor = (
        torch.arange(num_nodes, device=device)
        .unsqueeze(0)
        .expand(batch_size, num_nodes)
        .contiguous()
    )
    mask.scatter_(2, cursor.unsqueeze(2), True)
    for _ in range(int(max_depth)):
        cursor = node_parents.gather(1, cursor).clamp(min=0)
        mask.scatter_(2, cursor.unsqueeze(2), True)
    return mask


def build_dflash_tree_meta(
    *, ancestor_mask: torch.Tensor, prefix_lens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive kernel links and positions from an ancestor mask."""
    batch_size, num_nodes, _ = ancestor_mask.shape
    device = ancestor_mask.device
    if prefix_lens.dtype != torch.int64:
        raise ValueError(
            "DFLASH tree meta requires int64 prefix_lens (the CUDA kernel casts "
            f"the pointer unchecked), got {prefix_lens.dtype}."
        )

    positions = torch.empty((batch_size * num_nodes,), dtype=torch.int64, device=device)
    links = torch.full((3, batch_size, num_nodes), -1, dtype=torch.int64, device=device)
    retrive_index, retrive_next_token, retrive_next_sibling = links

    reconstruct_indices_from_tree_mask(
        ancestor_mask.contiguous(),
        prefix_lens,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        batch_size,
        num_nodes,
    )
    return positions, retrive_index, retrive_next_token, retrive_next_sibling
