"""Verify metadata for a DFLASH beam tree.

The beam walk (`models/dflash.py::_beam_walk_torch` and its triton twin) emits the
minimal representation of the tree: `node_tokens` and `node_parents`, both
`[bs, num_nodes]`, in BFS order so that `node_parents[i] < i`. Target verify needs
more than that, and this module derives all of it.

Two different masks are involved and they must not be confused:

- the **QLEN** mask, `[bs, N, N]` bool, is the ancestor closure over draft nodes
  alone. `reconstruct_indices_from_tree_mask` consumes exactly this; the committed
  prefix reaches it only through `prefix_lens`, which the kernel adds to the
  per-node depth to produce absolute positions.
- the **FULL_MASK**, a flat bool buffer, is what the attention backends consume.
  Its trailing `N x N` block per request *is* the QLEN mask, preceded by the
  request's committed prefix columns.

Spelling: the sgl_kernel op schema says `retrive_*`. This module keeps that
spelling on values that go straight into the op, matching the boundary
`eagle_utils.py::verify_tree_greedy_func` already draws, while
`DFlashVerifyInput` keeps the correct `retrieve_*` on its fields.
"""

from __future__ import annotations

import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask


def build_ancestor_mask(*, node_parents: torch.Tensor, max_depth: int) -> torch.Tensor:
    """`[bs, N, N]` bool ancestor closure, `mask[b, i, j] = j is an ancestor of i`.

    The diagonal is set (a node counts as its own ancestor) and column 0 is set on
    every row, because the root is an ancestor of everything.

    Walks parent pointers `max_depth` times rather than scanning the `N` nodes in
    order. `max_depth` is the deepest layer index -- `gamma = block_size - 1` -- so
    the iteration count is independent of the beam width, and a width sweep does
    not pay a growing number of kernel launches. Clamping a spent chain to node 0
    is harmless precisely because the root bit is already set on every row.
    """
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
    """`(positions, retrive_index, retrive_next_token, retrive_next_sibling)`.

    Derived from the mask rather than from `node_parents` directly, so a
    mis-built mask shows up as links that disagree with the parents instead of
    staying invisible until attention silently reads the wrong keys.

    Returned in the kernel's own argument order. `positions` is flat `[bs * N]`;
    the three link tensors are `[bs, N]`. All four are int64, which the CUDA
    kernel requires -- it casts the pointers without checking.
    """
    batch_size, num_nodes, _ = ancestor_mask.shape
    device = ancestor_mask.device
    if prefix_lens.dtype != torch.int64:
        raise ValueError(
            "DFLASH tree meta requires int64 prefix_lens (the CUDA kernel casts "
            f"the pointer unchecked), got {prefix_lens.dtype}."
        )

    positions = torch.empty((batch_size * num_nodes,), dtype=torch.int64, device=device)
    # -1 is the "no such link" sentinel; the kernel leaves absent links untouched.
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


def build_full_tree_mask(
    *, ancestor_mask: torch.Tensor, prefix_lens_cpu: torch.Tensor
) -> torch.Tensor:
    """The flat bool mask the attention backends consume.

    Per request: `N` rows of width `prefix + N`, the leading `prefix` columns all
    True (every draft node sees the whole committed prefix) and the trailing block
    the request's ancestor closure. Requests are concatenated, so the total is
    `sum(prefix) * N + N**2 * bs` -- the same formula
    `DFlashVerifyInput.generate_attn_arg_prefill` uses to size its padding.

    Takes the prefix lengths on the host because the row widths are Python-level
    loop bounds; passing the device copy would force a sync every step. These must
    be the *committed* lengths, not the temporarily verify-extended ones.
    """
    batch_size, num_nodes, _ = ancestor_mask.shape
    device = ancestor_mask.device
    rows = []
    for request, prefix_len in enumerate(prefix_lens_cpu.tolist()):
        prefix = torch.ones(
            (num_nodes, int(prefix_len)), dtype=torch.bool, device=device
        )
        rows.append(torch.cat([prefix, ancestor_mask[request]], dim=1).flatten())
    return torch.cat(rows)
