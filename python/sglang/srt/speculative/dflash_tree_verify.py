"""Decode steps shared by DFLASH chain and tree verification.

The helpers are stateless so the worker's branch points remain small.

Two width conventions meet here and must not be swapped:

- `verify_width` (`N = 1 + (block_size - 1) * tree_width`) is the node count --
  the verify forward's per-request token count, and the width of `candidates`,
  `predict`, `cache_loc_2d` and the ancestor mask.
- `block_size` is the longest root-to-leaf chain, hence the width of
  `accept_index` and of everything derived from it (`out_tokens`, the scheduler's
  output stride). The accepted run is a path, never wider than the tree is deep.
"""

from __future__ import annotations

import msgspec
import torch

from sglang.kernels.ops.speculative.dflash import write_dflash_tree_full_mask
from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens_func
from sglang.srt.layers.attention.verify_mask import fill_verify_mask_indptr
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_tree import (
    build_ancestor_mask,
    build_dflash_tree_meta,
)
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_utils import move_accept_tokens_to_target_kvcache


class TreeAccept(msgspec.Struct):
    """Results of greedy tree acceptance."""

    out_tokens: torch.Tensor
    commit_lens: torch.Tensor
    bonus_tokens: torch.Tensor
    accept_index: torch.Tensor
    predict: torch.Tensor


def build_tree_verify_input(
    *,
    node_tokens: torch.Tensor,
    node_parents: torch.Tensor,
    block_size: int,
    tree_width: int,
    prefix_lens: torch.Tensor,
    mask_buffer: torch.Tensor,
    mask_indptr: torch.Tensor,
) -> DFlashVerifyInput:
    """Build tree links, positions, and the device-side verify mask."""
    ancestor_mask = build_ancestor_mask(
        node_parents=node_parents, max_depth=block_size - 1
    )
    positions, retrive_index, retrive_next_token, retrive_next_sibling = (
        build_dflash_tree_meta(ancestor_mask=ancestor_mask, prefix_lens=prefix_lens)
    )
    num_nodes = int(node_parents.shape[1])
    custom_mask = write_dflash_tree_full_mask(
        ancestor_mask=ancestor_mask,
        mask_indptr=fill_verify_mask_indptr(
            mask_indptr=mask_indptr,
            seq_lens=prefix_lens,
            num_draft_tokens=num_nodes,
            bs=int(node_parents.shape[0]),
        ),
        seq_lens=prefix_lens,
        out=mask_buffer,
    )
    return DFlashVerifyInput(
        draft_token=node_tokens.reshape(-1),
        positions=positions,
        draft_token_num=num_nodes,
        topk=tree_width,
        block_size=block_size,
        custom_mask=custom_mask,
        retrieve_index=retrive_index,
        retrieve_next_token=retrive_next_token,
        retrieve_next_sibling=retrive_next_sibling,
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )


def accept_tree_greedy(
    *,
    verify_input: DFlashVerifyInput,
    next_token_logits: torch.Tensor,
    bs: int,
) -> TreeAccept:
    """Greedily accept a tree and lay the result out like a chain."""
    num_nodes = int(verify_input.draft_token_num)
    device = next_token_logits.device

    target_predict = torch.argmax(next_token_logits, dim=-1).view(bs, num_nodes)
    # Padded accept indices can still reach predict during logprob gathering.
    predict = torch.zeros((bs * num_nodes,), dtype=torch.int32, device=device)
    accept_index = torch.full(
        (bs, verify_input.max_tree_depth), -1, dtype=torch.int32, device=device
    )
    num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=device)
    verify_tree_greedy_func(
        predicts=predict,
        accept_index=accept_index,
        accept_token_num=num_correct_drafts,
        candidates=verify_input.draft_token.view(bs, num_nodes),
        retrieve_index=verify_input.retrieve_index,
        retrieve_next_token=verify_input.retrieve_next_token,
        retrieve_next_sibling=verify_input.retrieve_next_sibling,
        target_predict=target_predict,
        topk=verify_input.tree_topk,
    )

    # Keep padded gathers in range.
    out_tokens = predict[accept_index.to(torch.int64).clamp(min=0)]
    commit_lens = num_correct_drafts + 1
    bonus_tokens = torch.empty((bs,), dtype=torch.int32, device=device)
    fill_bonus_tokens_func(
        out_tokens,
        commit_lens,
        bonus_tokens,
        accept_index.shape[1],
        bs,
    )
    return TreeAccept(
        out_tokens=out_tokens,
        commit_lens=commit_lens,
        bonus_tokens=bonus_tokens,
        accept_index=accept_index,
        predict=predict,
    )


def move_accepted_target_kv(
    *,
    batch: ScheduleBatch,
    accepted: TreeAccept,
    token_to_kv_pool_allocator,
) -> None:
    """Move accepted tree-node KV to the contiguous commit slots."""
    move_accept_tokens_to_target_kvcache(
        batch,
        accepted.accept_index,
        accepted.commit_lens - 1,
        token_to_kv_pool_allocator,
    )


def compact_hidden_to_commit_layout(
    *,
    target_hidden: torch.Tensor,
    accept_index: torch.Tensor,
    bs: int,
    verify_width: int,
) -> torch.Tensor:
    """Compact accepted hidden rows into the chain commit layout."""
    from sglang.srt.speculative.eagle_worker_common import compact_accept_to_front

    return compact_accept_to_front(
        target_hidden, accept_index, bs, num_draft_tokens=verify_width
    )


def commit_positions(
    *, prefix_lens: torch.Tensor, verify_width: int
) -> torch.Tensor:
    """Return positions for the compacted chain commit layout."""
    offsets = torch.arange(verify_width, device=prefix_lens.device)
    return (prefix_lens.unsqueeze(1) + offsets).reshape(-1)
