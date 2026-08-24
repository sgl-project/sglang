"""The DFLASH decode steps that differ between chain verify and tree verify.

Tree width 1 keeps the chain path in `DFlashWorkerV2.forward_batch_generation`
byte-for-byte, so everything here runs only when the beam is wider -- or when
`SGLANG_DFLASH_FORCE_TREE_VERIFY` routes width 1 through here, which is how the
two paths are shown to agree (a width-1 beam *is* the chain).

Stateless on purpose: the worker hands each step the values it needs and assigns
the results back, so the fork points inside that long method stay readable as
`if tree: x = f(...)`.

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

from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens_func
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_tree import (
    build_ancestor_mask,
    build_dflash_tree_meta,
    build_full_tree_mask,
)
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_utils import move_accept_tokens_to_target_kvcache


class TreeAccept(msgspec.Struct):
    """What the greedy tree accept produces, in the worker's own vocabulary.

    `out_tokens` / `commit_lens` / `bonus_tokens` carry exactly what the chain
    path's `_commit_accept` returns, so the shared tail (KV writeback, seq-len
    advance, result assembly) does not have to branch. `accept_index` and
    `predict` are additionally exposed because the tree needs them for the steps
    a chain gets for free: hidden-state compaction, logprobs, mamba step index.
    """

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
    prefix_lens_cpu: torch.Tensor,
) -> DFlashVerifyInput:
    """The verify input for a beam tree: mask, tree links, depth-based positions.

    `prefix_lens` must be the *committed* lengths on both sides. The device copy
    feeds the kernel that turns per-node depth into an absolute position; the host
    copy feeds the per-request row widths of the flat attention mask. Passing the
    verify-extended lengths would shift every position by one block and widen
    every mask row past what the backend allocated.
    """
    ancestor_mask = build_ancestor_mask(
        node_parents=node_parents, max_depth=block_size - 1
    )
    positions, retrive_index, retrive_next_token, retrive_next_sibling = (
        build_dflash_tree_meta(ancestor_mask=ancestor_mask, prefix_lens=prefix_lens)
    )
    return DFlashVerifyInput(
        draft_token=node_tokens.reshape(-1),
        positions=positions,
        draft_token_num=int(node_parents.shape[1]),
        topk=tree_width,
        block_size=block_size,
        custom_mask=build_full_tree_mask(
            ancestor_mask=ancestor_mask, prefix_lens_cpu=prefix_lens_cpu
        ),
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
    """Walk the tree greedily and lay the accepted path out like a chain.

    Pure tensor work, so it is checkable against the chain accept without a model:
    at width 1 the tree *is* the chain and the two must return the same tokens.
    The KV that the accepted path implies is moved separately, by
    `move_accepted_target_kv` -- keep the two adjacent at the call site.

    `out_tokens` is gathered through `accept_index` rather than shifted left, which
    is the same "drafts shifted by one, bonus last" run the chain produces:
    `predict[node]` is the target's continuation *after* that node.
    """
    num_nodes = int(verify_input.draft_token_num)
    device = next_token_logits.device

    target_predict = torch.argmax(next_token_logits, dim=-1).view(bs, num_nodes)
    # Zero-initialized, not empty: `compute_spec_logprobs` gathers through the whole
    # accept_index including its -1 pad, and -1 indexes predict[-1] rather than
    # raising. An uninitialized slot there is a garbage token id and the logprob
    # gather walks off the vocabulary. Only the accepted prefix is ever read, so any
    # in-range value would do; 0 is what EAGLE relies on for the same reason.
    predict = torch.zeros((bs * num_nodes,), dtype=torch.int32, device=device)
    # -1 marks "no node accepted at this depth"; the kernel fills the prefix only.
    accept_index = torch.full(
        (bs, verify_input.max_tree_depth), -1, dtype=torch.int32, device=device
    )
    num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=device)
    verify_tree_greedy_func(
        predicts=predict,  # mutable
        accept_index=accept_index,  # mutable
        accept_token_num=num_correct_drafts,  # mutable
        candidates=verify_input.draft_token.view(bs, num_nodes),
        retrieve_index=verify_input.retrieve_index,
        retrieve_next_token=verify_input.retrieve_next_token,
        retrieve_next_sibling=verify_input.retrieve_next_sibling,
        target_predict=target_predict,
        topk=verify_input.tree_topk,
    )

    # Padded entries clamp to node 0; they land past commit_lens and are never read.
    out_tokens = predict[accept_index.to(torch.int64).clamp(min=0)]
    commit_lens = num_correct_drafts + 1
    bonus_tokens = torch.empty((bs,), dtype=torch.int32, device=device)
    fill_bonus_tokens_func(
        out_tokens,
        commit_lens,
        bonus_tokens,  # mutable
        # Stride of out_tokens, i.e. accept_index's width, NOT the node count.
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
    """Copy the accepted nodes' target KV down onto the contiguous front slots.

    The r-th accepted token is node `accept_index[i, r]`, but the next round reads
    position `prefix + r`, which `req_to_token` maps to the r-th *slot*. Chain
    verify has those coincide, which is why DFLASH has never needed this step; a
    tree does not, and skipping it leaves the target reading a sibling's KV.
    """
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
    """Move the accepted path's hidden rows to the front of each node block.

    This is what lets the draft-KV writeback stay untouched: it commits row `r`
    of request `i` into `cache_loc_2d[i, r]`, and after this gather row `r` holds
    the node accepted at depth `r`. Without it the tree would write node `r`'s
    features into depth `r`'s slot -- a silent mismatch that width 1 cannot show,
    because there the accepted nodes already are the first rows.
    """
    from sglang.srt.speculative.eagle_worker_common import compact_accept_to_front

    return compact_accept_to_front(
        target_hidden, accept_index, bs, num_draft_tokens=verify_width
    )


def commit_positions(
    *, prefix_lens: torch.Tensor, verify_width: int
) -> torch.Tensor:
    """Positions for the compacted commit layout, `[bs * verify_width]`.

    Not the same tensor the verify forward used: there a node's position is
    `prefix + depth(node)`, ordered by node. Here row `r` is whichever node was
    accepted at depth `r`, so its position is `prefix + r` no matter which node it
    was, and the layout is a plain chain again. Rows past the accepted run get
    positions that are never committed.
    """
    offsets = torch.arange(verify_width, device=prefix_lens.device)
    return (prefix_lens.unsqueeze(1) + offsets).reshape(-1)
