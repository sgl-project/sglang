from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_eagle_utils_module() -> Module:
    return load_jit(
        "eagle_utils",
        cuda_files=["speculative/eagle_utils.cuh"],
        cuda_wrappers=[
            ("build_tree_kernel_efficient", "build_tree_kernel_efficient"),
            ("verify_tree_greedy", "verify_tree_greedy"),
        ],
    )


@register_custom_op(
    op_name="build_tree_kernel_efficient_out",
    mutates_args=[
        "tree_mask",
        "positions",
        "retrive_index",
        "retrive_next_token",
        "retrive_next_sibling",
    ],
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
    tree_mask_mode: int,
) -> None:
    """
    Build EAGLE speculative decoding tree in-place.

    Args:
        parent_list:          [bs, topk*(depth-1)+1] int64 — parent node indices in tree
        selected_index:       [bs, draft_token_num-1] int64 — selected token indices
        verified_seq_len:     [bs] int64 — verified sequence lengths
        tree_mask:            tree attention mask tensor (shape depends on tree_mask_mode)
        positions:            [bs, draft_token_num] int64 — filled with token positions
        retrive_index:        [bs, draft_token_num] int64 — filled with retrieval indices
        retrive_next_token:   [bs, draft_token_num] int64 — filled with next token links
        retrive_next_sibling: [bs, draft_token_num] int64 — filled with sibling links
        topk:                 number of top-k candidates per step
        depth:                tree depth
        draft_token_num:      total number of draft tokens per batch element
        tree_mask_mode:       0=FULL_MASK, 1=QLEN_ONLY, 2=QLEN_ONLY_BITPACKING
    """
    module = _jit_eagle_utils_module()
    module.build_tree_kernel_efficient(
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
        tree_mask_mode,
    )


@register_custom_op(
    op_name="verify_tree_greedy_out",
    mutates_args=["predicts", "accept_index", "accept_token_num"],
)
def verify_tree_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    """
    Greedy tree verification for EAGLE speculative decoding.

    Args:
        predicts:              [tot_num_draft_tokens] int32 — filled with accepted/predicted tokens
        accept_index:          [bs, num_spec_step] int32 — filled with accepted token indices
        accept_token_num:      [bs] int32 — filled with number of accepted tokens per sequence
        candidates:            [bs, num_draft_tokens] int64 — draft token IDs
        retrive_index:         [bs, num_draft_tokens] int64 — tree node retrieval indices
        retrive_next_token:    [bs, num_draft_tokens] int64 — next token links in tree
        retrive_next_sibling:  [bs, num_draft_tokens] int64 — sibling links in tree
        target_predict:        [bs, num_draft_tokens] int64 — target model token predictions
    """
    module = _jit_eagle_utils_module()
    module.verify_tree_greedy(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
    )
