from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ngram_utils_module() -> Module:
    return load_jit(
        "ngram_utils",
        cuda_files=["speculative/ngram_utils.cuh"],
        cuda_wrappers=[
            ("reconstruct_indices_from_tree_mask", "reconstruct_indices_from_tree_mask"),
        ],
    )


@register_custom_op(
    op_name="reconstruct_indices_from_tree_mask_out",
    mutates_args=["positions", "retrive_index", "retrive_next_token", "retrive_next_sibling"],
)
def reconstruct_indices_from_tree_mask(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:
    """
    Reconstruct tree indices from a flat boolean tree mask.

    Args:
        tree_mask:            [bs * draft_token_num * draft_token_num] bool —
                              attention mask encoding the tree structure
        verified_seq_len:     [bs] int64 — verified sequence lengths
        positions:            [bs * draft_token_num] int64 — filled with token positions
        retrive_index:        [bs, draft_token_num] int64 — filled with retrieval indices
        retrive_next_token:   [bs, draft_token_num] int64 — filled with next token links
        retrive_next_sibling: [bs, draft_token_num] int64 — filled with sibling links
        batch_size:           number of sequences in the batch
        draft_token_num:      number of draft tokens per sequence
    """
    module = _jit_ngram_utils_module()
    module.reconstruct_indices_from_tree_mask(
        tree_mask,
        verified_seq_len,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        batch_size,
        draft_token_num,
    )
