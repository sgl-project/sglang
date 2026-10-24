# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See ubtree_components/LICENSE for the original license.

"""SGLang verify input for fixed-budget irregular UBTree proposals."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import TreeMaskMode


@dataclass
class UBTreeVerifyInput(EagleVerifyInput):
    # Direct local parent index for every verify node. UBTree already owns this
    # topology, so hybrid recurrent kernels must not reconstruct it per layer.
    retrieve_parent_token: torch.Tensor | None = None
    # Per-node depth (root=0). The KDA verify backend copies this into a
    # graph-stable buffer and uses it to launch one parallel wave per depth.
    tree_depths: torch.Tensor | None = None
    logical_max_depth: int = 1
    tree_mask_mode: TreeMaskMode = TreeMaskMode.QLEN_ONLY

    @property
    def uses_direct_parent(self) -> bool:
        # Also true for CUDA Graph dummy inputs where the live tensor is absent;
        # capture must compile the direct-parent kernel variant.
        return True

    @property
    def max_tree_depth(self) -> int:
        return int(self.logical_max_depth)

    @property
    def tree_topk(self) -> int:
        # The tree is irregular; the linked child/sibling arrays are authoritative.
        return -1


__all__ = ["UBTreeVerifyInput"]


def create_ubtree_verify_input(
    custom_mask, num_tokens_per_req: int
) -> UBTreeVerifyInput:
    """Build the fixed-width target input used during CUDA Graph capture."""
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    return UBTreeVerifyInput(
        draft_token=None,
        custom_mask=custom_mask,
        positions=None,
        retrieve_index=None,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        tree_depths=None,
        retrieve_cum_len=None,
        spec_steps=spec.speculative_num_steps,
        topk=spec.speculative_eagle_topk,
        draft_token_num=num_tokens_per_req,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        seq_lens_sum=None,
        seq_lens_cpu=None,
        logical_max_depth=spec.speculative_dflash_block_size,
        tree_mask_mode=TreeMaskMode.QLEN_ONLY,
    )
