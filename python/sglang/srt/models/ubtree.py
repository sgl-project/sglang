# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See ../speculative/ubtree_components/LICENSE for the original license.

"""SGLang-native DFlash draft backbone with the UBTree selector attached."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.ubtree_components.config import parse_ubtree_config
from sglang.srt.speculative.ubtree_components.selector import UBTreeSelector


class Qwen3UBTreeModel(DFlashDraftModel):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        parse_ubtree_config(config)
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.ubtree_selector = UBTreeSelector(config)
        self._ubtree_segment_graph_runner = None

    def attach_segment_graph_runner(self, runner) -> None:
        self._ubtree_segment_graph_runner = runner

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ):
        runner = self._ubtree_segment_graph_runner
        if runner is not None and input_embeds is None:
            output = runner(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
            )
            if output is not None:
                return output
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            get_embedding=get_embedding,
            pp_proxy_tensors=pp_proxy_tensors,
        )


class Ling3UBTreeModel(Qwen3UBTreeModel):
    """Released Ling3 checkpoint alias for the shared UBTree draft model."""


EntryClass = [Qwen3UBTreeModel, Ling3UBTreeModel]
