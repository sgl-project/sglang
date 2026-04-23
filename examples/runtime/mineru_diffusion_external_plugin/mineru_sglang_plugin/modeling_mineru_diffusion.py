"""Skeleton external model entry for MinerU diffusion architecture.

This file demonstrates wiring for an external multimodal architecture name via
`EntryClass`. Replace this skeleton with a full model port for production use.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration


class MinerUDiffusionForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """Minimal external-entry skeleton.

    Notes:
    - Keeps forward path aligned with built-in multimodal model behavior.
    - Intended as an example container for external registration and processor
      wiring.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ):
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            get_embedding=get_embedding,
        )


EntryClass = MinerUDiffusionForConditionalGeneration

__all__ = ["MinerUDiffusionForConditionalGeneration", "EntryClass"]
