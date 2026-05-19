# Copyright 2024 SGLang Team (internal fork)
# Early exit Qwen2 for sequence classification / embedding extraction.
# Runs only the first K transformer layers, then norm + pooler/logits.
#
# Usage:
#   SGLANG_EXIT_LAYER=20 python -m sglang.launch_server --model Qwen/Qwen2.5-72B ...

import logging
import os
from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

logger = logging.getLogger(__name__)

_DEFAULT_EXIT_LAYER: Optional[int] = (
    int(os.environ["SGLANG_EXIT_LAYER"])
    if os.environ.get("SGLANG_EXIT_LAYER")
    else None
)


class EarlyExitMixin:
    """Mixin that adds early-exit forward to any Qwen-style CausalLM.

    Expects the host class to have:
      - self.model.embed_tokens
      - self.model.layers
      - self.model.norm
      - self.logits_processor
      - self.pooler
    """

    def _get_exit_layer(self, forward_batch: ForwardBatch) -> Optional[int]:
        """Resolve exit_layer: per-request > env var > None (full forward)."""
        per_req = getattr(forward_batch, "exit_layer", None)
        if per_req is not None:
            return per_req
        return _DEFAULT_EXIT_LAYER

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        exit_layer = self._get_exit_layer(forward_batch)

        if exit_layer is None or exit_layer >= len(self.model.layers):
            # Full forward — delegate to base class
            return super().forward(
                input_ids, positions, forward_batch,
                input_embeds, get_embedding, **kwargs,
            )

        # --- Early exit: run first `exit_layer` layers ---
        if input_embeds is None:
            hidden_states = self.model.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for i in range(exit_layer):
            hidden_states, residual = self.model.layers[i](
                positions, hidden_states, forward_batch, residual,
            )

        # Apply final norm (same as full forward post-loop)
        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.model.norm(hidden_states)
            else:
                hidden_states, _ = self.model.norm(hidden_states, residual)

        if get_embedding:
            return self.pooler(hidden_states, forward_batch)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch,
        )


class Qwen2ForEarlyExitCausalLM(EarlyExitMixin, Qwen2ForCausalLM):
    pass


if _DEFAULT_EXIT_LAYER is not None:
    logger.info(f"Early exit enabled: exit_layer={_DEFAULT_EXIT_LAYER}")

EntryClass = Qwen2ForEarlyExitCausalLM
