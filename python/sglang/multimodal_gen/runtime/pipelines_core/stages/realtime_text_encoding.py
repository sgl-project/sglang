# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Robbyant/lingbot-world

"""
LingBot-World realtime text stages.

The reference lingbot_fast_server initializes prompt embeddings once per session.
Cache text encoder outputs across realtime chunks so condition sampling stays
closer to the actual denoising step.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _normalize_prompt_value(
    value: str | list[str] | None,
) -> str | tuple[str, ...] | None:
    if isinstance(value, list):
        return tuple(value)
    return value


def _copy_tensor_list(
    value: list[torch.Tensor] | None,
) -> list[torch.Tensor] | None:
    if value is None:
        return None
    return list(value)


def _copy_seq_lens(
    value: list[list[int]] | None,
) -> list[list[int]] | None:
    if value is None:
        return None
    return [list(seq_lens) for seq_lens in value]


class RealtimeTextState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.cache_key: tuple[Any, ...] | None = None
        self.prompt_embeds: list[torch.Tensor] | None = None
        self.pooled_embeds: list[torch.Tensor] | None = None
        self.prompt_attention_mask: list[torch.Tensor] | None = None
        self.prompt_embeds_mask: list[torch.Tensor] | None = None
        self.prompt_seq_lens: list[list[int]] | None = None
        self.negative_prompt_embeds: list[torch.Tensor] | None = None
        self.neg_pooled_embeds: list[torch.Tensor] | None = None
        self.negative_attention_mask: list[torch.Tensor] | None = None
        self.negative_prompt_embeds_mask: list[torch.Tensor] | None = None
        self.negative_prompt_seq_lens: list[list[int]] | None = None

    def clear_text_cache(self):
        self.cache_key = None
        self.prompt_embeds = None
        self.pooled_embeds = None
        self.prompt_attention_mask = None
        self.prompt_embeds_mask = None
        self.prompt_seq_lens = None
        self.negative_prompt_embeds = None
        self.neg_pooled_embeds = None
        self.negative_attention_mask = None
        self.negative_prompt_embeds_mask = None
        self.negative_prompt_seq_lens = None

    def dispose(self):
        super().dispose()
        self.clear_text_cache()


class RealtimeTextEncodingStage(TextEncodingStage):
    """Cache text encoder outputs across realtime chunks by prompt identity."""

    def _make_cache_key(self, batch: Req) -> tuple[Any, ...]:
        return (
            _normalize_prompt_value(batch.prompt),
            bool(batch.do_classifier_free_guidance),
            (
                _normalize_prompt_value(batch.negative_prompt)
                if batch.do_classifier_free_guidance
                else None
            ),
        )

    def _restore_cached_outputs(self, batch: Req, state: RealtimeTextState) -> Req:
        batch.prompt_embeds = _copy_tensor_list(state.prompt_embeds) or []
        batch.pooled_embeds = _copy_tensor_list(state.pooled_embeds) or []
        batch.prompt_attention_mask = _copy_tensor_list(state.prompt_attention_mask)
        batch.prompt_embeds_mask = _copy_tensor_list(state.prompt_embeds_mask)
        batch.prompt_seq_lens = _copy_seq_lens(state.prompt_seq_lens)
        batch.negative_prompt_embeds = _copy_tensor_list(state.negative_prompt_embeds)
        batch.neg_pooled_embeds = _copy_tensor_list(state.neg_pooled_embeds) or []
        batch.negative_attention_mask = _copy_tensor_list(state.negative_attention_mask)
        batch.negative_prompt_embeds_mask = _copy_tensor_list(
            state.negative_prompt_embeds_mask
        )
        batch.negative_prompt_seq_lens = _copy_seq_lens(state.negative_prompt_seq_lens)
        return batch

    def _store_outputs(self, batch: Req, state: RealtimeTextState) -> None:
        state.prompt_embeds = _copy_tensor_list(batch.prompt_embeds)
        state.pooled_embeds = _copy_tensor_list(batch.pooled_embeds)
        state.prompt_attention_mask = _copy_tensor_list(batch.prompt_attention_mask)
        state.prompt_embeds_mask = _copy_tensor_list(batch.prompt_embeds_mask)
        state.prompt_seq_lens = _copy_seq_lens(batch.prompt_seq_lens)
        state.negative_prompt_embeds = _copy_tensor_list(batch.negative_prompt_embeds)
        state.neg_pooled_embeds = _copy_tensor_list(batch.neg_pooled_embeds)
        state.negative_attention_mask = _copy_tensor_list(batch.negative_attention_mask)
        state.negative_prompt_embeds_mask = _copy_tensor_list(
            batch.negative_prompt_embeds_mask
        )
        state.negative_prompt_seq_lens = _copy_seq_lens(batch.negative_prompt_seq_lens)

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        if batch.session is None:
            return super().forward(batch, server_args)

        state = batch.session.get_or_create_state(RealtimeTextState)
        assert isinstance(state, RealtimeTextState)

        # cache the encoder results into BaseRealtimeState, restore when encoder inputs hits the cache
        cache_key = self._make_cache_key(batch)
        if state.cache_key == cache_key and state.prompt_embeds is not None:
            return self._restore_cached_outputs(batch, state)

        state.clear_text_cache()

        # perform regular text encoding
        batch = super().forward(batch, server_args)

        state.cache_key = cache_key
        self._store_outputs(batch, state)
        return batch
