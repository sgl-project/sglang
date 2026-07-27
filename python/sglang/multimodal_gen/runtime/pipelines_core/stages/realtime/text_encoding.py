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


_TEXT_CACHE_TENSOR_LIST_FIELDS = (
    "prompt_embeds",
    "pooled_embeds",
    "prompt_attention_mask",
    "prompt_embeds_mask",
    "negative_prompt_embeds",
    "neg_pooled_embeds",
    "negative_attention_mask",
    "negative_prompt_embeds_mask",
)
_TEXT_CACHE_SEQ_LENS_FIELDS = (
    "prompt_seq_lens",
    "negative_prompt_seq_lens",
)
_TEXT_CACHE_FIELDS = _TEXT_CACHE_TENSOR_LIST_FIELDS + _TEXT_CACHE_SEQ_LENS_FIELDS
_TEXT_CACHE_DEFAULT_EMPTY_LIST_FIELDS = {
    "prompt_embeds",
    "pooled_embeds",
    "neg_pooled_embeds",
}


def _copy_text_cache_field(name: str, value):
    if name in _TEXT_CACHE_SEQ_LENS_FIELDS:
        return _copy_seq_lens(value)
    return _copy_tensor_list(value)


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
        for field in _TEXT_CACHE_FIELDS:
            setattr(self, field, None)

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
        for field in _TEXT_CACHE_FIELDS:
            value = _copy_text_cache_field(field, getattr(state, field))
            if value is None and field in _TEXT_CACHE_DEFAULT_EMPTY_LIST_FIELDS:
                value = []
            setattr(batch, field, value)
        return batch

    def _store_outputs(self, batch: Req, state: RealtimeTextState) -> None:
        for field in _TEXT_CACHE_FIELDS:
            setattr(state, field, _copy_text_cache_field(field, getattr(batch, field)))

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
