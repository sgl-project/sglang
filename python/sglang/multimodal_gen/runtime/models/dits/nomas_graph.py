# SPDX-License-Identifier: Apache-2.0
"""Capture a cache-free Nomas prediction while evaluating the current event mask eagerly."""

from contextvars import ContextVar

import torch
from torch import nn

from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    BaseBreakableCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.models.dits.nomas import NomasSegmentContext
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)


class _SegmentPrediction(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.empty_cache = model.allocate_cache(0)
        self.attention = ContextVar("nomas_offline_attention")

    @eager_on_graph(
        output_factory=lambda self, q, k, v, *_: torch.empty_like(
            q.transpose(1, 2), memory_format=torch.contiguous_format
        ).transpose(1, 2)
    )
    def attend(self, q, k, v, layer_cache, valid_length):
        # The valid prefix and action windows can change without changing the shape.
        return self.attention.get()(q, k, v)

    def forward(self, hidden_states, timestep, encoder_hidden_states, text_mask, rope):
        context = NomasSegmentContext(
            self.attend, rope, tuple(hidden_states.shape[2:]), 0, 0
        )
        return self.model.predict(
            hidden_states,
            timestep,
            encoder_hidden_states,
            text_mask,
            cache=self.empty_cache,
            _context=context,
        )[0]


class NomasGraphRunner(BaseBreakableCudaGraphRunner):
    def __init__(self, model):
        super().__init__(_SegmentPrediction(model), model.device)

    def predict(self, hidden_states, timestep, condition, context, *, warmup):
        inputs = dict(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=condition.text_states,
            text_mask=condition.text_mask,
            rope=context.rope,
        )
        token = self.transformer.attention.set(context.attention)
        try:
            if warmup:
                self.capture(**inputs)
            return self(**inputs)
        finally:
            self.transformer.attention.reset(token)
