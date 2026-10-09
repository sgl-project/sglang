"""Clef prompt states share the lifetime of their KV cache slots."""

from typing import Any

import torch

from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool


class ClefHybridLinearKVPool(HybridLinearKVPool):
    def __init__(self, *, clef_hidden_size: int, **kwargs: Any) -> None:
        if kwargs.get("post_capture_active", False):
            raise ValueError("Clef hidden cache does not support post-capture resizing")
        super().__init__(**kwargs)
        self.clef_hidden_states = torch.empty(
            (self.size + self.page_size, clef_hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.mem_usage += self.clef_hidden_states.nbytes / (1 << 30)

    def store_clef_hidden(
        self, locations: torch.Tensor, hidden_states: torch.Tensor
    ) -> None:
        self.clef_hidden_states[locations] = hidden_states

    def gather_clef_hidden(self, locations: torch.Tensor) -> torch.Tensor:
        return self.clef_hidden_states[locations]
