"""Zero-parameter role proxies used while materializing Qwen weights."""

from __future__ import annotations

from typing import Any

from torch import nn


class AFDProxyAttention(nn.Module):
    def forward(
        self,
        positions: Any,
        hidden_states: Any,
        forward_batch: Any,
        *args,
        **kwargs,
    ):
        del positions, forward_batch, args, kwargs
        return hidden_states


class AFDProxyMLP(nn.Module):
    def forward(self, hidden_states: Any, *args, **kwargs):
        del args, kwargs
        return hidden_states
