"""Per-forward state for lookahead sparse-attention selectors.

The state is intentionally owned by ``ForwardBatch`` rather than by an
attention backend or a model module.  A backend may be shared by many model
layers and requests, while a forecast is valid only for the current forward
pass and the next layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_FORECAST_STATE_KEY = "lookahead_forecast_state"


@dataclass
class ForecastState:
    """Forecast query produced by one layer for the immediately next layer."""

    query: Optional[torch.Tensor] = None
    producer_layer: Optional[int] = None

    def reset(self) -> None:
        """Discard state from a previous model invocation."""
        self.query = None
        self.producer_layer = None

    def for_layer(self, layer_id: int) -> Optional[torch.Tensor]:
        """Return the forecast produced by ``layer_id - 1`` if available."""
        if self.producer_layer != layer_id - 1:
            return None
        return self.query

    def publish(self, layer_id: int, query: torch.Tensor) -> None:
        """Publish a forecast for the next layer in the current forward."""
        self.query = query
        self.producer_layer = layer_id


def get_forecast_state(forward_batch: ForwardBatch) -> ForecastState:
    """Get the request-local forecast state attached to ``forward_batch``."""
    if forward_batch.model_specific_states is None:
        forward_batch.model_specific_states = {}

    state = forward_batch.model_specific_states.get(_FORECAST_STATE_KEY)
    if state is None:
        state = ForecastState()
        forward_batch.model_specific_states[_FORECAST_STATE_KEY] = state
    if not isinstance(state, ForecastState):
        raise TypeError(
            f"ForwardBatch state key {_FORECAST_STATE_KEY!r} is already used by "
            f"{type(state).__name__}, expected ForecastState"
        )
    return state
