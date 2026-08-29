from __future__ import annotations

import torch


class DsparkTpSync:
    def __init__(self, tp_group) -> None:
        self._tp_group = tp_group
        self._enabled = tp_group.world_size > 1

    def sync(self, values: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return values
        return self._tp_group.broadcast(values, src=0)
