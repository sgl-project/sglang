from __future__ import annotations

from typing import Optional

import torch

_OPTIONAL_INT_NONE = -1
_OPTIONAL_INT_INVALID = -2


class DsparkTpSync:
    def __init__(self, tp_group) -> None:
        self._tp_group = tp_group
        self._enabled = tp_group.world_size > 1
        self._optional_int_buffer = (
            torch.empty((1,), dtype=torch.int64) if self._enabled else None
        )

    def sync(self, values: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return values
        return self._tp_group.broadcast_capture_safe(values, src=0)

    def broadcast_optional_int(self, value: Optional[int]) -> Optional[int]:
        if not self._enabled:
            if value is not None and value < 0:
                raise RuntimeError(f"DSpark received invalid optional integer {value}.")
            return value
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DSpark CPU decisions cannot run in CUDA graph capture.")

        packet = self._optional_int_buffer
        assert packet is not None
        if self._tp_group.rank_in_group == 0:
            if value is None:
                encoded = _OPTIONAL_INT_NONE
            else:
                encoded = value if value >= 0 else _OPTIONAL_INT_INVALID
            packet.fill_(encoded)
        torch.distributed.broadcast(
            packet,
            src=self._tp_group.ranks[0],
            group=self._tp_group.cpu_group,
        )
        received = int(packet.item())
        if received < _OPTIONAL_INT_NONE:
            raise RuntimeError(
                f"DSpark received invalid optional integer {received} from its "
                "tensor-parallel source rank."
            )
        return None if received == _OPTIONAL_INT_NONE else received
