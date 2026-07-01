from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParamInfo:
    """Metadata about an offloaded parameter, used as a buffer pool key."""

    name: str  # name relative to its containing submodule (may be dotted)
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype

    @property
    def num_bytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n * torch.empty((), dtype=self.dtype).element_size()


class StaticBufferPool:
    """Pre-allocated device buffer pool indexed by (name, shape, stride, dtype)."""

    def __init__(
        self,
        param_infos: List[ParamInfo],
        slot_capacity: int,
        device: torch.device,
    ) -> None:
        assert slot_capacity >= 1
        self.slot_capacity = slot_capacity
        self.device = device
        self.total_bytes = 0

        unique: Dict[Tuple, ParamInfo] = {}
        for info in param_infos:
            key = (info.name, info.shape, info.stride, info.dtype)
            unique.setdefault(key, info)

        self._buffers: Dict[Tuple, List[torch.Tensor]] = {}
        for key, info in unique.items():
            slots: List[torch.Tensor] = []
            for _ in range(slot_capacity):
                slots.append(
                    torch.empty_strided(
                        size=info.shape,
                        stride=info.stride,
                        dtype=info.dtype,
                        device=device,
                    )
                )
                self.total_bytes += info.num_bytes
            self._buffers[key] = slots

        logger.info(
            "[offloader] static buffer pool: %d unique keys x %d slots, %.3f GB",
            len(unique),
            slot_capacity,
            self.total_bytes / 1e9,
        )

    def get(
        self,
        name: str,
        shape: Tuple[int, ...],
        stride: Tuple[int, ...],
        dtype: torch.dtype,
        slot_idx: int,
    ) -> torch.Tensor:
        return self._buffers[(name, shape, stride, dtype)][
            slot_idx % self.slot_capacity
        ]
