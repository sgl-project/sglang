"""Model-neutral demand-cache policies for owner-sharded Shared KV."""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEMAND_CACHE_TAG_BYTES = 8
DEMAND_CACHE_EPOCH_BYTES = 4
_MAX_EPOCH = (1 << 31) - 1


@dataclass
class PoolDemandCache:
    """Demand cache backed by stable slots in a consumer-local pool.

    The model adapter chooses the pool slot for each source object. Complete
    historical objects remain resident, while mutable objects are refreshed.
    This cache has no hashing, probing, eviction, collision state, or
    first-writer arbitration.
    """

    data: torch.Tensor
    tags: torch.Tensor
    epoch_tensor: torch.Tensor
    keys: tuple[int, ...]
    epoch: int = 1

    @classmethod
    def create(
        cls,
        *,
        keys: tuple[int, ...],
        entries_per_key: int,
        entry_bytes: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> PoolDemandCache:
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Pool demand-cache keys must be unique")
        if entries_per_key <= 0 or entry_bytes <= 0:
            raise ValueError("Pool demand-cache geometry must be positive")
        return cls(
            data=torch.empty(
                (len(keys), entries_per_key, entry_bytes),
                dtype=dtype,
                device=device,
            ),
            tags=torch.zeros(
                (len(keys), entries_per_key),
                dtype=torch.int64,
                device=device,
            ),
            epoch_tensor=torch.ones((), dtype=torch.int32, device=device),
            keys=keys,
        )

    @property
    def key_to_slot(self) -> dict[int, int]:
        return {key: slot for slot, key in enumerate(self.keys)}

    @property
    def allocated_bytes(self) -> int:
        return self.data.nbytes + self.tags.nbytes + self.epoch_tensor.nbytes

    def storage_for(self, key: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            slot = self.key_to_slot[key]
        except KeyError as exc:
            raise ValueError(f"Demand-cache key {key} is not configured") from exc
        return self.data[slot], self.tags[slot]

    def invalidate(self) -> None:
        if self.epoch >= _MAX_EPOCH:
            self.tags.zero_()
            self.epoch = 1
        else:
            self.epoch += 1
        self.epoch_tensor.fill_(self.epoch)

    def clear(self) -> None:
        self.data = None
        self.tags = None
        self.epoch_tensor = None
