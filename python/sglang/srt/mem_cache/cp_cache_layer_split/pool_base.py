"""Common ownership contract for CP Cache LayerSplit pools."""

from __future__ import annotations

import logging

from sglang.srt.mem_cache.cp_cache_layer_split.utils import (
    get_global_layer_shard_range,
    get_layer_owner,
    get_layer_shard_range,
)

logger = logging.getLogger(__name__)


class CpCacheLayerSplitPoolBase:
    """Stage-local layer ownership shared by all Cache LayerSplit pools."""

    requires_descriptor_matched_transfer = False

    def _init_cp_cache_layer_split(
        self,
        *,
        cp_rank: int,
        cp_size: int,
        layer_shard_start_layer: int,
        layer_shard_layer_num: int,
    ) -> None:
        """Initialize the stage-local layer ownership state."""
        if cp_size <= 1:
            raise ValueError(f"Cache LayerSplit requires cp_size > 1, got {cp_size}")
        if not 0 <= cp_rank < cp_size:
            raise ValueError(f"Invalid cp_rank={cp_rank} for cp_size={cp_size}")
        if layer_shard_start_layer < 0 or layer_shard_layer_num <= 0:
            raise ValueError(
                "Invalid Cache LayerSplit stage: "
                f"start_layer={layer_shard_start_layer}, "
                f"layer_num={layer_shard_layer_num}"
            )

        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self._layer_shard_start_layer = layer_shard_start_layer
        self._layer_shard_layer_num = layer_shard_layer_num

    def _local_layer_idx(self, layer_id: int) -> int:
        local_layer_idx = layer_id - self._layer_shard_start_layer
        if not 0 <= local_layer_idx < self._layer_shard_layer_num:
            raise ValueError(
                f"Layer {layer_id} is outside Cache LayerSplit stage "
                f"[{self._layer_shard_start_layer}, "
                f"{self._layer_shard_start_layer + self._layer_shard_layer_num})"
            )
        return local_layer_idx

    def _owned_local_layer_range(self) -> tuple[int, int]:
        return get_layer_shard_range(
            self.cp_rank, self.cp_size, self._layer_shard_layer_num
        )

    def _owned_global_layer_range(self) -> tuple[int, int]:
        return get_global_layer_shard_range(
            self.cp_rank,
            self.cp_size,
            self._layer_shard_start_layer,
            self._layer_shard_layer_num,
        )

    def _is_layer_owned(self, layer_id: int) -> bool:
        local_idx = self._local_layer_idx(layer_id)
        owned_start, owned_end = self._owned_local_layer_range()
        return owned_start <= local_idx < owned_end

    def _get_layer_owner_rank(self, layer_id: int) -> int:
        return get_layer_owner(
            self._local_layer_idx(layer_id),
            self.cp_size,
            self._layer_shard_layer_num,
        )

    def _build_owned_layer_local_index_map(self) -> dict[int, int]:
        owned_start, owned_end = self._owned_global_layer_range()
        return {
            layer_id: layer_id - owned_start
            for layer_id in range(owned_start, owned_end)
        }

    def _log_layer_shard_plan(self) -> None:
        partitions = []
        for rank in range(self.cp_size):
            start, end = get_global_layer_shard_range(
                rank,
                self.cp_size,
                self._layer_shard_start_layer,
                self._layer_shard_layer_num,
            )
            partitions.append(f"r{rank}:[{start},{end})")
        owned_start, owned_end = self._owned_global_layer_range()
        logger.info(
            "Cache LayerSplit plan: stage=[%s,%s), cp_rank=%s, cp_size=%s, "
            "owned=[%s,%s), partitions=%s",
            self._layer_shard_start_layer,
            self._layer_shard_start_layer + self._layer_shard_layer_num,
            self.cp_rank,
            self.cp_size,
            owned_start,
            owned_end,
            "; ".join(partitions),
        )


def is_cp_cache_layer_split_pool(pool) -> bool:
    """True when ``pool`` participates in CP Cache LayerSplit (any model)."""
    return isinstance(pool, CpCacheLayerSplitPoolBase)
