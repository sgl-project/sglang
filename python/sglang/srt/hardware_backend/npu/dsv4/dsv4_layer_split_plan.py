"""Pure layer-ownership math for the DSV4 NPU cache layer-split pool.

Kept free of torch / torch_npu imports so unit tests can exercise the shard
plan on CPU. See ``dsv4_cache_layer_split.py`` for the pool that consumes it.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from sglang.srt.layers.cp.utils import get_layer_owner, get_layer_shard_range


def owned_bucket_range(
    bucket_layer_ids: List[int], owned_start: int, owned_end: int
) -> Tuple[int, int]:
    """Owned ``[start, end)`` sub-range of a monotonic bucket-id list.

    ``bucket_layer_ids[i]`` is the stage-local layer id of bucket buffer ``i``;
    ownership over stage-local ids is a contiguous range, so the owned bucket
    indices are contiguous too.
    """
    start = 0
    while start < len(bucket_layer_ids) and bucket_layer_ids[start] < owned_start:
        start += 1
    end = start
    while end < len(bucket_layer_ids) and bucket_layer_ids[end] < owned_end:
        end += 1
    return start, end


class DSV4LayerShardPlan:
    """Contiguous layer->CP-rank ownership for one DSV4 pool instance.

    Layers are split as evenly as possible over stage-local ids (the first
    ``num_layers % shard_size`` ranks own one extra layer). Because a layer's
    per-bucket id is a monotonic function of its stage-local id, each bucket's
    owned set is a contiguous sub-range of that bucket's buffer list.
    """

    def __init__(
        self,
        *,
        rank: int,
        shard_size: int,
        num_layers: int,
        stage_start: int,
        ratios: List[int],
    ):
        self.rank = rank
        self.shard_size = shard_size
        self.num_layers = num_layers
        self.stage_start = stage_start
        self.owned_start, self.owned_end = get_layer_shard_range(
            rank, shard_size, num_layers
        )
        # SWA buffers are 1:1 with stage-local layer ids.
        self._bucket_ids: Dict[str, List[int]] = {
            "swa": list(range(num_layers)),
            "c4": [i for i, r in enumerate(ratios) if r == 4],
            "c128": [i for i, r in enumerate(ratios) if r == 128],
        }
        self.shard_start = stage_start + self.owned_start
        self.shard_end = stage_start + self.owned_end

    def is_stage_local_owned(self, local_layer_idx: int) -> bool:
        return self.owned_start <= local_layer_idx < self.owned_end

    def is_layer_owned(self, layer_id: int) -> bool:
        return self.is_stage_local_owned(layer_id - self.stage_start)

    def owner_rank(self, layer_id: int) -> int:
        return get_layer_owner(
            layer_id - self.stage_start, self.shard_size, self.num_layers
        )

    def bucket_layer_ids(self, bucket: str) -> List[int]:
        return self._bucket_ids[bucket]

    def owned_bucket_range(self, bucket: str) -> Tuple[int, int]:
        """``[start, end)`` of the owned sub-range of a bucket buffer list."""
        return owned_bucket_range(
            self._bucket_ids[bucket], self.owned_start, self.owned_end
        )

    def owned_stage_local_range(self) -> Tuple[int, int]:
        return self.owned_start, self.owned_end

    def owned_stage_local_ids(self, bucket: str) -> List[int]:
        start, end = self.owned_bucket_range(bucket)
        return self._bucket_ids[bucket][start:end]

    def partition_summary(self) -> str:
        return "; ".join(
            f"r{r}:{get_layer_shard_range(r, self.shard_size, self.num_layers)}"
            for r in range(self.shard_size)
        )
