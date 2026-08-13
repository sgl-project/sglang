"""Model-neutral owner-sharded family construction and accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout
from sglang.srt.mem_cache.shared_kv.vmm import (
    RankMajorSharedSlab,
    _synchronize_vmm_stage,
    create_rank_major_shared_slab,
)


@dataclass(frozen=True)
class OwnerShardedFamilySpec:
    name: str
    num_layers: int
    logical_rows_per_layer: int
    ownership_granule: int
    storage_rows_per_granule: int
    row_shape: tuple[int, ...]
    dtype: torch.dtype
    map_rank_local: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.logical_rows_per_layer < 0:
            raise ValueError(
                "logical_rows_per_layer must be non-negative, "
                f"got {self.logical_rows_per_layer}"
            )
        if self.ownership_granule <= 0:
            raise ValueError(
                "ownership_granule must be positive, " f"got {self.ownership_granule}"
            )
        if self.storage_rows_per_granule <= 0:
            raise ValueError(
                "storage_rows_per_granule must be positive, "
                f"got {self.storage_rows_per_granule}"
            )
        if not self.row_shape or any(dimension <= 0 for dimension in self.row_shape):
            raise ValueError(
                f"row_shape must contain positive dimensions: {self.row_shape}"
            )


@dataclass(frozen=True)
class SharedFamilyAccounting:
    name: str
    logical_blocks_per_layer: int
    minimum_blocks_per_rank: int
    physical_blocks_per_rank: int
    logical_storage_bytes: int
    minimum_physical_bytes_per_rank: int
    mapped_bytes_per_rank: int
    alignment_overhead_bytes_per_rank: int


@dataclass
class OwnerShardedFamily:
    spec: OwnerShardedFamilySpec
    layout: OwnerShardedLayout
    slab: RankMajorSharedSlab
    _closed: bool = field(default=False, init=False)

    @classmethod
    def create(
        cls,
        *,
        spec: OwnerShardedFamilySpec,
        cp_size: int,
        cpu_group: ProcessGroup,
        zero_initialize: bool = True,
    ) -> OwnerShardedFamily:
        group_size = dist.get_world_size(group=cpu_group)
        if cp_size != group_size:
            raise ValueError(
                "Shared family cp_size must match the CPU process group size: "
                f"cp_size={cp_size}, group_size={group_size}"
            )
        minimum_layout = OwnerShardedLayout(
            cp_size=cp_size,
            ownership_granule=spec.ownership_granule,
            logical_rows=spec.logical_rows_per_layer,
        )
        storage_rows_per_layer = (
            minimum_layout.minimum_blocks_per_rank * spec.storage_rows_per_granule
        )
        slab = create_rank_major_shared_slab(
            (storage_rows_per_layer, *spec.row_shape),
            layer_num=spec.num_layers,
            dtype=spec.dtype,
            cpu_group=cpu_group,
            first_dim_multiple=spec.storage_rows_per_granule,
            map_rank_local=spec.map_rank_local,
        )
        try:
            if slab.rank_stride_rows % spec.storage_rows_per_granule != 0:
                raise RuntimeError(
                    f"Shared family {spec.name} rank stride {slab.rank_stride_rows} "
                    "is not divisible by storage_rows_per_granule "
                    f"{spec.storage_rows_per_granule}"
                )

            layout = OwnerShardedLayout(
                cp_size=cp_size,
                ownership_granule=spec.ownership_granule,
                logical_rows=spec.logical_rows_per_layer,
                physical_blocks_per_rank=(
                    slab.rank_stride_rows // spec.storage_rows_per_granule
                ),
            )
            if zero_initialize:
                initialization_error = None
                try:
                    slab.allocation.local_view.zero_()
                    torch.cuda.synchronize()
                except BaseException as error:
                    initialization_error = error
                _synchronize_vmm_stage(
                    cpu_group,
                    dist.get_rank(group=cpu_group),
                    "family zero initialization",
                    initialization_error,
                )
            return cls(spec=spec, layout=layout, slab=slab)
        except BaseException:
            slab.close()
            raise

    def _checked_layer(self, layer: int) -> int:
        if layer < 0 or layer >= self.spec.num_layers:
            raise IndexError(
                f"layer must be in [0, {self.spec.num_layers}), got {layer}"
            )
        return layer

    def layer_global(self, layer: int) -> torch.Tensor:
        return self.slab.global_views[self._checked_layer(layer)]

    def layer_rank_relative(self, layer: int) -> torch.Tensor:
        layer = self._checked_layer(layer)
        if not self.slab.rank_local_views:
            raise RuntimeError(
                f"Shared family {self.spec.name} has no rank-relative VMM alias"
            )
        return self.slab.rank_local_views[layer]

    def layer_owner_local(self, layer: int) -> torch.Tensor:
        return self.slab.local_views[self._checked_layer(layer)]

    def accounting(self) -> SharedFamilyAccounting:
        row_bytes = (
            math.prod(self.spec.row_shape)
            * torch.empty((), dtype=self.spec.dtype).element_size()
        )
        logical_storage_bytes = (
            self.spec.num_layers
            * self.layout.logical_blocks
            * self.spec.storage_rows_per_granule
            * row_bytes
        )
        minimum_physical_bytes = (
            self.spec.num_layers
            * self.layout.minimum_blocks_per_rank
            * self.spec.storage_rows_per_granule
            * row_bytes
        )
        mapped_bytes = int(self.slab.allocation.aligned_bytes_per_rank)
        return SharedFamilyAccounting(
            name=self.spec.name,
            logical_blocks_per_layer=self.layout.logical_blocks,
            minimum_blocks_per_rank=self.layout.minimum_blocks_per_rank,
            physical_blocks_per_rank=self.layout.blocks_per_rank,
            logical_storage_bytes=logical_storage_bytes,
            minimum_physical_bytes_per_rank=minimum_physical_bytes,
            mapped_bytes_per_rank=mapped_bytes,
            alignment_overhead_bytes_per_rank=(mapped_bytes - minimum_physical_bytes),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.slab.close()
