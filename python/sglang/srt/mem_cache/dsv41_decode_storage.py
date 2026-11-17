"""Immutable physical-pool contract for native C1/C2 Decode storage restores."""

import hashlib
import json
from dataclasses import asdict, dataclass

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)


@dataclass(frozen=True)
class DecodeStoragePoolSpec:
    name: PoolName
    layer_ids: tuple[int, ...]
    page_slots: int
    page_bytes: int
    layout: str
    buffers: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]


@dataclass(frozen=True)
class DecodeStoragePoolPlan:
    page_size: int
    layout: str
    compression_ratios: tuple[int, ...]
    source_layers: tuple[int, ...]
    specs: tuple[DecodeStoragePoolSpec, ...]

    @property
    def pool_names(self) -> frozenset[PoolName]:
        return frozenset(spec.name for spec in self.specs)

    @property
    def fingerprint(self) -> str:
        descriptor = dict(schema="dsv41-decode-persistent-v1", **asdict(self))
        return hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def transfers(self) -> list[PoolTransfer]:
        # Indices/keys belong to an operation, never to the immutable plan.
        return [
            PoolTransfer(
                name=spec.name,
                indices_from_pool=PoolName.KV,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            )
            for spec in self.specs
        ]

    def validate_transfers(self, transfers: list[PoolTransfer]) -> None:
        names = [transfer.name for transfer in transfers]
        if (
            len(names) != len(set(names))
            or set(names) != self.pool_names
            or any(
                transfer.indices_from_pool != PoolName.KV
                or transfer.hit_policy != PoolHitPolicy.ALL_PAGES
                or transfer.block_pages != 0
                or transfer.group_pages != 0
                for transfer in transfers
            )
        ):
            raise ValueError(
                "Decode storage requires the exact persistent ALL_PAGES plan"
            )


def build_decode_storage_pool_plan(pool, page_size, sidecar_specs):
    """Freeze native page geometry without addresses, capacity or private state."""
    ratios = tuple(int(value) for value in pool.compression_ratios)
    if not any(value in (1, 2) for value in ratios) or set(ratios) - {0, 1, 2}:
        raise ValueError("Decode storage only supports native C1/C2 persistent pools")
    allowed = {
        PoolName.DEEPSEEK_V4_C1,
        PoolName.DEEPSEEK_V4_C1_INDEXER,
        PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE,
        PoolName.DEEPSEEK_V4_C2,
        PoolName.DEEPSEEK_V4_C2_INDEXER,
        PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE,
    }
    infos = [
        info
        for info in pool.get_device_pool_infos()
        if info.indices_from_pool == PoolName.KV
    ]
    names = [info.pool_name for info in infos]
    if not names or len(names) != len(set(names)) or set(names) - allowed:
        raise ValueError("Unknown, empty or duplicate Decode persistent pool schema")
    specs = []
    for info in infos:
        region = info.buffer_info
        if (
            region.page_slots <= 0
            or page_size % region.page_slots
            or region.page_bytes <= 0
            or len(region.buffers) != len(info.layer_ids)
        ):
            raise ValueError("Invalid Decode persistent physical-page geometry")
        specs.append(
            DecodeStoragePoolSpec(
                info.pool_name,
                tuple(info.layer_ids),
                int(region.page_slots),
                int(region.page_bytes),
                str(region.layout),
                tuple(
                    (str(buffer.dtype), tuple(buffer.shape[1:]), tuple(buffer.stride()))
                    for buffer in region.buffers
                ),
            )
        )
    plan = DecodeStoragePoolPlan(
        int(page_size),
        pool.kv_layout.value,
        ratios,
        tuple(pool.kv_source_layers),
        tuple(sorted(specs, key=lambda spec: str(spec.name))),
    )
    plan.validate_transfers(
        [
            PoolTransfer(
                name=spec.pool_name,
                indices_from_pool=spec.indices_from_pool,
                hit_policy=spec.hit_policy,
            )
            for spec in sidecar_specs
            if spec.indices_from_pool == PoolName.KV
        ]
    )
    return plan
