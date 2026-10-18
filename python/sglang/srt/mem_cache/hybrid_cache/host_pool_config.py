"""HiCache capacity and transfer bindings prepared before host allocation.

The root keeps its existing KV host constructor. Sidecars consume captured
buffer information and never ask a device pool to describe host construction.
"""

from __future__ import annotations

from typing import Any, Optional

import msgspec

from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo, IndexKeyBufferInfo
from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec


class HostPoolConfig(msgspec.Struct, frozen=True, kw_only=True):
    """One sidecar's packed buffers and device/controller layer coordinates."""

    pool_name: PoolName
    buffer_info: IndexKeyBufferInfo
    device_to_host_layer: dict[int, int]
    layer_mapping: dict[int, int]

    def sidecar_spec(self) -> SidecarPoolSpec:
        return SidecarPoolSpec(pool_name=self.pool_name, indices_from_pool=PoolName.KV)


class HostPoolGroupConfig(msgspec.Struct, frozen=True, kw_only=True):
    root: Any
    transfer_page_size: int
    layer_mapping: dict[int, int]
    transfer_layer_id_max: int
    packed_draft_device_pools: tuple[Any, ...]
    sidecars: tuple[HostPoolConfig, ...]

    @property
    def pool_names(self) -> tuple[PoolName, ...]:
        return (PoolName.KV, *(config.pool_name for config in self.sidecars))


def with_packed_draft_layer_mapping(
    layer_mapping: dict[int, int],
    *,
    transfer_layer_start: int,
    target_device_layer_num: int,
    draft_layer_num: int,
) -> dict[int, int]:
    return layer_mapping | {
        transfer_layer_start + depth: target_device_layer_num + depth
        for depth in range(draft_layer_num)
    }


def collect_host_pool_infos(root: Any) -> tuple[DevicePoolInfo, ...]:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    # Other KV formats still use their existing host constructor.
    return root.get_device_pool_infos() if isinstance(root, DSATokenToKVPool) else ()


def _pool_infos_by_name(
    infos: tuple[DevicePoolInfo, ...],
) -> dict[PoolName, DevicePoolInfo]:
    pools = {info.pool_name: info for info in infos}
    if len(pools) != len(infos):
        raise ValueError(
            f"duplicate device pool names: {[info.pool_name for info in infos]}"
        )
    if not pools:
        return pools
    if PoolName.KV not in pools:
        raise ValueError(f"expected primary KV pool, got {tuple(pools)}")
    for info in infos:
        if info.indices_from_pool != PoolName.KV:
            raise ValueError(
                f"{info.pool_name}: expected indices_from_pool=KV, got {info.indices_from_pool}"
            )
        if info.pool_name != PoolName.KV and not isinstance(
            info.buffer_info, IndexKeyBufferInfo
        ):
            raise ValueError(
                f"{info.pool_name}: unsupported host buffer type {type(info.buffer_info).__name__}"
            )
    return pools


def _prepare_sidecar(
    target: DevicePoolInfo,
    drafts: tuple[DevicePoolInfo, ...],
    *,
    model_layer_ids: tuple[int, ...],
    owned_layer_range: tuple[int, int],
    layer_mapping: dict[int, int],
) -> HostPoolConfig:
    buffers = target.buffer_info
    if not isinstance(buffers, IndexKeyBufferInfo):
        raise TypeError(
            f"{target.pool_name}: expected index buffers, got {type(buffers).__name__}"
        )
    if len(target.layer_ids) != len(buffers.buffers):
        raise ValueError(
            f"{target.pool_name}: {len(target.layer_ids)} layers for {len(buffers.buffers)} buffers"
        )
    model_to_device = {layer: i for i, layer in enumerate(model_layer_ids)}
    if any(layer not in model_to_device for layer in target.layer_ids):
        raise ValueError(
            f"{target.pool_name}: layer_ids={target.layer_ids} outside KV layers {model_layer_ids}"
        )
    start, end = owned_layer_range
    owned = [
        (model_to_device[layer], buffer)
        for layer, buffer in zip(target.layer_ids, buffers.buffers)
        if start <= model_to_device[layer] < end
    ]
    packed_buffers = [buffer for _, buffer in owned]
    device_to_host = {layer: i for i, (layer, _) in enumerate(owned)}
    transfer_layers = {model_to_device[layer] for layer in target.layer_ids}
    for depth, draft in enumerate(drafts):
        draft_buffers = draft.buffer_info
        if not isinstance(draft_buffers, IndexKeyBufferInfo):
            raise TypeError(
                f"{target.pool_name}: draft[{depth}] has buffer type {type(draft_buffers).__name__}"
            )
        if len(draft.layer_ids) != 1 or len(draft_buffers.buffers) != 1:
            raise ValueError(
                f"{target.pool_name}: draft[{depth}] requires one owned layer, got {draft.layer_ids}"
            )
        if (
            draft_buffers.page_size,
            draft_buffers.compress_ratio,
            draft_buffers.key_format,
            draft_buffers.kernel_page_bytes,
        ) != (
            buffers.page_size,
            buffers.compress_ratio,
            buffers.key_format,
            buffers.kernel_page_bytes,
        ):
            raise ValueError(
                f"{target.pool_name}: draft[{depth}] index key format differs from target"
            )
        device_to_host[len(model_layer_ids) + depth] = len(packed_buffers)
        transfer_layers.add(len(model_layer_ids) + depth)
        packed_buffers.extend(draft_buffers.buffers)
    packed = msgspec.structs.replace(buffers, buffers=tuple(packed_buffers))
    if packed.kernel_page_bytes % packed.page_size:
        raise ValueError(
            f"{target.pool_name}: kernel_page_bytes={packed.kernel_page_bytes} must split over page_size={packed.page_size}"
        )
    # A layer-sharded rank can own no index buffers but still joins the group.
    if packed.buffers:
        packed.validate()
    return HostPoolConfig(
        pool_name=target.pool_name,
        buffer_info=packed,
        device_to_host_layer=device_to_host,
        layer_mapping={
            transfer: layer
            for transfer, layer in layer_mapping.items()
            if layer in transfer_layers
        },
    )


def prepare_host_pool_config(
    *,
    root: Any,
    full_layer_mapping: dict[int, int],
    transfer_layer_id_max: int,
    transfer_page_size: int,
    packed_draft_device_pools: tuple[Any, ...] = (),
) -> HostPoolGroupConfig:
    """Capture each provider once, check packing, and bind before allocating host memory."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    # Hybrid drafts pack their full-attention KV child into the target KV host.
    packed_draft_device_pools = tuple(
        pool.full_kv_pool if isinstance(pool, HybridLinearKVPool) else pool
        for pool in packed_draft_device_pools
    )
    for transfer_id, device_id in full_layer_mapping.items():
        if not 0 <= transfer_id < transfer_layer_id_max:
            raise ValueError(
                f"transfer layer {transfer_id} outside [0, {transfer_layer_id_max})"
            )
        if not 0 <= device_id < root.layer_num:
            raise ValueError(f"device layer {device_id} outside [0, {root.layer_num})")
    targets = _pool_infos_by_name(collect_host_pool_infos(root))
    drafts = tuple(
        _pool_infos_by_name(collect_host_pool_infos(pool))
        for pool in packed_draft_device_pools
    )
    for pool, pools in zip(packed_draft_device_pools, drafts):
        if set(pools) != set(targets):
            raise ValueError(
                f"packed draft pools {tuple(pools)} differ from target {tuple(targets)}"
            )
        if pool.layer_num != 1:
            raise ValueError(
                f"packed draft requires exactly one KV layer per draft pool, got {pool.layer_num}"
            )
    check_packed_kv_rows(kv_pool=root, drafts=packed_draft_device_pools)
    mapping = with_packed_draft_layer_mapping(
        full_layer_mapping,
        transfer_layer_start=transfer_layer_id_max,
        target_device_layer_num=root.layer_num,
        draft_layer_num=len(drafts),
    )
    owned_range = (
        root._owned_local_layer_range()
        if root.layer_shard_enabled
        else (0, root.layer_num)
    )
    sidecars = tuple(
        _prepare_sidecar(
            info,
            tuple(pools[name] for pools in drafts),
            model_layer_ids=root.model_layer_ids,
            owned_layer_range=owned_range,
            layer_mapping=mapping,
        )
        for name, info in targets.items()
        if name != PoolName.KV
    )
    return HostPoolGroupConfig(
        root=root,
        transfer_page_size=transfer_page_size,
        layer_mapping=mapping,
        transfer_layer_id_max=transfer_layer_id_max + len(drafts),
        packed_draft_device_pools=packed_draft_device_pools,
        sidecars=sidecars,
    )


def is_mla_pool(pool: Any) -> bool:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(pool, MLATokenToKVPool)


def _kv_row_signature(pool: Any, *, use_mla: bool) -> tuple:
    if use_mla:
        return (pool.kv_cache_dim, pool.store_dtype)
    return (pool.head_num, pool.head_dim, pool.v_head_dim, pool.store_dtype)


def check_packed_kv_rows(
    *, kv_pool: Any, drafts: tuple[Any, ...], use_mla: Optional[bool] = None
):
    if not drafts:
        return
    if use_mla is None:
        use_mla = is_mla_pool(kv_pool)
    target = _kv_row_signature(kv_pool, use_mla=use_mla)
    for draft in drafts:
        row = _kv_row_signature(draft, use_mla=use_mla)
        if row != target:
            raise ValueError(
                f"packed draft KV row {row} differs from target KV row {target}"
            )
