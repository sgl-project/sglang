"""Pure HiCache declaration validation and transfer-layer binding."""

from __future__ import annotations

from typing import Any, Optional

import msgspec

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.pool_host.host_pool_decl import HostPoolDecl


class HostPoolConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Validated device owners and transfer layers for one host pool."""

    decl: HostPoolDecl
    layer_mapping: dict[int, int]
    # Exclusive upper bound, including packed draft tails.
    transfer_layer_id_max: int
    # Draft pools whose same-named buffers are appended as tail layers, in depth order.
    packed_draft_device_pools: tuple[Any, ...] = ()


class HostPoolGroupConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Validated host pools and their shared transfer granularity."""

    # Original-token slots moved together. Individual buffers may use fewer rows.
    transfer_page_size: int
    # Dependency order with the unique primary KV root first.
    pools: tuple[HostPoolConfig, ...]


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


def _validate_packed_draft_pools(
    *, target_decls: tuple[HostPoolDecl, ...], draft_pools: tuple[Any, ...]
) -> tuple[tuple[HostPoolDecl, ...], ...]:
    """Collect draft declarations once and check they can be appended to target host
    pools: it declares the same pools with the same storage info and every
    packed layer owns its buffer. The draft plan already chose packing, so a
    draft that cannot be packed is an error, not a silent skip."""
    targets = {d.pool_name: d for d in target_decls}
    collected = []
    for pool in draft_pools:
        declarations = pool.host_pool_decls()
        collected.append(declarations)
        drafts = {d.pool_name: d for d in declarations}
        if len(drafts) != len(declarations):
            raise ValueError("packed draft declares duplicate host pool names")
        if set(drafts) != set(targets):
            raise ValueError(
                f"packed draft {type(pool).__name__} declares "
                f"{sorted(d.value for d in drafts)} but the target declares "
                f"{sorted(d.value for d in targets)}; every target pool needs a "
                "draft counterpart or the draft buffers are not restored"
            )
        for name, target in targets.items():
            draft = drafts[name]
            if draft.storage_info != target.storage_info:
                raise ValueError(
                    f"packed draft {name.value} storage {draft.storage_info} "
                    f"differs from target {target.storage_info}"
                )
            owned = draft.owned_device_layers
            if owned is not None and (
                len(set(owned)) != draft.device_pool.layer_num
                or set(owned) != set(range(draft.device_pool.layer_num))
            ):
                raise ValueError(
                    f"packed draft {name.value} owns buffers on {len(owned)} of "
                    f"{draft.device_pool.layer_num} layers; every packed layer "
                    "must own its buffer"
                )
        if layout_root(declarations).device_pool.layer_num != 1:
            raise ValueError(
                "packed draft requires exactly one KV layer per draft pool"
            )
    return tuple(collected)


def layout_root(decls: tuple[HostPoolDecl, ...]) -> HostPoolDecl:
    """The one declaration whose host pool decides the group's capacity."""
    roots = [d for d in decls if d.is_layout_root]
    if len(roots) != 1:
        raise ValueError(
            f"expected exactly one layout root, got {[d.pool_name for d in roots]}"
        )
    return roots[0]


def _find_pool_decl(*, decls: tuple[HostPoolDecl, ...], name: PoolName) -> HostPoolDecl:
    return next(d for d in decls if d.pool_name == name)


def _validate_and_order_decls(
    decls: tuple[HostPoolDecl, ...],
) -> tuple[HostPoolDecl, ...]:
    names = [decl.pool_name for decl in decls]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate host pool names: {names}")
    root = layout_root(decls)
    if not root.is_primary or root.pool_name != PoolName.KV:
        raise ValueError(
            f"expected the layout root to be the primary KV pool, got {root.pool_name}"
        )
    for decl in decls:
        if not decl.is_layout_root and (
            decl.host_pool_builder is None or decl.storage_info is None
        ):
            raise ValueError(
                f"{decl.pool_name}: dependent pool needs builder and storage_info"
            )
        owned = decl.owned_device_layers
        if owned is not None and (
            len(set(owned)) != len(owned)
            or any(layer < 0 or layer >= decl.device_pool.layer_num for layer in owned)
        ):
            raise ValueError(f"{decl.pool_name}: invalid owned_device_layers {owned}")
        if decl.is_primary:
            continue
        if decl.indices_from_pool != root.pool_name:
            raise ValueError(
                f"{decl.pool_name}.indices_from_pool must be {root.pool_name}, "
                f"got {decl.indices_from_pool}"
            )
        if decl.is_layout_root:
            continue
        if decl.layout_source == decl.pool_name:
            raise ValueError(f"{decl.pool_name}.layout_source must name another pool")
        if decl.layout_source not in names:
            raise ValueError(
                f"{decl.pool_name} references undeclared pool {decl.layout_source}"
            )
    # Topologically sort by layout_source so dependencies are built first.
    ordered = []
    pending = list(decls)
    built = set()
    while pending:
        ready = [
            decl
            for decl in pending
            if decl.is_layout_root or decl.layout_source in built
        ]
        if not ready:
            raise ValueError(
                f"cyclic layout_source dependencies: {[decl.pool_name for decl in pending]}"
            )
        for decl in ready:
            ordered.append(decl)
            built.add(decl.pool_name)
            pending.remove(decl)
    return tuple(ordered)


def prepare_host_pool_config(
    *,
    decls: tuple[HostPoolDecl, ...],
    full_layer_mapping: dict[int, int],
    transfer_layer_id_max: int,
    transfer_page_size: int,
    packed_draft_device_pools: tuple[Any, ...] = (),
) -> HostPoolGroupConfig:
    """Validate declarations and buffers, then append packed draft transfer layers.

    The input mapping and limit describe target layers only. No host memory is
    allocated here. config.pools are in host construction order.
    """
    decls = _validate_and_order_decls(decls)
    root = decls[0]
    target_layers = root.device_pool.layer_num
    for transfer_id, device_id in full_layer_mapping.items():
        if not 0 <= transfer_id < transfer_layer_id_max:
            raise ValueError(
                f"transfer layer {transfer_id} outside [0, {transfer_layer_id_max})"
            )
        if not 0 <= device_id < target_layers:
            raise ValueError(f"device layer {device_id} outside [0, {target_layers})")
    packed_draft_decls = _validate_packed_draft_pools(
        target_decls=decls, draft_pools=packed_draft_device_pools
    )
    mapping = with_packed_draft_layer_mapping(
        full_layer_mapping,
        transfer_layer_start=transfer_layer_id_max,
        target_device_layer_num=target_layers,
        draft_layer_num=len(packed_draft_decls),
    )
    configs = tuple(
        HostPoolConfig(
            decl=d,
            layer_mapping=_filter_owned_layer_mapping(
                mapping=mapping,
                owned_device_layers=d.owned_device_layers,
                target_layer_num=target_layers,
            ),
            transfer_layer_id_max=transfer_layer_id_max + len(packed_draft_decls),
            packed_draft_device_pools=tuple(
                _find_pool_decl(decls=decls_of_draft, name=d.pool_name).device_pool
                for decls_of_draft in packed_draft_decls
            ),
        )
        for d in decls
    )
    check_packed_kv_rows(
        kv_pool=root.device_pool,
        drafts=configs[0].packed_draft_device_pools,
    )
    for config in configs[1:]:
        config.decl.host_pool_builder.validate(
            decl=config.decl,
            transfer_page_size=transfer_page_size,
            packed_draft_device_pools=config.packed_draft_device_pools,
        )
    return HostPoolGroupConfig(transfer_page_size=transfer_page_size, pools=configs)


def _filter_owned_layer_mapping(
    *,
    mapping: dict[int, int],
    owned_device_layers: Optional[tuple[int, ...]],
    target_layer_num: int,
) -> dict[int, int]:
    """Drop transfer layers whose device layer owns no buffer for this pool.
    Packed draft tails (device index >= target layer count) are kept."""
    if owned_device_layers is None:
        return mapping
    owned = set(owned_device_layers)
    return {
        t: dev for t, dev in mapping.items() if dev >= target_layer_num or dev in owned
    }


def is_mla_pool(pool: Any) -> bool:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(pool, MLATokenToKVPool)


def _kv_row_signature(pool: Any, *, use_mla: bool) -> tuple:
    if use_mla:
        return (pool.kv_cache_dim, pool.store_dtype)
    return (pool.head_num, pool.head_dim, pool.v_head_dim, pool.store_dtype)


def packed_kv_rows_match(
    *, kv_pool: Any, drafts: tuple[Any, ...], use_mla: Optional[bool] = None
) -> bool:
    """Whether every draft KV row has the target's shape and dtype."""
    if use_mla is None:
        use_mla = is_mla_pool(kv_pool)
    target = _kv_row_signature(kv_pool, use_mla=use_mla)
    return all(_kv_row_signature(d, use_mla=use_mla) == target for d in drafts)


def check_packed_kv_rows(
    *, kv_pool: Any, drafts: tuple[Any, ...], use_mla: Optional[bool] = None
):
    """Packed draft KV layers share the target's host row, so their device
    rows must have the same shape and dtype."""
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
