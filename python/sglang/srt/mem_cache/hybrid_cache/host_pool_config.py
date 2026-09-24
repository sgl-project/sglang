"""Pure HiCache declaration validation and transfer-layer binding."""

from __future__ import annotations

from typing import Any, Optional

import msgspec

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.pool_host.host_pool_decl import HostPoolDecl


class LayerBinding(msgspec.Struct, frozen=True, kw_only=True):
    """Transfer layer (what the controller iterates) to device pool layer.

    Compact host layer indices stay inside the host pool; packed draft
    remapping stays inside the controller.
    """

    transfer_to_device: dict[int, int]
    # Exclusive upper bound; transfer IDs may contain holes.
    transfer_layer_id_max: int


class HostPoolBuildConfig(msgspec.Struct, frozen=True, kw_only=True):
    """A declaration bound to one stack: what the assembler builds an entry from."""

    decl: HostPoolDecl
    layer_binding: LayerBinding
    # Draft pools whose same-named buffers are appended as tail layers, in depth order.
    packed_draft_device_pools: tuple[Any, ...] = ()


def validate_packed_draft_pools(
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
    ordered = []
    pending = list(decls)
    built = set()
    while pending:
        ready = [d for d in pending if d.is_layout_root or d.layout_source in built]
        if not ready:
            raise ValueError(
                f"cyclic layout_source dependencies: {[d.pool_name for d in pending]}"
            )
        for decl in ready:
            ordered.append(decl)
            built.add(decl.pool_name)
            pending.remove(decl)
    return tuple(ordered)


def prepare_host_pool_configs(
    *,
    decls: tuple[HostPoolDecl, ...],
    full_layer_mapping: dict[int, int],
    transfer_layer_id_max: int,
    packed_draft_decls: tuple[tuple[HostPoolDecl, ...], ...] = (),
) -> tuple[HostPoolBuildConfig, ...]:
    """Bind declarations to a stack: the primary KV pool is the one layout root
    that anchors the others' capacity, and every sidecar takes its indices from
    it (HostPoolGroup resolves no chains).

    ``packed_draft_decls`` have passed validate_packed_draft_pools; each config
    carries the draft objects that own its same-named buffers."""
    names = [d.pool_name for d in decls]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate host pool names: {names}")
    root = layout_root(decls)
    if not root.is_primary or root.pool_name != PoolName.KV:
        raise ValueError(
            f"expected the layout root to be the primary KV pool, got {root.pool_name}"
        )
    for d in decls:
        if d.is_primary:
            continue
        if d.indices_from_pool != root.pool_name:
            raise ValueError(
                f"{d.pool_name}.indices_from_pool must be {root.pool_name}, "
                f"got {d.indices_from_pool}"
            )
        if d.is_layout_root:
            continue
        if d.layout_source == d.pool_name:
            raise ValueError(f"{d.pool_name}.layout_source must name another pool")
        if d.layout_source not in names:
            raise ValueError(
                f"{d.pool_name} references undeclared pool {d.layout_source}"
            )
    decls = _validate_and_order_decls(decls)
    target_layers = root.device_pool.layer_num
    device_layer_limit = target_layers + len(packed_draft_decls)
    for transfer_id, device_id in full_layer_mapping.items():
        if not 0 <= transfer_id < transfer_layer_id_max:
            raise ValueError(
                f"transfer layer {transfer_id} outside [0, {transfer_layer_id_max})"
            )
        if not 0 <= device_id < device_layer_limit:
            raise ValueError(
                f"device layer {device_id} outside [0, {device_layer_limit})"
            )
    return tuple(
        HostPoolBuildConfig(
            decl=d,
            layer_binding=LayerBinding(
                transfer_to_device=_filter_owned_layer_mapping(
                    mapping=full_layer_mapping,
                    owned_device_layers=d.owned_device_layers,
                    target_layer_num=root.device_pool.layer_num,
                ),
                transfer_layer_id_max=transfer_layer_id_max,
            ),
            packed_draft_device_pools=tuple(
                _find_pool_decl(decls=decls_of_draft, name=d.pool_name).device_pool
                for decls_of_draft in packed_draft_decls
            ),
        )
        for d in decls
    )


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
