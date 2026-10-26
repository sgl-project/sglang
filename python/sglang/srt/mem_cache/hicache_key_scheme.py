# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unified L3 key scheme (``--hicache-storage-key-scheme unified``).

Replaces the rank/topology key suffixes (``_{tp_rank}_{tp_size}``,
``_{pp_size}_{pp_rank}``, ``_cp{r}_{s}``) with one topology-free coordinate::

    {page_hash}_{digest}_L{start}-{end}[_H{head_group}]

The coordinate names what an object HOLDS -- a layer-range x head-range
rectangle of one page, both K and V -- never who wrote it, so any deployment
whose shard tiles the grid derives the same keys for the same bytes. A TP2
rank's objects are exactly the union of two TP4 ranks'; a PP1 rank's are the
union of the stages'.

Two fleet-wide agreements set the grid: ``head_group`` (kv heads per chunk) and
``layer_partition`` (layers per chunk). They are not local tuning knobs --
they fix chunk boundaries, and boundaries are namespace identity. Left unset,
each falls back to this rank's own shard, which shares only with its own
topology.

The digest prefixes every key, so any identity mismatch misses rather than
collides. dtype is the LOGICAL one: fp8_e4m3 and fp8_e5m2 never share a
keyspace even though both store as uint8.
"""

from __future__ import annotations

import hashlib
import logging

import msgspec

from sglang.srt.mem_cache.pool_host.page_unified import PAGE_UNIFIED_OBJECT_LAYOUT

logger = logging.getLogger(__name__)

# The digest covers the encoded struct, so any schema change changes every key.
# Bump on any field change.
_SCHEMA_VERSION = 1


class KVCacheNamespace(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Immutable identity of one shared L3 KV keyspace.

    Everything that must match for two deployments' KV bytes to be
    interchangeable. Field names and order are part of the encoding: append
    only, and bump ``schema_version``.
    """

    schema_version: int = _SCHEMA_VERSION
    model_id: str
    # Logical dtype, not the storage view: fp8 variants all store as uint8
    # and must not share a keyspace.
    dtype: str
    page_size: int
    # MLA-family pools: KV is replicated across attn-TP ranks, so there is no
    # head axis (total_kv_heads and head_group are 0).
    rank_replicated: bool
    total_kv_heads: int
    # Layers per chunk; 0 = this rank's own range (same-split sharing only).
    # A stage must START on a multiple of it; only the last PP stage may end
    # short, forming the model's trailing remainder chunk.
    layer_partition: int = 0
    head_group: int
    # Byte order of the stored objects. Layouts serialize a page in different
    # orders at EQUAL sizes, so omitting this would let two deployments
    # exchange byte-permuted KV under identical keys.
    object_layout: str


def namespace_digest(namespace: KVCacheNamespace) -> str:
    """Digest of the namespace encoding, used as the key prefix.

    msgpack encoding of a Struct is deterministic given the class definition,
    which is why the schema versions the encoding.
    """
    encoded = msgspec.msgpack.encode(namespace)
    return f"ukv{_SCHEMA_VERSION}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def derive_namespace(
    *,
    model_id: str,
    dtype: str,
    page_size: int,
    rank_replicated: bool,
    total_kv_heads: int,
    head_group: int,
    layer_partition: int = 0,
    object_layout: str = PAGE_UNIFIED_OBJECT_LAYOUT,
) -> KVCacheNamespace:
    """Derive the namespace from deployment facts plus the fleet agreements."""
    namespace = KVCacheNamespace(
        model_id=model_id,
        dtype=dtype,
        page_size=page_size,
        rank_replicated=rank_replicated,
        total_kv_heads=0 if rank_replicated else total_kv_heads,
        head_group=0 if rank_replicated else head_group,
        layer_partition=layer_partition,
        object_layout=object_layout,
    )
    _validate_grid(namespace)
    return namespace


def _validate_grid(namespace: KVCacheNamespace) -> None:
    if namespace.layer_partition < 0:
        raise ValueError(f"layer_partition must be non-negative: {namespace}")
    if namespace.page_size <= 0:
        raise ValueError(f"page_size must be positive: {namespace}")
    if not namespace.model_id:
        raise ValueError(
            "the unified key scheme requires a non-empty model_id: an empty id "
            "would merge different models into one keyspace."
        )
    if namespace.rank_replicated:
        if namespace.head_group != 0 or namespace.total_kv_heads != 0:
            raise ValueError(
                f"rank_replicated namespaces have no head axis; set "
                f"total_kv_heads=0 and head_group=0: {namespace}"
            )
        return
    if namespace.head_group <= 0 or namespace.total_kv_heads <= 0:
        raise ValueError(
            f"sharded-KV namespaces need positive total_kv_heads/head_group: "
            f"{namespace}"
        )
    if namespace.total_kv_heads % namespace.head_group != 0:
        raise ValueError(
            f"head_group={namespace.head_group} must divide "
            f"total_kv_heads={namespace.total_kv_heads}."
        )


def normalize_dtype(dtype: object) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``."""
    return str(dtype).removeprefix("torch.")


class UnifiedKVPlan(msgspec.Struct, frozen=True, kw_only=True):
    """One rank's unified-key objects and where their bytes sit in a page.

    ``suffixes`` is layer-major / head-minor, and ``layer_ranges`` x
    ``head_ranges`` is the same cross product in the same order: entry
    ``i * len(head_ranges) + j`` of ``suffixes`` names
    ``(layer_ranges[i], head_ranges[j])``. Everything downstream relies on that
    pairing, so the two are built together here rather than re-derived.

    The ranges are LOCAL to this rank's pool, so they index the host pool
    directly; the suffixes carry the absolute, model-global coordinates.
    """

    namespace: KVCacheNamespace
    suffixes: list[str]
    layer_ranges: list[tuple[int, int]]
    # Head groups, or a single whole-pool range for rank-replicated pools.
    head_ranges: list[tuple[int, int]]

    @property
    def head_group_num(self) -> int:
        return len(self.head_ranges)


def plan_unified_kv(
    *,
    model_id: str,
    dtype: str,
    page_size: int,
    rank_replicated: bool,
    local_kv_heads: int,
    attn_tp_rank: int,
    attn_tp_size: int,
    attn_cp_size: int,
    start_layer: int,
    end_layer: int,
    is_final_stage: bool,
    head_group: int | None = None,
    layer_partition: int | None = None,
) -> UnifiedKVPlan:
    """Derive the namespace and this rank's chunk plan from deployment facts.

    Raises rather than degrading: a rank whose shard does not tile the declared
    grid would write objects a reader cannot reassemble, and the symptom would
    be silently wrong KV rather than a miss.
    """
    if attn_cp_size > 1:
        raise NotImplementedError(
            "the unified key scheme does not support attention context "
            "parallelism yet: NSA-CP ranks hold sub-page slices (needs the "
            "token-granule extension) and replicated-CP needs writer election. "
            "Use --hicache-storage-key-scheme rank-suffix."
        )
    if not 0 <= start_layer < end_layer:
        raise ValueError(f"invalid layer range [{start_layer}, {end_layer}).")

    total_kv_heads = 0 if rank_replicated else local_kv_heads * attn_tp_size
    resolved_head_group = _resolve_head_group(
        rank_replicated=rank_replicated,
        local_kv_heads=local_kv_heads,
        attn_tp_size=attn_tp_size,
        total_kv_heads=total_kv_heads,
        head_group=head_group,
    )
    namespace = derive_namespace(
        model_id=model_id,
        dtype=dtype,
        page_size=page_size,
        rank_replicated=rank_replicated,
        total_kv_heads=total_kv_heads,
        head_group=resolved_head_group,
        layer_partition=layer_partition or 0,
    )

    layer_coords, layer_ranges = _layer_chunks(
        namespace.layer_partition, start_layer, end_layer, is_final_stage
    )
    digest = namespace_digest(namespace)
    if rank_replicated:
        return UnifiedKVPlan(
            namespace=namespace,
            suffixes=[f"{digest}_{coord}" for coord in layer_coords],
            layer_ranges=layer_ranges,
            head_ranges=[(0, 1)],
        )

    if local_kv_heads % resolved_head_group != 0:
        raise ValueError(
            f"this rank's {local_kv_heads} kv heads do not tile "
            f"head_group={resolved_head_group}."
        )
    groups_per_rank = local_kv_heads // resolved_head_group
    first_head_index = attn_tp_rank * groups_per_rank
    return UnifiedKVPlan(
        namespace=namespace,
        # Layer-major / head-minor, the order the page block packs them in.
        suffixes=[
            f"{digest}_{coord}_H{first_head_index + i}"
            for coord in layer_coords
            for i in range(groups_per_rank)
        ],
        layer_ranges=layer_ranges,
        head_ranges=[
            (i * resolved_head_group, (i + 1) * resolved_head_group)
            for i in range(groups_per_rank)
        ],
    )


def _resolve_head_group(
    *,
    rank_replicated: bool,
    local_kv_heads: int,
    attn_tp_size: int,
    total_kv_heads: int,
    head_group: int | None,
) -> int:
    if rank_replicated:
        return 0
    if head_group is None:
        if local_kv_heads == 1 and attn_tp_size > 1:
            # Ambiguous: 1 head/rank could equally mean kv-head replication,
            # where two ranks hold the same head and would race on one key.
            # An explicit head_group is the operator attesting it is sharding.
            raise NotImplementedError(
                "the unified key scheme cannot derive a namespace at 1 kv head "
                "per rank; set --hicache-storage-head-group, or use "
                "--hicache-storage-key-scheme rank-suffix."
            )
        if local_kv_heads != total_kv_heads:
            # Correct but unshareable: another attn-TP size derives a different
            # digest and misses everything. Nothing else reports this -- the
            # symptom is a 0% hit rate against a populated store.
            logger.warning(
                "unified key scheme: --hicache-storage-head-group is unset, so "
                "this namespace is keyed to this rank's %d of %d kv heads and "
                "can only share objects with deployments at attn-TP %d. Set it "
                "to total_kv_heads / lcm of the fleet's attn-TP sizes to share "
                "across TP sizes; a rank owning several groups simply owns "
                "several chunks.",
                local_kv_heads,
                total_kv_heads,
                attn_tp_size,
            )
        return local_kv_heads
    if head_group <= 0:
        raise ValueError(f"head_group must be positive: {head_group}")
    if local_kv_heads % head_group != 0 or head_group > local_kv_heads:
        raise ValueError(
            f"head_group={head_group} must divide this rank's {local_kv_heads} "
            f"kv heads. As a fleet grid it must divide every member's local "
            f"kv-head count, so pick total_kv_heads / lcm(the fleet's attn-TP "
            f"sizes) -- here total_kv_heads={total_kv_heads}."
        )
    return head_group


def _layer_chunks(
    layer_partition: int, start_layer: int, end_layer: int, is_final_stage: bool
) -> tuple[list[str], list[tuple[int, int]]]:
    """Absolute layer coordinates and their pool-local ranges.

    Coordinates are ABSOLUTE, so any PP split -- uneven stages included --
    yields collision-free names, and a differing split misses rather than
    colliding.
    """
    if not layer_partition:
        return [f"L{start_layer}-{end_layer}"], [(0, end_layer - start_layer)]
    if start_layer % layer_partition != 0:
        raise ValueError(
            f"this rank's layer range [{start_layer}, {end_layer}) does not "
            f"start on a multiple of layer_partition={layer_partition}; stages "
            f"whose boundaries do not align to the layer unit cannot share a "
            f"partitioned namespace (drop layer_partition for per-stage ranges)."
        )
    if end_layer % layer_partition != 0 and not is_final_stage:
        raise ValueError(
            f"this rank's layer range [{start_layer}, {end_layer}) ends off the "
            f"layer_partition={layer_partition} grid; only the FINAL pipeline "
            f"stage may end short (the model's trailing remainder forms the "
            f"short last chunk)."
        )
    bounds = [
        (a, min(a + layer_partition, end_layer))
        for a in range(start_layer, end_layer, layer_partition)
    ]
    return (
        [f"L{a}-{b}" for a, b in bounds],
        [(a - start_layer, b - start_layer) for a, b in bounds],
    )
