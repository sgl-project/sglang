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

    {page_hash}_{digest}_L{start}-{end}[_H{head_group}]_{k|v}

The trailing component is appended by the backend; this module derives the
rest. The coordinate names what an object HOLDS -- a layer-range x
head-range rectangle of one page -- never who wrote it, so any deployment
whose shard tiles the grid derives the same keys for the same data.

``head_group`` / ``layer_partition`` in the extra config switch on the layout
adapter: a rank then owns the (layer window x head group) cross product, and
every object uses one byte order regardless of host layout, so ``page_first``
and ``page_first_direct`` share a keyspace (``object_layout`` = "unified-v2").
Without them the raw page block IS the object, so the host layout is part of
the identity instead.

The digest prefixes every key, so any identity mismatch misses rather than
colliding. Note dtype is the LOGICAL one: fp8_e4m3 and fp8_e5m2 never share a
keyspace even though both store as uint8.
"""

from __future__ import annotations

import hashlib
import logging

import msgspec

logger = logging.getLogger(__name__)

# The digest is computed over the encoded struct, so any schema change must
# change every key. Bump on any field change.
_SCHEMA_VERSION = 1


class KVCacheNamespace(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Immutable identity of one shared L3 KV keyspace.

    Everything that must match for two deployments' KV bytes to be
    interchangeable. Field order is part of the encoding: append only, and
    bump ``schema_version``.
    """

    schema_version: int = _SCHEMA_VERSION
    model_id: str
    # Logical dtype, not the storage view: fp8 variants all store as uint8
    # and must not share a keyspace.
    dtype: str
    page_size: int
    # MLA-family pools: KV replicated across attn-TP ranks, so no head axis
    # (total_kv_heads and head_group are 0).
    rank_replicated: bool
    total_kv_heads: int
    # Layers per chunk; 0 = per-stage ranges (same-partition sharing only).
    # A stage must START on a multiple of it; only the last PP stage may end
    # short, forming the model's trailing remainder chunk.
    layer_group: int = 0
    head_group: int
    # Optional build/ABI digest, to keep deployments with different numerics
    # in different namespaces.
    numerics_id: str = ""
    # Byte order of the stored objects: the raw host layout without the
    # adapter, else "unified-v2". Layouts serialize a page in different orders
    # at EQUAL sizes, so omitting this would let two deployments exchange
    # byte-permuted KV under identical keys.
    object_layout: str


def namespace_digest(namespace: KVCacheNamespace) -> str:
    """Digest of the namespace encoding, used as the key prefix.

    msgpack encoding of a Struct is deterministic given the class definition,
    which is why the schema versions the encoding.
    """
    encoded = msgspec.msgpack.encode(namespace)
    return f"ukv{_SCHEMA_VERSION}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def load_namespace_descriptor(path: str) -> KVCacheNamespace:
    """Load and strictly decode an out-of-band descriptor file (JSON).

    Not wired to a CLI flag yet -- the extra-config entries cover today's
    grids. Kept and tested so the schema and digest semantics are pinned from
    the first release of the key format.
    """
    with open(path, "rb") as f:
        raw = f.read()
    namespace = msgspec.json.decode(raw, type=KVCacheNamespace)
    if namespace.schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"Namespace descriptor {path} has schema_version="
            f"{namespace.schema_version}, this build supports {_SCHEMA_VERSION}."
        )
    _validate_grid(namespace)
    return namespace


def derive_namespace(
    *,
    model_id: str,
    dtype: str,
    page_size: int,
    rank_replicated: bool,
    total_kv_heads: int,
    head_group: int,
    object_layout: str,
    layer_group: int = 0,
) -> KVCacheNamespace:
    """Derive the namespace from deployment facts plus the fleet agreements.

    ``head_group`` and ``layer_group`` are the two fleet-wide agreements:
    deployments passing the same values share a keyspace across TP and PP
    sizes respectively. Unset, each falls back to this rank's own shard, which
    only same-topology deployments match. Every other mismatch (model, dtype,
    page size) simply partitions the keyspace -- never a geometry collision.
    """
    namespace = KVCacheNamespace(
        model_id=model_id,
        dtype=dtype,
        page_size=page_size,
        rank_replicated=rank_replicated,
        total_kv_heads=0 if rank_replicated else total_kv_heads,
        head_group=0 if rank_replicated else head_group,
        object_layout=object_layout,
        layer_group=layer_group,
    )
    _validate_grid(namespace)
    return namespace


def _validate_grid(namespace: KVCacheNamespace) -> None:
    if namespace.layer_group < 0:
        raise ValueError(f"layer_group must be non-negative: {namespace}")
    if namespace.page_size <= 0:
        raise ValueError(f"page_size must be positive: {namespace}")
    if not namespace.model_id:
        raise ValueError(
            "the unified key scheme requires a non-empty model_id: an empty id would "
            "merge different models into one keyspace."
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


def build_unified_suffixes(
    namespace: KVCacheNamespace,
    *,
    attn_tp_rank: int,
    attn_tp_size: int,
    attn_cp_size: int,
    start_layer: int,
    end_layer: int,
    local_kv_heads: int,
    dtype: str,
    page_size: int,
    model_id: str,
    rank_replicated: bool,
    object_layout: str,
    is_final_stage: bool = True,
) -> list[str]:
    """Validate this rank against the namespace; return its owned key suffixes.

    The rank's shard must tile the grid and every identity field must match;
    anything else raises with the remedy, never degrades silently.

    One suffix per owned chunk, layer-major / head-minor. A rank whose kv-head
    shard is coarser than ``head_group`` owns ``local_kv_heads // head_group``
    of them, starting at ``attn_tp_rank * that``. Rank-replicated namespaces
    get one suffix per layer window.
    """
    _validate_grid(namespace)
    _check_identity(
        namespace,
        dtype=dtype,
        page_size=page_size,
        model_id=model_id,
        rank_replicated=rank_replicated,
        object_layout=object_layout,
    )
    if attn_cp_size > 1:
        raise NotImplementedError(
            "the unified key scheme does not support attention context parallelism "
            "yet: NSA-CP ranks hold sub-page slices (needs the token-granule "
            "extension) and replicated-CP needs writer election. Use "
            "--hicache-storage-key-scheme rank-suffix with the file backend."
        )

    if not 0 <= start_layer < end_layer:
        raise ValueError(f"invalid layer range [{start_layer}, {end_layer}).")
    # Layer coordinates are ABSOLUTE ranges, so any PP split -- uneven stages
    # included -- yields collision-free names, and a differing split misses
    # rather than colliding. A declared layer unit is what lets a reader
    # consume chunks written under a different split.
    if namespace.layer_group:
        lg = namespace.layer_group
        if start_layer % lg != 0:
            raise ValueError(
                f"this rank's layer range [{start_layer}, {end_layer}) does "
                f"not start on a multiple of layer_partition={lg}; stages "
                f"whose boundaries do not align to the layer unit cannot "
                f"share a partitioned namespace (drop layer_partition for "
                f"per-stage ranges)."
            )
        if end_layer % lg != 0 and not is_final_stage:
            raise ValueError(
                f"this rank's layer range [{start_layer}, {end_layer}) ends "
                f"off the layer_partition={lg} grid; only the FINAL pipeline "
                f"stage may end short (the model's trailing remainder forms "
                f"the short last chunk)."
            )
        layer_coords = [
            f"L{a}-{min(a + lg, end_layer)}" for a in range(start_layer, end_layer, lg)
        ]
    else:
        layer_coords = [f"L{start_layer}-{end_layer}"]

    digest = namespace_digest(namespace)
    if namespace.rank_replicated:
        return [f"{digest}_{coord}" for coord in layer_coords]

    if namespace.total_kv_heads != local_kv_heads * attn_tp_size:
        raise ValueError(
            f"namespace total_kv_heads={namespace.total_kv_heads} != "
            f"local_kv_heads({local_kv_heads}) x attn_tp_size({attn_tp_size}): "
            f"either the namespace does not match this model/parallelism, or "
            f"kv heads are replicated across ranks (tp_size > model kv heads),"
            f" which needs writer election (follow-up). Use "
            f"--hicache-storage-key-scheme rank-suffix meanwhile."
        )
    if local_kv_heads % namespace.head_group != 0:
        raise ValueError(
            f"this rank's {local_kv_heads} kv heads do not tile "
            f"head_group={namespace.head_group}."
        )
    chunks_per_rank = local_kv_heads // namespace.head_group
    first_head_index = attn_tp_rank * chunks_per_rank
    # Cross product, layer-major / head-minor -- the same order the layout
    # adapter packs bytes in. Single-axis fan-out is the degenerate case.
    return [
        f"{digest}_{coord}_H{first_head_index + i}"
        for coord in layer_coords
        for i in range(chunks_per_rank)
    ]


def _check_identity(
    namespace: KVCacheNamespace,
    *,
    dtype: str,
    page_size: int,
    model_id: str,
    rank_replicated: bool,
    object_layout: str,
) -> None:
    if namespace.object_layout != object_layout:
        raise ValueError(
            f"namespace object_layout={namespace.object_layout!r} != this "
            f"deployment's --hicache-mem-layout {object_layout!r}; object "
            f"bytes would be permuted, not just misplaced."
        )
    if namespace.model_id != model_id:
        raise ValueError(
            f"namespace model_id={namespace.model_id!r} != served model "
            f"{model_id!r}; refusing to mix models in one keyspace."
        )
    if namespace.dtype != dtype:
        raise ValueError(
            f"namespace dtype={namespace.dtype} != stored KV dtype {dtype}."
        )
    if namespace.page_size != page_size:
        raise ValueError(
            f"namespace page_size={namespace.page_size} != deployment "
            f"page_size {page_size}."
        )
    if namespace.rank_replicated != rank_replicated:
        raise ValueError(
            f"namespace rank_replicated={namespace.rank_replicated} != "
            f"deployment pool replication ({rank_replicated}); wrong "
            f"descriptor for this model."
        )


def normalize_dtype(dtype: object) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"`` (descriptor-file friendly)."""
    return str(dtype).removeprefix("torch.")


class UnifiedKVPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Everything the storage layer needs for one rank's unified-key objects."""

    namespace: KVCacheNamespace
    # One suffix per owned chunk (layer-major, head-minor).
    suffixes: list[str]
    # Any partition config is set, so objects use the unified byte order.
    adapter: bool
    # LOCAL half-open chunk ranges for the adapter, or None without it.
    layer_ranges: list[tuple[int, int]] | None = None
    head_ranges: list[tuple[int, int]] | None = None


# Host layouts the adapter can present in the unified byte order. page_head
# and split K/V pools have no unified page view at all. `layer_first` does,
# but is deliberately excluded and IS reachable (kernel_ascend keeps it, and
# the runtime attach endpoint can switch backends without re-running arg
# resolution), so the raise below is load-bearing rather than defensive.
ADAPTER_LAYOUTS = ("page_first", "page_first_direct")


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
    pool_layout: str,
    head_group_config: int | None = None,
    layer_partition: int | None = None,
) -> UnifiedKVPlan:
    """Derive the namespace and this rank's chunk plan from deployment facts.

    Any partition config switches on adapter mode, which puts chunks in their
    own keyspace but shares it across host layouts.
    """
    adapter = layer_partition is not None or (
        head_group_config is not None and not rank_replicated
    )
    if adapter and pool_layout not in ADAPTER_LAYOUTS:
        raise ValueError(
            f"the unified key scheme does not support the {pool_layout!r} "
            f"host layout with partition configs; use --hicache-mem-layout with "
            f"one of {ADAPTER_LAYOUTS}. Every supported layout emits the same "
            f"object bytes and shares one keyspace; these are the ones with a "
            f"unified page view."
        )

    # head_group is the FLEET's chunk size, not a local tuning option: it
    # fixes chunk boundaries, and boundaries are namespace identity. Its
    # default -- this rank's head count -- means "share only with my own TP
    # size", since a TP2 rank's [0, H/2) is not a TP4 rank's [0, H/4).
    total_kv_heads = local_kv_heads * attn_tp_size
    head_group = local_kv_heads
    if head_group_config is not None and not rank_replicated:
        if head_group_config <= 0:
            raise ValueError(f"head_group must be positive: {head_group_config}")
        if (
            local_kv_heads % head_group_config != 0
            or head_group_config > local_kv_heads
        ):
            raise ValueError(
                f"head_group={head_group_config} must divide this rank's "
                f"{local_kv_heads} kv heads. As a fleet grid it must divide "
                f"every member's local kv-head count, so pick "
                f"total_kv_heads / lcm(the fleet's attn-TP sizes) -- here "
                f"total_kv_heads={total_kv_heads}."
            )
        head_group = head_group_config
    elif not rank_replicated and adapter and local_kv_heads != total_kv_heads:
        # Correct but unshareable: another attn-TP size derives a different
        # digest and misses everything. Nothing else reports this -- the
        # symptom is a 0% hit rate against a populated store.
        logger.warning(
            "unified key scheme: head_group is unset, so this namespace is "
            "keyed to this rank's %d of %d kv heads and can only share objects "
            "with deployments at attn-TP %d. Set head_group in "
            "--hicache-storage-backend-extra-config (total_kv_heads / lcm of "
            "the fleet's attn-TP sizes) to share across TP sizes; a rank that "
            "owns several groups simply owns several chunks.",
            local_kv_heads,
            total_kv_heads,
            attn_tp_size,
        )
    if (
        not rank_replicated
        and local_kv_heads == 1
        and attn_tp_size > 1
        and head_group_config is None
    ):
        # Ambiguous: 1 head/rank could mean kv-head replication. An explicit
        # head_group is the operator attesting that it is sharding.
        raise NotImplementedError(
            "the unified key scheme cannot derive a namespace at 1 kv head per rank; "
            "set head_group in the extra config, or use "
            "--hicache-storage-key-scheme rank-suffix."
        )

    # Under the adapter every layout emits the same bytes and differs only in
    # descriptor count, so the layout must NOT enter the namespace. Without it
    # the raw layout IS the wire format and stays part of the identity.
    object_layout = "unified-v2" if adapter else pool_layout
    namespace = derive_namespace(
        model_id=model_id,
        dtype=dtype,
        page_size=page_size,
        rank_replicated=rank_replicated,
        total_kv_heads=total_kv_heads,
        head_group=0 if rank_replicated else head_group,
        object_layout=object_layout,
        layer_group=layer_partition or 0,
    )
    suffixes = build_unified_suffixes(
        namespace,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_cp_size=attn_cp_size,
        start_layer=start_layer,
        end_layer=end_layer,
        local_kv_heads=local_kv_heads,
        dtype=dtype,
        page_size=page_size,
        model_id=model_id,
        rank_replicated=rank_replicated,
        object_layout=object_layout,
        is_final_stage=is_final_stage,
    )

    layer_ranges = None
    head_ranges = None
    if adapter:
        lg = layer_partition or (end_layer - start_layer)
        layer_ranges = [
            (a - start_layer, min(a + lg, end_layer) - start_layer)
            for a in range(start_layer, end_layer, lg)
        ]
        if not rank_replicated:
            chunks = local_kv_heads // head_group
            head_ranges = [
                (i * head_group, (i + 1) * head_group) for i in range(chunks)
            ]
            assert len(suffixes) == len(layer_ranges) * len(head_ranges)
        else:
            assert len(suffixes) == len(layer_ranges)

    return UnifiedKVPlan(
        namespace=namespace,
        suffixes=list(suffixes),
        adapter=adapter,
        layer_ranges=layer_ranges,
        head_ranges=head_ranges,
    )
