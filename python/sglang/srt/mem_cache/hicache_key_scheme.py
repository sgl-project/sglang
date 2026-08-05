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

"""Canonical-grid L3 key scheme (``--hicache-storage-key-scheme canonical-grid``).

Replaces the per-backend rank/topology key suffixes (``_{tp_rank}_{tp_size}``,
``_{pp_size}_{pp_rank}``, ``_cp{r}_{s}``, ...) with a topology-free canonical
cell coordinate::

    {page_hash}_{namespace_digest}_L{start_layer}-{end_layer}[_H{head_group_index}]

The coordinate names *what data the object holds* (a model-global layer-range x
kv-head-range rectangle of one page), never *who wrote it*. Any deployment
whose shard tiles the namespace grid derives identical keys for identical
data, which makes cross-topology reuse a pure key-selection problem. Design:
``DESIGN_l3_canonical_shard_grid.md``.

Without partition knobs, a rank owns one cell per page (its absolute layer
range, its head shard) and objects keep the raw pool-layout bytes. Setting
``head_group`` (heads per cell) and/or ``layer_partition`` (layers per cell)
in the extra config switches the namespace to the **cell adapter**
(:func:`plan_canonical_cells`): a rank owns the (layer window x head group)
cross product of cells, and every object carries a layout-neutral canonical
byte order — (head, layer, token, dim) per K/V half — regardless of the host
pool layout (``object_layout`` becomes the constant ``cell-v1``). The pool
adapters convert on the fly and skip the copy for slabs that are already
canonical-contiguous (e.g. MLA on page_first_direct stays zero-copy).

The namespace digest prefixes every key: deployments share objects iff every
identity field matches, so configuration differences partition into disjoint
keyspaces instead of colliding. Notably the *logical* dtype is an identity
field — fp8_e4m3 and fp8_e5m2 never share a keyspace even though both store
as uint8. :func:`load_namespace_descriptor` is the out-of-band descriptor
API, kept for richer future identities (numerics_id pinning).
"""

from __future__ import annotations

import hashlib
import logging

import msgspec

logger = logging.getLogger(__name__)

# Bump when the struct schema or its canonical encoding changes: the digest is
# computed over the encoded struct, so any schema change must change every key.
_SCHEMA_VERSION = 1


class KVCacheNamespace(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Immutable identity of one shared L3 KV keyspace.

    Everything that must be equal for two deployments' KV bytes to be
    interchangeable, plus the canonical grid that fixes cell boundaries.
    Field order is part of the canonical encoding — append new fields only,
    and bump ``schema_version`` when doing so. ``forbid_unknown_fields``
    makes descriptor-file typos a decode error instead of a silently
    different keyspace.
    """

    schema_version: int = _SCHEMA_VERSION
    model_id: str
    # Logical torch dtype of the KV cache, normalized (e.g. "bfloat16",
    # "float8_e4m3fn") — NOT the storage view dtype (fp8 variants all store
    # as uint8 and must not share a keyspace).
    dtype: str
    page_size: int
    # True for MLA-family pools whose KV is replicated across attn-TP ranks;
    # such namespaces have no head axis (total_kv_heads/head_group are 0).
    rank_replicated: bool
    total_kv_heads: int
    # Head grid: kv heads per cell. Layer grid: layer_group > 0 = layers per
    # cell (the fleet's layer unit). Every stage must START on a multiple of
    # layer_group; the model's trailing remainder simply forms a short final
    # cell (allowed only on the last PP stage, where the stage end is the
    # model total). 0 = per-stage ranges (same-partition sharing only).
    layer_group: int = 0
    head_group: int
    # Optional kernel/build ABI digest; deployments whose numerics must not
    # mix set distinct values and thereby get distinct namespaces.
    numerics_id: str = ""
    # Host memory-pool layout of the stored object bytes (page_first,
    # page_head, page_first_direct, ...). Different layouts serialize a page
    # in different byte orders with EQUAL sizes, so without this field two
    # deployments could exchange byte-permuted KV under identical keys.
    # Identity field: mismatched layouts miss instead of corrupting.
    object_layout: str


def namespace_digest(namespace: KVCacheNamespace) -> str:
    """Digest of the canonical descriptor encoding, used as the key prefix.

    msgspec's msgpack encoding of a Struct is deterministic given the class
    definition (fields in declaration order), which is why the schema itself
    versions the encoding.
    """
    encoded = msgspec.msgpack.encode(namespace)
    return f"ukv{_SCHEMA_VERSION}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def load_namespace_descriptor(path: str) -> KVCacheNamespace:
    """Load and strictly decode an out-of-band descriptor file (JSON).

    Not wired to a CLI flag: the extra-config knobs (head_group,
    layer_partition) cover today's grids, so fleet descriptor files remain a
    follow-up for richer identities (numerics_id pinning, per-component
    grids). Kept (and tested) now so the schema, strictness, and digest
    semantics are pinned from the first release of the key format.
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

    ``head_group`` is the head-grid agreement (the ``head_group`` extra-config
    knob): deployments passing the same value land in one keyspace; without
    it, ``head_group`` = the rank's local head count and only same-TP
    deployments share. ``layer_group`` is the layer-unit agreement
    (``layer_partition`` in the extra config, layers per cell) that enables
    PP read-back across different pipeline splits — the model's trailing
    remainder forms a short final cell; without it, layer cells are
    per-stage ranges and only same-partition deployments share.
    All other mismatches (model, logical dtype, page size) partition into
    disjoint keyspaces — safe, never a geometry collision.
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
            "canonical-grid requires a non-empty model_id: an empty id would "
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


def build_canonical_cell_suffixes(
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
    """Validate this rank against the namespace; return its owned cell suffixes.

    The attach-time compatibility check of the design's section 2.2: the
    rank's shard must tile the grid, and every identity field must match.
    Raises with the remedy in the message; never degrades silently.

    Returns one suffix per owned cell, in ascending head-group order. A rank
    whose kv-head shard is coarser than ``head_group`` owns
    ``local_kv_heads // head_group`` cells (head fan-out): rank ``r`` owns
    head groups ``[r * n, (r + 1) * n)`` — the same arithmetic as the
    mooncake split-heads virtual ranks, re-keyed canonically. Rank-replicated
    namespaces always return exactly one suffix.
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
            "canonical-grid does not support attention context parallelism "
            "yet: NSA-CP ranks hold sub-page slices (needs the token-granule "
            "extension) and replicated-CP needs writer election. Use "
            "--hicache-storage-key-scheme rank-suffix with the file backend."
        )

    if not 0 <= start_layer < end_layer:
        raise ValueError(f"invalid layer range [{start_layer}, {end_layer}).")
    # Layer coordinates are absolute ranges: any PP partition (uneven stages
    # included) yields valid, collision-free names; differing partitions miss
    # instead of colliding. With a layer unit declared, the stage must start
    # on the grid and owns one cell per contained window; the model's
    # trailing remainder forms a short final cell (legal only on the last
    # pipeline stage, whose end IS the model total) — this is what lets a
    # reader consume cells written under a different pipeline split.
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
                f"the short last cell)."
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
    cells_per_rank = local_kv_heads // namespace.head_group
    first_head_index = attn_tp_rank * cells_per_rank
    # The general case is the cross product, layer-major / head-minor — the
    # same order the cell-adapter arena packs bytes in. Single-axis fan-outs
    # are the degenerate cases (one coordinate list has length 1).
    return [
        f"{digest}_{coord}_H{first_head_index + i}"
        for coord in layer_coords
        for i in range(cells_per_rank)
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


class CanonicalCellPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Everything the storage layer needs for one rank's canonical cells."""

    namespace: KVCacheNamespace
    # One suffix per owned cell (layer-major, head-minor).
    suffixes: list[str]
    # True when any partition knob is set: objects then use the canonical
    # byte order via the gather/scatter adapter (which skips the copy when
    # the pool view already matches).
    adapter: bool
    # LOCAL half-open cell ranges for the adapter, or None without it.
    layer_ranges: Optional[list[tuple[int, int]]] = None
    head_ranges: Optional[list[tuple[int, int]]] = None


# Host layouts the adapter can view canonically.
ADAPTER_LAYOUTS = ("layer_first", "page_first", "page_first_direct", "page_head")


def plan_canonical_cells(
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
    head_group_knob: Optional[int] = None,
    layer_partition: Optional[int] = None,
) -> CanonicalCellPlan:
    """Derive the namespace and this rank's cell plan from deployment facts.

    Any partition knob switches the namespace to adapter mode: objects carry
    the layout-neutral canonical byte order (object_layout "cell-v1"), so
    the pool layout never constrains which fleets can share.
    """
    adapter = layer_partition is not None or (
        head_group_knob is not None and not rank_replicated
    )
    if adapter and pool_layout not in ADAPTER_LAYOUTS:
        raise ValueError(
            f"the cell adapter does not support the {pool_layout!r} host "
            f"layout; use one of {ADAPTER_LAYOUTS}."
        )

    head_group = local_kv_heads
    if head_group_knob is not None and not rank_replicated:
        if head_group_knob <= 0:
            raise ValueError(f"head_group must be positive: {head_group_knob}")
        if local_kv_heads % head_group_knob != 0 or head_group_knob > local_kv_heads:
            raise ValueError(
                f"head_group={head_group_knob} must divide this rank's "
                f"{local_kv_heads} kv heads."
            )
        head_group = head_group_knob
    if (
        not rank_replicated
        and local_kv_heads == 1
        and attn_tp_size > 1
        and head_group_knob is None
    ):
        # 1 head/rank is ambiguous (could be kv-head replication); an
        # explicit head_group is the operator's attestation of sharding.
        raise NotImplementedError(
            "canonical-grid cannot derive a namespace at 1 kv head per rank; "
            "set head_group in the extra config, or use "
            "--hicache-storage-key-scheme rank-suffix."
        )

    object_layout = "cell-v1" if adapter else pool_layout
    namespace = derive_namespace(
        model_id=model_id,
        dtype=dtype,
        page_size=page_size,
        rank_replicated=rank_replicated,
        total_kv_heads=local_kv_heads * attn_tp_size,
        head_group=0 if rank_replicated else head_group,
        object_layout=object_layout,
        layer_group=layer_partition or 0,
    )
    suffixes = build_canonical_cell_suffixes(
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
            cells = local_kv_heads // head_group
            head_ranges = [(i * head_group, (i + 1) * head_group) for i in range(cells)]
            assert len(suffixes) == len(layer_ranges) * len(head_ranges)
        else:
            assert len(suffixes) == len(layer_ranges)

    return CanonicalCellPlan(
        namespace=namespace,
        suffixes=list(suffixes),
        adapter=adapter,
        layer_ranges=layer_ranges,
        head_ranges=head_ranges,
    )
