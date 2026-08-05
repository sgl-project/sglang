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

Scope (enforced in :func:`build_canonical_cell_suffixes`): the layer
coordinate is the rank's absolute layer range — any pipeline partition works,
uneven stages included (61-layer models at any pp_size); partitions that
differ simply derive disjoint keys and miss instead of colliding. On the head
axis a rank owns ``local_kv_heads // head_group`` cells per page (head
fan-out): fleets agree on the head grid via the existing ``tp_lcm_size``
extra-config knob (``head_group = total_kv_heads / tp_lcm_size``), which this
scheme subsumes — the canonical H indices coincide with the split-heads
virtual ranks, so the proven multi-key read/write machinery
(``get_split_heads_page_buffer_meta``, mooncake key fan-out) carries over
under canonical names. Rank-replicated pools (MLA-family) have no head axis,
so cross-TP-size reuse needs no head-grid agreement at all. Uniform layer
grids with layer fan-out remain the follow-up and are rejected explicitly.

The namespace is derived from deployment facts (model, logical KV dtype,
page size, head grid) and its digest prefixes every key: deployments share
objects iff every identity field matches, so configuration differences
partition into disjoint keyspaces instead of colliding (fail-safe, and
observable via the digest logged at attach). Notably the *logical* dtype is
an identity field — fp8_e4m3 and fp8_e5m2 caches never share a keyspace even
though both store as uint8. :func:`load_namespace_descriptor` is the
out-of-band fleet-descriptor API (a JSON file shared by deployments that
must agree on a head grid finer than their own shards); it becomes reachable
from the CLI together with head fan-out.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

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
    # Head grid: kv heads per cell. Layer cells are absolute per-stage layer
    # ranges in v1; layer_group must stay 0 (uniform layer grids with
    # fan-out are the follow-up).
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
    # Canonical layer partition (the fleet agreement for PP read-back):
    # strictly increasing boundaries starting at 0 and ending at the model's
    # layer count, e.g. [0, 30, 61]. Every deployment's stage must start and
    # end on boundaries; a stage spanning several ranges owns one cell per
    # range (layer fan-out). Empty = per-stage ranges (same-partition
    # deployments share; differing partitions miss).
    layer_boundaries: list[int] = []


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

    Not wired to a CLI flag yet: fleet descriptor files ship together with
    head fan-out, where deployments must agree on a head grid finer than
    their own shards. Kept (and tested) now so the schema, strictness, and
    digest semantics are pinned from the first release of the key format.
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
    layer_boundaries: Optional[list[int]] = None,
) -> KVCacheNamespace:
    """Derive the namespace from deployment facts plus the fleet agreements.

    ``head_group`` is the head-grid agreement: deployments that pass the same
    ``tp_lcm_size`` derive ``head_group = total_kv_heads / tp_lcm_size`` and
    land in one keyspace; without it, ``head_group`` = the rank's local head
    count and only same-TP deployments share. ``layer_boundaries`` is the
    layer-partition agreement (``layer_partition`` in the extra config) that
    enables PP read-back across different pipeline splits; without it, layer
    cells are per-stage ranges and only same-partition deployments share.
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
        layer_boundaries=list(layer_boundaries) if layer_boundaries else [],
    )
    _validate_grid(namespace)
    return namespace


def _validate_grid(namespace: KVCacheNamespace) -> None:
    if namespace.layer_group != 0:
        raise NotImplementedError(
            f"layer_group={namespace.layer_group}: uniform layer grids with "
            f"multi-cell fan-out are not implemented; v1 uses per-stage layer "
            f"ranges (layer_group=0)."
        )
    boundaries = namespace.layer_boundaries
    if boundaries:
        if len(boundaries) < 2 or boundaries[0] != 0:
            raise ValueError(
                f"layer_boundaries must start at 0 and contain at least one "
                f"range: {boundaries}"
            )
        if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
            raise ValueError(
                f"layer_boundaries must be strictly increasing: {boundaries}"
            )
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
    # instead of colliding. With a canonical layer partition, the stage must
    # align to boundaries and owns one cell per contained range (layer
    # fan-out — this is what lets a reader consume cells written under a
    # different pipeline split).
    if namespace.layer_boundaries:
        boundaries = namespace.layer_boundaries
        if start_layer not in boundaries or end_layer not in boundaries:
            raise ValueError(
                f"this rank's layer range [{start_layer}, {end_layer}) does "
                f"not start and end on canonical layer boundaries "
                f"{boundaries}; every deployment sharing this namespace must "
                f"partition layers on these boundaries."
            )
        first = boundaries.index(start_layer)
        last = boundaries.index(end_layer)
        layer_coords = [
            f"L{boundaries[i]}-{boundaries[i + 1]}" for i in range(first, last)
        ]
    else:
        layer_coords = [f"L{start_layer}-{end_layer}"]

    digest = namespace_digest(namespace)
    if namespace.rank_replicated:
        return [f"{digest}_{coord}" for coord in layer_coords]

    if len(layer_coords) > 1:
        raise NotImplementedError(
            "layer fan-out is only supported for rank-replicated (MLA-family)"
            " pools in this PR: an MHA layer-range cell is not contiguous "
            "under the layouts the head axis requires. Partition this "
            "deployment's stages on the canonical boundaries, or drop "
            "layer_partition."
        )

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
    return [
        f"{digest}_{layer_coords[0]}_H{first_head_index + i}"
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
