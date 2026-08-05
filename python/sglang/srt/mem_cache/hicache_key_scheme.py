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

v1 scope (enforced in :func:`build_canonical_cell_suffix`): exactly one cell
per rank per page. The layer coordinate is the rank's absolute layer range —
any pipeline partition works, uneven stages included (61-layer models at any
pp_size); partitions that differ simply derive disjoint keys and miss instead
of colliding. The head axis requires ``head_group`` == the rank's local
kv-head count. Objects are therefore byte-identical to the rank-suffix
scheme's; only the names change. Uniform layer grids with multi-cell fan-out
(and head fan-out subsuming the mooncake ``tp_lcm_size`` split-heads
mechanism) are the follow-up and are rejected with explicit errors here.
Rank-replicated pools (MLA-family) have no head axis, so cross-TP-size reuse
works immediately.

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
    local_kv_heads: int,
) -> KVCacheNamespace:
    """Default descriptor when no file is given: head grid == this deployment.

    Deployments with different topologies then derive different digests and
    land in disjoint keyspaces — safe (never a geometry collision) but with no
    cross-topology sharing. Fleets that want sharing distribute one descriptor
    file instead. Layer cells are per-stage ranges, so the derived namespace
    is PP-partition-independent: only the head grid enters the identity.
    """
    namespace = KVCacheNamespace(
        model_id=model_id,
        dtype=dtype,
        page_size=page_size,
        rank_replicated=rank_replicated,
        total_kv_heads=0 if rank_replicated else total_kv_heads,
        head_group=0 if rank_replicated else local_kv_heads,
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


def build_canonical_cell_suffix(
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
) -> str:
    """Validate this rank against the namespace and return its cell suffix.

    The attach-time compatibility check of the design's section 2.2: the
    rank's shard must tile the grid, and every identity field must match.
    Raises with the remedy in the message; never degrades silently.
    """
    _validate_grid(namespace)
    _check_identity(
        namespace,
        dtype=dtype,
        page_size=page_size,
        model_id=model_id,
        rank_replicated=rank_replicated,
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
    # Absolute per-stage range: any PP partition (uneven stages included)
    # yields valid, collision-free names; differing partitions miss instead
    # of colliding.
    layer_coord = f"L{start_layer}-{end_layer}"

    digest = namespace_digest(namespace)
    if namespace.rank_replicated:
        return f"{digest}_{layer_coord}"

    if namespace.total_kv_heads != local_kv_heads * attn_tp_size:
        raise ValueError(
            f"namespace total_kv_heads={namespace.total_kv_heads} != "
            f"local_kv_heads({local_kv_heads}) x attn_tp_size({attn_tp_size}): "
            f"either the descriptor does not match this model/parallelism, or "
            f"kv heads are replicated across ranks (tp_size > model kv heads),"
            f" which needs writer election (follow-up). Use "
            f"--hicache-storage-key-scheme rank-suffix meanwhile."
        )
    if local_kv_heads % namespace.head_group != 0:
        raise ValueError(
            f"this rank's {local_kv_heads} kv heads do not tile "
            f"head_group={namespace.head_group}."
        )
    if local_kv_heads != namespace.head_group:
        raise NotImplementedError(
            f"head fan-out is not implemented: this rank holds "
            f"{local_kv_heads} kv heads but the namespace head_group is "
            f"{namespace.head_group} ({local_kv_heads // namespace.head_group} "
            f"cells per page). v1 requires head_group == the rank's local kv "
            f"heads; for cross-TP-size reuse today use the rank-suffix scheme "
            f"with tp_lcm_size split-heads."
        )
    head_index = (attn_tp_rank * local_kv_heads) // namespace.head_group
    return f"{digest}_{layer_coord}_H{head_index}"


def _check_identity(
    namespace: KVCacheNamespace,
    *,
    dtype: str,
    page_size: int,
    model_id: str,
    rank_replicated: bool,
) -> None:
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
