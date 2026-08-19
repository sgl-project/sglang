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

Replaces the per-backend rank/topology key suffixes (``_{tp_rank}_{tp_size}``,
``_{pp_size}_{pp_rank}``, ``_cp{r}_{s}``, ...) with one topology-free
coordinate::

    {page_hash}_{namespace_digest}_L{start_layer}-{end_layer}[_H{head_group_index}]

The coordinate names *what data the object holds* (a model-global layer-range x
kv-head-range rectangle of one page), never *who wrote it*. Any deployment
whose shard tiles the namespace grid derives identical keys for identical
data, which makes cross-topology reuse a pure key-selection problem.

Without partition knobs, a rank owns one chunk per page (its absolute layer
range, its head shard) and objects keep the raw pool-layout bytes. Setting
``head_group`` (heads per chunk) and/or ``layer_partition`` (layers per
chunk) in the extra config switches the namespace to the **layout adapter**
(:func:`plan_unified_kv`): a rank owns the (layer window x head group)
cross product of chunks, and every object carries the same unified byte
order — (layer, token, head, dim) per K/V half, MLA (layer, token, dim) —
which is ``page_first_direct``'s own page block, regardless of the host
pool layout (``object_layout`` becomes the constant ``unified-v2``). The
pool adapters convert on the fly and skip the copy for slabs that already
sit in that order contiguously — which on ``page_first_direct`` is every
slab whenever the fleet grid does not cut the kv-head axis, MLA included.

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

# Bump when the struct schema or its encoding changes: the digest is
# computed over the encoded struct, so any schema change must change every key.
_SCHEMA_VERSION = 1


class KVCacheNamespace(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Immutable identity of one shared L3 KV keyspace.

    Everything that must be equal for two deployments' KV bytes to be
    interchangeable, plus the shared grid that fixes chunk boundaries.
    Field order is part of the encoding — append new fields only,
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
    # Head grid: kv heads per chunk. Layer grid: layer_group > 0 = layers
    # per chunk (the fleet's layer unit). Every stage must START on a
    # multiple of layer_group; the model's trailing remainder simply forms a
    # short final chunk (allowed only on the last PP stage, where the stage
    # end is the model total). 0 = per-stage ranges (same-partition sharing
    # only).
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
    """Digest of the namespace encoding, used as the key prefix.

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
    (``layer_partition`` in the extra config, layers per chunk) that enables
    PP read-back across different pipeline splits — the model's trailing
    remainder forms a short final chunk; without it, layer chunks are
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

    The attach-time compatibility check of the design's section 2.2: the
    rank's shard must tile the grid, and every identity field must match.
    Raises with the remedy in the message; never degrades silently.

    Returns one suffix per owned chunk, in ascending head-group order. A
    rank whose kv-head shard is coarser than ``head_group`` owns
    ``local_kv_heads // head_group`` chunks (head fan-out): rank ``r`` owns
    head groups ``[r * n, (r + 1) * n)`` — the same arithmetic as the
    mooncake split-heads virtual ranks, re-keyed topology-free.
    Rank-replicated namespaces return one suffix per layer window.
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
    # Layer coordinates are absolute ranges: any PP partition (uneven stages
    # included) yields valid, collision-free names; differing partitions miss
    # instead of colliding. With a layer unit declared, the stage must start
    # on the grid and owns one chunk per contained window; the model's
    # trailing remainder forms a short final chunk (legal only on the last
    # pipeline stage, whose end IS the model total) — this is what lets a
    # reader consume chunks written under a different pipeline split.
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
    # The general case is the cross product, layer-major / head-minor — the
    # same order the layout adapter packs bytes in. Single-axis fan-outs
    # are the degenerate cases (one coordinate list has length 1).
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
    # True when any partition knob is set: objects then use the unified
    # byte order via the gather/scatter adapter (which skips the copy when
    # the pool view already matches).
    adapter: bool
    # LOCAL half-open chunk ranges for the adapter, or None without it.
    layer_ranges: Optional[list[tuple[int, int]]] = None
    head_ranges: Optional[list[tuple[int, int]]] = None


# Host layouts the adapter can present in the unified byte order.
ADAPTER_LAYOUTS = ("layer_first", "page_first", "page_first_direct", "page_head")


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
    head_group_knob: Optional[int] = None,
    layer_partition: Optional[int] = None,
) -> UnifiedKVPlan:
    """Derive the namespace and this rank's chunk plan from deployment facts.

    Any partition knob switches the namespace to adapter mode: objects carry
    the unified byte order (object_layout "unified-v2"), so the pool layout
    never constrains which fleets can share.
    """
    adapter = layer_partition is not None or (
        head_group_knob is not None and not rank_replicated
    )
    if adapter and pool_layout not in ADAPTER_LAYOUTS:
        raise ValueError(
            f"the KV layout adapter does not support the {pool_layout!r} host "
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
            "the unified key scheme cannot derive a namespace at 1 kv head per rank; "
            "set head_group in the extra config, or use "
            "--hicache-storage-key-scheme rank-suffix."
        )

    object_layout = "unified-v2" if adapter else pool_layout
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


class KVCacheLayoutAdapter:
    """Backend-neutral staging machinery for unified-layout IO.

    The host pools expose the unified gather/scatter primitives; this
    class owns everything else a backend needs to serve a partitioned
    (unified-v2) namespace: the per-chunk key fan-out, the sub-batch
    geometry, and one pinned staging buffer per IO direction (backup and
    prefetch run on concurrent controller threads). A backend brings only
    its own pointer-based batch put/get and, for RDMA transports, a
    ``register_buffer`` hook for the staging buffers.

    When every slab is already pool-contiguous no staging is allocated and
    all transfers resolve to pool addresses (pure zero-copy).
    """

    def __init__(self, mem_pool_host, storage_config, register_buffer=None):
        self.pool = mem_pool_host
        self.page_size = mem_pool_host.page_size
        self.layer_ranges = storage_config.unified_layer_ranges
        self.head_ranges = storage_config.unified_head_ranges
        suffixes = storage_config.unified_suffix
        assert isinstance(suffixes, list) and self.layer_ranges is not None
        self.suffixes = suffixes
        # Rank-replicated (MLA-family) chunks are single objects; sharded
        # pools store one K and one V object per chunk.
        self.split_kv = self.head_ranges is not None
        self.keys_per_page = len(suffixes) * (2 if self.split_kv else 1)
        self.staging_set = None
        self.staging_get = None
        self.staging_pages = 0
        if self.pool.unified_zero_copy(self.layer_ranges, self.head_ranges):
            logger.info(
                "HiCache KV layout adapter: everything pool-contiguous, "
                "zero-copy (no staging buffers)."
            )
            return
        extra = storage_config.extra_config or {}
        staging_mb = extra.get("staging_buffer_mb", 256)
        page_bytes = self.pool.unified_bytes_per_page(
            self.layer_ranges, self.head_ranges
        )
        staging_bytes = max(int(staging_mb) << 20, page_bytes)
        self.staging_pages = staging_bytes // page_bytes
        staging_numel = self.staging_pages * page_bytes
        self.staging_set = self._alloc_staging(staging_numel)
        self.staging_get = self._alloc_staging(staging_numel)
        if register_buffer is not None:
            register_buffer(self.staging_set)
            register_buffer(self.staging_get)
        logger.info(
            "HiCache KV layout adapter: 2 x %d-page staging buffers "
            "(%.1f MB each) for the backup and prefetch threads.",
            self.staging_pages,
            staging_numel / (1 << 20),
        )

    def _alloc_staging(self, numel):
        # Pinned so RDMA transports can register and DMA it directly.
        # (Overridable seam: CPU-only tests allocate unpinned.)
        import torch

        return torch.empty(numel, dtype=torch.uint8, pin_memory=True)

    def chunk_keys(self, page_keys: list) -> list:
        """Fan page keys out to chunk keys, in the pools' slab order:
        page-major, suffix (layer-major, head-minor), K then V."""
        key_list = []
        if not self.split_kv:
            for key_ in page_keys:
                for suffix in self.suffixes:
                    key_list.append(f"{key_}_{suffix}_k")
            return key_list
        for key_ in page_keys:
            for suffix in self.suffixes:
                key_list.append(f"{key_}_{suffix}_k")
                key_list.append(f"{key_}_{suffix}_v")
        return key_list

    def sub_batches(self, keys: list, host_indices):
        """Split a batch into staging-sized (page_keys, indices) pieces;
        each piece reuses the staging buffers from offset 0."""
        if not keys:
            return
        pages_per_batch = self.staging_pages or len(keys)
        for start in range(0, len(keys), pages_per_batch):
            page_keys = keys[start : start + pages_per_batch]
            indices = host_indices[
                start * self.page_size : (start + len(page_keys)) * self.page_size
            ]
            yield page_keys, indices

    def gather(self, indices):
        """Write path: (ptrs, sizes) per slab — unified-order bytes, staged
        into staging_set only where the pool view is not already
        contiguous."""
        return self.pool.gather_unified_chunks(
            indices, self.layer_ranges, self.head_ranges, self.staging_set
        )

    def read_metas(self, indices):
        """Read path targets: direct slabs fetch straight into the pool,
        staged slabs into staging_get."""
        return self.pool.get_unified_chunk_meta(
            indices, self.layer_ranges, self.head_ranges, self.staging_get
        )

    def scatter(self, indices, page_ok):
        """Read path finalize: copy staged slabs of successful pages from
        staging_get into the pool (direct slabs already landed in place)."""
        if self.staging_get is None or not any(page_ok):
            return
        self.pool.scatter_unified_chunks(
            indices, self.layer_ranges, self.head_ranges, self.staging_get, page_ok
        )
