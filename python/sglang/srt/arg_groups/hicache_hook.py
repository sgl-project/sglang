# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the hierarchical KV cache."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
    use_mla_backend,
)

logger = logging.getLogger(__name__)


def handle_hicache(server_args: Any):
    """Normalize hicache-related knobs into a valid runtime configuration.

    Resolution order:
    1) Layout <-> I/O compatibility for direct conflicts.
    2) Storage <-> layout compatibility (may rewrite layout).
    """
    cfg = resolving_view(server_args)
    if cfg.enable_unified_cache_external_linker:
        if cfg.enable_hierarchical_cache:
            raise ValueError(
                "--enable-unified-cache-external-linker and "
                "--enable-hierarchical-cache are mutually exclusive."
            )
        if cfg.hicache_storage_backend is not None:
            raise ValueError(
                "--enable-unified-cache-external-linker does not use "
                "--hicache-storage-backend."
            )
        return

    # Skip all normalization when neither hicache nor decode-offload path is active.
    if not (
        cfg.enable_hierarchical_cache
        or cfg.disaggregation_decode_enable_offload_kvcache
        or (
            cfg.disaggregation_mode == "decode"
            and cfg.disaggregation_decode_retraction_backup in (None, "host_pool")
        )
    ):
        return

    validate_hicache_host_memory_mode(server_args)

    # Step 0: the unified key scheme pins the layout and io backend, so it has
    # to run before the compatibility rewrites below get a say.
    resolve_hicache_key_scheme(server_args)

    # Step 1: Initial layout-io compatibility normalization.
    resolve_layout_io_compatibility(server_args)

    # Step 2: Storage-layout normalization without changing io backend.
    resolve_storage_layout_compatibility(server_args)

    # Step 3: DCP compatibility for the L2 (device<->host) path.
    resolve_hicache_dcp_compatibility(server_args)


def handle_hicache_ratio_default(server_args: Any):
    """Default the host/device ratio per host memory mode.

    Runs before the dummy-model boundary: direct HostKVCache consumers
    (unit fixtures, dummy-model launches) must never see a None ratio.
    buffer_only stages in flight rather than retaining, so it needs only
    enough to cover the write backlog plus parked prefetches.

    A decode server keeps the ratio unset here: kv_cache_builder resolves
    it against the retraction-backup backend (1.0 for host_pool, else 2.0).
    """
    cfg = resolving_view(server_args)
    if cfg.hicache_ratio is None and cfg.disaggregation_mode != "decode":
        declare_resolution(
            server_args,
            "_handle_hicache_ratio_default",
            hicache_ratio=(
                1.2 if cfg.hicache_host_memory_mode == "buffer_only" else 2.0
            ),
        )


def resolve_hicache_dcp_compatibility(server_args: Any):

    cfg = resolving_view(server_args)
    if cfg.dcp_size <= 1 or not cfg.enable_hierarchical_cache:
        return
    if cfg.hicache_storage_backend is not None:
        raise NotImplementedError(
            "--hicache-storage-backend (L3) with --dcp-size > 1 is not "
            "supported yet: under DCP each rank holds a distinct "
            "interleaved MLA KV shard, so the rank-0-only replicated-MLA "
            "backup and the storage keys must become dcp_rank-aware "
            "first. Run HiCache+DCP with L1/L2 only."
        )
    if cfg.speculative_algorithm not in (None, "DSPARK"):
        raise NotImplementedError(
            "HiCache with --dcp-size > 1 only supports DSPARK speculative "
            "decoding; other draft-model host pools have no DCP index "
            "translation."
        )
    if cfg.enable_lmcache:
        raise NotImplementedError(
            "--enable-lmcache with --dcp-size > 1 is not supported: "
            "LMCache has no DCP-aware index translation."
        )
    if cfg.enable_hisparse:
        raise NotImplementedError(
            "--enable-hisparse with --dcp-size > 1 is not supported: the "
            "HiSparse host pool is constructed without DCP translation."
        )
    if not use_mla_backend(server_args):
        raise NotImplementedError(
            "HiCache with --dcp-size > 1 is only supported for MLA models: "
            "the index translation lives in MLATokenToKVPoolHost, and the "
            "MHA host pool has none."
        )
    logger.info(
        "HiCache + DCP enabled (L1/L2 only): host pool uses widened "
        "logical slot accounting with per-rank physical translation at "
        "the transfer boundary (dcp_size=%d).",
        cfg.dcp_size,
    )


def resolve_hicache_key_scheme(server_args: Any):
    """Validate --hicache-storage-key-scheme and resolve what it implies."""
    cfg = resolving_view(server_args)
    unified = cfg.hicache_storage_key_scheme == "unified"
    partition_configs = (
        cfg.hicache_storage_head_group is not None
        or cfg.hicache_storage_layer_partition is not None
    )
    if not unified:
        if partition_configs:
            raise ValueError(
                "--hicache-storage-head-group / --hicache-storage-layer-partition "
                "require --hicache-storage-key-scheme unified."
            )
        return

    for name in ("hicache_storage_head_group", "hicache_storage_layer_partition"):
        value = getattr(cfg, name)
        if value is not None and value <= 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be positive, got {value}."
            )
    if cfg.hicache_storage_backend is None:
        raise ValueError(
            "--hicache-storage-key-scheme unified names L3 objects, so it needs "
            "--hicache-storage-backend."
        )
    if cfg.hicache_storage_backend not in ("file", "mooncake"):
        raise NotImplementedError(
            f"--hicache-storage-key-scheme unified supports the file and "
            f"mooncake backends; got {cfg.hicache_storage_backend!r}."
        )
    if cfg.speculative_algorithm is not None:
        # Draft layers are appended to the host pool's layer axis but are not
        # part of the model-global grid, so they would land inside a named
        # chunk unnamed.
        raise NotImplementedError(
            "--hicache-storage-key-scheme unified does not support speculative "
            "decoding: the draft layers are not part of the L3 layer grid."
        )
    if cfg.dcp_size > 1 or cfg.attn_cp_size > 1:
        raise NotImplementedError(
            "--hicache-storage-key-scheme unified does not support context "
            "parallelism: a CP rank holds sub-page slices, which needs the "
            "token-granule extension."
        )

    # The layout IS part of the object identity, so it is pinned rather than
    # checked: a unified deployment that silently kept another layout would
    # publish byte-permuted KV under keys a reader trusts.
    resolutions = {}
    if cfg.hicache_mem_layout != "page_unified":
        logger.warning(
            "--hicache-storage-key-scheme unified stores objects in the "
            "page_unified byte order; replacing --hicache-mem-layout %r.",
            cfg.hicache_mem_layout,
        )
        resolutions["hicache_mem_layout"] = "page_unified"
    if cfg.hicache_io_backend != "kernel":
        # Only the kernel backend relayouts; the copy engine moves a page block
        # verbatim and cannot produce this order.
        logger.warning(
            "--hicache-storage-key-scheme unified needs the kernel io backend; "
            "replacing --hicache-io-backend %r.",
            cfg.hicache_io_backend,
        )
        resolutions["hicache_io_backend"] = "kernel"
    if resolutions:
        declare_resolution(server_args, "_resolve_hicache_key_scheme", **resolutions)


def resolve_layout_io_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.hicache_mem_layout == "page_unified":
        # page_unified is only ever driven by the kernel backend's relayout
        # pair; the rewrites below are about the other page layouts.
        if cfg.hicache_io_backend != "kernel":
            raise ValueError(
                "--hicache-mem-layout page_unified requires "
                "--hicache-io-backend kernel: the direct backend copies a page "
                "block verbatim and cannot produce this byte order."
            )
        return
    if (
        cfg.hicache_mem_layout == "page_first_direct"
        and cfg.hicache_io_backend == "kernel"
    ):
        declare_resolution(
            server_args,
            "_resolve_layout_io_compatibility",
            hicache_io_backend="direct",
        )
        logger.warning(
            "Kernel io backend does not support page first direct layout, switching to direct io backend"
        )

    if cfg.hicache_mem_layout == "page_first" and cfg.hicache_io_backend == "direct":
        declare_resolution(
            server_args,
            "_resolve_layout_io_compatibility",
            hicache_mem_layout="page_first_direct",
        )
        logger.warning(
            "Page first layout is not supported with direct IO backend, switching to page first direct layout"
        )


def resolve_storage_layout_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    if (
        cfg.hicache_storage_backend not in ("mooncake", "npu_memcache")
        or cfg.hicache_mem_layout != "layer_first"
    ):
        return

    if cfg.hicache_io_backend == "direct":
        new_layout = "page_first_direct"
    elif cfg.hicache_io_backend == "kernel":
        new_layout = "page_first"
    else:
        # Keep current behavior for unknown backends (e.g., kernel_ascend).
        new_layout = cfg.hicache_mem_layout

    declare_resolution(
        server_args,
        "_resolve_storage_layout_compatibility",
        hicache_mem_layout=new_layout,
    )
    logger.warning(
        f"Mooncake/Ascend MemCache storage backend does not support layer_first layout, "
        f"switching to {new_layout} layout for {cfg.hicache_io_backend} io backend"
    )


def validate_hicache_host_memory_mode(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.hicache_host_memory_mode not in ("cache", "buffer_only"):
        raise ValueError(
            "hicache_host_memory_mode must be 'cache' or 'buffer_only', "
            f"got {cfg.hicache_host_memory_mode!r}"
        )

    # Both modes are defaulted upstream (a decode server resolves the
    # ratio later, in kv_cache_builder), so this fires only if that
    # defaulting regresses -- never build an unsized host pool.
    if (
        cfg.hicache_size <= 0
        and cfg.hicache_ratio is None
        and cfg.disaggregation_mode != "decode"
    ):
        raise ValueError(
            f"--hicache-host-memory-mode {cfg.hicache_host_memory_mode} "
            "requires a host pool size: pass --hicache-size or "
            "--hicache-ratio."
        )

    if cfg.hicache_host_memory_mode == "cache":
        return

    if cfg.hicache_storage_backend is None:
        raise ValueError(
            "--hicache-host-memory-mode buffer_only requires a storage backend "
            "(--hicache-storage-backend): host memory is only a staging buffer "
            "and all cached data lives in storage."
        )
    if cfg.hicache_write_policy == "write_back":
        raise ValueError(
            "--hicache-host-memory-mode buffer_only does not support "
            "--hicache-write-policy write_back; use write_through or "
            "write_through_selective."
        )
    if cfg.disaggregation_mode == "decode":
        raise ValueError(
            "--hicache-host-memory-mode buffer_only is not supported on "
            "decode instances: the decode-side prefetch and offload paths "
            "bypass the buffer-mode pipeline, fetching without its prefix "
            "context and never consuming its staged holds. Prefill "
            "instances share the standard scheduler path and are supported."
        )
