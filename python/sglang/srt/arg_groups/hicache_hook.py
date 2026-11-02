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
    config = cfg.unified_cache_external_linker_config
    if config is not None and not isinstance(config, dict):
        raise ValueError(
            "--unified-cache-external-linker-config must be a JSON object."
        )
    if cfg.unified_cache_external_linker_backend == "dynamic" and (
        cfg.enable_unified_cache_external_linker or config is not None
    ):
        from sglang.srt.mem_cache.unified_cache.linker_config import (
            UnifiedCacheLinkerConfig,
        )

        UnifiedCacheLinkerConfig.from_dict(config)
    if cfg.enable_linker_mla_dedup and (
        not cfg.enable_unified_cache_external_linker
        or cfg.unified_cache_external_linker_backend != "mooncake"
    ):
        raise ValueError("--enable-linker-mla-dedup requires the Mooncake linker.")
    if cfg.enable_unified_cache_external_linker:
        if cfg.disable_radix_cache or cfg.radix_cache_backend is not None:
            raise ValueError(
                "--enable-unified-cache-external-linker requires the default "
                "UnifiedRadixCache with radix caching enabled."
            )
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

    if cfg.disaggregation_decode_host_receive_threshold < 1:
        # The shared host pool must match KV transfer's per-layer buffers.
        declare_resolution(
            server_args, "handle_hicache", hicache_mem_layout="layer_first"
        )

    # The key scheme pins the layout and I/O backend before compatibility checks.
    resolve_hicache_key_scheme(server_args)

    # Step 1: Initial layout-io compatibility normalization.
    resolve_layout_io_compatibility(server_args)

    # Step 2: Storage-layout normalization without changing io backend.
    resolve_storage_layout_compatibility(server_args)
    resolve_unified_memory_layout_io(server_args)
    if (
        cfg.disaggregation_decode_host_receive_threshold < 1
        and cfg.hicache_mem_layout != "layer_first"
    ):
        raise ValueError(
            f"The resolved HiCache storage layout {cfg.hicache_mem_layout!r} "
            "cannot share the layer_first decode host pool used by KV transfer"
        )

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

    An explicit --hicache-ratio or --hicache-size is honored as given, so it
    resolves --hicache-host-memory-fraction to None (auto-sizing off).
    """
    cfg = resolving_view(server_args)
    fraction = cfg.hicache_host_memory_fraction
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("--hicache-host-memory-fraction must be in (0, 1].")
    fields = {}
    if cfg.hicache_ratio is None and cfg.disaggregation_mode != "decode":
        fields["hicache_ratio"] = (
            1.2 if cfg.hicache_host_memory_mode == "buffer_only" else 2.0
        )
    if cfg.hicache_ratio is not None or cfg.hicache_size > 0:
        fields["hicache_host_memory_fraction"] = None
    if fields:
        declare_resolution(server_args, "_handle_hicache_ratio_default", **fields)


def resolve_hicache_dcp_compatibility(server_args: Any):

    cfg = resolving_view(server_args)
    if cfg.dcp_size <= 1 or not cfg.enable_hierarchical_cache:
        return
    if (
        cfg.hicache_storage_backend is not None
        and cfg.hicache_host_memory_mode == "buffer_only"
    ):
        raise NotImplementedError(
            "HiCache L3 with DCP does not support buffer_only host memory."
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
        "HiCache + DCP enabled: host pool uses widened "
        "logical slot accounting with per-rank physical translation at "
        "the transfer boundary (dcp_size=%d).",
        cfg.dcp_size,
    )


def resolve_hicache_key_scheme(server_args: Any):
    """Validate --hicache-storage-key-scheme and resolve what it implies."""
    cfg = resolving_view(server_args)
    unified = cfg.hicache_storage_key_scheme == "unified"
    partition_configs = (
        cfg.hicache_storage_head_partition is not None
        or cfg.hicache_storage_layer_partition is not None
    )
    if not unified:
        if (
            cfg.hicache_mem_layout == "page_unified"
            and cfg.hicache_storage_backend is not None
        ):
            raise ValueError(
                "--hicache-mem-layout page_unified with storage requires "
                "--hicache-storage-key-scheme unified to distinguish its byte order."
            )
        if partition_configs:
            raise ValueError(
                "--hicache-storage-head-partition / --hicache-storage-layer-partition "
                "require --hicache-storage-key-scheme unified."
            )
        return

    for name in ("hicache_storage_head_partition", "hicache_storage_layer_partition"):
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
    if cfg.hicache_storage_backend != "file":
        raise NotImplementedError(
            f"--hicache-storage-key-scheme unified supports only the file "
            f"backend; got {cfg.hicache_storage_backend!r}."
        )
    if cfg.speculative_algorithm is not None:
        # Draft layers are appended to the host pool's layer axis but are not
        # part of the model-global grid, so they would land inside a named
        # chunk unnamed.
        raise NotImplementedError(
            "--hicache-storage-key-scheme unified does not support speculative "
            "decoding: the draft layers are not part of the L3 layer grid."
        )
    if cfg.dcp_size > 1:
        raise NotImplementedError(
            "--hicache-storage-key-scheme unified does not support decode "
            "context parallelism (--dcp-size > 1): a DCP rank holds "
            "sub-page slices, which need token-granule storage keys."
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
    if cfg.hicache_storage_backend == "mooncake" and cfg.dcp_size > 1:
        # DCP MLA pages support Mooncake multi-buffer transfers. Keep startup
        # and runtime attachment on the same requested host layout.
        return
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


def resolve_unified_memory_layout_io(server_args: Any):
    # The unified pool's token-major entries make each layer's device rows
    # strided; the direct page_first_direct copies assume packed rows.
    cfg = resolving_view(server_args)
    if not cfg.enable_unified_memory or cfg.hicache_mem_layout != "page_first_direct":
        return
    declare_resolution(
        server_args,
        "_resolve_unified_memory_layout_io",
        hicache_mem_layout="page_first",
        hicache_io_backend="kernel",
    )
    logger.warning(
        "--enable-unified-memory: the direct io backend cannot copy the unified "
        "pool's strided KV rows, switching to the page_first layout with the "
        "kernel io backend"
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
        # buffer_only has no host tier to write back to; pages go to storage.
        declare_resolution(
            server_args,
            "_validate_hicache_host_memory_mode",
            hicache_write_policy="write_through",
        )
        logger.info(
            "--hicache-host-memory-mode buffer_only resolves "
            "--hicache-write-policy write_back to write_through"
        )
    if cfg.disaggregation_mode == "decode":
        raise ValueError(
            "--hicache-host-memory-mode buffer_only is not supported on "
            "decode instances: the decode-side prefetch and offload paths "
            "bypass the buffer-mode pipeline, fetching without its prefix "
            "context and never consuming its staged holds. Prefill "
            "instances share the standard scheduler path and are supported."
        )
