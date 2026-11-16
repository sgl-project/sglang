# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the hierarchical KV cache."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import resolving_view, use_mla_backend
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

UNIFIED_KEY_SCHEME_STORAGE_BACKENDS = ("file", "mooncake")


def handle_hicache(server_args: Any):
    """Normalize hicache-related knobs into a valid runtime configuration.

    Resolution order:
    1) Layout <-> I/O compatibility for direct conflicts.
    2) Storage <-> layout compatibility (may rewrite layout).
    """
    cfg = resolving_view(server_args)
    from sglang.srt.mem_cache.unified_cache.linker_config import (
        UnifiedCacheLinkerConfig,
        parse_linker_load_options,
    )

    config = cfg.unified_cache_external_linker_config
    _, _, backend_config = parse_linker_load_options(config)
    if cfg.unified_cache_external_linker_backend == "dynamic" and (
        cfg.enable_unified_cache_external_linker or config is not None
    ):
        UnifiedCacheLinkerConfig.from_dict(
            backend_config if config is not None else None
        )
    if cfg.enable_linker_mla_dedup and (
        not cfg.enable_unified_cache_external_linker
        or cfg.unified_cache_external_linker_backend != "mooncake"
    ):
        raise ValueError("--enable-linker-mla-dedup requires the Mooncake linker.")
    if cfg.enable_unified_cache_external_linker:
        if cfg.pp_size > 1:
            raise ValueError(
                "--enable-unified-cache-external-linker does not support "
                "pipeline parallelism; use --pp-size 1."
            )
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
    if not _hicache_resolution_active(cfg):
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


def _hicache_resolution_active(cfg: Any) -> bool:
    """Whether any path that builds a HiCache host pool is on."""
    return bool(
        cfg.enable_hierarchical_cache
        or cfg.disaggregation_decode_enable_offload_kvcache
        or (
            cfg.disaggregation_mode == "decode"
            and cfg.disaggregation_decode_retraction_backup in (None, "host_pool")
        )
    )


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
    if cfg.speculative_algorithm not in (None, "DSPARK", "DFLASH"):
        raise NotImplementedError(
            "HiCache with --dcp-size > 1 only supports DSPARK and DFLASH "
            "speculative decoding."
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
    if cfg.hicache_storage_backend not in UNIFIED_KEY_SCHEME_STORAGE_BACKENDS:
        raise NotImplementedError(
            f"--hicache-storage-key-scheme unified supports the "
            f"{', '.join(UNIFIED_KEY_SCHEME_STORAGE_BACKENDS)} backends; got "
            f"{cfg.hicache_storage_backend!r}."
        )
    # Speculative decoding is checked by validate_unified_key_scheme_speculative()
    # once the speculative hook has resolved the algorithm.
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


# Speculative algorithms that keep no draft KV in HiCache: their host pages
# hold the target KV alone, so the unified keys of a deployment without
# speculative decoding name every byte.
_UNIFIED_KEY_SCHEME_DRAFT_FREE_ALGORITHMS = ("FROZEN_KV_MTP",)
# Algorithms whose draft KV is stored under unified keys. EAGLE covers
# NEXTN-style MTP, whose one-layer draft is packed onto the target's page
# block; an EAGLE draft that does not pack is refused once it is loaded.
# DFLASH joins them when it runs without its draft KV ring.
_UNIFIED_KEY_SCHEME_DRAFT_KV_ALGORITHMS = ("EAGLE",)


def validate_unified_key_scheme_speculative(server_args: Any) -> None:
    """Refuse speculative algorithms whose draft KV has no unified L3 keys.

    Runs after handle_speculative_decoding(): which HiCache draft mode an
    algorithm takes depends on the resolved algorithm (NEXTN is an alias, a
    Gemma4 assistant is promoted to FROZEN_KV_MTP) and on the resolved DFLASH
    draft KV ring, none of which handle_hicache() can see.
    """
    cfg = resolving_view(server_args)
    if (
        cfg.hicache_storage_key_scheme != "unified"
        or cfg.enable_unified_cache_external_linker
        or not _hicache_resolution_active(cfg)
    ):
        if cfg.hicache_storage_draft_id is not None:
            raise ValueError(
                "--hicache-storage-draft-id names draft KV in unified L3 keys; it "
                "needs --hicache-storage-key-scheme unified on the hierarchical "
                "cache."
            )
        return
    algorithm = cfg.speculative_algorithm
    stores_draft_kv = algorithm in _UNIFIED_KEY_SCHEME_DRAFT_KV_ALGORITHMS or (
        # A ring holds no prefix, so no draft KV follows a page to L3. Without
        # it the draft KV lives in a host pool of its own, beside the target's.
        algorithm == "DFLASH" and not cfg.speculative_dflash_draft_kv_ring
    )
    if not stores_draft_kv:
        if cfg.hicache_storage_draft_id is not None:
            raise ValueError(
                "--hicache-storage-draft-id is set, but "
                f"--speculative-algorithm {algorithm} stores no draft KV under "
                "unified L3 keys."
            )
    if algorithm is None:
        return
    if (
        not cfg.enable_hierarchical_cache
        or cfg.disaggregation_decode_enable_offload_kvcache
    ):
        # Decode offload and retraction backup build their own host pools and
        # controllers; speculative decoding is not covered there.
        raise NotImplementedError(
            "--hicache-storage-key-scheme unified supports speculative decoding "
            "only with --enable-hierarchical-cache, and not with decode KV "
            "cache offload."
        )
    if algorithm in _UNIFIED_KEY_SCHEME_DRAFT_FREE_ALGORITHMS:
        return
    if stores_draft_kv:
        _validate_unified_draft_kv(server_args)
        return
    if algorithm == "DFLASH":
        return
    raise NotImplementedError(
        f"--hicache-storage-key-scheme unified does not support "
        f"--speculative-algorithm {algorithm} (supported: NEXTN-style MTP, "
        f"FROZEN_KV_MTP, DFLASH). Use --hicache-storage-key-scheme rank-suffix."
    )


def _validate_unified_draft_kv(server_args: Any) -> None:
    """Requirements of an algorithm whose draft KV is stored under unified keys.

    Draft chunks are required for a hit and, once written, never rewritten,
    so a deployment must not be able to publish draft rows that are missing,
    random, or produced on a path whose draft KV was not verified.
    """
    cfg = resolving_view(server_args)
    scheme = "--hicache-storage-key-scheme unified"
    algorithm = cfg.speculative_algorithm
    draft_load_format = cfg.speculative_draft_load_format or cfg.load_format
    refusals = (
        (
            cfg.enable_multi_layer_eagle,
            "multi-layer EAGLE: the rows of a deeper draft layer depend on "
            "tokens that the page's key does not cover",
        ),
        (
            cfg.hicache_host_memory_mode == "buffer_only",
            "--hicache-host-memory-mode buffer_only",
        ),
        (
            draft_load_format == "dummy",
            "a draft with random weights (load format 'dummy'): its KV differs "
            "from process to process",
        ),
        (cfg.attn_cp_size > 1, "prefill context parallelism (--attn-cp-size > 1)"),
        (
            # Decided by the server arguments, so not left to the controller,
            # which refuses the draft KV pool only once the weights are loaded.
            algorithm == "DFLASH" and cfg.pp_size > 1,
            "pipeline parallelism (--pp-size > 1): the draft KV pool needs a "
            "single pipeline stage",
        ),
        (
            cfg.disaggregation_mode != "null",
            f"PD disaggregation (--disaggregation-mode {cfg.disaggregation_mode})",
        ),
        (cfg.enable_streaming_session, "--enable-streaming-session"),
        (cfg.enable_session_radix_cache, "--enable-session-radix-cache"),
        (
            algorithm == "EAGLE" and envs.SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND.get(),
            "SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND=1: skipped steps leave draft "
            "rows unwritten",
        ),
        (
            algorithm == "DFLASH" and cfg.hicache_storage_backend == "mooncake",
            "the Mooncake backend, which has no unified keys for the draft KV "
            "pool of DFLASH without its draft KV ring (the 'DFLASH draft KV "
            "ring' log line says why the ring is off)",
        ),
    )
    for refused, what in refusals:
        if refused:
            raise NotImplementedError(
                f"{scheme} stores the draft KV of --speculative-algorithm "
                f"{algorithm} and does not support {what}. Use "
                f"--hicache-storage-key-scheme rank-suffix."
            )

    draft_id = cfg.hicache_storage_draft_id
    if draft_id is None:
        # Without a draft path the draft loads from the target checkpoint, at
        # the target's revision (ModelConfig.from_server_args).
        target_revision = cfg.revision or "main"
        draft_path = cfg.speculative_draft_model_path or cfg.model_path
        draft_revision = cfg.speculative_draft_model_revision or (
            target_revision if cfg.speculative_draft_model_path is None else "main"
        )
        if draft_path == cfg.model_path and draft_revision == target_revision:
            # Part of the target checkpoint, which the target namespace names.
            draft_id = "bundled"
        else:
            draft_id = f"{draft_path}@{draft_revision}"
        if cfg.speculative_draft_model_quantization:
            # The same checkpoint quantized is another draft.
            draft_id += f"+{cfg.speculative_draft_model_quantization}"
        declare_resolution(
            server_args,
            "validate_unified_key_scheme_speculative",
            hicache_storage_draft_id=draft_id,
        )
    elif not draft_id:
        raise ValueError("--hicache-storage-draft-id must not be empty.")
    logger.info("HiCache unified L3 keys: draft id %r for %s.", draft_id, algorithm)


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
