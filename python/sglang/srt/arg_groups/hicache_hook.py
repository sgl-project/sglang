# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the hierarchical KV cache."""

from __future__ import annotations

import json
import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
    use_mla_backend,
)

logger = logging.getLogger(__name__)


def handle_hicache(server_args: Any):
    """Normalize hicache-related configs into a valid runtime configuration.

    Resolution order:
    1) Layout <-> I/O compatibility for direct conflicts.
    2) Storage <-> layout compatibility (may rewrite layout).
    3) The unified layout / io backend, which reads the layout both of the
       above may have rewritten.
    """
    cfg = resolving_view(server_args)
    # Step 0: L3 key-scheme validation. A no-op for the default rank-suffix
    # scheme, so it runs ahead of every early return below and a unified flag
    # is never silently inert -- including under the external linker, which
    # has no L3 of its own.
    resolve_hicache_key_scheme(server_args)

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

    # Step 1: Initial layout-io compatibility normalization.
    resolve_layout_io_compatibility(server_args)

    # Step 2: Storage-layout normalization without changing io backend.
    resolve_storage_layout_compatibility(server_args)

    # Step 3: the unified layout / io backend, after every rewrite above.
    resolve_unified_layout_io(server_args)

    # Step 4: DCP compatibility for the L2 (device<->host) path.
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


def resolve_hicache_key_scheme(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.hicache_storage_key_scheme == "rank-suffix":
        return
    if not (
        cfg.enable_hierarchical_cache
        or cfg.disaggregation_decode_enable_offload_kvcache
    ):
        raise ValueError(
            "--hicache-storage-key-scheme unified has no effect "
            "without --enable-hierarchical-cache (or decode KV offload); "
            "refusing a silently inert flag."
        )
    if cfg.hicache_storage_backend is None:
        raise ValueError(
            "--hicache-storage-key-scheme unified requires an L3 "
            "backend (--hicache-storage-backend)."
        )
    if cfg.hicache_storage_backend not in ("file", "mooncake"):
        raise NotImplementedError(
            "the unified key scheme v1 supports --hicache-storage-backend file "
            f"or mooncake; got {cfg.hicache_storage_backend!r}. Other "
            "backends need chunk-granular key support first."
        )
    if cfg.speculative_algorithm is not None:
        raise NotImplementedError(
            "the unified key scheme does not cover speculative-decoding draft "
            "pools yet; use --hicache-storage-key-scheme rank-suffix."
        )
    # TODO: a unified chunk names whole pages, but both context-parallel modes
    # shard a page below that granularity -- DCP interleaves tokens across
    # ranks, attention CP holds sub-page slices (NSA) or replicated pages. Both
    # need the token-granule extension; support them once the layouts converge.
    # Checked here, not only at attach, because the decode-offload attach path
    # has no CP/DCP group wired into its controller.
    if cfg.dcp_size > 1:
        raise NotImplementedError(
            "the unified key scheme with --dcp-size > 1 is not supported: each "
            "DCP rank holds an interleaved token shard (needs the "
            "token-granule extension)."
        )
    # Read through the resolved view: prefill-CP overrides stash attn_cp_size
    # without mutating the raw field.
    if cfg.attn_cp_size > 1:
        raise NotImplementedError(
            "the unified key scheme with --attn-cp-size > 1 is not supported: "
            "CP ranks hold sub-page slices or replicated pages (needs "
            "token-granule chunks / writer election)."
        )

    configs = _unified_extra_config(server_args)
    if configs is not None:
        tp_lcm_size = configs.get("tp_lcm_size")
        head_group = configs.get("head_group")
        layer_partition = configs.get("layer_partition")
        if tp_lcm_size:
            raise ValueError(
                "tp_lcm_size is the legacy rank-suffix split-heads config; "
                "the unified key scheme uses head_group in the extra "
                "config (heads per chunk)."
            )

        _validate_partition_config("head_group", head_group)
        _validate_partition_config("layer_partition", layer_partition)
        # head_group is ignored on rank-replicated (MLA-family) pools, so a
        # shared fleet extra-config must not be rejected for them here.
        adapter = layer_partition is not None or (
            head_group is not None and not use_mla_backend(server_args)
        )
        if adapter and cfg.hicache_storage_backend != "mooncake":
            # Not "the file backend cannot do unified" -- it can, and v1
            # supports it. What it cannot do is the per-chunk key fan-out these
            # two configs create: it stores one object per page.
            raise NotImplementedError(
                "unified-scheme partition configs (head_group / "
                "layer_partition) need a multi-key-per-page backend; only "
                "mooncake supports them (--hicache-storage-backend "
                f"{cfg.hicache_storage_backend!r} stores one object per page). "
                "Drop them to run the unified scheme on this backend."
            )


def _validate_partition_config(name: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"{name} in --hicache-storage-backend-extra-config must be a "
            f"positive integer, got {value!r}."
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


def _unified_extra_config(server_args: Any) -> dict | None:
    cfg = resolving_view(server_args)
    if cfg.hicache_storage_key_scheme != "unified":
        return None
    extra = cfg.hicache_storage_backend_extra_config
    if not extra or extra.startswith("@"):
        return None
    try:
        parsed = json.loads(extra)
    except (ValueError, AttributeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _unified_scheme(server_args: Any) -> bool:
    cfg = resolving_view(server_args)
    return cfg.hicache_storage_key_scheme == "unified"


def _unified_head_cut(server_args: Any) -> bool:
    configs = _unified_extra_config(server_args)
    if configs is None:
        return False
    return configs.get("head_group") is not None and not use_mla_backend(server_args)


def resolve_layout_io_compatibility(server_args: Any):
    """Settle (layout, io_backend) into a combination the pools can serve."""
    cfg = resolving_view(server_args)
    head_cut = _unified_head_cut(server_args)
    if (
        cfg.hicache_mem_layout == "page_first_direct"
        and cfg.hicache_io_backend == "kernel"
        and not head_cut
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


def resolve_unified_layout_io(server_args: Any):
    """Pin the unified scheme's host layout and io backend, overriding both.

    Layout is always page_first_direct: it enters the namespace digest, so
    operator choice would split one model across keyspaces, and it is the only
    layout that serves an L3 chunk in a single descriptor rather than one per
    (layer, token). The io backend then follows the head cut -- kernel to read
    head-group-major page blocks, direct otherwise.
    """
    cfg = resolving_view(server_args)
    if not _unified_scheme(server_args):
        return
    # The Ascend io backend reads exactly one layout per pool family
    # (page_first_direct for MHA, page_first_kv_split for MLA) and is pinned by
    # the platform hook, which runs BEFORE this one. Rewriting the layout under
    # it would produce a pair its transfer arms reject -- at the first L2
    # transfer, not at launch -- so refuse here instead.
    if cfg.hicache_io_backend == "kernel_ascend":
        raise NotImplementedError(
            "the unified key scheme does not support --hicache-io-backend "
            f"kernel_ascend (--hicache-mem-layout {cfg.hicache_mem_layout}): "
            "the Ascend transfer path reads only the layout the NPU platform "
            "pins, which the unified object order cannot be normalized onto. "
            "Use --hicache-storage-key-scheme rank-suffix."
        )
    # Read the layout being replaced BEFORE declaring: `cfg` is a live view of
    # the declarations, so after declare_resolution it already reports the new
    # value and the warning would name page_first_direct as its own origin.
    previous_layout = cfg.hicache_mem_layout
    if previous_layout != "page_first_direct":
        declare_resolution(
            server_args,
            "_resolve_unified_layout_io",
            hicache_mem_layout="page_first_direct",
        )
        logger.warning(
            "The unified key scheme serves an L3 chunk at one descriptor only "
            "from page_first_direct page blocks; switching "
            "--hicache-mem-layout from %s. (Other layouts hold the same bytes "
            "but serve each chunk as layers x tokens separate runs.)",
            previous_layout,
        )
    # Only a HEAD cut makes page blocks head-group-major, and only the pfdhg
    # transfer kernels can read that; the copy-engine 'direct' path moves a
    # page block verbatim. A layer partition alone leaves the natural order.
    if _unified_head_cut(server_args):
        if cfg.hicache_io_backend != "kernel":
            declare_resolution(
                server_args,
                "_resolve_unified_layout_io",
                hicache_io_backend="kernel",
            )
            logger.warning(
                "The unified key scheme with head_group stores host pages "
                "head-group-major, which only the kernel io backend can read; "
                "switching to the kernel io backend"
            )
    elif cfg.hicache_io_backend == "kernel":
        # Re-apply the rule the first resolver would have: it saw the layout
        # BEFORE this one rewrote it, so without this an operator asking for
        # page_first + kernel and one asking for page_first_direct + kernel end
        # on the same layout but different io backends.
        declare_resolution(
            server_args,
            "_resolve_unified_layout_io",
            hicache_io_backend="direct",
        )
        logger.warning(
            "Kernel io backend does not support page first direct layout, "
            "switching to direct io backend"
        )


def resolve_storage_layout_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    if (
        cfg.hicache_storage_backend != "mooncake"
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
        f"Mooncake storage backend does not support layer_first layout, "
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
