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
    """Normalize hicache-related knobs into a valid runtime configuration.

    Resolution order:
    1) Layout <-> I/O compatibility for direct conflicts.
    2) Storage <-> layout compatibility (may rewrite layout).
    """
    cfg = resolving_view(server_args)
    # Step 0: L3 key-scheme validation. Runs before the early return so
    # unified flags are never silently inert.
    resolve_hicache_key_scheme(server_args)

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
    # Topologies whose at-rest KV is not a dense per-rank rectangle of
    # whole pages cannot be named by unified chunks yet. Checked here
    # (not only at attach) because the decode-offload attach path has no
    # CP/DCP group wired into its controller.
    if cfg.dcp_size > 1:
        raise NotImplementedError(
            "the unified key scheme with --dcp-size > 1 is not supported: each "
            "DCP rank holds an interleaved token shard (needs the "
            "token-granule extension)."
        )
    # attn_cp_size is resolvable (prefill-CP overrides stash it without
    # mutating the raw field), so read it through the resolved view.
    if cfg.attn_cp_size > 1:
        raise NotImplementedError(
            "the unified key scheme with --attn-cp-size > 1 is not supported: "
            "CP ranks hold sub-page slices or replicated pages (needs "
            "token-granule chunks / writer election)."
        )
    # Best-effort early check of the partition knobs when the extra
    # config is inline JSON (the '@file' form is re-validated at attach).
    # Any knob selects the KV layout adapter: objects use the unified
    # byte order, so there is no host-layout requirement, but the
    # per-chunk key fan-out needs a multi-key backend.
    extra = cfg.hicache_storage_backend_extra_config
    if extra and not extra.startswith("@"):
        try:
            extra_dict = json.loads(extra)
            tp_lcm_size = extra_dict.get("tp_lcm_size")
            head_group = extra_dict.get("head_group")
            layer_partition = extra_dict.get("layer_partition")
        except (ValueError, AttributeError):
            tp_lcm_size = None
            head_group = None
            layer_partition = None
        if tp_lcm_size:
            raise ValueError(
                "tp_lcm_size is the legacy rank-suffix split-heads knob; "
                "the unified key scheme uses head_group in the extra "
                "config (heads per chunk)."
            )
        # head_group is ignored on rank-replicated (MLA-family) pools, so
        # a shared fleet extra-config must not be rejected for them here.
        adapter = layer_partition is not None or (
            head_group and not use_mla_backend(server_args)
        )
        if adapter and cfg.hicache_storage_backend != "mooncake":
            raise NotImplementedError(
                "unified-scheme partition knobs (head_group / "
                "layer_partition) need a multi-key-per-page backend; "
                "only mooncake supports them."
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


def _unified_adapter_knobs(server_args: Any) -> tuple[bool, bool]:
    """Return ``(adapter, head_cut)`` for the unified L3 partition knobs.

    ``adapter`` -- any partition knob is set, so L3 objects use the unified
    byte order and the host layout must be ``page_first_direct`` (the only
    layout whose page block IS that order).

    ``head_cut`` -- the kv-head axis is partitioned, so page blocks become
    head-group-major and the KV transfer must run through the permuting pfdhg
    kernels, i.e. the ``kernel`` io backend. MLA has no kv-head axis to cut,
    so its pages stay in the natural order and only need the layout.

    Read from inline JSON only; the '@file' form and the runtime attach
    endpoint are re-checked in HiCacheController._build_unified_suffix, which
    is where a mismatch raises.
    """
    cfg = resolving_view(server_args)
    if cfg.hicache_storage_key_scheme != "unified":
        return False, False
    extra = cfg.hicache_storage_backend_extra_config
    if not extra or extra.startswith("@"):
        return False, False
    try:
        parsed = json.loads(extra)
        head_group = parsed.get("head_group")
        layer_partition = parsed.get("layer_partition")
    except (ValueError, AttributeError):
        return False, False
    head_cut = bool(head_group) and not use_mla_backend(server_args)
    adapter = head_cut or layer_partition is not None
    return adapter, head_cut


def resolve_layout_io_compatibility(server_args: Any):
    cfg = resolving_view(server_args)
    adapter, head_cut = _unified_adapter_knobs(server_args)
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

    if adapter:
        # Every partition knob needs page_first_direct: it is the only host
        # layout whose page block already IS the unified byte order, so an L3
        # chunk is one contiguous range. On page_first no chunk is contiguous
        # at any page size > 1, which is why it is no longer an adapter layout.
        # Steer here rather than let attach raise -- the mooncake default
        # steering below would otherwise land a layer_partition-only
        # deployment on page_first and fail at startup.
        if cfg.hicache_mem_layout != "page_first_direct":
            declare_resolution(
                server_args,
                "_resolve_layout_io_compatibility",
                hicache_mem_layout="page_first_direct",
            )
            logger.warning(
                "The unified key scheme's partition knobs require the "
                "page_first_direct host layout, switching to it"
            )
    if head_cut:
        # Head-group-major page blocks are only readable by the pfdhg transfer
        # kernels, which are the 'kernel' io backend; the copy-engine 'direct'
        # path can only move a page block verbatim.
        if cfg.hicache_io_backend != "kernel":
            declare_resolution(
                server_args,
                "_resolve_layout_io_compatibility",
                hicache_io_backend="kernel",
            )
            logger.warning(
                "The unified key scheme with head_group stores host pages "
                "head-group-major, which only the kernel io backend can read; "
                "switching to the kernel io backend"
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
