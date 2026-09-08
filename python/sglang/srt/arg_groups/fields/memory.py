"""Config fields of the ``memory`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``memory`` bag, which is what ``get_memory()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import dataclasses
import json
from typing import (
    Any,
    Dict,
    Optional,
)

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
)
from sglang.srt.arg_groups.choices import RADIX_EVICTION_POLICY_CHOICES


@dataclasses.dataclass
class Memory:
    """Namespace ``memory``."""

    _NS_PATH = "memory"
    radix_eviction_policy: A[
        str,
        Arg(
            help=(
                "The eviction policy of radix trees. 'lru' stands for Least "
                "Recently Used, 'lfu' stands for Least Frequently Used, 'slru' "
                "stands for Segmented Least Recently Used, and 'priority' evicts "
                "lower-priority requests first. See "
                "https://docs.sglang.io/docs/advanced_features/radix_eviction_policy "
                "for what each policy optimizes for."
            ),
            choices=RADIX_EVICTION_POLICY_CHOICES,
        ),
    ] = "lru"
    radix_eviction_policy_config: A[
        Optional[Dict[str, Any]],
        Arg(
            help=(
                "Tuning parameters for --radix-eviction-policy, as a json object "
                "passed to the policy as keyword arguments. Only 'slru' takes any "
                "today: protected_threshold (int, default 2), e.g. "
                "'{\"protected_threshold\": 4}'. An unrecognized key fails at "
                "startup, naming the key and the policy. See "
                "https://docs.sglang.io/docs/advanced_features/radix_eviction_policy#policy-parameters "
                "for the full parameter list."
            ),
            type_parser=json.loads,
        ),
    ] = None
    disable_radix_cache: A[
        bool,
        Arg(
            help="Disable RadixAttention for prefix caching.",
            resolvable=True,
        ),
    ] = False
    enable_page_major_kv_layout: A[
        bool,
        "Enable the page-major KV layout: lay out the Mamba state and full/SWA "
        "KV caches in a page-granularity envelope (page is the outermost axis, "
        "layer-major within a page) instead of the default per-layer "
        "(layer-major) layout. Requires the Triton attention / linear-attn / "
        "Mamba backends.",
    ] = False
    enable_unified_memory: A[
        bool,
        "Replace the statically-partitioned hybrid-model pools (full-attn KV + "
        "SWA/Mamba state) with one byte buffer split dynamically between "
        "sub-pools. Requires the Triton attention / linear-attn / Mamba "
        "backends; not yet compatible with PD disaggregation or speculative "
        "decoding.",
    ] = False
    enable_session_radix_cache: A[
        bool,
        "Track per-session references on UnifiedRadixCache KV: eviction consumes unreferenced entries before referenced ones, and closing a session only dereferences its KV.",
    ] = False
    radix_cache_backend: A[
        Optional[str],
        "Name of a radix-cache backend previously registered via register_radix_cache_backend. Omit this flag to use the built-in default cache selection chain.",
    ] = None

    # -------------------------------------------------------------------------
    # Hierarchical cache
    # -------------------------------------------------------------------------
    enable_hierarchical_cache: A[bool, "Enable hierarchical cache"] = False
    hicache_host_memory_mode: A[
        str,
        Arg(
            help="Whether host memory is a persistent HiCache tier (cache) or a transient staging buffer between GPU and the storage backend (buffer_only). buffer_only requires --hicache-storage-backend.",
            choices=["cache", "buffer_only"],
        ),
    ] = "cache"
    hicache_ratio: A[
        Optional[float],
        "The ratio of the size of host KV cache memory pool to the size of device pool. Defaults to 2.0 in cache mode, 1.2 in buffer_only mode, or 0.2 for backup-only host-pool decode retraction.",
    ] = None
    hicache_size: A[
        int,
        "The size of host KV cache memory pool in gigabytes. Overrides --hicache-ratio in either host memory mode.",
    ] = 0
    hicache_write_policy: A[
        str,
        Arg(
            help="The write policy of hierarchical cache.",
            choices=["write_back", "write_through", "write_through_selective"],
        ),
    ] = "write_through"
    hicache_io_backend: A[
        str,
        Arg(
            help="The IO backend for KV cache transfer between CPU and GPU",
            choices=["direct", "kernel", "kernel_ascend"],
        ),
    ] = "kernel"
    hicache_mem_layout: A[
        str,
        Arg(
            help="The layout of host memory pool for hierarchical cache.",
            choices=[
                "layer_first",
                "page_first",
                "page_first_direct",
                "page_first_kv_split",
                "page_head",
            ],
        ),
    ] = "page_first"
    hicache_storage_backend: A[
        Optional[str],
        Arg(
            help="The storage backend for hierarchical KV cache. Built-in backends: file, mooncake, hf3fs, nixl, aibrix. For dynamic backend, use --hicache-storage-backend-extra-config to specify: backend_name (custom name), module_path (Python module path), class_name (backend class name).",
            choices=[
                "file",
                "sim",
                "mooncake",
                "hf3fs",
                "nixl",
                "aibrix",
                "dynamic",
                "eic",
                "simm",
                "mori",
                "shm",
            ],
        ),
    ] = None
    hicache_storage_prefetch_policy: A[
        str,
        Arg(
            help="Control when prefetching from the storage backend should stop.",
            choices=["best_effort", "wait_complete", "timeout"],
        ),
    ] = "timeout"
    hicache_storage_backend_extra_config: A[
        Optional[str],
        "A dictionary in JSON string format, or a string starting with a leading '@' and a config file in JSON/YAML/TOML format, containing extra configuration for the storage backend.",
    ] = None
    hicache_storage_prefetch_retry_poll_interval: A[
        int,
        Arg(
            help=(
                "Scheduling passes a queued request waits after a storage "
                "prefetch miss before the availability check is retried "
                "(under load the first check can run before the needed "
                "backup commits). 0 disables retries."
            ),
        ),
    ] = 0
    hicache_storage_prefetch_retry_max_attempts: A[
        int,
        "Maximum storage prefetch retries per request when --hicache-storage-prefetch-retry-poll-interval is set.",
    ] = 4

    # -------------------------------------------------------------------------
    # Unified Radix Cache
    # -------------------------------------------------------------------------
    enable_unified_cache_external_linker: A[
        bool,
        "Link UnifiedRadixCache directly to an external KV store (direct L3), with no host cache tier.",
    ] = False
    unified_cache_external_linker_backend: A[
        str,
        Arg(
            help="Storage backend for --enable-unified-cache-external-linker.",
            choices=["mooncake", "mori"],
        ),
    ] = "mooncake"

    # -------------------------------------------------------------------------
    # Hierarchical sparse attention
    # -------------------------------------------------------------------------
    enable_hisparse: A[bool, "Enable hierarchical sparse attention"] = False
    hisparse_config: A[
        Optional[str],
        Arg(
            help='A dictionary in JSON string format for hierarchical sparse attention configuration. Example: \'{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 2}\'',
            aliases=["--hierarchical-sparse-attention-extra-config"],
        ),
    ] = None

    # -------------------------------------------------------------------------
    # LMCache
    # -------------------------------------------------------------------------
    enable_lmcache: A[
        bool, "Using LMCache as an alternative hierarchical cache solution"
    ] = False
    lmcache_config_file: A[
        Optional[str],
        "Path to the LMCache YAML configuration file",
    ] = None

    # -------------------------------------------------------------------------
    # FlexKV
    # -------------------------------------------------------------------------
    enable_flexkv: A[
        bool,
        (
            "Route the default RadixCache through FlexKV's KVManager for "
            "host-tier (CPU / SSD / Remote) KV cache offload. Equivalent "
            "to --radix-cache-backend=flexkv but also participates in the "
            "auto-selection chain alongside --enable-lmcache."
        ),
    ] = False
    flexkv_config_file: A[
        Optional[str],
        (
            "Path to the FlexKV YAML / JSON configuration file. "
            "Equivalent to setting the FLEXKV_CONFIG_PATH environment "
            "variable."
        ),
    ] = None
