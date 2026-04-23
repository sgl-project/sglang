"""UMBPStore — HiCache L3 storage backend using UMBP (local DRAM + SSD).

Follows the same pattern as MooncakeStore:
- Zero-copy v1 interface (batch_get_v1 / batch_set_v1)
- Uses mem_pool_host.get_page_buffer_meta() for pointer/size extraction
- Key suffix generation per TP rank / PP rank
"""

import logging
import os
import socket
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


def _import_umbp_client():
    """Import UMBPClient from mori.umbp (requires mori built with BUILD_UMBP=ON)."""
    import mori.umbp as umbp_mod

    UMBPClient = umbp_mod.UMBPClient
    UMBPConfig = umbp_mod.UMBPConfig
    UMBPRole = umbp_mod.UMBPRole
    UMBPIoBackend = getattr(umbp_mod, "UMBPIoBackend", None)
    UMBPDurabilityMode = getattr(umbp_mod, "UMBPDurabilityMode", None)
    UMBPDistributedConfig = getattr(umbp_mod, "UMBPDistributedConfig", None)

    return (
        UMBPClient,
        UMBPConfig,
        UMBPRole,
        UMBPIoBackend,
        UMBPDurabilityMode,
        UMBPDistributedConfig,
    )


def _optional_env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value is not None else None


def _optional_env_str(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None and value != "" else None


def _bool_from_any(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _default_node_address() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def _select_rank_config_value(
    value: Any,
    rank_index: int,
    field_name: str,
    cast_type,
    auto_increment_scalar: bool = False,
):
    if value is None:
        raise ValueError(f"{field_name} must not be None")

    candidates = value
    if isinstance(value, str) and "," in value:
        candidates = [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(candidates, (list, tuple)):
        if not candidates:
            raise ValueError(f"{field_name} must not be empty")
        if rank_index >= len(candidates):
            raise ValueError(
                f"{field_name} has {len(candidates)} entries, but rank_index={rank_index}"
            )
        return cast_type(candidates[rank_index])

    selected = cast_type(candidates)
    if auto_increment_scalar:
        selected = cast_type(selected + rank_index)
    return selected


class UMBPStore(HiCacheStorage):
    """Local DRAM+SSD storage backend for HiCache L3 caching.

    Compatible with the zero-copy v1 interface used by CacheController.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig = None,
        mem_pool_host: HostKVCache = None,
    ):
        (
            UMBPClient,
            UMBPConfig,
            UMBPRole,
            UMBPIoBackend,
            UMBPDurabilityMode,
            UMBPDistributedConfig,
        ) = _import_umbp_client()

        if storage_config is not None:
            self.is_mla_backend = storage_config.is_mla_model
            self.local_rank = storage_config.tp_rank
            self.pp_rank = storage_config.pp_rank
            self.pp_size = storage_config.pp_size
            self.tp_size = storage_config.tp_size
        else:
            self.is_mla_backend = False
            self.local_rank = 0
            self.pp_rank = 0
            self.pp_size = 1
            self.tp_size = 1

        cfg = UMBPConfig.from_environment()
        # UMBPStore owns role selection explicitly. Do not inherit LOCAL_RANK /
        # UMBP_ROLE-based multi-process defaults from mori here, otherwise
        # ordinary multi-rank sglang runs can accidentally become follower-only
        # and skip writes.
        cfg.role = UMBPRole.Standalone
        extra = getattr(storage_config, "extra_config", None) or {}
        explicit_tenant_id = (
            os.getenv("UMBP_SPDK_PROXY_TENANT_ID") is not None
            or "spdk_proxy_tenant_id" in extra
        )
        tenant_id_base = (
            int(extra["spdk_proxy_tenant_id_base"])
            if "spdk_proxy_tenant_id_base" in extra
            else _optional_env_int("UMBP_SPDK_PROXY_TENANT_ID_BASE")
        )
        dp_rank_hint = _optional_env_int("SGLANG_DP_RANK")
        dp_size_hint = _optional_env_int("SGLANG_DP_SIZE")
        local_rank_hint = _optional_env_int("LOCAL_RANK")

        if dp_rank_hint is None:
            try:
                from sglang.srt.layers.dp_attention import (
                    get_attention_dp_rank,
                    get_attention_dp_size,
                    is_dp_attention_enabled,
                )

                if is_dp_attention_enabled():
                    dp_rank_hint = get_attention_dp_rank()
                    dp_size_hint = get_attention_dp_size()
            except (ImportError, AssertionError):
                pass

        if local_rank_hint is not None:
            unique_rank = local_rank_hint
        else:
            base_rank = dp_rank_hint if dp_rank_hint is not None else 0
            unique_rank = ((base_rank * max(self.pp_size, 1)) + self.pp_rank) * max(
                self.tp_size, 1
            ) + self.local_rank

        # Load settings from extra_config if available
        if "dram_capacity_bytes" in extra:
            cfg.dram.capacity_bytes = int(extra["dram_capacity_bytes"])
        if "ssd_enabled" in extra:
            cfg.ssd.enabled = bool(extra["ssd_enabled"])
        if "ssd_storage_dir" in extra:
            cfg.ssd.storage_dir = str(extra["ssd_storage_dir"])
        if "ssd_capacity_bytes" in extra:
            cfg.ssd.capacity_bytes = int(extra["ssd_capacity_bytes"])
        if "copy_to_ssd_async" in extra:
            cfg.copy_pipeline.async_enabled = bool(extra["copy_to_ssd_async"])
        if "copy_to_ssd_queue_depth" in extra:
            cfg.copy_pipeline.queue_depth = int(extra["copy_to_ssd_queue_depth"])
        if "ssd_segment_size_bytes" in extra:
            cfg.ssd.segment_size_bytes = int(extra["ssd_segment_size_bytes"])
        if "ssd_batch_max_ops" in extra:
            cfg.copy_pipeline.batch_max_ops = int(extra["ssd_batch_max_ops"])
        if "ssd_queue_depth" in extra:
            cfg.ssd.io.queue_depth = int(extra["ssd_queue_depth"])
        if "ssd_writer_threads" in extra:
            cfg.copy_pipeline.worker_threads = int(extra["ssd_writer_threads"])
        if "ssd_enable_background_gc" in extra:
            cfg.ssd.durability.enable_background_gc = bool(
                extra["ssd_enable_background_gc"]
            )
        if "auto_promote_on_read" in extra:
            cfg.eviction.auto_promote_on_read = bool(extra["auto_promote_on_read"])
        if "eviction_policy" in extra:
            cfg.eviction.policy = str(extra["eviction_policy"])
        if "eviction_candidate_window" in extra:
            cfg.eviction.candidate_window = int(extra["eviction_candidate_window"])
        if "ssd_io_backend" in extra and UMBPIoBackend is not None:
            backend = str(extra["ssd_io_backend"]).lower()
            if backend in ("pthread", "posix"):
                cfg.ssd.io.backend = UMBPIoBackend.PThread
            elif backend in ("io_uring", "uring"):
                cfg.ssd.io.backend = UMBPIoBackend.IoUring
        if "ssd_durability_mode" in extra and UMBPDurabilityMode is not None:
            durability = str(extra["ssd_durability_mode"]).lower()
            if durability in ("strict", "sync"):
                cfg.ssd.durability.mode = UMBPDurabilityMode.Strict
            elif durability in ("relaxed", "async"):
                cfg.ssd.durability.mode = UMBPDurabilityMode.Relaxed
        if "ssd_backend" in extra:
            ssd_backend = str(extra["ssd_backend"]).strip().lower()
            if ssd_backend not in ("posix", "spdk", "spdk_proxy"):
                raise ValueError(
                    "extra_config['ssd_backend'] must be one of: "
                    "posix, spdk, spdk_proxy"
                )
            cfg.ssd_backend = ssd_backend
        if "spdk_nvme_pci_addr" in extra:
            cfg.spdk_nvme_pci_addr = str(extra["spdk_nvme_pci_addr"])
        if "spdk_proxy_shm_name" in extra:
            cfg.spdk_proxy_shm_name = str(extra["spdk_proxy_shm_name"])
        if "spdk_proxy_startup_timeout_ms" in extra:
            cfg.spdk_proxy_startup_timeout_ms = int(
                extra["spdk_proxy_startup_timeout_ms"]
            )
        if "spdk_proxy_bin" in extra:
            cfg.spdk_proxy_bin = str(extra["spdk_proxy_bin"])
        if "spdk_proxy_tenant_id" in extra:
            cfg.spdk_proxy_tenant_id = int(extra["spdk_proxy_tenant_id"])
        if "spdk_proxy_tenant_quota_bytes" in extra:
            cfg.spdk_proxy_tenant_quota_bytes = int(
                extra["spdk_proxy_tenant_quota_bytes"]
            )
        if "spdk_proxy_max_channels" in extra:
            cfg.spdk_proxy_max_channels = int(extra["spdk_proxy_max_channels"])
        if "spdk_proxy_data_per_channel_mb" in extra:
            cfg.spdk_proxy_data_per_channel_mb = int(
                extra["spdk_proxy_data_per_channel_mb"]
            )
        if "spdk_proxy_auto_start" in extra:
            cfg.spdk_proxy_auto_start = bool(extra["spdk_proxy_auto_start"])
        if "spdk_proxy_idle_exit_timeout_ms" in extra:
            cfg.spdk_proxy_idle_exit_timeout_ms = int(
                extra["spdk_proxy_idle_exit_timeout_ms"]
            )
        if "spdk_proxy_allow_borrow" in extra:
            cfg.spdk_proxy_allow_borrow = bool(extra["spdk_proxy_allow_borrow"])
        if "spdk_proxy_reserved_shared_bytes" in extra:
            cfg.spdk_proxy_reserved_shared_bytes = int(
                extra["spdk_proxy_reserved_shared_bytes"]
            )

        master_address = extra.get("master_address", _optional_env_str("UMBP_MASTER_ADDRESS"))
        if master_address and UMBPDistributedConfig is not None:
            dist_cfg = UMBPDistributedConfig()
            dist_cfg.master_config.master_address = str(master_address)

            node_address = extra.get("node_address", _optional_env_str("UMBP_NODE_ADDRESS"))
            if node_address is None:
                node_address = _default_node_address()
            else:
                node_address = _select_rank_config_value(
                    node_address,
                    unique_rank,
                    "node_address",
                    str,
                )
            dist_cfg.master_config.node_address = node_address

            node_id = extra.get("node_id", _optional_env_str("UMBP_NODE_ID"))
            if node_id is None:
                dist_cfg.master_config.node_id = (
                    f"{node_address}:dp{dp_rank_hint if dp_rank_hint is not None else 0}"
                    f":pp{self.pp_rank}:tp{self.local_rank}"
                )
            else:
                dist_cfg.master_config.node_id = _select_rank_config_value(
                    node_id,
                    unique_rank,
                    "node_id",
                    str,
                )

            if "auto_heartbeat" in extra:
                dist_cfg.master_config.auto_heartbeat = _bool_from_any(extra["auto_heartbeat"])

            io_engine_host = extra.get(
                "io_engine_host", _optional_env_str("UMBP_IO_ENGINE_HOST")
            )
            if io_engine_host is None:
                io_engine_host = node_address
            else:
                io_engine_host = _select_rank_config_value(
                    io_engine_host,
                    unique_rank,
                    "io_engine_host",
                    str,
                )
            dist_cfg.io_engine.host = io_engine_host

            io_engine_port = extra.get(
                "io_engine_port", _optional_env_str("UMBP_IO_ENGINE_PORT")
            )
            if io_engine_port is not None:
                dist_cfg.io_engine.port = _select_rank_config_value(
                    io_engine_port,
                    unique_rank,
                    "io_engine_port",
                    int,
                    auto_increment_scalar=True,
                )

            if "staging_buffer_size" in extra:
                dist_cfg.staging_buffer_size = int(extra["staging_buffer_size"])

            peer_service_port = extra.get(
                "peer_service_port", _optional_env_str("UMBP_PEER_SERVICE_PORT")
            )
            if peer_service_port is not None:
                dist_cfg.peer_service_port = _select_rank_config_value(
                    peer_service_port,
                    unique_rank,
                    "peer_service_port",
                    int,
                    auto_increment_scalar=True,
                )

            cache_remote_fetches = extra.get(
                "cache_remote_fetches",
                _optional_env_str("UMBP_CACHE_REMOTE_FETCHES"),
            )
            if cache_remote_fetches is not None:
                dist_cfg.cache_remote_fetches = _bool_from_any(cache_remote_fetches)

            # Auto-compute master's PageBitmapAllocator page_size so every
            # UMBPStore Put/Get maps to exactly one master page (no partial
            # tail, 1 RDMA per page).  Resolution order:
            #   1. extra_config["dram_page_size"] — explicit operator override
            #      (escape hatch for debugging / forced experiments).
            #   2. derived from mem_pool_host (the normal production path).
            #   3. left at 0 when neither source is available; mori's
            #      UMBPDistributedConfig.dram_page_size defaults to 0, which
            #      delegates to the master-side ClientRegistryConfig
            #      .default_dram_page_size (2 MiB by default). The
            #      partial-tail safety net in PoolClient handles any
            #      size mismatch.
            page_byte_size = None
            if "dram_page_size" in extra:
                page_byte_size = int(extra["dram_page_size"])
            elif mem_pool_host is not None:
                # Probe element_size from the same buffer-meta helper that
                # batch_preprocess will actually use; this matches per-call
                # Put/Get size byte-for-byte for MHA / MHA-split / MLA / NSA
                # without per-case formulas (NSA in particular: get_ksize_per_token
                # would over-count by the indexer buffer that is never put to UMBP).
                dummy = torch.zeros(mem_pool_host.page_size, dtype=torch.int64)
                if self.is_mla_backend:
                    _, esz = mem_pool_host.get_page_buffer_meta(dummy)
                elif storage_config is not None and getattr(
                    storage_config, "should_split_heads", False
                ):
                    sf = storage_config.tp_lcm_size // storage_config.tp_size
                    _, esz = mem_pool_host.get_split_heads_page_buffer_meta(dummy, sf)
                else:
                    _, esz = mem_pool_host.get_page_buffer_meta(dummy)
                page_byte_size = int(esz[0]) if esz else 0

            if page_byte_size is not None and page_byte_size > 0 and hasattr(
                dist_cfg, "dram_page_size"
            ):
                dist_cfg.dram_page_size = int(page_byte_size)
                logger.info(
                    "UMBPStore: setting master dram_page_size=%d "
                    "(ksize_per_token=%s × page_size=%s%s)",
                    dist_cfg.dram_page_size,
                    (
                        mem_pool_host.get_ksize_per_token()
                        if mem_pool_host is not None
                        else "n/a"
                    ),
                    (
                        mem_pool_host.page_size
                        if mem_pool_host is not None
                        else "n/a"
                    ),
                    (
                        f" / split_factor={storage_config.tp_lcm_size // storage_config.tp_size}"
                        if (
                            mem_pool_host is not None
                            and storage_config is not None
                            and getattr(storage_config, "should_split_heads", False)
                        )
                        else ""
                    ),
                )

            cfg.distributed = dist_cfg
            logger.info(
                "UMBPStore distributed mode: master=%s, node_id=%s, node_addr=%s, "
                "io=%s:%s, peer_port=%s",
                dist_cfg.master_config.master_address,
                dist_cfg.master_config.node_id,
                dist_cfg.master_config.node_address,
                dist_cfg.io_engine.host,
                dist_cfg.io_engine.port,
                dist_cfg.peer_service_port,
            )

        self.storage_config = storage_config

        # MLA + TP > 1: shared SSD mode (standalone only).
        # In distributed mode every rank is a peer of the master-led pool; we
        # must NOT short-circuit followers (would leave their DRAM pool empty
        # while the master still routes keys to them, causing Get misses).
        self.is_mla_follower = False
        tp_size = self.tp_size
        use_spdk = cfg.ssd_backend in ("spdk", "spdk_proxy")
        distributed_enabled = cfg.distributed is not None
        if not distributed_enabled and self.is_mla_backend and tp_size > 1:
            cfg.ssd.enabled = True
            if self.local_rank == 0:
                # Leader: copy every DRAM write to shared SSD.
                cfg.role = UMBPRole.SharedSSDLeader
            else:
                # Follower: read-only access.
                cfg.role = UMBPRole.SharedSSDFollower
                self.is_mla_follower = True
                # SPDK: follower must use the proxy path rather than direct
                # SpdkSsdTier.  Give a longer startup timeout so followers can
                # wait for the shared proxy service to become READY.
                if use_spdk:
                    cfg.ssd_backend = "spdk_proxy"
                    if cfg.spdk_proxy_startup_timeout_ms < 60000:
                        cfg.spdk_proxy_startup_timeout_ms = 60000
            logger.info(
                "UMBPStore MLA+TP>1: rank=%d, role=%s, ssd_backend=%s, shared_ssd=%s",
                self.local_rank,
                "leader" if self.local_rank == 0 else "follower",
                cfg.ssd_backend,
                cfg.ssd.storage_dir,
            )

        try:
            from sglang.srt.layers.dp_attention import (
                get_attention_dp_rank,
                get_attention_dp_size,
                is_dp_attention_enabled,
            )

            if is_dp_attention_enabled():
                dp_rank = get_attention_dp_rank()
                dp_size = get_attention_dp_size()
                dp_rank_hint = dp_rank
                dp_size_hint = dp_size
                if cfg.ssd.enabled:
                    if cfg.ssd_backend in ("spdk", "spdk_proxy"):
                        # DP + SPDK must always use the proxy service path.
                        # Direct SpdkSsdTier is single-process and cannot
                        # provide tenant isolation across DP ranks.
                        cfg.ssd_backend = "spdk_proxy"
                        if cfg.spdk_proxy_startup_timeout_ms < 60000:
                            cfg.spdk_proxy_startup_timeout_ms = 60000
                        if tenant_id_base is not None:
                            cfg.spdk_proxy_tenant_id = tenant_id_base + dp_rank
                        elif not explicit_tenant_id:
                            cfg.spdk_proxy_tenant_id = dp_rank
                        elif dp_size > 1:
                            logger.warning(
                                "UMBPStore DP isolation: using explicit fixed tenant_id=%s "
                                "with dp_size=%d; all DP groups will share one tenant "
                                "unless you set spdk_proxy_tenant_id_base",
                                cfg.spdk_proxy_tenant_id,
                                dp_size,
                            )
                        if cfg.spdk_proxy_tenant_quota_bytes <= 0 and dp_size > 1:
                            # Reserve 5% headroom for offset allocator bin
                            # rounding (small-float bins round up each
                            # allocation by up to ~12.5%).
                            safe_cap = int(cfg.ssd.capacity_bytes * 0.95)
                            cfg.spdk_proxy_tenant_quota_bytes = max(
                                1, safe_cap // dp_size
                            )
                        # Validate: total tenant quotas must fit within SSD
                        # capacity after allocator rounding.
                        if dp_size > 1:
                            total_quota = cfg.spdk_proxy_tenant_quota_bytes * dp_size
                            if total_quota > cfg.ssd.capacity_bytes:
                                old_quota = cfg.spdk_proxy_tenant_quota_bytes
                                safe_cap = int(cfg.ssd.capacity_bytes * 0.95)
                                cfg.spdk_proxy_tenant_quota_bytes = max(
                                    1, safe_cap // dp_size
                                )
                                logger.warning(
                                    "UMBPStore: tenant_quota_bytes=%d × dp_size=%d = %d "
                                    "exceeds ssd_capacity=%d. Reduced to %d to "
                                    "avoid SPDK proxy NO_SPACE. Consider "
                                    "increasing UMBP_SSD_BYTES.",
                                    old_quota,
                                    dp_size,
                                    total_quota,
                                    cfg.ssd.capacity_bytes,
                                    cfg.spdk_proxy_tenant_quota_bytes,
                                )
                        logger.info(
                            "UMBPStore DP isolation: dp_rank=%d, dp_size=%d, tenant_id=%s, tenant_quota_bytes=%s",
                            dp_rank,
                            dp_size,
                            getattr(cfg, "spdk_proxy_tenant_id", "n/a"),
                            getattr(cfg, "spdk_proxy_tenant_quota_bytes", "n/a"),
                        )
                    else:
                        cfg.ssd.storage_dir = f"{cfg.ssd.storage_dir}/dp{dp_rank}"
                        logger.info(
                            "UMBPStore DP isolation: dp_rank=%d, dp_size=%d, ssd_dir=%s",
                            dp_rank,
                            dp_size,
                            cfg.ssd.storage_dir,
                        )
        except (ImportError, AssertionError):
            pass

        if (
            cfg.ssd.enabled
            and not self.is_mla_follower
            and not (self.is_mla_backend and tp_size > 1)
            and cfg.ssd_backend not in ("spdk", "spdk_proxy")
        ):
            rank_dir_parts = []
            if dp_rank_hint is not None:
                rank_dir_parts.append(f"dp{dp_rank_hint}")
            if self.pp_size > 1:
                rank_dir_parts.append(f"pp{self.pp_rank}")
            if self.tp_size > 1:
                rank_dir_parts.append(f"tp{self.local_rank}")
            if not rank_dir_parts and unique_rank != 0:
                rank_dir_parts.append(f"rank{unique_rank}")
            if rank_dir_parts:
                cfg.ssd.storage_dir = os.path.join(
                    cfg.ssd.storage_dir, "_".join(rank_dir_parts)
                )
                logger.info(
                    "UMBPStore local SSD isolation: unique_rank=%d, ssd_dir=%s",
                    unique_rank,
                    cfg.ssd.storage_dir,
                )

        if cfg.ssd.enabled and cfg.ssd_backend in ("spdk", "spdk_proxy"):
            if dp_rank_hint is not None and tenant_id_base is not None:
                cfg.spdk_proxy_tenant_id = tenant_id_base + dp_rank_hint
            elif dp_rank_hint is not None and not explicit_tenant_id:
                cfg.spdk_proxy_tenant_id = dp_rank_hint
            if (
                dp_rank_hint is not None
                and dp_size_hint is not None
                and cfg.spdk_proxy_tenant_quota_bytes <= 0
                and dp_size_hint > 1
            ):
                safe_cap = int(cfg.ssd.capacity_bytes * 0.95)
                cfg.spdk_proxy_tenant_quota_bytes = max(1, safe_cap // dp_size_hint)

        self.client = UMBPClient(cfg)
        if mem_pool_host is not None:
            self.register_mem_pool_host(mem_pool_host)

        self.enable_pp = self.pp_size > 1
        if self.enable_pp:
            self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
            self.mla_suffix = f"{self.pp_rank}"
        else:
            self.mha_suffix = f"{self.local_rank}"
            self.mla_suffix = ""

        self.split_factor = 0
        if storage_config and storage_config.should_split_heads:
            self.split_factor = storage_config.tp_lcm_size // storage_config.tp_size
            base_rank = self.local_rank * self.split_factor
            target_ranks = [base_rank + i for i in range(self.split_factor)]
            if self.enable_pp:
                self.mha_suffix = [f"{rank}_{self.pp_rank}" for rank in target_ranks]
            else:
                self.mha_suffix = [f"{rank}" for rank in target_ranks]

        logger.info(
            "UMBPStore initialized: dram=%d MB, ssd=%s, mla=%s, rank=%d, ssd_backend=%s",
            cfg.dram.capacity_bytes // (1024 * 1024),
            cfg.ssd.enabled,
            self.is_mla_backend,
            self.local_rank,
            cfg.ssd_backend,
        )

    # ------------------------------------------------------------------
    # Host memory pool registration
    # ------------------------------------------------------------------
    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        assert self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
            "page_head",
        ], "UMBP store only supports page_first, page_first_direct, or page_head layout"

        # In distributed mode, pre-register the entire host KV buffer with the
        # underlying RDMA IOEngine so PoolClient can take the zero-copy path
        # for batch_get_into_ptr / batch_put_from_ptr (skips the staging
        # buffer memcpy + lock and removes the per-call `staging_buffer_size`
        # cap).  Standalone returns true as no-op by IUMBPClient contract;
        # we still gate on is_distributed() below to avoid a pointless call.
        self._zero_copy_registered = False
        if self.client is None:
            return
        try:
            is_distributed = bool(self.client.is_distributed())
        except Exception:
            is_distributed = False
        if not is_distributed:
            return
        if not hasattr(self.client, "register_memory"):
            return
        try:
            kv_buffer = mem_pool_host.kv_buffer
            host_ptr = int(kv_buffer.data_ptr())
            host_size = int(kv_buffer.numel() * kv_buffer.element_size())
            ok = bool(self.client.register_memory(host_ptr, host_size))
        except Exception as exc:
            logger.warning(
                "UMBPStore: register_memory failed (%s); falling back to staging "
                "buffer path. Per-transfer size will be capped by "
                "distributed.staging_buffer_size.",
                exc,
            )
            return
        if ok:
            self._zero_copy_registered = True
            logger.info(
                "UMBPStore: registered host KV buffer for RDMA zero-copy "
                "(ptr=0x%x, size=%d MB)",
                host_ptr,
                host_size // (1024 * 1024),
            )
        else:
            logger.warning(
                "UMBPStore: register_memory returned false; staying on staging "
                "buffer fallback path."
            )

    # ------------------------------------------------------------------
    # Key suffix generation — mirrors MooncakeStore
    # ------------------------------------------------------------------
    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mha_suffix}_k")
            key_list.append(f"{key_}_{self.mha_suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mha_split_heads_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = (
            self.mem_pool_host.get_split_heads_page_buffer_meta(
                indices, self.split_factor
            )
        )
        key_list = []
        for key_ in keys:
            for suffix in self.mha_suffix:
                key_list.append(f"{key_}_{suffix}_k")
                key_list.append(f"{key_}_{suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mla_suffix}_k")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        else:
            if self.storage_config and self.storage_config.should_split_heads:
                return self._get_mha_split_heads_buffer_meta(keys, host_indices)
            else:
                return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(self, results: List[bool], is_set_operate=False):
        """Convert per-key-component results to per-page results.

        For MHA: each page has K+V → group pairs.
        For MLA: each page has K only.
        """
        if self.is_mla_backend:
            return list(results)
        else:
            if self.storage_config and self.storage_config.should_split_heads:
                group_size = self.split_factor * 2
                groups = [
                    results[i : i + group_size]
                    for i in range(0, len(results), group_size)
                ]
                return [all(g) for g in groups]
            else:
                # Group K/V pairs
                kv_pairs = zip(results[::2], results[1::2])
                return [k and v for k, v in kv_pairs]

    # ------------------------------------------------------------------
    # Zero-copy v1 interface
    # ------------------------------------------------------------------
    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)

        # Normalize sizes to list of per-key sizes
        if isinstance(buffer_sizes, int):
            sizes = [buffer_sizes] * len(key_strs)
        elif isinstance(buffer_sizes, list) and len(buffer_sizes) == 1:
            sizes = buffer_sizes * len(key_strs)
        else:
            sizes = list(buffer_sizes)

        get_results = self.client.batch_get_into_ptr(key_strs, list(buffer_ptrs), sizes)
        return self._batch_postprocess(get_results)

    def _compute_expanded_depths(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo]
    ) -> List[int]:
        """Compute per-expanded-key depth values from prefix_keys metadata.

        depth = len(prefix_keys) + page_index_within_node.
        All key variants of the same page (K, V, multi-rank) share the same depth.
        Returns an empty list if no metadata is available (caller falls back to plain LRU).
        """
        prefix_keys = getattr(extra_info, "prefix_keys", None) if extra_info else None
        if prefix_keys is None:
            return []

        prefix_len = len(prefix_keys)
        depths_per_page = [prefix_len + i for i in range(len(keys))]

        # Expand to match the key_strs layout produced by _batch_preprocess.
        expanded = []
        for d in depths_per_page:
            if self.is_mla_backend:
                expanded.append(d)  # MLA: 1 key per page
            elif self.storage_config and self.storage_config.should_split_heads:
                # split heads: 2 keys per split rank, split_factor ranks per page
                for _ in range(self.split_factor):
                    expanded.append(d)
                    expanded.append(d)
            else:
                expanded.append(d)  # K
                expanded.append(d)  # V
        return expanded

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Follower never writes (CacheController also sets backup_skip, but guard here too)
        if self.is_mla_follower:
            page_count = len(host_indices) // self.mem_pool_host.page_size
            return [True] * page_count

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)

        if isinstance(buffer_sizes, int):
            sizes = [buffer_sizes] * len(key_strs)
        elif isinstance(buffer_sizes, list) and len(buffer_sizes) == 1:
            sizes = buffer_sizes * len(key_strs)
        else:
            sizes = list(buffer_sizes)

        expanded_depths = self._compute_expanded_depths(keys, extra_info)

        if expanded_depths:
            put_results = self.client.batch_put_from_ptr_with_depth(
                key_strs, list(buffer_ptrs), sizes, expanded_depths
            )
        else:
            put_results = self.client.batch_put_from_ptr(
                key_strs, list(buffer_ptrs), sizes
            )

        return self._batch_postprocess(put_results, is_set_operate=True)

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """Return count of consecutive existing keys from start."""
        if self.is_mla_backend:
            query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]
            key_multiplier = 1
        else:
            query_keys = []
            if self.storage_config and self.storage_config.should_split_heads:
                for key in keys:
                    for suffix in self.mha_suffix:
                        query_keys.append(f"{key}_{suffix}_k")
                        query_keys.append(f"{key}_{suffix}_v")
                key_multiplier = 2 * self.split_factor
            else:
                for key in keys:
                    query_keys.append(f"{key}_{self.mha_suffix}_k")
                    query_keys.append(f"{key}_{self.mha_suffix}_v")
                key_multiplier = 2

        hit_count = self.client.batch_exists_consecutive(query_keys)
        return hit_count // key_multiplier

    # ------------------------------------------------------------------
    # Legacy ABC interface (required by HiCacheStorage)
    # ------------------------------------------------------------------
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        if target_location is None or target_sizes is None:
            return None
        ok = self.client.get_into_ptr(key, target_location, target_sizes)
        return target_location if ok else None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        if not keys:
            return 0
        assert len(keys) == len(target_locations) == len(target_sizes)
        results = self.client.batch_get_into_ptr(
            keys,
            list(target_locations),
            list(target_sizes),
        )
        for i, ok in enumerate(results):
            if not ok:
                return i
        return len(keys)

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.is_mla_follower:
            return True
        if target_location is None or target_sizes is None:
            return False
        return self.client.put_from_ptr(key, target_location, target_sizes)

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if not keys:
            return False
        if self.is_mla_follower:
            return True
        assert len(keys) == len(target_locations) == len(target_sizes)
        results = self.client.batch_put_from_ptr(
            keys,
            list(target_locations),
            list(target_sizes),
        )
        return all(results)

    def exists(self, key: str) -> bool:
        return self.client.exists(key)

    def clear(self) -> None:
        self.client.clear()

    def flush(self) -> bool:
        if self.client is None or not hasattr(self.client, "flush"):
            return True
        return bool(self.client.flush())

    def close(self) -> None:
        if getattr(self, "client", None) is None:
            return
        try:
            self.flush()
        except Exception:
            logger.exception("UMBPStore flush during close failed")
        self.client = None
