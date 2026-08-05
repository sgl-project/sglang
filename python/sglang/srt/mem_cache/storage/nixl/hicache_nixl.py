import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host import HostKVCache
from sglang.srt.mem_cache.pool_host.mha import AsymmetricMHATokenToKVPoolHost
from sglang.srt.mem_cache.storage.mmap import alloc_mmap
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
    MEM_BACKEND_HUGEPAGE,
    tensor_mem_backend,
)
from sglang.srt.mem_cache.storage.nixl.nixl_cleaner import HiCacheL3Cleaner

from .nixl_registry import NixlRegistry
from .nixl_utils import NixlBackendConfig, NixlBackendSelection, NixlFileManager

try:
    from nixl._api import nixl_agent, nixl_agent_config, nixlBind
except ImportError as e:
    raise ImportError(
        "Please install NIXL by following the instructions at "
        "https://github.com/ai-dynamo/nixl/blob/main/README.md "
        "to use HiCacheNixl storage backend."
    ) from e

logger = logging.getLogger(__name__)


def _parse_storage_dirs(raw: Optional[str]) -> List[str]:
    """Split NIXL FILE storage directory config into ordered unique paths."""
    if not raw:
        return []
    candidates = [path.strip() for path in raw.split(",")]
    candidates = [path for path in candidates if path]
    seen: dict[str, str] = {}
    ordered: List[str] = []
    for path in candidates:
        real_path = os.path.realpath(path)
        if real_path in seen:
            raise ValueError(
                "SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR contains duplicate "
                f"path {path!r} (same mount as {seen[real_path]!r})."
            )
        seen[real_path] = path
        ordered.append(path)
    return ordered


@dataclass
class _HybridPoolContext:
    host_pool: HostKVCache
    is_zero_copy: bool
    bounce_set: Optional[torch.Tensor] = None
    bounce_get: Optional[torch.Tensor] = None
    bounce_page_bytes: int = 0
    bounce_payload_numel: int = 0


class HiCacheNixl(HiCacheStorage):
    """HiCacheNixl provides high-performance storage using NIXL plugins."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        file_path: str = "/tmp/hicache_storage",
    ):
        """Initialize NIXL storage connector."""

        # create nixlconfig from the --hicache-storage-backend-extra-config
        nixlconfig = NixlBackendConfig(storage_config.extra_config)

        # select the NIXL backend plugin from extra_config or environment variable
        plugin = nixlconfig.get_specified_plugin()

        use_direct_io = nixlconfig.get_use_direct_io()

        # Might be better to be unified across HiCache backends and moved to HiCacheController
        storage_dirs = _parse_storage_dirs(
            envs.SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR.get() or file_path
        )
        self.file_manager = (
            NixlFileManager(storage_dirs, use_direct_io=use_direct_io)
            if plugin not in NixlBackendSelection.OBJ_PLUGINS
            else None
        )

        tp_rank, tp_size, model_name = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
        )

        self.is_mla_model = storage_config.is_mla_model
        self.is_zero_copy = False
        self.storage_config = storage_config
        self.backup_skip = self.is_mla_model and storage_config.tp_rank != 0

        model_name = "-".join(model_name.split("/")) if model_name else ""

        if self.is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        sync_mode = getattr(
            nixlBind, "NIXL_THREAD_SYNC_RW", nixlBind.NIXL_THREAD_SYNC_STRICT
        )
        agent_config = nixl_agent_config(backends=[])
        self.agent_name = f"hicache_nixl_{str(uuid.uuid4())}"
        self.agent = nixl_agent(self.agent_name, agent_config)
        bind_cfg = nixlBind.nixlAgentConfig()
        bind_cfg.useProgThread = agent_config.enable_pthread
        bind_cfg.useListenThread = agent_config.enable_listen
        bind_cfg.listenPort = agent_config.port
        bind_cfg.syncMode = sync_mode
        bind_cfg.pthrDelay = 0
        bind_cfg.lthrDelay = 100000
        bind_cfg.captureTelemetry = agent_config.capture_telemetry
        self.agent.agent = nixlBind.nixlAgent(self.agent_name, bind_cfg)
        self.agent.plugin_list = self.agent.agent.getAvailPlugins()

        self.backend_selector = NixlBackendSelection(plugin, nixlconfig)
        if not self.backend_selector.create_backend(self.agent):
            raise RuntimeError("Failed to create NIXL backend")

        self.registry = NixlRegistry(
            self.agent,
            self.backend_selector.mem_type,
            self.file_manager,
        )
        # O_DIRECT requires OS-page-aligned I/O buffers on all file-based backends
        # (POSIX, GDS, GDS_MT, 3FS).
        # DOCA_MEMOS is an OBJ backend but has the same address/length requirement.
        self.needs_page_alignment = (
            use_direct_io and self.file_manager is not None
        ) or self.backend_selector.backend_name == "DOCA_MEMOS"
        if self.needs_page_alignment:
            logger.info(
                "HiCacheNixl: %s requires 4KiB aligned host descriptor addresses "
                "and lengths.",
                self.backend_selector.backend_name,
            )
        # Pre-registered host regions (set by register_mem_pool_host):
        # zero-copy: registrations covering every mem_pool_host data buffer
        # non-zero-copy: two registrations, one bounce buffer per direction
        # (set/get) so the two storage threads never share slots.
        self._host_regs: List[Any] = []
        self._bounce_set: Optional[torch.Tensor] = None
        self._bounce_get: Optional[torch.Tensor] = None
        self._bounce_page_bytes: Optional[int] = None
        self._bounce_payload_numel: Optional[int] = None
        self._logical_anchor = False
        self._hybrid_pool_ctx: dict[PoolName, _HybridPoolContext] = {}
        self.registered_pools: dict[PoolName, HostKVCache] = {}
        cleanup_dirs = (
            self.file_manager.iter_all_base_dirs()
            if self.file_manager is not None
            else []
        )
        cleaner_config = nixlconfig.get_l3_cleaner_config()
        self._l3_cleaner: Optional[HiCacheL3Cleaner] = (
            HiCacheL3Cleaner(
                cleanup_dirs,
                tp_rank,
                high_watermark=cleaner_config["high_watermark"],
                low_watermark=cleaner_config["low_watermark"],
            )
            if (
                cleanup_dirs
                and self.file_manager is not None
                and cleaner_config["enabled"]
            )
            else None
        )
        if self._l3_cleaner is not None:
            self._l3_cleaner.start()

    def _format_key(self, key: str) -> str:
        if self.backend_selector.backend_name == "DOCA_MEMOS":
            return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        if self.backend_selector.mem_type == "FILE":
            return self.file_manager.get_file_path(key)
        return key

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _get_component_key(
        self, key: str, component_name: Optional[PoolName] = None
    ) -> str:
        if component_name in (None, PoolName.KV):
            return self._get_suffixed_key(key)
        return f"{self._get_suffixed_key(key)}_{component_name}"

    def _get_component_keys(
        self, keys: List[str], pool_name: Optional[PoolName] = None
    ) -> List[str]:
        return [self._get_component_key(key, pool_name) for key in keys]

    def _get_hybrid_component_keys(
        self,
        keys: List[str],
        pool_name: PoolName,
        key_multiplier: int,
        host_pool: HostKVCache,
    ) -> List[str]:
        if key_multiplier == 1:
            return self._get_component_keys(keys, pool_name)

        if pool_name == PoolName.MAMBA:
            mamba_has_temporal = host_pool.temporal_state_elem_size > 0
            if mamba_has_temporal:
                suffixes = [f"_{pool_name}_temporal"] + [
                    f"_{pool_name}_conv_{i}" for i in range(key_multiplier - 1)
                ]
            else:
                suffixes = [f"_{pool_name}_conv_{i}" for i in range(key_multiplier)]
        elif key_multiplier == 2:
            suffixes = [f"_{pool_name}_k", f"_{pool_name}_v"]
        else:
            suffixes = [f"_{pool_name}_{i}" for i in range(key_multiplier)]

        return [
            f"{self._get_suffixed_key(key)}{suffix}"
            for key in keys
            for suffix in suffixes
        ]

    def _create_query_tuple(self, key: str) -> tuple:
        """Build the NIXL query_memory tuple for a single key."""
        return (0, 0, 0, self._format_key(key))

    def _query_keys_exist(self, keys: List[str]) -> List[bool]:
        if not keys:
            return []
        tuples = [self._create_query_tuple(key) for key in keys]
        query_res = self.agent.query_memory(
            tuples,
            self.backend_selector.backend_name,
            mem_type=self.backend_selector.mem_type,
        )
        return [res is not None for res in query_res]

    def _xfer_and_wait(
        self,
        host_descs: Any,
        storage_descs: Any,
        direction: str,
    ) -> bool:
        """Initialize and poll a NIXL transfer to completion."""
        try:
            xfer_req = self.agent.initialize_xfer(
                direction, host_descs, storage_descs, self.agent_name
            )
        except Exception as e:
            logger.error(f"Failed to create transfer request: {e}")
            return False

        try:
            state = self.agent.transfer(xfer_req)
            while state != "DONE":
                state = self.agent.check_xfer_state(xfer_req)
                if state == "ERR":
                    logger.error("Transfer failed")
                    return False
                # Best would be to have a better notification mechanism from NIXL,
                # but we only have polling for now.
                time.sleep(0.0001)
            return True
        except Exception as e:
            logger.error(f"Failed to execute transfer: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            self.agent.release_xfer_handle(xfer_req)

    def _xfer_pre_registered(
        self,
        host_buffers: List[tuple],
        keys: List[str],
        direction: str,
    ) -> bool:
        """Run a transfer where the host side is already pre-registered.

        ``host_buffers`` is a list of ``(addr, size)`` tuples within the
        pre-registered host region (kv_buffer for zero-copy, bounce buffer
        otherwise). Only the storage side is registered per transfer.
        """
        if len(host_buffers) != len(keys):
            logger.error("Mismatch between number of host buffers and keys")
            return False

        host_descs = self.agent.get_xfer_descs(
            [(addr, size, 0) for (addr, size) in host_buffers], "DRAM"
        )
        if host_descs is None:
            logger.error("Failed to build host xfer descs")
            return False

        with self.registry.storage(host_buffers, keys, direction) as storage_descs:
            if storage_descs is None:
                return False
            return self._xfer_and_wait(host_descs, storage_descs, direction)

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError("deprecated; use batch_get_v1")

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        raise NotImplementedError("deprecated; use batch_get_v1")

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError("deprecated; use batch_set_v1")

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError("deprecated; use batch_set_v1")

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self._logical_anchor = False

        # enable zero-copy automatically if mem layout is page_first or page_first_direct
        self.is_zero_copy = mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
        ]

        kv = getattr(mem_pool_host, "kv_buffer", None)
        if kv is None:
            # DeepSeek V4 uses a LogicalHostPool as the KV anchor. It has no
            # actual KV bytes; component pools carry the data through v2 APIs.
            # Still write a small marker object per page so batch_exists_v2 can
            # use the anchor key to gate sidecar lookups.
            super().register_mem_pool_host(mem_pool_host)
            self.is_zero_copy = False
            self._logical_anchor = True
            marker_numel = 4096 if self.needs_page_alignment else 1
            pin_memory = bool(getattr(mem_pool_host, "pin_memory", False))
            self._bounce_page_bytes = marker_numel
            self._bounce_set = self._alloc_registered(
                marker_numel, torch.uint8, pin_memory, "logical_anchor_set"
            )
            self._bounce_get = self._alloc_registered(
                marker_numel, torch.uint8, pin_memory, "logical_anchor_get"
            )
            self._bounce_set.fill_(1)
            logger.info(
                "HiCacheNixl: registered logical anchor pool with %d-byte markers",
                self._bounce_page_bytes,
            )
            return

        if self.backend_selector.backend_name == "DOCA_MEMOS":
            buffers = (
                mem_pool_host.get_hybrid_pool_buffer()
                if hasattr(mem_pool_host, "get_hybrid_pool_buffer")
                else kv
            )
            buffer_list = buffers if isinstance(buffers, (list, tuple)) else [buffers]
            kv_buffers = [buf for buf in buffer_list if buf.numel() > 0]
        else:
            kv_buffers = [kv]
        super().register_mem_pool_host(mem_pool_host)

        if self.needs_page_alignment and self.is_zero_copy:
            # Check that the kv_buffer base AND per-page strides are multiples of
            # the OS page size so every pointer passed to NIXL (base + p * stride)
            # is page-aligned. The base is whatever torch.empty() happened to give
            # us -- it is not guaranteed to be page-aligned. Fall back to copy mode
            # if either condition fails.
            # 4096: O_DIRECT alignment is FS-dependent (some allow 512 B); 4 KiB
            # is the safe lower bound all known FSes accept, and real page-sizes meet it.
            if not mem_pool_host.is_stride_page_aligned(4096):
                logger.warning(
                    "HiCacheNixl: %s requires aligned descriptors but the host "
                    "kv_buffer is not 4KiB aligned. Falling back to copy mode.",
                    self.backend_selector.backend_name,
                )
                self.is_zero_copy = False
            elif self.backend_selector.backend_name == "DOCA_MEMOS" and any(
                tensor_mem_backend(buf) != MEM_BACKEND_HUGEPAGE for buf in kv_buffers
            ):
                logger.warning(
                    "HiCacheNixl: DOCA_MEMOS zero-copy requires hugetlb-backed "
                    "host buffers. Falling back to copy mode."
                )
                self.is_zero_copy = False

        if self.is_zero_copy:
            for i, buf in enumerate(kv_buffers):
                self._pre_register_host(
                    buf.data_ptr(),
                    buf.numel() * buf.element_size(),
                    f"kv_buffer_{i}",
                )
        else:
            # One bounce buffer per direction so set/get run lock-free across
            # the prefetch and backup threads. The logical payload matches one
            # flat data page; DOCA_MEMOS pads each physical slot to 4 KiB.
            asymmetric_pool = (
                self._get_asymmetric_pool(mem_pool_host)
                if self.backend_selector.backend_name == "DOCA_MEMOS"
                else None
            )
            if asymmetric_pool is not None:
                page_numel = (
                    asymmetric_pool.layer_num
                    * asymmetric_pool.page_size
                    * asymmetric_pool.head_num
                    * (asymmetric_pool.head_dim + asymmetric_pool.v_head_dim)
                )
                page_dtype = asymmetric_pool.dtype
            else:
                sample = mem_pool_host.get_dummy_flat_data_page()
                page_numel = sample.numel()
                page_dtype = sample.dtype
                del sample
            self._bounce_payload_numel = page_numel
            pin_memory = bool(getattr(mem_pool_host, "pin_memory", False))
            self._bounce_page_bytes = page_numel * page_dtype.itemsize
            if self.backend_selector.backend_name == "DOCA_MEMOS":
                self._bounce_page_bytes = (
                    (self._bounce_page_bytes + 4095) // 4096 * 4096
                )
            slot_numel = self._bounce_page_bytes // page_dtype.itemsize
            self._bounce_set = self._alloc_registered(
                slot_numel, page_dtype, pin_memory, "bounce_set"
            )
            self._bounce_get = self._alloc_registered(
                slot_numel, page_dtype, pin_memory, "bounce_get"
            )

        logger.info(
            f"HiCacheNixl: pre-registered host regions for "
            f"layout={mem_pool_host.layout} zero_copy={self.is_zero_copy}"
        )

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if host_pool_name == PoolName.KV:
            return

        is_doca_memos = self.backend_selector.backend_name == "DOCA_MEMOS"
        is_zero_copy = hasattr(host_pool, "get_page_buffer_meta") and hasattr(
            host_pool, "get_hybrid_pool_buffer"
        )
        if is_doca_memos and getattr(host_pool, "layout", None) not in (
            "page_first",
            "page_first_direct",
        ):
            is_zero_copy = False
        buffers = []
        if is_zero_copy:
            buffers = host_pool.get_hybrid_pool_buffer()
            if is_doca_memos:
                buffers = [buf for buf in buffers if buf.numel() > 0]
            is_zero_copy = bool(buffers)
        if is_zero_copy and self.needs_page_alignment:
            is_zero_copy = host_pool.is_stride_page_aligned(4096)
            if not is_zero_copy:
                logger.warning(
                    "HiCacheNixl: %s requires aligned descriptors but hybrid "
                    "pool %s is not 4096-byte aligned; using bounce buffers.",
                    self.backend_selector.backend_name,
                    host_pool_name,
                )
        if is_zero_copy and is_doca_memos:
            is_zero_copy = all(
                tensor_mem_backend(buf) == MEM_BACKEND_HUGEPAGE for buf in buffers
            )
            if not is_zero_copy:
                logger.warning(
                    "HiCacheNixl: DOCA_MEMOS zero-copy pool %s is not "
                    "hugetlb-backed; using bounce buffers.",
                    host_pool_name,
                )
        if is_zero_copy:
            for i, buf in enumerate(buffers):
                self._pre_register_host(
                    buf.data_ptr(),
                    buf.numel() * buf.element_size(),
                    f"{host_pool_name}_buffer_{i}",
                )
            self._hybrid_pool_ctx[host_pool_name] = _HybridPoolContext(
                host_pool=host_pool, is_zero_copy=True
            )
        else:
            sample = host_pool.get_dummy_flat_data_page()
            page_numel = sample.numel()
            pin_memory = bool(getattr(host_pool, "pin_memory", False))
            page_bytes = page_numel * sample.element_size()
            if is_doca_memos:
                page_bytes = (page_bytes + 4095) // 4096 * 4096
            slot_numel = page_bytes // sample.element_size()
            bounce_set = self._alloc_registered(
                slot_numel,
                sample.dtype,
                pin_memory,
                f"{host_pool_name}_bounce_set",
            )
            bounce_get = self._alloc_registered(
                slot_numel,
                sample.dtype,
                pin_memory,
                f"{host_pool_name}_bounce_get",
            )
            del sample
            self._hybrid_pool_ctx[host_pool_name] = _HybridPoolContext(
                host_pool=host_pool,
                is_zero_copy=False,
                bounce_set=bounce_set,
                bounce_get=bounce_get,
                bounce_page_bytes=page_bytes,
                bounce_payload_numel=page_numel,
            )

        super().register_mem_host_pool_v2(host_pool, host_pool_name)

        logger.info(
            "HiCacheNixl: registered hybrid host pool %s zero_copy=%s",
            host_pool_name,
            is_zero_copy,
        )

    @staticmethod
    def _get_asymmetric_pool(
        host_pool: HostKVCache,
    ) -> Optional[AsymmetricMHATokenToKVPoolHost]:
        anchor_entry = getattr(host_pool, "anchor_entry", None)
        data_pool = getattr(anchor_entry, "host_pool", host_pool)
        if not isinstance(data_pool, AsymmetricMHATokenToKVPoolHost):
            return None
        return data_pool

    def _get_asymmetric_flat_data_page(
        self, asymmetric_pool: AsymmetricMHATokenToKVPoolHost, index: int
    ) -> torch.Tensor:
        if asymmetric_pool.layout == "page_first":
            k_page = asymmetric_pool.k_buffer[index : index + asymmetric_pool.page_size]
            v_page = asymmetric_pool.v_buffer[index : index + asymmetric_pool.page_size]
        else:
            page_index = index // asymmetric_pool.page_size
            k_page = asymmetric_pool.k_buffer[page_index]
            v_page = asymmetric_pool.v_buffer[page_index]
        return torch.cat((k_page.reshape(-1), v_page.reshape(-1)))

    def _set_asymmetric_flat_data_page(
        self,
        asymmetric_pool: AsymmetricMHATokenToKVPoolHost,
        index: int,
        data_page: torch.Tensor,
    ) -> None:
        if asymmetric_pool.layout == "page_first":
            k_page = asymmetric_pool.k_buffer[index : index + asymmetric_pool.page_size]
            v_page = asymmetric_pool.v_buffer[index : index + asymmetric_pool.page_size]
        else:
            page_index = index // asymmetric_pool.page_size
            k_page = asymmetric_pool.k_buffer[page_index]
            v_page = asymmetric_pool.v_buffer[page_index]
        flat_data = data_page.reshape(-1)
        k_numel = k_page.numel()
        k_page.copy_(flat_data[:k_numel].reshape_as(k_page))
        v_page.copy_(flat_data[k_numel:].reshape_as(v_page))

    @staticmethod
    def _copy_to_bounce_slot(
        slot: torch.Tensor, src: torch.Tensor, payload_numel: int
    ) -> None:
        src = src.reshape(-1)
        slot[:payload_numel].copy_(src)
        if payload_numel < slot.numel():
            slot[payload_numel:].zero_()

    def _get_bounce_slot_buffers(
        self, buf: torch.Tensor, page_bytes: int, page_num: int
    ) -> List[tuple]:
        base = buf.data_ptr()
        return [(base + i * page_bytes, page_bytes) for i in range(page_num)]

    def _get_hybrid_key_multiplier(
        self, pool_name: PoolName, host_pool: HostKVCache
    ) -> int:
        if pool_name == PoolName.MAMBA:
            return int(host_pool.temporal_state_elem_size > 0) + len(
                host_pool.conv_buffer
            )
        if hasattr(host_pool, "v_buffer"):
            return 2
        return 1

    def _get_hybrid_zero_copy_buffers(
        self,
        pool_name: PoolName,
        keys: List[str],
        host_indices: torch.Tensor,
        ctx: _HybridPoolContext,
    ) -> tuple[List[str], List[tuple], int]:
        """Build NIXL keys and memory descriptors for zero-copy hybrid transfers.

        The host pool returns one or more physical buffers per logical cache page
        depending on the pool type, for example K/V buffers for SWA or temporal
        plus convolution buffers for Mamba. This helper expands each logical page
        key into component-level storage keys, validates that the expanded keys
        match the host-pool metadata, and returns `(key_strs, host_buffers,
        key_multiplier)`.
        """
        ptr_list, size_list = ctx.host_pool.get_page_buffer_meta(host_indices)
        page_num = len(keys)
        if page_num == 0 or len(ptr_list) % page_num != 0:
            logger.error(
                "HiCacheNixl: hybrid pool %s metadata mismatch: pages=%s ptrs=%s",
                pool_name,
                page_num,
                len(ptr_list),
            )
            return [], [], 0
        key_multiplier = len(ptr_list) // page_num
        key_strs = self._get_hybrid_component_keys(
            keys,
            pool_name,
            key_multiplier,
            ctx.host_pool,
        )
        if len(key_strs) != len(ptr_list):
            logger.error(
                "HiCacheNixl: hybrid pool %s key/meta mismatch: keys=%s ptrs=%s",
                pool_name,
                len(key_strs),
                len(ptr_list),
            )
            return [], [], 0
        return key_strs, list(zip(ptr_list, size_list)), key_multiplier

    def _prepare_pool_transfer(
        self,
        transfer: PoolTransfer,
        keys: List[str],
        host_indices: Optional[torch.Tensor],
        for_write: bool,
    ) -> tuple[Optional[HostKVCache], List[str], List[tuple], List[int], int]:
        ctx = self._hybrid_pool_ctx.get(transfer.name)
        if ctx is None:
            logger.error("Host pool %s is not registered in HiCacheNixl", transfer.name)
            return None, [], [], [], 0

        host_pool = ctx.host_pool
        page_size = getattr(host_pool, "page_size", 1) or 1
        expected = len(keys) * page_size
        if host_indices is None or host_indices.numel() != expected:
            logger.error(
                "Pool %s indices length mismatch: expected %s, got %s",
                transfer.name,
                expected,
                host_indices.numel() if host_indices is not None else 0,
            )
            return host_pool, [], [], [], 0

        if ctx.is_zero_copy:
            key_strs, host_buffers, key_multiplier = self._get_hybrid_zero_copy_buffers(
                transfer.name, keys, host_indices, ctx
            )
            page_offsets = [
                host_indices[i * page_size].item() for i in range(len(keys))
            ]
            return host_pool, key_strs, host_buffers, page_offsets, key_multiplier

        if len(keys) > STORAGE_BATCH_SIZE:
            logger.error(
                "HiCacheNixl: hybrid pool %s batch size %s exceeds bounce buffer capacity %s",
                transfer.name,
                len(keys),
                STORAGE_BATCH_SIZE,
            )
            return host_pool, [], [], [], 0

        page_offsets = [host_indices[i * page_size].item() for i in range(len(keys))]
        bounce = ctx.bounce_set if for_write else ctx.bounce_get
        if bounce is None:
            logger.error(
                "Hybrid pool %s bounce buffer is not registered", transfer.name
            )
            return host_pool, [], [], [], 0

        if for_write:
            for i, page_offset in enumerate(page_offsets):
                src = host_pool.get_data_page(page_offset, flat=True)
                self._copy_to_bounce_slot(bounce[i], src, ctx.bounce_payload_numel)

        host_buffers = self._get_bounce_slot_buffers(
            bounce, ctx.bounce_page_bytes, len(page_offsets)
        )
        key_strs = self._get_component_keys(keys, transfer.name)
        return host_pool, key_strs, host_buffers, page_offsets, 1

    def _iter_storage_pool_transfer_chunks(
        self, transfer: PoolTransfer
    ) -> Iterator[tuple[List[str], Optional[torch.Tensor]]]:
        """Yield storage-only ``(keys, host_indices)`` chunks."""
        keys = transfer.keys or []
        host_indices = transfer.host_indices
        ctx = self._hybrid_pool_ctx.get(transfer.name)
        if ctx is None or len(keys) <= STORAGE_BATCH_SIZE:
            yield keys, host_indices
            return

        page_size = getattr(ctx.host_pool, "page_size", 1) or 1
        for start in range(0, len(keys), STORAGE_BATCH_SIZE):
            end = min(start + STORAGE_BATCH_SIZE, len(keys))
            index_start = start * page_size
            index_end = end * page_size
            yield keys[start:end], host_indices[index_start:index_end]

    def _alloc_registered(
        self,
        page_numel: int,
        dtype: torch.dtype,
        pin_memory: bool,
        kind: str,
    ) -> torch.Tensor:
        """Allocate a ``(STORAGE_BATCH_SIZE, page_numel)`` bounce buffer and
        pre-register it as a DRAM region with NIXL. Uses alloc_mmap so the
        buffer is page-aligned -- required by O_DIRECT file backends and
        DOCA_MEMOS. pin_memory is currently unused (alloc_mmap does not support
        it)."""
        buf = alloc_mmap((STORAGE_BATCH_SIZE, page_numel), dtype)
        if (
            self.backend_selector.backend_name == "DOCA_MEMOS"
            and tensor_mem_backend(buf) != MEM_BACKEND_HUGEPAGE
        ):
            raise RuntimeError(
                "HiCache NIXL DOCA_MEMOS bounce buffers must be hugetlb-backed. "
                "Set SGLANG_HUGEPAGE_MODE=required, SGLANG_HUGEPAGE_SIZE=2MB, "
                "and reserve enough 2 MiB huge pages."
            )
        self._pre_register_host(buf.data_ptr(), buf.numel() * buf.element_size(), kind)
        return buf

    def _pre_register_host(self, base_addr: int, total_size: int, kind: str) -> None:
        """Register a single DRAM region up-front and remember the handle."""
        reg_descs = self.agent.get_reg_descs([(base_addr, total_size, 0, "")], "DRAM")
        if reg_descs is None:
            raise RuntimeError(f"Failed to build reg descs for host {kind}")
        try:
            self._host_regs.append(self.agent.register_memory(reg_descs))
        except Exception as e:
            raise RuntimeError(f"Failed to pre-register host {kind} with NIXL") from e

    def clear(self) -> None:
        if self.file_manager is None:
            return
        self.file_manager.clear()

    def close(self):
        if self._l3_cleaner is not None:
            self._l3_cleaner.stop()
            self._l3_cleaner = None
        while self._host_regs:
            reg = self._host_regs.pop()
            try:
                self.agent.deregister_memory(reg)
            except Exception as e:
                logger.debug("deregister of pre-registered host region failed: %s", e)
        self._bounce_set = None
        self._bounce_get = None
        self._bounce_page_bytes = None
        self._bounce_payload_numel = None
        self._hybrid_pool_ctx.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def exists(self, key: str) -> bool:
        results = self.batch_exists([key])
        return results > 0

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        if self.is_zero_copy:
            key_list = self._get_key_list_from_meta(keys)
            key_denominator = (
                1 if self.is_mla_model else 2
            )  # MLA: 1 key per page (_k only), non-MLA: 2 NIXL keys per page (_k + _v)
        else:
            key_list = [self._get_suffixed_key(key) for key in keys]
            key_denominator = 1

        exists_results = self._query_keys_exist(key_list)

        for i, exists in enumerate(exists_results):
            if not exists:
                return i // key_denominator
        return len(exists_results) // key_denominator

    def _get_key_list_from_meta(self, keys: List[str]) -> List[str]:
        # Each key maps to a `_k` entry, plus a `_v` entry on non-MLA models
        # (MLA stores k/v interleaved in a single buffer).
        key_list = []
        for key in keys:
            suffixed_key = self._get_suffixed_key(key)
            key_list.append(f"{suffixed_key}_k")
            if not self.is_mla_model:
                key_list.append(f"{suffixed_key}_v")
        return key_list

    def _get_location_and_size_list_from_meta(
        self, keys: List[str], host_indices: torch.Tensor
    ):
        # zero copy: mem_pool_host.get_data_page() does not work due to non-contiguous tensors, causing issues for NIXL transfer
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(
            host_indices
        )
        key_list = self._get_key_list_from_meta(keys)

        if len(key_list) != len(ptr_list):
            logger.error(
                f"HiCacheNixl: mismatch between number of keys and number of buffer meta entries, keys: {len(keys)}, key_list: {len(key_list)}, buffer meta entries: {len(ptr_list)}"
            )
            return [], [], []

        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys: List[str], host_indices: torch.Tensor, op: str):
        """Build (key_list, host_buffers) for the v1 path.

        For zero-copy: ``host_buffers`` are ``(addr, size)`` tuples inside the
        pre-registered ``kv_buffer``.
        For non-zero-copy: ``host_buffers`` are slots of the direction-specific
        pre-registered bounce buffer (``_bounce_set`` for set, ``_bounce_get``
        for get); for ``op == "set"`` we copy the host pages into those slots
        here so the subsequent transfer reads from the bounce buffer.
        Returns ``([], [])`` on validation failure.
        """
        page_size = self.mem_pool_host.page_size
        page_num = len(host_indices) // page_size

        if len(keys) == 0 or len(keys) != page_num:
            logger.warning(
                f"HiCacheNixl: empty keys or mismatch in keys and host_indices lengths. keys: {len(keys)}, host_indices: {len(host_indices)}, page_size: {page_size}"
            )
            return [], []

        if self.is_zero_copy:
            key_list, ptr_list, size_list = self._get_location_and_size_list_from_meta(
                keys, host_indices
            )
            host_buffers = list(zip(ptr_list, size_list))
            return key_list, host_buffers

        if page_num > STORAGE_BATCH_SIZE:
            logger.error(
                f"HiCacheNixl: batch size {page_num} exceeds bounce buffer capacity {STORAGE_BATCH_SIZE}"
            )
            return [], []

        bounce = self._bounce_set if op == "set" else self._bounce_get
        if op == "set":
            if self._logical_anchor:
                bounce[:page_num].fill_(1)
            else:
                asymmetric_pool = (
                    self._get_asymmetric_pool(self.mem_pool_host)
                    if self.backend_selector.backend_name == "DOCA_MEMOS"
                    else None
                )
                for i in range(page_num):
                    index = host_indices[i * page_size]
                    if asymmetric_pool is not None:
                        src = self._get_asymmetric_flat_data_page(
                            asymmetric_pool, index
                        )
                    else:
                        src = self.mem_pool_host.get_data_page(index, flat=True)
                    self._copy_to_bounce_slot(
                        bounce[i], src, self._bounce_payload_numel
                    )

        host_buffers = self._get_bounce_slot_buffers(
            bounce, self._bounce_page_bytes, page_num
        )
        key_list = [self._get_suffixed_key(key) for key in keys]
        return key_list, host_buffers

    def _batch_xfer(
        self,
        keys: List[str],
        key_strs: List[str],
        host_buffers: List[tuple],
        direction: str,
    ) -> List[bool]:
        """Run a batch READ or WRITE for the v1 path against the pre-registered
        host region (no per-transfer host registration).
        """
        if not key_strs or not host_buffers:
            return [False] * len(keys)

        if len(key_strs) != len(host_buffers):
            logger.error("Mismatch between number of key_strs and host_buffers")
            return [False] * len(keys)

        storage_keys = [self._format_key(k) for k in key_strs]
        success = self._xfer_pre_registered(host_buffers, storage_keys, direction)

        # READ results are consumed by _batch_get_postprocess, which pairs
        # entries 2*i / 2*i+1 for non-MLA zero-copy: it needs one bool per
        # key_str (i.e. per `_k`/`_v` buffer). WRITE results map 1:1 to
        # pages, i.e. to `keys`.
        result_len = len(key_strs) if direction == "READ" else len(keys)
        return [success] * result_len

    def _batch_get_postprocess(
        self,
        host_indices: torch.Tensor,
        results: List[bool],
    ) -> List[bool]:
        page_size = self.mem_pool_host.page_size
        page_num = len(host_indices) // page_size

        if self.is_zero_copy:
            # zero copy: update final results based on the boolean results from NIXL transfer
            if self.is_mla_model:
                return results
            return [(results[2 * i] and results[2 * i + 1]) for i in range(page_num)]

        if self._logical_anchor:
            return results

        # non zero copy: copy data from the get-side bounce buffer to mem_pool_host
        asymmetric_pool = (
            self._get_asymmetric_pool(self.mem_pool_host)
            if self.backend_selector.backend_name == "DOCA_MEMOS"
            else None
        )
        for i in range(page_num):
            if not results[i]:
                break
            index = host_indices[i * page_size]
            data_page = self._bounce_get[i].reshape(-1)[: self._bounce_payload_numel]
            if asymmetric_pool is not None:
                self._set_asymmetric_flat_data_page(asymmetric_pool, index, data_page)
            else:
                self.mem_pool_host.set_from_flat_data_page(index, data_page)
        return results

    def _log_xfer_stats(
        self,
        op_name: str,
        num_keys: int,
        host_indices: torch.Tensor,
        buffer_sizes: List[int],
        elapsed_ms: float,
    ) -> None:
        total_bytes = sum(s for s in buffer_sizes if s is not None)
        bw = total_bytes / (elapsed_ms / 1000) / (1024 * 1024) if elapsed_ms else 0.0
        logger.debug(
            f"HiCacheNixl {op_name} transferred: {num_keys} keys (pages), "
            f"{host_indices.numel()} host_indices, {total_bytes} bytes, "
            f"total time: {elapsed_ms:.3f} ms, effective bandwidth: {bw:.2f} MB/s"
        )

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if not self._host_regs:
            logger.error(
                "HiCacheNixl batch_get_v1: register_mem_pool_host must be called first"
            )
            return [False] * len(keys)

        key_strs, host_buffers = self._batch_preprocess(keys, host_indices, "get")
        if not key_strs or not host_buffers:
            return [False] * len(keys)

        start_time = time.perf_counter()
        results = self._batch_xfer(keys, key_strs, host_buffers, "READ")
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._log_xfer_stats(
            "batch_get_v1",
            len(keys),
            host_indices,
            [s for _, s in host_buffers],
            elapsed_ms,
        )

        return self._batch_get_postprocess(host_indices, results)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # skip on MLA backup rank
        if self.backup_skip:
            return [True] * len(keys)

        if len(keys) == 0:
            return []

        if not self._host_regs:
            logger.error(
                "HiCacheNixl batch_set_v1: register_mem_pool_host must be called first"
            )
            return [False] * len(keys)

        key_strs, host_buffers = self._batch_preprocess(keys, host_indices, "set")
        if not key_strs or not host_buffers:
            return [False] * len(keys)

        start_time = time.perf_counter()
        results = self._batch_xfer(keys, key_strs, host_buffers, "WRITE")
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._log_xfer_stats(
            "batch_set_v1",
            len(keys),
            host_indices,
            [s for _, s in host_buffers],
            elapsed_ms,
        )

        return results

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        kv_pages = self.batch_exists(keys, extra_info)
        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            if transfer.name not in self.registered_pools:
                final_pages = 0
                break

            ctx = self._hybrid_pool_ctx.get(transfer.name)
            if ctx is None:
                final_pages = 0
                break
            key_multiplier = (
                self._get_hybrid_key_multiplier(transfer.name, ctx.host_pool)
                if ctx.is_zero_copy
                else 1
            )
            component_keys = self._get_hybrid_component_keys(
                keys[:kv_pages], transfer.name, key_multiplier, ctx.host_pool
            )
            exists_results = self._query_keys_exist(component_keys)
            page_exists = self._page_results(exists_results, key_multiplier)

            boundary = 0
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                try:
                    boundary = page_exists.index(False)
                except ValueError:
                    boundary = kv_pages
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        page_exists[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break

            if boundary:
                hit_count[transfer.name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    @staticmethod
    def _page_results(results: List[bool], key_multiplier: int) -> List[bool]:
        if key_multiplier <= 1:
            return results
        return [
            all(results[i : i + key_multiplier])
            for i in range(0, len(results), key_multiplier)
        ]

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            pool_results: List[bool] = []
            for (
                chunk_keys,
                chunk_host_indices,
            ) in self._iter_storage_pool_transfer_chunks(transfer):
                host_pool, key_strs, host_buffers, page_offsets, key_multiplier = (
                    self._prepare_pool_transfer(
                        transfer,
                        chunk_keys,
                        chunk_host_indices,
                        for_write=False,
                    )
                )
                if host_pool is None or not key_strs:
                    pool_results.extend([False] * len(chunk_keys))
                    break

                start_time = time.perf_counter()
                transfer_results = self._batch_xfer(
                    key_strs, key_strs, host_buffers, "READ"
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._log_xfer_stats(
                    f"batch_get_v2[{transfer.name}]",
                    len(chunk_keys),
                    chunk_host_indices,
                    [size for _, size in host_buffers],
                    elapsed_ms,
                )
                ctx = self._hybrid_pool_ctx[transfer.name]
                page_results = self._page_results(transfer_results, key_multiplier)
                if not ctx.is_zero_copy:
                    for ok, page_offset, data_page in zip(
                        page_results, page_offsets, ctx.bounce_get
                    ):
                        if not ok:
                            break
                        host_pool.set_from_flat_data_page(
                            page_offset,
                            data_page.reshape(-1)[: ctx.bounce_payload_numel],
                        )
                pool_results.extend(page_results)
                if not all(page_results):
                    break
            if len(pool_results) < len(transfer.keys or []):
                pool_results.extend(
                    [False] * (len(transfer.keys or []) - len(pool_results))
                )
            results[transfer.name] = pool_results
        return results

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            pool_results: List[bool] = []
            for (
                chunk_keys,
                chunk_host_indices,
            ) in self._iter_storage_pool_transfer_chunks(transfer):
                _, key_strs, host_buffers, _, key_multiplier = (
                    self._prepare_pool_transfer(
                        transfer,
                        chunk_keys,
                        chunk_host_indices,
                        for_write=True,
                    )
                )
                if not key_strs:
                    pool_results.extend([False] * len(chunk_keys))
                    break

                start_time = time.perf_counter()
                transfer_results = self._batch_xfer(
                    key_strs, key_strs, host_buffers, "WRITE"
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._log_xfer_stats(
                    f"batch_set_v2[{transfer.name}]",
                    len(chunk_keys),
                    chunk_host_indices,
                    [size for _, size in host_buffers],
                    elapsed_ms,
                )
                page_results = self._page_results(transfer_results, key_multiplier)
                pool_results.extend(page_results)
                if not all(page_results):
                    break
            if len(pool_results) < len(transfer.keys or []):
                pool_results.extend(
                    [False] * (len(transfer.keys or []) - len(pool_results))
                )
            results[transfer.name] = pool_results
        return results
