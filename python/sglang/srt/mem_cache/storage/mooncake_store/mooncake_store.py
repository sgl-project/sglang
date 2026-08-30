import ctypes
import json
import logging
import os
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import requests
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host import HostKVCache, HostTensorAllocator
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.observability.metrics_collector import StorageMetrics

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB
SETUP_TIMEOUT = 600  # 10min
DEFAULT_TENANT_ID = "default"

logger = logging.getLogger(__name__)


class MooncakeHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        from mooncake.store import MooncakeHostMemAllocator

        self.allocator = MooncakeHostMemAllocator()
        self.ptr = None

    def allocate(
        self, dims: tuple, dtype: torch.dtype, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Allocates memory using MooncakeHostMemAllocator and wraps it in a PyTorch tensor.
        """
        self.dims = dims
        self.dtype = dtype
        size = 1
        for d in dims:
            size *= d
        size *= torch.tensor([], dtype=self.dtype).element_size()
        ptr_int = self.allocator.alloc(size)
        self.ptr = ptr_int
        c_type = ctypes.c_byte * size
        c_array = c_type.from_address(ptr_int)

        tensor = torch.frombuffer(c_array, dtype=torch.uint8, count=size)

        if dtype != torch.uint8:
            element_size = torch.tensor([], dtype=dtype).element_size()
            assert size % element_size == 0, "Size must be divisible by element size"
            tensor = tensor.view(dtype)

        return tensor.view(dims)


def _parse_global_segment_size(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("gb"):
            num = s[:-2].strip()
            if not num:
                raise ValueError(
                    "Invalid global_segment_size: missing number before 'gb'"
                )
            return int(num) * 1024 * 1024 * 1024
        return int(s)
    return int(value)


def _normalize_tenant_id(value) -> str:
    if value is None:
        return DEFAULT_TENANT_ID
    tenant_id = str(value).strip()
    return tenant_id if tenant_id else DEFAULT_TENANT_ID


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    protocol: str
    device_name: str
    master_server_address: str
    master_metrics_port: int
    check_server: bool
    standalone_storage: bool
    client_server_address: str
    enable_ssd_offload: bool = False
    ssd_offload_path: Optional[str] = None
    tenant_id: str = DEFAULT_TENANT_ID

    @staticmethod
    def from_file() -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        if not envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
            raise RuntimeError(
                f"Config file path not set. Please set {envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.name}"
            )
        file_path = envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.get()
        try:
            with open(file_path) as fin:
                config = json.load(fin)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {str(e)}")

        if (
            "master_server_address" not in config
            and "client_server_address" not in config
        ):
            raise ValueError(
                "Either master_server_address or client_server_address is required in config file"
            )

        return MooncakeStoreConfig(
            local_hostname=config.get(
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default
            ),
            metadata_server=config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            protocol=config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=config.get(
                "master_server_address", envs.MOONCAKE_MASTER.default
            ),
            master_metrics_port=config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=config.get("check_server", envs.MOONCAKE_CHECK_SERVER.default),
            standalone_storage=config.get(
                "standalone_storage", envs.MOONCAKE_STANDALONE_STORAGE.default
            ),
            client_server_address=config.get(
                "client_server_address", envs.MOONCAKE_CLIENT.default
            ),
            enable_ssd_offload=config.get(
                "enable_ssd_offload", envs.MOONCAKE_ENABLE_SSD_OFFLOAD.default
            ),
            ssd_offload_path=config.get(
                "ssd_offload_path", envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.default
            ),
            tenant_id=_normalize_tenant_id(
                config.get("tenant_id", envs.MOONCAKE_TENANT_ID.default)
            ),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable.
        export MOONCAKE_MASTER=10.13.3.232:50051
        export MOONCAKE_PROTOCOL="rdma"
        export MOONCAKE_DEVICE=""
        export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
        """
        # other required environment variables...
        if not envs.MOONCAKE_MASTER.is_set() and not envs.MOONCAKE_CLIENT.is_set():
            raise ValueError(
                "Either the environment variable 'MOONCAKE_MASTER' or 'MOONCAKE_CLIENT' is not set."
            )

        # Special handling for local_hostname: try MOONCAKE_LOCAL_HOSTNAME first,
        # then fall back to LOCAL_HOSTNAME if not set.
        # This is for forward compatibility with the legacy LOCAL_HOSTNAME environment variable.
        if envs.MOONCAKE_LOCAL_HOSTNAME.is_set():
            local_hostname = envs.MOONCAKE_LOCAL_HOSTNAME.get()
        else:
            local_hostname = os.getenv(
                "LOCAL_HOSTNAME", envs.MOONCAKE_LOCAL_HOSTNAME.default
            )

        return MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=envs.MOONCAKE_TE_META_DATA_SERVER.get(),
            global_segment_size=_parse_global_segment_size(
                envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.get()
            ),
            protocol=envs.MOONCAKE_PROTOCOL.get(),
            device_name=envs.MOONCAKE_DEVICE.get(),
            master_server_address=envs.MOONCAKE_MASTER.get(),
            master_metrics_port=envs.MOONCAKE_MASTER_METRICS_PORT.get(),
            check_server=envs.MOONCAKE_CHECK_SERVER.get(),
            standalone_storage=envs.MOONCAKE_STANDALONE_STORAGE.get(),
            client_server_address=envs.MOONCAKE_CLIENT.get(),
            enable_ssd_offload=envs.MOONCAKE_ENABLE_SSD_OFFLOAD.get(),
            ssd_offload_path=envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.get(),
            tenant_id=_normalize_tenant_id(envs.MOONCAKE_TENANT_ID.get()),
        )

    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "MooncakeStoreConfig":
        """Load config from extra_config dictionary."""
        if (
            "master_server_address" not in extra_config
            and "client_server_address" not in extra_config
        ):
            raise ValueError(
                "Either master_server_address or client_server_address is required in extra_config"
            )

        return MooncakeStoreConfig(
            local_hostname=extra_config.get(
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default
            ),
            metadata_server=extra_config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                extra_config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            protocol=extra_config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=extra_config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=extra_config.get(
                "master_server_address", envs.MOONCAKE_MASTER.default
            ),
            master_metrics_port=extra_config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=extra_config.get(
                "check_server", envs.MOONCAKE_CHECK_SERVER.default
            ),
            standalone_storage=extra_config.get(
                "standalone_storage", envs.MOONCAKE_STANDALONE_STORAGE.default
            ),
            client_server_address=extra_config.get(
                "client_server_address", envs.MOONCAKE_CLIENT.default
            ),
            enable_ssd_offload=extra_config.get(
                "enable_ssd_offload", envs.MOONCAKE_ENABLE_SSD_OFFLOAD.default
            ),
            ssd_offload_path=extra_config.get(
                "ssd_offload_path", envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.default
            ),
            tenant_id=_normalize_tenant_id(
                extra_config.get("tenant_id", envs.MOONCAKE_TENANT_ID.default)
            ),
        )


class MooncakeBaseStore:
    def __init__(self):
        self.store = None
        self.config = None

    def _import_mooncake_store(self):
        try:
            from mooncake.store import MooncakeDistributedStore

            return MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html "
                "to run SGLang with MooncakeConnector."
            ) from e

    def _import_mooncake_group_semantics(self):
        try:
            from mooncake.store import ReplicateConfig
        except ImportError:
            return None, False

        supports_group_ids = hasattr(ReplicateConfig, "group_ids")
        if not supports_group_ids:
            try:
                supports_group_ids = hasattr(ReplicateConfig(), "group_ids")
            except Exception:
                supports_group_ids = False
        return ReplicateConfig, supports_group_ids

    def _load_config(self, storage_config: Any = None):
        extra_config = (
            getattr(storage_config, "extra_config", None) if storage_config else None
        )

        if extra_config and (
            extra_config.get("master_server_address") is not None
            or extra_config.get("client_server_address") is not None
        ):
            config = MooncakeStoreConfig.load_from_extra_config(extra_config)
            logger.info("Mooncake Configuration loaded from extra_config successfully.")

        elif envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
            config = MooncakeStoreConfig.from_file()
            logger.info("Mooncake Configuration loaded from file successfully.")

        else:
            config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded from env successfully.")

        return config

    def register_buffer(self, tensor: torch.Tensor):
        if self.store is None:
            raise RuntimeError("Mooncake store is not initialized.")
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        if size == 0:
            # conv-only models have a 0-element ssm state; nothing to register
            return
        ret_code = self.store.register_buffer(ptr, size)
        if ret_code != 0:
            logger.error(f"Failed to register buffer, error code: {ret_code}")
            raise RuntimeError(
                f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
            )


class MooncakeStore(HiCacheStorage, MooncakeBaseStore):

    @staticmethod
    def _standalone_required_bytes(mem_pool: Any) -> int:
        """Compute total bytes of host buffers that must be visible to the real client.

        In standalone (dummy client) mode, the real mooncake_client process needs
        to map any host buffers we will later pass by pointer via register_buffer().
        For hybrid models, that includes KV + sidecar pools (e.g. Mamba temporal/conv).
        """
        # Prefer a generic "hybrid pool" accessor when present.
        total = 0
        seen_ptrs: set[int] = set()

        def _add_tensor(t: Optional[torch.Tensor]):
            nonlocal total
            if t is None:
                return
            try:
                ptr = int(t.data_ptr())
            except Exception:
                return
            if ptr in seen_ptrs:
                return
            seen_ptrs.add(ptr)
            total += int(t.numel() * t.element_size())

        # Always include the anchor KV buffer if present.
        _add_tensor(getattr(mem_pool, "kv_buffer", None))

        # HostPoolGroup: include each pool's hybrid buffers when available.
        entries = getattr(mem_pool, "entries", None)
        if entries:
            for entry in entries:
                host_pool = getattr(entry, "host_pool", None)
                if host_pool is None:
                    continue
                # KV pool anchor memory is already covered, but harmless if added twice.
                _add_tensor(getattr(host_pool, "kv_buffer", None))
                for buf in getattr(host_pool, "get_hybrid_pool_buffer", lambda: [])():
                    _add_tensor(buf)
            return total

        # Single HostKVCache-like pool: add its sidecar buffers if any.
        for buf in getattr(mem_pool, "get_hybrid_pool_buffer", lambda: [])():
            _add_tensor(buf)
        return total

    def __init__(
        self, storage_config: HiCacheStorageConfig = None, mem_pool: HostKVCache = None
    ):
        MooncakeBaseStore.__init__(self)
        MooncakeDistributedStore = self._import_mooncake_store()
        self._replicate_config_cls, self._supports_group_ids = (
            self._import_mooncake_group_semantics()
        )
        try:
            self.store = MooncakeDistributedStore()

            self.config = self._load_config(storage_config)
            extra_config = (
                getattr(storage_config, "extra_config", None)
                if storage_config
                else None
            )
            self.enable_group_semantics = bool(
                extra_config.get("enable_group_semantics", False)
                if extra_config
                else False
            )
            self._use_group_semantics = (
                self.enable_group_semantics
                and self._supports_group_ids
                and self._replicate_config_cls is not None
            )
            if self.enable_group_semantics and not self._supports_group_ids:
                logger.warning(
                    "Mooncake group semantics is enabled, but the installed "
                    "Mooncake package does not support ReplicateConfig.group_ids. "
                    "Falling back to the existing batch_put_from path."
                )
            rank_scale_factor = (
                1
                if storage_config is None
                else (storage_config.tp_size * storage_config.pp_size)
            )

            per_rank_global_segment_size = (
                self.config.global_segment_size // rank_scale_factor
            )

            # Use the backend tag and model name as a prefix to isolate tenants
            # and models sharing one store.
            self.config_prefix = None
            config_prefix_parts = []
            if extra_config and extra_config.get("extra_backend_tag") is not None:
                config_prefix_parts.append(str(extra_config["extra_backend_tag"]))
            if storage_config is not None and storage_config.model_name:
                model_name = "-".join(storage_config.model_name.split("/"))
                config_prefix_parts.append(model_name)
            if config_prefix_parts:
                self.config_prefix = "_".join(config_prefix_parts)
                logger.info(f"Using Mooncake config prefix: {self.config_prefix}")

            # Check server status
            if self.config.check_server:
                self.check_server()

            # Handle JSON device_name configuration
            device_name = self.config.device_name
            if device_name and device_name.strip().startswith("{"):
                try:
                    device_config = json.loads(device_name)
                    if storage_config and hasattr(storage_config, "tp_rank"):
                        tp_rank = storage_config.tp_rank
                        # Try both integer and string keys since JSON parsing may convert keys
                        device_name = device_config.get(tp_rank, "")
                        if not device_name:
                            device_name = device_config.get(str(tp_rank), "")
                    else:
                        device_name = ""
                except (json.JSONDecodeError, AttributeError):
                    logger.warning(
                        f"Failed to parse device_name as JSON: {device_name}"
                    )
                    device_name = ""
            if self.config.standalone_storage:
                if not isinstance(mem_pool.allocator, MooncakeHostTensorAllocator):
                    raise RuntimeError(
                        "MooncakeStore with standalone_storage=True requires MooncakeHostTensorAllocator. "
                        "Please set standalone_storage=False "
                        "or upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'."
                    )
                required_bytes = self._standalone_required_bytes(mem_pool)
                ret_code = self.store.setup_dummy(
                    required_bytes,
                    DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer
                    self.config.client_server_address,
                )
            else:
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_mooncake_transfer_engine,
                    )

                    self._shared_mooncake_transfer_engine = (
                        get_mooncake_transfer_engine()
                    )
                except Exception:
                    self._shared_mooncake_transfer_engine = None
                    logger.debug("Failed to reuse initialized mooncake transfer engine")

                # Only reuse the shared MooncakeTransferEngine when its
                # configuration matches the one used by MooncakeStore.
                if (
                    self._shared_mooncake_transfer_engine is not None
                    and device_name
                    == self._shared_mooncake_transfer_engine.get_ib_device()
                    and self.config.metadata_server == "P2PHANDSHAKE"
                    and self.config.protocol == "rdma"
                ):
                    client_hostname = (
                        self._shared_mooncake_transfer_engine.get_session_id()
                    )
                    transfer_engine = self._shared_mooncake_transfer_engine.get_engine()
                    logger.info(
                        f"Reuse initialized mooncake transfer engine: {self._shared_mooncake_transfer_engine}"
                    )
                else:
                    client_hostname = self.config.local_hostname
                    transfer_engine = None

                setup_kwargs = {}
                if self.config.enable_ssd_offload:
                    setup_kwargs["enable_ssd_offload"] = True
                if self.config.ssd_offload_path is not None:
                    setup_kwargs["ssd_offload_path"] = self.config.ssd_offload_path
                if self.config.tenant_id != DEFAULT_TENANT_ID:
                    setup_kwargs["tenant_id"] = self.config.tenant_id

                while True:
                    try:
                        ret_code = self.store.setup(
                            client_hostname,
                            self.config.metadata_server,
                            per_rank_global_segment_size,
                            DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer
                            self.config.protocol,
                            device_name,
                            self.config.master_server_address,
                            transfer_engine,
                            **setup_kwargs,
                        )
                        break
                    except TypeError as e:
                        unsupported_kwargs = [
                            key for key in list(setup_kwargs) if key in str(e)
                        ]
                        if not unsupported_kwargs:
                            raise
                        if "tenant_id" in unsupported_kwargs:
                            raise RuntimeError(
                                "The installed Mooncake version does not support "
                                "tenant_id in MooncakeDistributedStore.setup(). "
                                "Please upgrade Mooncake to use non-default "
                                "Mooncake tenants with SGLang."
                            ) from e
                        logger.warning(
                            "The installed Mooncake version does not support the "
                            f"{', '.join(unsupported_kwargs)} parameter(s) in setup(). "
                            f"Retrying without {', '.join(unsupported_kwargs)}. "
                            "Please upgrade Mooncake to enable SSD offload support."
                        )
                        for key in unsupported_kwargs:
                            setup_kwargs.pop(key, None)
            if ret_code:
                raise RuntimeError(
                    f"Failed to setup Mooncake store, error code: {ret_code}"
                )
            logger.info("Mooncake store setup successfully.")

            self.local_rank = (
                storage_config.tp_rank if storage_config is not None else 0
            )
            self.warmup()
            logger.info("Mooncake store warmup successfully.")

            self.enable_storage_metrics = False
            if storage_config is not None:
                self.is_mla_backend = storage_config.is_mla_model
                self.pp_rank = storage_config.pp_rank
                self.pp_size = storage_config.pp_size
                self.attn_cp_rank = storage_config.attn_cp_rank
                self.attn_cp_size = storage_config.attn_cp_size
                self.enable_storage_metrics = storage_config.enable_storage_metrics
            else:
                self.is_mla_backend = False
                self.local_rank = 0
                self.pp_rank = 0
                self.pp_size = 1
                self.attn_cp_rank = 0
                self.attn_cp_size = 1

            self.enable_pp = self.pp_size > 1
            if self.enable_pp:
                self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
                self.mla_suffix = f"{self.pp_rank}"
            else:
                self.mha_suffix = f"{self.local_rank}"
                self.mla_suffix = ""

            # DSA cache layer split: each CP rank only holds a disjoint layer
            # slice, so L3 objects must be sharded per rank. With a shared key
            # ranks would overwrite each other (first-writer-wins) and a
            # self-read would return another rank's slice. The rank suffix
            # also keeps these objects out of the key space of non-sharded
            # instances of the same model, whose page objects have a
            # different intra-page byte layout.
            if (
                storage_config is not None
                and storage_config.is_mla_model
                and getattr(storage_config, "is_dsa_layer_split", False)
                and self.attn_cp_size > 1
            ):
                self.mla_suffix = f"{self.mla_suffix}_ls{self.attn_cp_rank}"

            # Optional cross-layout L3 fallback (experimental): let a
            # layer-sharded instance serve misses from the canonical page
            # objects written by a non-sharded instance of the same model,
            # and vice versa. Opt-in via extra_config because it assumes
            # both instances share one store and one model checkpoint.
            _xcfg = extra_config or {}
            self.cross_layout_fallback = bool(
                _xcfg.get("enable_cross_layout_fallback", False)
            )
            self.cross_layout_shard_size = int(
                _xcfg.get("cross_layout_shard_size", 0) or 0
            )

            self.storage_config = storage_config
            self.should_split_heads = storage_config.should_split_heads
            self.split_factor = 0
            if self.should_split_heads:
                self.split_factor = (
                    self.storage_config.tp_lcm_size // self.storage_config.tp_size
                )
                base_rank = self.local_rank * self.split_factor
                target_ranks = [base_rank + i for i in range(self.split_factor)]
                if self.enable_pp:
                    self.mha_suffix = [
                        f"{rank}_{self.pp_rank}" for rank in target_ranks
                    ]
                else:
                    self.mha_suffix = [f"{rank}" for rank in target_ranks]

            self.registered_pools = {}

            self.gb_per_page = None
            self.prefetch_pgs = []
            self.backup_pgs = []
            self.prefetch_bandwidth = []
            self.backup_bandwidth = []

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    @staticmethod
    def _iter_host_pool_buffers(host_pool: HostKVCache):
        get_buffers = getattr(
            host_pool,
            "get_hybrid_pool_buffer",
            lambda: [getattr(host_pool, "kv_buffer", None)],
        )
        for buf in get_buffers():
            if buf is not None:
                yield buf

    def check_server(self):
        master_server_ip = self.config.master_server_address.split(":")[0]
        segments_url = f"http://{master_server_ip}:{self.config.master_metrics_port}/get_all_segments"
        start_time = time.perf_counter()

        check_result = False
        while time.perf_counter() - start_time < SETUP_TIMEOUT:
            try:
                check_segments_resp = requests.get(segments_url, timeout=3)
            except Exception:
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            if check_segments_resp.text == "":
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            logger.info("Mooncake store server started successfully.")
            check_result = True
            break

        if not check_result:
            logger.error("Launch mooncake store server timeout")
            raise ValueError("Launch mooncake store server timeout")

    def warmup(self):
        warmup_key = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex
        warmup_value = bytes(4 * 1024)  # 4 KB

        # Retry logic to handle Transfer Engine startup race condition
        max_retries = 10
        retry_delay = 1.0  # seconds

        for attempt in range(max_retries):
            ret = self.store.put(warmup_key, warmup_value)
            if ret == 0:
                break
            logger.warning(
                f"[TP{self.local_rank}] Warmup put failed (attempt {attempt + 1}/{max_retries}), "
                f"ret={ret}, retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)
        else:
            raise RuntimeError(
                f"[TP{self.local_rank}] Warmup put failed after {max_retries} attempts, "
                "Transfer Engine might not be ready"
            )

        assert self.store.is_exist(warmup_key) == 1
        assert self.store.get(warmup_key) == warmup_value

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        if getattr(self.mem_pool_host, "kv_buffer", None) is None:
            # Hybrid logical anchors only own allocation indices. Their physical
            # tensors are registered through register_mem_host_pool_v2().
            return
        try:
            for buffer in self._iter_host_pool_buffers(self.mem_pool_host):
                super().register_buffer(buffer)
        except TypeError as err:
            logger.error("Failed to register buffer to Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Register Buffer Error.") from err

        bytes_per_page = mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size
        self.gb_per_page = bytes_per_page / (1 << 30)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        # KV anchor memory is already registered via register_mem_pool_host().
        # v2 here only registers additional hybrid pools.
        if host_pool_name == PoolName.KV:
            return
        if host_pool_name == PoolName.DRAFT:
            self.registered_pools[host_pool_name] = host_pool
            super().register_buffer(host_pool.kv_buffer)
            return

        # Keep a name->pool mapping so batch v2 can resolve PoolTransfer.name to
        # the corresponding host pool implementation at runtime.
        self.registered_pools[host_pool_name] = host_pool

        # Non-anchor pools are either sidecar-specific pools with their own
        # accessor, or ordinary KV-like host pools used as SWA side pools.
        for buf in self._iter_host_pool_buffers(host_pool):
            super().register_buffer(buf)

    def _tag_keys(self, keys: List[str]) -> List[str]:
        if self.config_prefix is None:
            return keys
        return [f"{self.config_prefix}_{key}" for key in keys]

    def _can_use_group_semantics(self) -> bool:
        return self._use_group_semantics

    def _make_group_id(self, logical_key: str) -> str:
        return f"sglang-hicache:{logical_key}"

    def _expand_group_ids(
        self, logical_keys: List[str], key_multiplier: int
    ) -> List[str]:
        group_ids = []
        for key in logical_keys:
            group_ids.extend([self._make_group_id(key)] * key_multiplier)
        return group_ids

    def _filter_group_ids(
        self, group_ids: Optional[List[str]], indices: List[int]
    ) -> Optional[List[str]]:
        if group_ids is None:
            return None
        return [group_ids[i] for i in indices]

    def _get_hybrid_page_component_keys(
        self, page_keys: List[str], transfer: PoolTransfer
    ) -> Tuple[List[str], int]:
        host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
        if host_pool is None:
            raise ValueError(f"Unregistered Mooncake hybrid pool: {transfer.name}")

        # Suffix order must match get_page_buffer_meta() for one page, because
        # Mooncake zips object keys with registered buffer pointers.
        pool_name = transfer.name
        suffixes = []
        if pool_name == PoolName.MAMBA:
            # Mamba stores one temporal object plus one object per conv state.
            # conv-only models have no ssm state; drop the 0-element temporal
            # object (mooncake rejects 0-size puts). get_page_buffer_meta drops
            # its temporal pointer under the same condition to stay aligned.
            conv_num = len(getattr(host_pool, "conv_buffer", None) or [])
            suffixes = [f"_{self.mha_suffix}_conv_{i}" for i in range(conv_num)]
            if getattr(host_pool, "temporal_state_elem_size", 1) > 0:
                suffixes = [f"_{self.mha_suffix}_temporal"] + suffixes
        elif pool_name == PoolName.DRAFT:
            # Draft pool's MLA/MHA layout is independent from the target
            # (e.g. EAGLE-MHA draft on top of an MLA target), so pick the
            # suffix scheme from the draft pool's own class. The `_draft`
            # tag is what keeps these keys from colliding with target's
            # `{rank}_k` / `{rank}_k` + `{rank}_v` keys.
            draft_pool = self.registered_pools.get(PoolName.DRAFT)
            if isinstance(draft_pool, MLATokenToKVPoolHost):
                suffixes = [f"_{self.mla_suffix}_{PoolName.DRAFT}_k"]
            else:
                suffixes = [
                    f"_{self.mha_suffix}_{PoolName.DRAFT}_k",
                    f"_{self.mha_suffix}_{PoolName.DRAFT}_v",
                ]
        elif pool_name == PoolName.DRAFT_SWA:
            from sglang.srt.mem_cache.memory_pool_host import (
                DeepSeekV4PagedHostPool,
            )
            from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

            if isinstance(
                host_pool,
                (DeepSeekV4PagedHostPool, MLATokenToKVPoolHost),
            ):
                suffixes = [f"_{self.mla_suffix}_{pool_name}"]
            elif isinstance(host_pool, MHATokenToKVPoolHost):
                suffixes = [
                    f"_{self.mha_suffix}_{pool_name}_k",
                    f"_{self.mha_suffix}_{pool_name}_v",
                ]
        elif pool_name in (
            PoolName.INDEXER,
            PoolName.DRAFT_INDEXER,
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C128,
            PoolName.DEEPSEEK_V4_C4_STATE,
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            PoolName.DEEPSEEK_V4_C128_STATE,
        ):
            # DSA indexer and DeepSeek V4 side pools are page-packed
            # single-object pools.
            suffixes = [f"_{self.mla_suffix}_{pool_name}"]
        elif pool_name == PoolName.SWA:
            if not self.is_mla_backend and hasattr(host_pool, "v_buffer"):
                # Ordinary MHA SWA mirrors a K/V pool.
                suffixes = [
                    f"_{self.mha_suffix}_{pool_name}_k",
                    f"_{self.mha_suffix}_{pool_name}_v",
                ]
            elif self.is_mla_backend:
                suffixes = [f"_{self.mla_suffix}_{pool_name}"]

        if not suffixes:
            raise ValueError(
                f"Unsupported Mooncake hybrid pool name: {pool_name}, "
                f"host_pool={type(host_pool)}"
            )
        key_multiplier = len(suffixes)
        component_keys = [
            f"{page_key}{suffix}" for page_key in page_keys for suffix in suffixes
        ]
        return component_keys, key_multiplier

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        if self.mem_pool_host.kv_buffer is None:
            # Logical anchor: no physical KV object exists in Mooncake, so the
            # usable prefix is determined entirely by required sidecar objects.
            kv_pages = len(keys)
        else:
            kv_pages = self.batch_exists(keys, extra_info)

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            component_keys, key_multiplier = self._get_hybrid_page_component_keys(
                keys, transfer
            )
            component_keys = self._tag_keys(component_keys)
            ex = self._batch_exist(component_keys)
            ex = self._cross_layout_exists_fallback(component_keys, ex)
            if key_multiplier > 0:
                page_exists = [
                    all(
                        r == 1
                        for r in ex[i * key_multiplier : (i + 1) * key_multiplier]
                    )
                    for i in range(kv_pages)
                ]
            else:
                page_exists = [False] * kv_pages
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

    def _batch_io_v2(self, transfers: List[PoolTransfer], is_set: bool):
        # Unified v2 I/O path: each PoolTransfer can expand to one or more
        # storage objects per logical page, but API still reports page-level result.
        results: dict = {}
        for transfer in transfers:
            host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
            keys = transfer.keys
            page_size = getattr(host_pool, "page_size", 1) or 1
            host_indices = transfer.host_indices
            assert len(keys) > 0
            assert len(keys) == len(host_indices) // page_size

            tagged_keys = self._tag_keys(keys)
            key_strs, key_multiplier = self._get_hybrid_page_component_keys(
                keys, transfer
            )
            key_strs = self._tag_keys(key_strs)
            ptr_list, element_size_list = host_pool.get_page_buffer_meta(host_indices)
            # layer_first pools return one buffer per (page, layer); pack them
            # into a single multi-buffer object per page key.
            if transfer.name == PoolName.DEEPSEEK_V4_C4 or (
                len(ptr_list) > len(key_strs)
            ):
                ptr_list, element_size_list = self._pack_multi_buffer_meta(
                    key_strs, ptr_list, element_size_list
                )

            if is_set:
                group_ids = (
                    self._expand_group_ids(tagged_keys, key_multiplier)
                    if self._can_use_group_semantics()
                    else None
                )
                exist_result = self._batch_exist(key_strs)
                io_results = [0 if state == 1 else -1 for state in exist_result]
                missing_idx = [i for i, state in enumerate(exist_result) if state != 1]
                if missing_idx:
                    put_results = self._put_batch_zero_copy_impl(
                        [key_strs[i] for i in missing_idx],
                        [ptr_list[i] for i in missing_idx],
                        [element_size_list[i] for i in missing_idx],
                        self._filter_group_ids(group_ids, missing_idx),
                    )
                    for i, res in zip(missing_idx, put_results):
                        io_results[i] = res
            else:
                io_results = self._get_batch_zero_copy_impl(
                    key_strs, ptr_list, element_size_list
                )
                io_results = self._cross_layout_get_fallback(
                    key_strs,
                    ptr_list,
                    element_size_list,
                    io_results,
                    token_major=False,
                    full_layer_num=getattr(host_pool, "layer_num", 0),
                )
            results[transfer.name] = self._batch_postprocess(
                io_results, is_set_operate=is_set, key_multiplier=key_multiplier
            )
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        return self._batch_io_v2(transfers, is_set=False)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        return self._batch_io_v2(transfers, is_set=True)

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

    @staticmethod
    def _uses_multi_buffer(buffer_ptrs: List[Any]) -> bool:
        return bool(buffer_ptrs) and isinstance(buffer_ptrs[0], Sequence)

    @staticmethod
    def _pack_multi_buffer_meta(
        key_strs: List[str],
        ptr_list: List[int],
        element_size_list: List[int],
    ) -> Tuple[List[Any], List[Any]]:
        if len(ptr_list) == len(key_strs):
            return ptr_list, element_size_list

        assert len(key_strs) > 0
        assert len(ptr_list) == len(element_size_list)
        assert len(ptr_list) % len(key_strs) == 0

        nbuf = len(ptr_list) // len(key_strs)
        return [ptr_list[i : i + nbuf] for i in range(0, len(ptr_list), nbuf)], [
            element_size_list[i : i + nbuf]
            for i in range(0, len(element_size_list), nbuf)
        ]

    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mha_suffix}_k")
            key_list.append(f"{key_}_{self.mha_suffix}_v")
        if len(key_list) != len(ptr_list):
            raise RuntimeError(
                "Mooncake layer_first multi-buffer is only supported for MLA "
                "host KV pool. Use page_first/page_first_direct for MHA."
            )
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mla_suffix}_k")
        ptr_list, element_size_list = self._pack_multi_buffer_meta(
            key_list, ptr_list, element_size_list
        )
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        else:
            if self.should_split_heads:
                return self._get_mha_split_heads_buffer_meta(keys, host_indices)
            else:
                return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(
        self, results: List[int], is_set_operate=False, key_multiplier=None
    ):
        """
        refer to https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-store/include/pybind_client.h
        for batch_get_into, results is Vector of integers,
            where each element is the number of bytes read on success, or a negative value on error
        for batch_put_from, results is Vector of integers,
            where each element is 0 on success, or a negative value on error
        """
        if key_multiplier is None:
            if self.is_mla_backend:
                key_multiplier = 1
            else:
                key_multiplier = 2
                if self.should_split_heads:
                    key_multiplier *= self.split_factor

        result_groups = [
            results[i : i + key_multiplier]
            for i in range(0, len(results), key_multiplier)
        ]
        return [
            (
                all(res == 0 for res in group)
                if is_set_operate
                else all(res > 0 for res in group)
            )
            for group in result_groups
        ]

    def _cross_layout_active(self) -> bool:
        if not getattr(self, "cross_layout_fallback", False):
            return False
        if not self.is_mla_backend:
            return False
        if "_ls" in (self.mla_suffix or ""):
            return True  # sharded reader, canonical peer objects
        return self.cross_layout_shard_size > 1  # canonical reader, shards

    @staticmethod
    def _cross_layout_key_tail(key: str) -> Optional[str]:
        for tail in ("_k", "_indexer"):
            if key.endswith(tail):
                return tail
        return None

    def _cross_layout_get_batch(self, keys: List[str], chunk: int) -> List[bytes]:
        """get_batch with per-key retry: large batches have been observed to
        return empty payloads for a subset of keys, so refetch holes
        individually before treating them as misses."""
        blobs: List[bytes] = []
        for c in range(0, len(keys), chunk):
            part = keys[c : c + chunk]
            try:
                blobs.extend(self.store.get_batch(part))
            except Exception:
                blobs.extend([b""] * len(part))
        for j, b in enumerate(blobs):
            if b is None or len(b) == 0:
                try:
                    blobs[j] = self.store.get(keys[j])
                except Exception:
                    pass
        return [bytes(b or b"") for b in blobs]

    def _cross_layout_exists_fallback(self, query_keys, exist_result):
        """Treat a page as present when the peer layout's object(s) exist."""
        if not self._cross_layout_active():
            return exist_result
        fixed = list(exist_result)
        sharded_self = "_ls" in (self.mla_suffix or "")
        miss = [
            i
            for i, r in enumerate(fixed)
            if r != 1 and self._cross_layout_key_tail(query_keys[i]) is not None
        ]
        if sharded_self:
            miss = [i for i in miss if self.mla_suffix in query_keys[i]]
            if not miss:
                return fixed
            plain = [
                query_keys[i].replace(self.mla_suffix, "", 1) for i in miss
            ]
            pex = self._batch_exist(plain)
            for i, pe in zip(miss, pex):
                if pe == 1:
                    fixed[i] = 1
            return fixed
        n = self.cross_layout_shard_size
        miss = [i for i in miss if "_ls" not in query_keys[i]]
        if not miss:
            return fixed
        probe = []
        for i in miss:
            tail = self._cross_layout_key_tail(query_keys[i])
            probe.extend(
                query_keys[i][: -len(tail)] + f"_ls{r}" + tail for r in range(n)
            )
        pex = self._batch_exist(probe)
        for j, i in enumerate(miss):
            if all(pex[j * n + r] == 1 for r in range(n)):
                fixed[i] = 1
        return fixed

    def _cross_layout_get_fallback(
        self,
        key_strs,
        ptr_list,
        element_size_list,
        io_results,
        token_major: bool,
        full_layer_num: int,
    ):
        """Serve get() misses from the peer layout's L3 objects.

        Canonical MLA KV page objects are token-major
        (page_size, layers, record); sidecar pools such as the DSA indexer
        are layer-major (layers, page_stride). Layer shards written by a
        layer-sharded instance are contiguous layer ranges of the same
        per-layer page bytes (see get_layer_shard_range), so translating
        between the two is a deterministic gather/scatter. Correctness was
        validated byte-for-byte against real objects; deeper layers may
        differ numerically between engines of different parallel topologies
        (reduction order under fp8), which is semantically benign -- PD
        disaggregation consumes cross-engine KV the same way.
        """
        if not self._cross_layout_active():
            return io_results
        import ctypes

        import numpy as np

        sharded_self = "_ls" in (self.mla_suffix or "")
        page_tokens = getattr(self.mem_pool_host, "page_size", 64)
        fixed = list(io_results)
        todo = []
        for i, res in enumerate(io_results):
            if res > 0:
                continue
            key = key_strs[i]
            tail = self._cross_layout_key_tail(key)
            if tail is None:
                continue
            if sharded_self:
                if self.mla_suffix not in key:
                    continue
                todo.append((i, key.replace(self.mla_suffix, "", 1), tail))
            else:
                if "_ls" in key:
                    continue
                todo.append((i, key, tail))
        if not todo:
            return fixed
        t0 = time.perf_counter()
        filled = 0
        if sharded_self:
            rank, size = self.attn_cp_rank, self.attn_cp_size
            blobs = self._cross_layout_get_batch([pk for _, pk, _ in todo], 8)
            for (i, pk, tail), obj in zip(todo, blobs):
                if not obj:
                    continue
                ptrs = ptr_list[i]
                sizes = element_size_list[i]
                if not isinstance(ptrs, (list, tuple)):
                    ptrs, sizes = [ptrs], [sizes]
                block_bytes = int(sizes[0])
                if block_bytes <= 0 or len(obj) % block_bytes:
                    continue
                total_layers = len(obj) // block_bytes
                base, rem = total_layers // size, total_layers % size
                start = rank * base + min(rank, rem)
                own = base + (1 if rank < rem else 0)
                ok = True
                if token_major:
                    rec = block_bytes // page_tokens
                    arr = np.frombuffer(obj, dtype=np.uint8).reshape(
                        page_tokens, total_layers, rec
                    )
                    for b in range(len(ptrs)):
                        if b >= own:
                            continue  # trailing padding blocks stay empty
                        blk = np.ascontiguousarray(arr[:, start + b, :])
                        if blk.nbytes != int(sizes[b]):
                            ok = False
                            break
                        ctypes.memmove(int(ptrs[b]), blk.ctypes.data, blk.nbytes)
                else:
                    for b in range(len(ptrs)):
                        if b >= own:
                            continue
                        off = (start + b) * block_bytes
                        blk = obj[off : off + block_bytes]
                        if len(blk) != int(sizes[b]):
                            ok = False
                            break
                        ctypes.memmove(
                            int(ptrs[b]),
                            (ctypes.c_char * len(blk)).from_buffer_copy(blk),
                            len(blk),
                        )
                if ok:
                    fixed[i] = 1
                    filled += 1
        else:
            n = self.cross_layout_shard_size
            flat = []
            for i, key, tail in todo:
                flat.extend(
                    key[: -len(tail)] + f"_ls{r}" + tail for r in range(n)
                )
            blobs = self._cross_layout_get_batch(flat, 48)
            base, rem = full_layer_num // n, full_layer_num % n
            for t, (i, key, tail) in enumerate(todo):
                shards = blobs[t * n : (t + 1) * n]
                if any(len(b) == 0 for b in shards):
                    continue
                ptrs = ptr_list[i]
                sizes = element_size_list[i]
                if isinstance(ptrs, (list, tuple)):
                    if len(ptrs) != 1:
                        continue
                    ptrs, sizes = ptrs[0], sizes[0]
                expected = int(sizes)
                if full_layer_num <= 0 or expected % full_layer_num:
                    continue
                block_bytes = expected // full_layer_num
                ok = True
                if token_major:
                    rec = block_bytes // page_tokens
                    out = np.empty(
                        (page_tokens, full_layer_num, rec), dtype=np.uint8
                    )
                    for r in range(n):
                        own = base + (1 if r < rem else 0)
                        start = r * base + min(r, rem)
                        if len(shards[r]) < own * block_bytes:
                            ok = False
                            break
                        arr = np.frombuffer(
                            shards[r][: own * block_bytes], dtype=np.uint8
                        ).reshape(own, page_tokens, rec)
                        for b in range(own):
                            out[:, start + b, :] = arr[b]
                    blob = out.tobytes() if ok else b""
                else:
                    parts = []
                    for r in range(n):
                        own = base + (1 if r < rem else 0)
                        if len(shards[r]) < own * block_bytes:
                            ok = False
                            break
                        parts.append(shards[r][: own * block_bytes])
                    blob = b"".join(parts) if ok else b""
                if not ok or len(blob) != expected:
                    continue
                ctypes.memmove(
                    int(ptrs),
                    (ctypes.c_char * len(blob)).from_buffer_copy(blob),
                    len(blob),
                )
                fixed[i] = 1
                filled += 1
        if filled:
            logger.debug(
                "cross-layout fallback filled %d/%d pages in %.2fs",
                filled,
                len(todo),
                time.perf_counter() - t0,
            )
        return fixed

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if self.mem_pool_host.kv_buffer is None:
            # DeepSeek V4's KV anchor is logical only; v2 side pools carry data.
            return [True] * len(keys)

        # Apply config prefix if available.
        keys = self._tag_keys(keys)

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)

        start_time = time.perf_counter()
        get_results = self._get_batch_zero_copy_impl(
            key_strs, buffer_ptrs, buffer_sizes
        )
        get_results = self._cross_layout_get_fallback(
            key_strs,
            buffer_ptrs,
            buffer_sizes,
            get_results,
            token_major=True,
            full_layer_num=getattr(self.mem_pool_host, "layer_num", 0),
        )
        end_time = time.perf_counter()

        if self.enable_storage_metrics:
            self.prefetch_pgs.append(len(keys))
            self.prefetch_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        return self._batch_postprocess(get_results, is_set_operate=False)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        if self.mem_pool_host.kv_buffer is None:
            # DeepSeek V4's KV anchor is logical only; v2 side pools carry data.
            return [True] * len(keys)

        # Apply config prefix if available.
        keys = self._tag_keys(keys)

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        key_multiplier = len(key_strs) // len(keys)
        group_ids = (
            self._expand_group_ids(keys, key_multiplier)
            if self._can_use_group_semantics()
            else None
        )
        exist_result = self._batch_exist(key_strs)

        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [-1] * len(key_strs)
        for i in range(len(key_strs)):
            if exist_result[i] != 1:
                set_keys.append(key_strs[i])
                set_buffer_ptrs.append(buffer_ptrs[i])
                set_buffer_sizes.append(buffer_sizes[i])
                set_indices.append(i)
            else:
                set_results[i] = 0

        # Only set non-existing keys to storage
        if len(set_keys) > 0:
            start_time = time.perf_counter()
            put_results = self._put_batch_zero_copy_impl(
                set_keys,
                set_buffer_ptrs,
                set_buffer_sizes,
                self._filter_group_ids(group_ids, set_indices),
            )
            end_time = time.perf_counter()

            if self.enable_storage_metrics:
                self.backup_pgs.append(len(set_keys))
                self.backup_bandwidth.append(
                    len(set_keys) / (end_time - start_time) * self.gb_per_page
                )

            for i in range(len(set_indices)):
                set_results[set_indices[i]] = put_results[i]

        return self._batch_postprocess(set_results, is_set_operate=True)

    def set(
        self,
        key,
        value: Optional[Any] = None,
        target_location: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        # Only support zero copy set for now
        assert target_location is not None and target_sizes is not None
        exist_result = self._batch_exist([key])
        if exist_result[0] == 1:
            return True
        put_result = self._put_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return put_result[0] == 0

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        # Only support zero copy set for now
        assert target_locations is not None and target_sizes is not None
        assert len(keys) == len(target_locations) == len(target_sizes)

        if len(keys) == 0:
            return False

        for i in range(len(keys)):
            if (
                keys[i] is None
                or target_locations[i] is None
                or target_sizes[i] is None
            ):
                return False

        exist_result = self._batch_exist(keys)
        set_keys = []
        set_target_locations = []
        set_target_sizes = []
        set_indices = []
        for i in range(len(keys)):
            if exist_result[i] != 1:
                set_keys.append(keys[i])
                set_target_locations.append(target_locations[i])
                set_target_sizes.append(target_sizes[i])
                set_indices.append(i)
        # Only set non-existing keys to storage
        start_time = time.perf_counter()
        put_result = self._put_batch_zero_copy_impl(
            set_keys, set_target_locations, set_target_sizes
        )
        end_time = time.perf_counter()

        if self.enable_storage_metrics:
            self.backup_pgs.append(len(set_keys))
            self.backup_bandwidth.append(
                len(set_keys) / (end_time - start_time) * self.gb_per_page
            )

        for i in range(len(set_indices)):
            if put_result[i] == 0:
                exist_result[set_indices[i]] = 1

        success_count = 0
        for i in range(len(keys)):
            if exist_result[i] == 0:
                break
            success_count += 1
        # TODO: return the number of consecutive successful operations from the start.
        return success_count == len(keys)

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        assert target_location is not None and target_sizes is not None
        get_result = self._get_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return get_result[0] >= 0

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        assert len(keys) == len(target_locations) == len(target_sizes)
        if len(keys) == 0:
            return 0

        start_time = time.perf_counter()
        get_result = self._get_batch_zero_copy_impl(
            keys, target_locations, target_sizes
        )
        end_time = time.perf_counter()

        if self.is_mla_backend:
            key_multiplier = 1
        else:
            key_multiplier = 2

        if self.enable_storage_metrics:
            self.prefetch_pgs.append(len(keys))
            self.prefetch_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        for i in range(len(keys)):
            if get_result[i] < 0:
                return i // key_multiplier
        return len(keys) // key_multiplier

    def exists(self, key) -> bool:
        exist_result = self._batch_exist([key])
        return exist_result[0] == 1

    def batch_exists(
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        # Apply config prefix if available.
        keys = self._tag_keys(keys)

        if self.is_mla_backend:
            query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]
            key_multiplier = 1
        else:
            query_keys = []
            if self.should_split_heads:
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

        exist_result = self._batch_exist(query_keys)
        exist_result = self._cross_layout_exists_fallback(query_keys, exist_result)
        for i in range(len(query_keys)):
            if exist_result[i] != 1:
                return i // key_multiplier
        return len(query_keys) // key_multiplier

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        pass

    def clear(self) -> None:
        self.store.remove_all()

    def _put_batch_zero_copy_impl(
        self,
        key_strs: List[str],
        buffer_ptrs: List[Any],
        buffer_sizes: List[Any],
        group_ids: Optional[List[str]] = None,
    ) -> List[int]:
        config = None
        if self._can_use_group_semantics() and group_ids is not None:
            if len(group_ids) != len(key_strs):
                raise ValueError(
                    "Mooncake group_ids length must match key_strs length: "
                    f"{len(group_ids)} != {len(key_strs)}"
                )
            config = self._replicate_config_cls()
            config.group_ids = group_ids

        if self._uses_multi_buffer(buffer_ptrs):
            config = config or self._replicate_config_cls()
            return self.store.batch_put_from_multi_buffers(
                key_strs, buffer_ptrs, buffer_sizes, config
            )
        elif config is not None:
            return self.store.batch_put_from(
                key_strs, buffer_ptrs, buffer_sizes, config
            )
        else:
            return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)

    def _get_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[Any], buffer_sizes: List[Any]
    ) -> List[int]:
        if self._uses_multi_buffer(buffer_ptrs):
            return self.store.batch_get_into_multi_buffers(
                key_strs, buffer_ptrs, buffer_sizes
            )
        return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)

    def _batch_exist(self, key_strs: List[str]) -> List[int]:
        return self.store.batch_is_exist(key_strs)

    def get_stats(self):
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return storage_metrics
