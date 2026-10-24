# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict

from sglang.srt.mem_cache.hicache_storage import (
    DCP_CAP_DRAFT_MLA_SHARDED,
    DCP_CAP_EXACT_RESTORABLE_PREFIX_PAGES,
    DCP_CAP_LOGICAL_PAGE_INDICES,
    DCP_CAP_MAMBA_TP_SHARDED,
    DCP_CAP_MLA_SHARDED,
    HiCacheStorage,
    HiCacheStorageConfig,
    PoolName,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StorageBackendFactory:
    """Factory for creating storage backend instances with support for dynamic loading."""

    _registry: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _load_backend_class(
        module_path: str, class_name: str, backend_name: str
    ) -> type[HiCacheStorage]:
        """Load and validate a backend class from module path."""
        try:
            module = importlib.import_module(module_path)
            backend_class = getattr(module, class_name)
            if not issubclass(backend_class, HiCacheStorage):
                raise TypeError(
                    f"Backend class {class_name} must inherit from HiCacheStorage"
                )
            return backend_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import backend '{backend_name}' from '{module_path}': {e}"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_path}': {e}"
            ) from e

    @classmethod
    def register_backend(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a storage backend with lazy loading.

        Args:
            name: Backend identifier
            module_path: Python module path containing the backend class
            class_name: Name of the backend class
        """
        if name in cls._registry:
            logger.warning(f"Backend '{name}' is already registered, overwriting")

        def loader() -> type[HiCacheStorage]:
            """Lazy loader function to import the backend class."""
            return cls._load_backend_class(module_path, class_name, name)

        cls._registry[name] = {
            "loader": loader,
            "module_path": module_path,
            "class_name": class_name,
        }

    @classmethod
    def validate_backend(
        cls, backend_name, storage_config, host_pools, backend_class=None
    ):
        """Check the pool set against the backend before any native client I/O.

        ``backend_class`` is the dynamic backend's class; registered backends
        are validated by name alone. Also runs when a hybrid controller
        registers a sidecar after startup.
        """
        if storage_config.dcp_size > 1:
            cls._validate_dcp_host_pools(backend_name, storage_config, host_pools)
        if backend_name == "dynamic":
            cls._validate_dynamic_backend(backend_class, storage_config, host_pools)

    @staticmethod
    def _validate_dcp_host_pools(backend_name, storage_config, host_pools):
        if backend_name not in ("file", "mooncake", "dynamic"):
            raise NotImplementedError("Backend does not support DCP storage.")
        from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        anchor = host_pools.get(PoolName.KV)
        if anchor is None:
            raise NotImplementedError("DCP storage requires a KV anchor.")
        mla_pools = {PoolName.KV}
        if backend_name == "dynamic":
            mla_pools.add(PoolName.DRAFT)
        for name, pool in host_pools.items():
            if name == PoolName.MAMBA:
                if not isinstance(pool, MambaPoolHost):
                    raise NotImplementedError("Unsupported DCP storage sidecar.")
                continue
            if name not in mla_pools:
                raise NotImplementedError("Unsupported DCP storage sidecar.")
            if not isinstance(pool, MLATokenToKVPoolHost) or pool.kv_buffer is None:
                raise NotImplementedError(
                    "DCP storage requires materialized MLA pools."
                )
            if pool.layout not in ("layer_first", "page_first", "page_first_direct"):
                raise NotImplementedError(
                    "DCP storage requires generic MLA page layout."
                )
            # Materialized MLA host pools reject separate FP4 scales while
            # capturing MLABufferInfo; they no longer retain a device_pool.
            if backend_name == "dynamic" and (
                pool.dcp_size != storage_config.dcp_size
                or pool.dcp_rank != storage_config.dcp_rank
                or pool.logical_page_size != storage_config.logical_page_size
                or pool.page_size * storage_config.dcp_size != pool.logical_page_size
                or pool.layout != anchor.layout
            ):
                raise NotImplementedError(
                    "DCP storage pool geometry or layout mismatch."
                )

    @staticmethod
    def _validate_dynamic_backend(backend_class, storage_config, host_pools):
        dcp = storage_config.dcp_size > 1
        if dcp:
            required = {DCP_CAP_MLA_SHARDED, DCP_CAP_LOGICAL_PAGE_INDICES}
            if PoolName.MAMBA in host_pools:
                required.update(
                    (DCP_CAP_MAMBA_TP_SHARDED, DCP_CAP_EXACT_RESTORABLE_PREFIX_PAGES)
                )
            if PoolName.DRAFT in host_pools:
                required.add(DCP_CAP_DRAFT_MLA_SHARDED)
            missing = required - backend_class.dcp_storage_capabilities
            if missing:
                raise NotImplementedError(
                    f"Dynamic backend lacks DCP storage capabilities: {sorted(missing)}"
                )
        # Legacy plugins that declare nothing keep their DCP1 path untouched.
        if dcp or backend_class.dcp_storage_capabilities:
            backend_class.validate_dcp_config(storage_config, host_pools)

    @classmethod
    def create_backend(
        cls,
        backend_name: str,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
        storage_host_pools=None,
        **kwargs,
    ) -> HiCacheStorage:
        """Create a storage backend instance.
        Args:
            backend_name: Name of the backend to create
            storage_config: Storage configuration
            mem_pool_host: Memory pool host object
            storage_host_pools: PoolName -> host pool map validated against the
                backend; defaults to ``{PoolName.KV: mem_pool_host}``
            **kwargs: Additional arguments passed to external backends
        Returns:
            Initialized storage backend instance
        Raises:
            ValueError: If backend is not registered and cannot be dynamically loaded
            ImportError: If backend module cannot be imported
            Exception: If backend initialization fails
        """
        if storage_host_pools is None:
            storage_host_pools = {PoolName.KV: mem_pool_host}
        if backend_name == "dynamic":
            if storage_config.extra_config is None:
                raise ValueError("Dynamic storage backend requires extra_config.")
            return cls._create_dynamic_backend(
                storage_config.extra_config,
                storage_config,
                storage_host_pools,
                **kwargs,
            )
        cls.validate_backend(backend_name, storage_config, storage_host_pools)
        # First check if backend is already registered
        if backend_name in cls._registry:
            registry_entry = cls._registry[backend_name]
            backend_class = registry_entry["loader"]()
            logger.info(
                f"Creating storage backend '{backend_name}' "
                f"({registry_entry['module_path']}.{registry_entry['class_name']})"
            )
            return cls._create_builtin_backend(
                backend_name, backend_class, storage_config, mem_pool_host
            )

        # Backend not found
        available_backends = list(cls._registry.keys())

        raise ValueError(
            f"Unknown storage backend '{backend_name}'. "
            f"Registered backends: {available_backends}. "
        )

    @classmethod
    def _create_dynamic_backend(
        cls,
        backend_config: Dict[str, Any],
        storage_config: HiCacheStorageConfig,
        storage_host_pools: Dict[PoolName, Any],
        **kwargs,
    ) -> HiCacheStorage:
        """Create a backend dynamically from configuration."""
        required_fields = ["backend_name", "module_path", "class_name"]
        for field in required_fields:
            if field not in backend_config:
                raise ValueError(
                    f"Missing required field '{field}' in backend config for 'dynamic' backend"
                )

        backend_name = backend_config["backend_name"]
        module_path = backend_config["module_path"]
        class_name = backend_config["class_name"]

        try:
            # Import the backend class
            backend_class = cls._load_backend_class(
                module_path, class_name, backend_name
            )

            logger.info(
                f"Creating dynamic storage backend '{backend_name}' "
                f"({module_path}.{class_name})"
            )

            # Validation precedes __init__: plugins may open native clients there.
            cls.validate_backend(
                "dynamic",
                storage_config,
                storage_host_pools,
                backend_class=backend_class,
            )
            return backend_class(storage_config, kwargs)
        except Exception as e:
            logger.error(
                f"Failed to create dynamic storage backend '{backend_name}': {e}"
            )
            raise

    @classmethod
    def _create_builtin_backend(
        cls,
        backend_name: str,
        backend_class: type[HiCacheStorage],
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
    ) -> HiCacheStorage:
        """Create built-in backend with original initialization logic."""
        if backend_name in ("file", "sim"):
            return backend_class(storage_config)
        elif backend_name == "nixl":
            return backend_class(storage_config)
        elif backend_name == "mooncake":
            backend = backend_class(storage_config, mem_pool_host)
            return backend
        elif backend_name == "npu_memcache":
            backend = backend_class(storage_config, mem_pool_host)
            return backend
        elif backend_name == "aibrix":
            backend = backend_class(storage_config, mem_pool_host)
            return backend
        elif backend_name == "hf3fs":
            # Calculate bytes_per_page based on memory pool layout
            if mem_pool_host.layout in ["page_first", "page_first_direct"]:
                bytes_per_page = (
                    mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size
                )
            elif mem_pool_host.layout == "layer_first":
                bytes_per_page = (
                    mem_pool_host.get_size_per_token() * mem_pool_host.page_size
                )

            dtype = mem_pool_host.dtype
            return backend_class.from_env_config(bytes_per_page, dtype, storage_config)
        elif backend_name == "eic":
            return backend_class(storage_config, mem_pool_host)
        elif backend_name == "simm":
            return backend_class(storage_config, mem_pool_host)
        elif backend_name == "mori":
            return backend_class(storage_config, mem_pool_host)
        elif backend_name == "shm":
            return backend_class(storage_config, mem_pool_host)
        elif backend_name == "seaweedfs":
            return backend_class(storage_config, mem_pool_host)
        elif backend_name == "tensorcast":
            return backend_class(storage_config)
        else:
            raise ValueError(f"Unknown built-in backend: {backend_name}")


# Register built-in storage backends
StorageBackendFactory.register_backend(
    "file", "sglang.srt.mem_cache.hicache_storage", "HiCacheFile"
)

StorageBackendFactory.register_backend(
    "sim", "sglang.srt.mem_cache.storage.sim_storage", "SimHiCacheStorage"
)

StorageBackendFactory.register_backend(
    "nixl",
    "sglang.srt.mem_cache.storage.nixl.hicache_nixl",
    "HiCacheNixl",
)

StorageBackendFactory.register_backend(
    "mooncake",
    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store",
    "MooncakeStore",
)

StorageBackendFactory.register_backend(
    "npu_memcache",
    "sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store",
    "NpuMemcacheStore",
)

StorageBackendFactory.register_backend(
    "hf3fs",
    "sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs",
    "HiCacheHF3FS",
)

StorageBackendFactory.register_backend(
    "aibrix",
    "sglang.srt.mem_cache.storage.aibrix_kvcache.aibrix_kvcache_storage",
    "AibrixKVCacheStorage",
)

StorageBackendFactory.register_backend(
    "eic",
    "sglang.srt.mem_cache.storage.eic.eic_storage",
    "EICStorage",
)

StorageBackendFactory.register_backend(
    "simm",
    "sglang.srt.mem_cache.storage.simm.hicache_simm",
    "HiCacheSiMM",
)

StorageBackendFactory.register_backend(
    "mori",
    "sglang.srt.mem_cache.storage.umbp.umbp_store",
    "UMBPStore",
)

StorageBackendFactory.register_backend(
    "shm",
    "sglang.srt.mem_cache.storage.shm",
    "HiCacheShm",
)

StorageBackendFactory.register_backend(
    "tensorcast",
    "sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store",
    "TensorcastStore",
)

StorageBackendFactory.register_backend(
    "seaweedfs",
    "sglang.srt.mem_cache.storage.seaweedfs.seaweedfs_store",
    "SeaweedFSStore",
)
