# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

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
    def create_backend(
        cls,
        backend_name: str,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
        **kwargs,
    ) -> HiCacheStorage:
        """Create a storage backend instance.
        Args:
            backend_name: Name of the backend to create
            storage_config: Storage configuration
            mem_pool_host: Memory pool host object
            **kwargs: Additional arguments passed to external backends
        Returns:
            Initialized storage backend instance
        Raises:
            ValueError: If backend is not registered and cannot be dynamically loaded
            ImportError: If backend module cannot be imported
            Exception: If backend initialization fails
        """
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

        # Try to dynamically load backend from extra_config
        if backend_name == "dynamic" and storage_config.extra_config is not None:
            backend_config = storage_config.extra_config
            return cls._create_dynamic_backend(
                backend_config, storage_config, mem_pool_host, **kwargs
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
        mem_pool_host: Any,
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

            # Create the backend instance with storage_config
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
        if backend_name == "file":
            return backend_class(storage_config)
        elif backend_name == "nixl":
            return backend_class(storage_config)
        elif backend_name == "mooncake":
            backend = backend_class(storage_config)
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
        else:
            raise ValueError(f"Unknown built-in backend: {backend_name}")


# Register built-in storage backends
StorageBackendFactory.register_backend(
    "file", "sglang.srt.mem_cache.hicache_storage", "HiCacheFile"
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
