import logging
import os
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.nixl.nixl_routing import route_key

logger = logging.getLogger(__name__)

_SGLANG_NIXL_CONFIG_KEYS = {
    "use_direct_io",
    "l3_cleaner_enabled",
    "l3_cleaner_high_watermark",
    "l3_cleaner_low_watermark",
}


class NixlBackendConfig:
    """Handles NIXL backend configurations"""

    def __init__(self, config: Optional[dict[str, str]] = None):
        """Initialize backend configuration.
        Args:
            config: configurations in a dictionary. This config comes from --hicache-storage-backend-extra-config

            config can be in two forms:
            1. fully qualified form (for all plugins, some of them are enabled, others not):
                {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            2. flat form (for a specific selected plugin), assuming all params apply to a selected plugin
                {'param1': 'value1', 'param2': 'value2', ...}
        """
        self.config = config or {}

    def get_use_direct_io(self) -> bool:
        """Return True if O_DIRECT should be requested when opening files.

        Checks the top-level ``use_direct_io`` key in the long-form JSON config first,
        then falls back to the ``SGLANG_HICACHE_NIXL_USE_DIRECT_IO`` environment variable
        (default: enabled).
        """
        if "use_direct_io" in self.config:
            return bool(self.config["use_direct_io"])
        return envs.SGLANG_HICACHE_NIXL_USE_DIRECT_IO.get()

    def get_l3_cleaner_config(self) -> dict:
        """Return typed NIXL FILE L3 cleaner options from top-level config."""
        config = {
            "enabled": True,
            "high_watermark": 80.0,
            "low_watermark": 70.0,
        }
        if "l3_cleaner_enabled" in self.config:
            enabled = self.config["l3_cleaner_enabled"]
            if not isinstance(enabled, bool):
                raise ValueError("l3_cleaner_enabled must be a boolean")
            config["enabled"] = enabled
        key_map = {
            "l3_cleaner_high_watermark": ("high_watermark", float),
            "l3_cleaner_low_watermark": ("low_watermark", float),
        }
        for raw_key, (cleaner_key, parser) in key_map.items():
            if raw_key in self.config:
                config[cleaner_key] = parser(self.config[raw_key])
        return config

    def get_specified_plugin(self) -> str:
        """decide which plugin to use: either config or SGLANG_HICACHE_NIXL_BACKEND_PLUGIN specifies the plugin, if not, use "auto" """

        if "plugin" in self.config:
            # fully qualified form: {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            # choose the FIRST active plugin
            for key, item in self.config["plugin"].items():
                if item.get("active", False) in [True, "true", "True"]:
                    plugin = key.upper()
                    break
        else:
            # config is empty, or in flat form {'param1': 'value1', 'param2': 'value2', ...}
            plugin = os.getenv("SGLANG_HICACHE_NIXL_BACKEND_PLUGIN", "auto")

        return plugin

    def get_backend_initparams(self, backend_name) -> dict:
        """Get initialization parameters from config of NIXL backend for backend creation.
        Args:
            backend_name: a specific backend's name (already converted "auto" into a specific backend name)

        """

        initparams = {}

        # config can be in two forms:
        if "plugin" in self.config:
            # fully qualified form: {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            if backend_name.lower() in self.config["plugin"]:
                config_data = self.config["plugin"][backend_name.lower()]
            else:
                logger.debug(
                    f"No specific config found for plugin {backend_name} in extra_config. Use default init params."
                )
                config_data = {}
        else:
            # flat form {'param1': 'value1', 'param2': 'value2', ...}
            config_data = self.config

        for key, value in config_data.items():
            # These keys are consumed by SGLang itself, not by NIXL plugins.
            if key in _SGLANG_NIXL_CONFIG_KEYS:
                continue
            initparams[key] = str(value)

        return initparams


class NixlBackendSelection:
    """Handles NIXL backend selection and creation."""

    # Priority order for File-based plugins in case of auto selection
    FILE_PLUGINS = ["3FS", "POSIX", "GDS_MT", "GDS"]
    # Priority order for File-based plugins in case of auto selection (add more as needed)
    OBJ_PLUGINS = ["OBJ"]  # Based on Amazon S3 SDK

    def __init__(
        self, plugin: str = "auto", nixlconfig: Optional[NixlBackendConfig] = None
    ):
        """Initialize backend selection.
        Args:
            plugin: Plugin to use (default "auto" selects best available).
                   Can be a file plugin (3FS, POSIX, GDS, GDS_MT) or
                   an object plugin (OBJ).
        """
        self.plugin = plugin
        self.backend_name = None
        self.mem_type = None
        self.nixlconfig = nixlconfig

    def create_backend(self, agent) -> bool:
        """Create the appropriate NIXL backend based on configuration."""
        try:
            plugin_list = agent.get_plugin_list()
            logger.debug(f"Available NIXL plugins: {plugin_list}")

            # Handle explicit plugin selection or auto priority
            if self.plugin == "auto":
                # Try all file plugins first
                for plugin in self.FILE_PLUGINS:
                    if plugin in plugin_list:
                        self.backend_name = plugin
                        break
                # If no file plugin found, try object plugins
                if not self.backend_name:
                    for plugin in self.OBJ_PLUGINS:
                        if plugin in plugin_list:
                            self.backend_name = plugin
                            break
            else:
                # Use explicitly requested plugin
                self.backend_name = self.plugin

            if self.backend_name not in plugin_list:
                logger.error(
                    f"Backend {self.backend_name} not available in plugins: {plugin_list}"
                )
                return False

            # obtain initparams for the backend from the NIXL config
            initparams = (
                self.nixlconfig.get_backend_initparams(self.backend_name)
                if self.nixlconfig
                else {}
            )

            # Create backend and set memory type
            if self.backend_name in self.OBJ_PLUGINS and "bucket" not in initparams:
                bucket = os.environ.get("AWS_DEFAULT_BUCKET")
                if not bucket:
                    logger.error(
                        "AWS_DEFAULT_BUCKET environment variable must be set for object storage"
                    )
                    return False

                initparams["bucket"] = bucket

            # create backend using initialization parameters
            agent.create_backend(self.backend_name, initparams)

            logger.info(
                f"NixlBackendSelection.create_backend: backend_name {self.backend_name} initparams {initparams} customParams {agent.get_backend_params(self.backend_name)} supported plugins {plugin_list}"
            )

            self.mem_type = "OBJ" if self.backend_name in self.OBJ_PLUGINS else "FILE"
            logger.debug(
                f"Created NIXL backend: {self.backend_name} with memory type: {self.mem_type}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to create NIXL backend: {e}, backend_name {self.backend_name}, supported plugins {plugin_list} initparams {initparams}"
            )
            return False


class NixlFileManager:
    """Handles file system operations for NIXL."""

    def __init__(self, base_dir: "list[str] | str", use_direct_io: bool = True):
        """
        Initialize file manager.
        Args:
            base_dir: Base directory or ordered base directories for tensor files.
            use_direct_io: If True, open files with O_DIRECT (bypasses OS page cache).
                Falls back to buffered I/O with a warning when O_DIRECT is unavailable.
        """
        if isinstance(base_dir, str):
            self.base_dirs = [base_dir] if base_dir else []
        else:
            self.base_dirs = [d for d in base_dir if d]
        self.use_direct_io = use_direct_io
        self._created_bucket_dirs: set[str] = set()
        if not self.base_dirs:
            logger.debug(
                f"Initialized file manager without a base directory. Direct I/O: {use_direct_io}"
            )
        else:
            for base in self.base_dirs:
                os.makedirs(base, exist_ok=True)
            logger.debug(
                f"Initialized file manager with base directories: {self.base_dirs}. Direct I/O: {use_direct_io}"
            )

    def clear(self) -> None:
        """Clear all files below every configured base directory."""
        if not self.base_dirs:
            logger.warning("Base directories are empty, skipping clear operation")
            return

        for base in self.base_dirs:
            try:
                for root, _dirs, files in os.walk(base):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                        except OSError as e:
                            logger.warning(f"Failed to remove file {file_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to clear base directory {base}: {e}")
        logger.debug(f"Cleared all files in base directories: {self.base_dirs}")

    def iter_all_base_dirs(self) -> list[str]:
        """Return base directories that may contain NIXL FILE cache entries."""
        return list(self.base_dirs)

    def get_file_path(self, key: str) -> str:
        """Get full file path for a given key."""
        if not self.base_dirs:
            return key
        disk_idx, bucket = route_key(key, len(self.base_dirs))
        return os.path.join(self.base_dirs[disk_idx], bucket, key)

    def open_file(self, file_path: str, create: bool = False) -> Optional[int]:
        """Open a file and return its file descriptor.

        If ``create`` is True, the file is created if it does not exist
        (mode 0o644, no truncation). When ``self.use_direct_io`` is True,
        the file is opened with ``O_DIRECT`` (bypasses the OS page cache);
        falls back to buffered I/O with a warning if ``O_DIRECT`` is
        unavailable on this platform.
        """
        flags = os.O_RDWR | os.O_CREAT if create else os.O_RDWR
        if self.use_direct_io:
            if hasattr(os, "O_DIRECT"):
                flags |= os.O_DIRECT
            else:
                logger.warning(
                    "use_direct_io is True, but O_DIRECT is not available on "
                    "this system. Falling back to buffered I/O."
                )
        try:
            if create:
                parent = os.path.dirname(file_path)
                if parent and parent not in self._created_bucket_dirs:
                    os.makedirs(parent, exist_ok=True)
                    self._created_bucket_dirs.add(parent)
            return os.open(file_path, flags, 0o644)
        except Exception as e:
            logger.error(f"Failed to open file {file_path}: {e}")
            return None

    def close_file(self, fd: int) -> bool:
        """Close a file descriptor."""
        try:
            os.close(fd)
            return True
        except Exception as e:
            logger.error(f"Failed to close file descriptor {fd}: {e}")
            return False
