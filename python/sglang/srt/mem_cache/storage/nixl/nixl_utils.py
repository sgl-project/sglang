import ctypes
import logging
import os
import uuid
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.nixl.nixl_routing import (
    _BUCKET_MASK,
    BUCKET_HEX_CHARS,
    route_key,
)

logger = logging.getLogger(__name__)

# 0 when the platform has no O_DIRECT (macOS); patched in tests.
_O_DIRECT = getattr(os, "O_DIRECT", 0)
# 0 before Linux 3.11 and on non-Linux; patched in tests.
_O_TMPFILE = getattr(os, "O_TMPFILE", 0)

# Consumed by SGLang, never forwarded to a plugin. "active" is deliberately absent:
# it selects the plugin, so listing it would make promotion flag every valid config.
_SGLANG_NIXL_CONFIG_KEYS = {
    "use_direct_io",
    "l3_cleaner_enabled",
    "l3_cleaner_high_watermark",
    "l3_cleaner_low_watermark",
}

# SGLang's own plugin selector, read by get_specified_plugin(). No NIXL plugin
# declares it, so forwarding it to create_backend passes an unknown option.
_PLUGIN_SELECTOR_KEY = "active"


def _normalize_initparam(key: str, value) -> str:
    """NIXL plugins match their boolean-like options case-sensitively, so a JSON
    ``true`` stringified to ``"True"`` is accepted and then silently ignored."""
    if isinstance(value, bool):
        return "true" if value else "false"
    # Only Python's bool repr is folded, not every casing: an option value is an
    # opaque case-sensitive string to NIXL, so a value that really is "TRUE"
    # reaches the plugin unchanged. Folding this one is still a guess, so say so.
    if value in ("True", "False"):
        logger.warning(
            "NIXL extra-config: %s is the string %r, which NIXL would compare "
            "case-sensitively against its lowercase boolean options; sending %r "
            "instead. Write a JSON boolean to make this unambiguous.",
            key,
            value,
            value.lower(),
        )
        return value.lower()
    return str(value)


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
        # Copied so promoting a misplaced key cannot mutate the caller's extra_config.
        self.config = dict(config or {})
        self._promote_misplaced_sglang_keys()

    def _promote_misplaced_sglang_keys(self) -> None:
        nested: dict[str, list] = {}
        plugins = self.config.get("plugin")
        if isinstance(plugins, dict):
            for plugin_name, plugin_config in plugins.items():
                if not isinstance(plugin_config, dict):
                    continue
                for key in sorted(_SGLANG_NIXL_CONFIG_KEYS & plugin_config.keys()):
                    nested.setdefault(key, []).append((plugin_name, plugin_config[key]))

        for key, occurrences in nested.items():
            where = ", ".join(f"plugin.{name}" for name, _ in occurrences)
            values = [value for _, value in occurrences]
            if key in self.config:
                resolution = "the top-level value is used instead"
            elif any(value != values[0] for value in values[1:]):
                resolution = "it is set to conflicting values there and is ignored"
            else:
                resolution = "honoring it as a top-level option"
                self.config[key] = values[0]
            logger.warning(
                "NIXL extra-config: %s under %s is consumed by SGLang itself, not by "
                "the NIXL plugin; %s. Move it to the top level of "
                "--hicache-storage-backend-extra-config.",
                key,
                where,
                resolution,
            )

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

        plugins = self.config.get("plugin")
        if isinstance(plugins, dict):
            # fully qualified form: {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            # choose the FIRST active plugin
            for key, item in plugins.items():
                if isinstance(item, dict) and item.get("active", False) in [
                    True,
                    "true",
                    "True",
                ]:
                    return key.upper()
            logger.warning(
                "NIXL extra-config: no plugin under 'plugin' is marked active; "
                "selecting the backend from SGLANG_HICACHE_NIXL_BACKEND_PLUGIN instead."
            )
        elif plugins is not None:
            logger.warning(
                "NIXL extra-config: 'plugin' is a %s, not a mapping of plugin name "
                "to options; selecting the backend from "
                "SGLANG_HICACHE_NIXL_BACKEND_PLUGIN instead.",
                type(plugins).__name__,
            )

        # config is empty, or in flat form {'param1': 'value1', 'param2': 'value2', ...}
        return os.getenv("SGLANG_HICACHE_NIXL_BACKEND_PLUGIN", "auto")

    def get_backend_initparams(self, backend_name) -> dict:
        """Get initialization parameters from config of NIXL backend for backend creation.
        Args:
            backend_name: a specific backend's name (already converted "auto" into a specific backend name)

        """

        initparams = {}

        # config can be in two forms:
        plugins = self.config.get("plugin")
        if plugins is not None:
            # fully qualified form: {'plugin': { 'posix': {...}, 'gds': {...}, ...}}
            config_data = {}
            if isinstance(plugins, dict):
                # A malformed section carries no options; the plugin gets its defaults.
                section = plugins.get(backend_name.lower())
                config_data = section if isinstance(section, dict) else {}
            if not config_data:
                logger.debug(
                    f"No specific config found for plugin {backend_name} in extra_config. Use default init params."
                )
        else:
            # flat form {'param1': 'value1', 'param2': 'value2', ...}
            config_data = self.config

        for key, value in config_data.items():
            # These keys are consumed by SGLang itself, not by NIXL plugins.
            if key in _SGLANG_NIXL_CONFIG_KEYS or key == _PLUGIN_SELECTOR_KEY:
                continue
            initparams[key] = _normalize_initparam(key, value)

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
            self.ensure_all_bucket_dirs()
            logger.debug(
                f"Initialized file manager with base directories: {self.base_dirs}. Direct I/O: {use_direct_io}"
            )

    def direct_io_error(self, addr: int, size: int) -> Optional[str]:
        """Return why an O_DIRECT write sourced from ``addr`` cannot be used, or
        None if it succeeds on every base directory. ``addr`` must be
        OS-page-aligned and ``size`` a multiple of the page size.

        Every base directory is probed: they are separate mounts, so O_DIRECT can
        be usable on one and rejected on the next.
        """
        if not self.base_dirs:
            return None
        if not _O_DIRECT:
            return "O_DIRECT is not available on this platform"
        for base in self.base_dirs:
            error = self._direct_io_dir_error(base, addr, size)
            if error is not None:
                return error
        return None

    def _open_probe_file(self, base: str) -> "tuple[int, Optional[str]]":
        """Open a scratch file for the probe, returning its fd and the path to
        unlink (None when the file has no directory entry to begin with).

        An O_TMPFILE probe is unlinked by construction, so a SIGKILL between open
        and unlink cannot leave a file behind: the L3 cleaner only walks bucket
        directories, so a stray file in a base directory is never reclaimed.
        """
        if _O_TMPFILE:
            try:
                fd = os.open(base, os.O_WRONLY | _O_TMPFILE | _O_DIRECT, 0o644)
                return fd, None
            except OSError:
                # Not every filesystem implements O_TMPFILE; a named file still
                # answers the question this probe asks.
                pass
        path = os.path.join(
            base, f".direct_io_probe.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        return (
            os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_DIRECT, 0o644),
            path,
        )

    def _direct_io_dir_error(self, base: str, addr: int, size: int) -> Optional[str]:
        fd = None
        path = None
        try:
            try:
                fd, path = self._open_probe_file(base)
            except OSError as e:
                return (
                    f"opening a scratch file under {base} with O_DIRECT failed: "
                    f"{e.strerror} (errno {e.errno})"
                )
            try:
                written = os.pwrite(fd, (ctypes.c_char * size).from_address(addr), 0)
            except OSError as e:
                return (
                    f"a page-aligned {size}-byte O_DIRECT write under {base} failed: "
                    f"{e.strerror} (errno {e.errno})"
                )
            if written != size:
                return (
                    f"a page-aligned {size}-byte O_DIRECT write under {base} moved "
                    f"only {written} bytes"
                )
            return None
        finally:
            if fd is not None:
                os.close(fd)
            if path is not None:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def disable_direct_io(self, reason: str) -> None:
        logger.warning(
            "NixlFileManager: disabling O_DIRECT and falling back to buffered "
            "tier-3 I/O: %s",
            reason,
        )
        self.use_direct_io = False

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

    def ensure_all_bucket_dirs(self) -> None:
        """Pre-create every possible bucket directory under each base dir.

        Called once when path mode is active so NIXL O_CREAT writes never
        fail due to a missing parent directory.
        """
        for base in self.base_dirs:
            for i in range(_BUCKET_MASK + 1):
                os.makedirs(
                    os.path.join(base, f"{i:0{BUCKET_HEX_CHARS}x}"),
                    exist_ok=True,
                )

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
            if _O_DIRECT:
                flags |= _O_DIRECT
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
