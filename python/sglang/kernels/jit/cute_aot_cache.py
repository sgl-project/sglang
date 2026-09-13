# Manage Ahead-of-Time (AOT) compiled kernels
"""In-memory and persistent caches for CuTe DSL JIT functions.

Compiled objects persist under ``SGLANG_CUTE_AOT_CACHE_DIR`` (default
``{SGLANG_CACHE_DIR}/cute_aot``) and are shared across process restarts. Set
the variable to an empty string to keep compilation process-local. The
directory must be trusted: cached object files are loaded into the process.
"""

import ctypes
import fcntl
import hashlib
import logging
import os
import pickle
import platform
import sys
import time
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Hashable, TypeAlias

logger = logging.getLogger(__name__)

_runtime_library_handles: list[Any] = []
_loaded_modules: list[Any] = []

CompileKeyType: TypeAlias = tuple[Hashable, ...]
CallableFunction: TypeAlias = Any

_UNSET_CACHE_DIR = object()


def _normalize_disk_key(value: Any) -> Any:
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ == "device":
        return ("torch.device", value.type)
    if isinstance(value, tuple):
        return tuple(_normalize_disk_key(item) for item in value)
    return value


@lru_cache(maxsize=None)
def _compute_source_fingerprint(
    source_paths: tuple[str, ...], enable_tvm_ffi: bool, target_arch: str
) -> str:
    """
    Hash all CuTe Python sources plus runtime ABI stamps into a short fingerprint.

    The fingerprint changes with the supplied sources, Python/CuTe versions,
    selected ABI, CUDA version, or target architecture.

    Computed once per process and cached.
    """
    import cutlass

    h = hashlib.sha256()

    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    # Exported objects contain host machine code, not just GPU code.
    h.update(f"host={sys.platform}-{platform.machine()}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"cuda={getattr(cutlass, 'CUDA_VERSION', 'unknown')}".encode())
    h.update(f"tvm_ffi={enable_tvm_ffi}".encode())
    h.update(f"arch={target_arch}".encode())
    if enable_tvm_ffi:
        import tvm_ffi

        h.update(f"tvm_ffi_version={tvm_ffi.__version__}".encode())

    for index, raw_path in enumerate(source_paths):
        source_path = Path(raw_path).resolve()
        if source_path.is_dir():
            sources = sorted(source_path.rglob("*.py"))
            root = source_path
        elif source_path.is_file():
            sources = [source_path]
            root = source_path.parent
        else:
            raise FileNotFoundError(source_path)
        for src in sources:
            if not src.is_file():
                continue
            h.update(f"{index}:{src.relative_to(root).as_posix()}".encode())
            content = src.read_bytes()
            h.update(len(content).to_bytes(8, "little"))
            h.update(content)

    return h.hexdigest()


def _resolve_target_arch() -> str:
    if target_arch := os.getenv("CUTE_DSL_ARCH"):
        return target_arch

    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


# Pre-load cute DSL runtime libraries with RTLD_GLOBAL so that their symbols
# (e.g. _cudaLibraryLoadData) are visible to .so modules loaded later via dlopen.
# Upstream cute.runtime.load_module loads these without RTLD_GLOBAL, which causes
# "undefined symbol" errors when loading cached kernels from disk.
@lru_cache(maxsize=2)
def _preload_runtime_libraries(enable_tvm_ffi: bool) -> None:
    import cutlass.cute as cute

    for raw_path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=enable_tvm_ffi):
        path = Path(raw_path)
        if path.is_file():
            _runtime_library_handles.append(
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            )


def _load_object(
    object_path: Path, function_prefix: str, enable_tvm_ffi: bool
) -> CallableFunction:
    import cutlass.cute as cute

    _preload_runtime_libraries(enable_tvm_ffi)
    module = cute.runtime.load_module(str(object_path), enable_tvm_ffi=enable_tvm_ffi)
    try:
        function = module[function_prefix]
    except (KeyError, TypeError):
        function = getattr(module, function_prefix)
    _loaded_modules.append(module)
    return function


class FileLock:
    """Context manager for advisory file locks using fcntl.flock.

    Supports exclusive (write) and shared (read) locks.
    Always blocks with polling until the lock is acquired or timeout is reached.

    Usage:
        with FileLock(lock_path, exclusive=True, timeout=15, label="abc"):
            # do work under lock
    """

    def __init__(
        self,
        lock_path: Path,
        exclusive: bool,
        timeout: float = 15,
        label: str = "",
    ):
        """
        Args:
            lock_path: Path to the lock file on disk.
            exclusive: True for exclusive (write) lock, False for shared (read) lock.
            timeout: Max seconds to wait for lock acquisition before raising RuntimeError.
            label: Optional human-readable label for error messages.
        """
        self.lock_path: Path = lock_path
        self.exclusive: bool = exclusive
        self.timeout: float = timeout
        self.label: str = label
        self._fd: int = -1

    @property
    def _lock_label(self) -> str:
        kind = "exclusive" if self.exclusive else "shared"
        return f"{kind} {self.label}" if self.label else kind

    def __enter__(self) -> "FileLock":
        open_flags = (
            os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        )
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH

        self._fd = os.open(str(self.lock_path), open_flags)

        deadline = time.monotonic() + self.timeout
        acquired = False
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(0.1)
        if not acquired:
            os.close(self._fd)
            self._fd = None
            raise RuntimeError(
                f"Timed out after {self.timeout}s waiting for "
                f"{self._lock_label} lock: {self.lock_path}"
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class JITCache:
    """
    In-memory cache for compiled functions.
    """

    def __init__(self):
        self.cache: dict[CompileKeyType, CallableFunction] = {}

    def __setitem__(self, key: CompileKeyType, fn: CallableFunction) -> None:
        self.cache[key] = fn

    def __getitem__(self, key: CompileKeyType) -> CallableFunction:
        return self.cache[key]

    def __contains__(self, key: CompileKeyType) -> bool:
        return key in self.cache

    def clear(self) -> None:
        """
        Clear in-memory cache of compiled functions
        """
        self.cache.clear()


class JITPersistentCache(JITCache):
    """
    In-memory cache for compiled functions, which is also backed by persistent storage.

    ``cache_path`` may be a path or a zero-argument callable returning one; a
    callable is resolved on first storage access, so constructing a cache at
    import time never probes the GPU or hashes sources.
    """

    EXPORT_FUNCTION_PREFIX = "func"
    LOCK_TIMEOUT_SECONDS = 15

    def __init__(
        self,
        cache_path: Path | Callable[[], Path],
        *,
        enable_tvm_ffi: bool = True,
    ):
        super().__init__()
        self._cache_path_source = cache_path
        self._resolved_cache_path: Path | None = None
        self.enable_tvm_ffi = enable_tvm_ffi

    @property
    def cache_path(self) -> Path:
        if self._resolved_cache_path is None:
            source = self._cache_path_source
            path = Path(source() if callable(source) else source)
            path.mkdir(parents=True, exist_ok=True)
            self._resolved_cache_path = path
        return self._resolved_cache_path

    def __setitem__(self, key: CompileKeyType, fn: CallableFunction) -> None:
        JITCache.__setitem__(self, key, fn)
        self._try_export_to_storage(key, fn)

    def __getitem__(self, key: CompileKeyType) -> CallableFunction:
        # Use __contains__ to try populating in-memory cache with persistent storage
        self.__contains__(key)
        return JITCache.__getitem__(self, key)

    def __contains__(self, key: CompileKeyType) -> bool:
        # Checks in-memory cache first, then tries loading from storage.
        # When returning True, guarantees the in-memory cache is populated.
        if JITCache.__contains__(self, key):
            return True
        return self._try_load_from_storage(key)

    def _try_load_from_storage(self, key: CompileKeyType) -> bool:
        """
        Try to load a function from persistent storage into in-memory cache.
        Returns True if loaded successfully, False if not found on disk.
        Holds a shared lock during loading to prevent concurrent writes.
        """
        sha256_hex = self._key_to_hash(key)
        obj_path = self.cache_path / f"{sha256_hex}.o"
        invalid_inode = None
        with FileLock(
            self._lock_path(sha256_hex),
            exclusive=False,
            timeout=self.LOCK_TIMEOUT_SECONDS,
            label=sha256_hex,
        ):
            if obj_path.exists():
                logger.debug("Loading compiled function from disk: %s", obj_path)
                try:
                    fn = _load_object(
                        obj_path, self.EXPORT_FUNCTION_PREFIX, self.enable_tvm_ffi
                    )
                except Exception as error:
                    logger.warning("Invalid cache object %s: %s", obj_path, error)
                    try:
                        invalid_inode = obj_path.stat().st_ino
                    except OSError:
                        pass
                else:
                    JITCache.__setitem__(self, key, fn)
                    return True
            else:
                logger.debug("Cache miss on disk for key hash %s", sha256_hex)
        if invalid_inode is not None:
            self._discard_invalid_object(sha256_hex, obj_path, invalid_inode)
        return False

    def _discard_invalid_object(
        self, sha256_hex: str, obj_path: Path, invalid_inode: int
    ) -> None:
        """Unlink a failed object under an exclusive lock.

        The shared load lock is released first (flock cannot upgrade), so a
        writer may republish in the gap; the inode check keeps a fresh object
        intact. Eviction is best-effort: on lock timeout the object is left
        for the next process.
        """
        try:
            with FileLock(
                self._lock_path(sha256_hex),
                exclusive=True,
                timeout=self.LOCK_TIMEOUT_SECONDS,
                label=sha256_hex,
            ):
                try:
                    if obj_path.stat().st_ino == invalid_inode:
                        obj_path.unlink()
                except OSError:
                    return
        except RuntimeError as error:
            logger.warning("Could not evict invalid object %s: %s", obj_path, error)

    def _try_export_to_storage(self, key: CompileKeyType, fn: CallableFunction) -> None:
        """Export a compiled function to persistent storage under exclusive lock."""
        sha256_hex = self._key_to_hash(key)
        with FileLock(
            self._lock_path(sha256_hex),
            exclusive=True,
            timeout=self.LOCK_TIMEOUT_SECONDS,
            label=sha256_hex,
        ):
            obj_path = self.cache_path / f"{sha256_hex}.o"
            if obj_path.exists():
                # Another process already exported.
                logger.debug("Skipping export, already on disk: %s", obj_path)
                return
            logger.debug("Exporting compiled function to disk: %s", obj_path)
            temp_key = f".{sha256_hex}.tmp"
            temp_obj_path = self.cache_path / f"{temp_key}.o"
            temp_obj_path.unlink(missing_ok=True)
            try:
                if self.enable_tvm_ffi:
                    fn.export_to_c(
                        object_file_path=str(temp_obj_path),
                        function_name=self.EXPORT_FUNCTION_PREFIX,
                    )
                else:
                    fn.export_to_c(
                        str(self.cache_path),
                        temp_key,
                        function_prefix=self.EXPORT_FUNCTION_PREFIX,
                    )
                os.replace(temp_obj_path, obj_path)
            finally:
                temp_obj_path.unlink(missing_ok=True)
            logger.debug(
                "Successfully exported compiled function to disk: %s", obj_path
            )

    def _key_to_hash(self, key: CompileKeyType) -> str:
        disk_key = (self.enable_tvm_ffi, _normalize_disk_key(key))
        return hashlib.sha256(pickle.dumps(disk_key)).hexdigest()

    def _lock_path(self, sha256_hex: str) -> Path:
        return self.cache_path / f"{sha256_hex}.lock"

    def clear(self) -> None:
        """
        Not only clear the in-memory cache. Also purge persistent compilation cache.
        """
        logger.debug("Clearing persistent cache at %s", self.cache_path)
        super().clear()
        for child in self.cache_path.iterdir():
            child.unlink()


def get_jit_cache(
    name: str | None = None,
    *,
    cache_dir: Any = _UNSET_CACHE_DIR,
    source_paths: Sequence[str | os.PathLike[str]] = (),
    enable_tvm_ffi: bool = True,
) -> JITCache:
    """
    JIT cache factory.
    `name` is an optional identifier to create subdirectories to manage cache.

    ``cache_dir`` defaults to ``SGLANG_CUTE_AOT_CACHE_DIR``; pass ``None`` (or
    set the variable to an empty string) for a process-local cache.

    When persistent caching is enabled, artifacts are namespaced under a
    source fingerprint directory so that code or dependency changes
    automatically invalidate stale entries.
    """
    if cache_dir is _UNSET_CACHE_DIR:
        from sglang.srt.environ import envs

        cache_dir = envs.SGLANG_CUTE_AOT_CACHE_DIR.get() or None
    if cache_dir is None:
        logger.debug("Persistent cache disabled, using in-memory JIT cache")
        return JITCache()

    def resolve_cache_path() -> Path:
        paths = (str(Path(__file__).resolve()),) + tuple(
            str(Path(path).resolve()) for path in source_paths
        )
        path = Path(cache_dir).expanduser() / _compute_source_fingerprint(
            paths, enable_tvm_ffi, _resolve_target_arch()
        )
        if name:
            path = path / name
        logger.debug("Creating persistent JIT cache at %s", path)
        return path

    return JITPersistentCache(resolve_cache_path, enable_tvm_ffi=enable_tvm_ffi)
