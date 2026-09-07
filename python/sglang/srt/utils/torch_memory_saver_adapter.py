import importlib.util
import logging
import os
from abc import ABC
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import torch_memory_saver

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


def _loaded_libcudart_dirs() -> List[str]:
    """Directories of CUDA runtime libraries already mapped into this process.

    Once torch is imported, ``/proc/self/maps`` names the exact libcudart the
    parent uses, so the subprocess can resolve the same one. Returns an empty
    list on platforms without ``/proc`` or in CPU-only processes.
    """
    try:
        with open("/proc/self/maps") as f:
            maps = f.read()
    except OSError:
        return []

    dirs: List[str] = []
    for line in maps.splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        path = fields[5]
        if "libcudart.so" not in os.path.basename(path):
            continue
        parent = os.path.dirname(path)
        if parent not in dirs:
            dirs.append(parent)
    return dirs


def _pip_nvidia_roots() -> List[Path]:
    """Candidate ``site-packages/nvidia`` roots holding pip CUDA wheels."""
    roots: List[Path] = []
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        spec = None
    for location in getattr(spec, "submodule_search_locations", None) or []:
        root = Path(location)
        if root not in roots:
            roots.append(root)

    # Also look next to torch, which covers environments where the ``nvidia``
    # namespace package is not importable by name.
    try:
        import torch

        root = Path(torch.__file__).resolve().parent.parent / "nvidia"
        if root not in roots:
            roots.append(root)
    except Exception:
        pass
    return roots


def _cudart_lib_dirs(nvidia_roots: Iterable[Path]) -> List[str]:
    """``<root>/*/lib`` directories that contain a CUDA runtime library.

    Covers both pip wheel layouts: CUDA 12 component wheels
    (``nvidia/cuda_runtime/lib/libcudart.so.12``) and consolidated CUDA 13
    wheels (``nvidia/cu13/lib/libcudart.so.13``). Only directories that hold
    libcudart are returned, so unrelated libraries are not exposed to the
    subprocess.
    """
    dirs: List[str] = []
    for root in nvidia_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for lib_dir in sorted(root.glob("*/lib")):
            if not lib_dir.is_dir():
                continue
            if any(lib_dir.glob("libcudart.so*")):
                path = str(lib_dir)
                if path not in dirs:
                    dirs.append(path)
    return dirs


def _prepend_ld_library_path(dirs: List[str], current: Optional[str]) -> str:
    """Prepend ``dirs`` to a ``:``-separated path string without duplicates."""
    merged: List[str] = []
    for entry in [*dirs, *(current or "").split(":")]:
        if entry and entry not in merged:
            merged.append(entry)
    return ":".join(merged)


@contextmanager
def _cuda_runtime_ld_library_path():
    """Expose the CUDA runtime directory to subprocesses via LD_LIBRARY_PATH.

    torch_memory_saver injects its preload hook into child processes with
    LD_PRELOAD, so the dynamic loader must resolve the hook's
    ``libcudart.so.<major>`` dependency before Python (and torch's
    RPATH-carrying libraries) are loaded. The hook ships without an RPATH,
    and pip installs the CUDA runtime under ``site-packages/nvidia/*/lib``,
    which is normally not on LD_LIBRARY_PATH, so the child dies with
    ``libcudart.so.13: cannot open shared object file`` (issue #36533).
    Prepending the discovered runtime directories lets the loader find the
    same libcudart the parent already uses. No-op when nothing is discovered
    (e.g. CPU-only builds) or the directories are already present.
    """
    lib_dirs = _loaded_libcudart_dirs()
    for lib_dir in _cudart_lib_dirs(_pip_nvidia_roots()):
        if lib_dir not in lib_dirs:
            lib_dirs.append(lib_dir)

    old_value = os.environ.get("LD_LIBRARY_PATH")
    new_value = _prepend_ld_library_path(lib_dirs, old_value)
    if not lib_dirs or new_value == old_value:
        yield
        return

    logger.debug(
        "memory saver subprocess LD_LIBRARY_PATH gains CUDA runtime dirs: %s",
        lib_dirs,
    )
    os.environ["LD_LIBRARY_PATH"] = new_value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = old_value


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def cuda_graph(self, **kwargs):
        raise NotImplementedError

    def disable(self):
        raise NotImplementedError

    def pause(self, tag: str):
        raise NotImplementedError

    def resume(self, tag: str):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control"""

    @contextmanager
    def configure_subprocess(self):
        with _cuda_runtime_ld_library_path():
            with torch_memory_saver.configure_subprocess():
                yield

    def region(self, tag: str, enable_cpu_backup: bool = False):
        return _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)

    def cuda_graph(self, **kwargs):
        return _memory_saver.cuda_graph(**kwargs)

    def disable(self):
        return _memory_saver.disable()

    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        return _memory_saver.resume(tag=tag)

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        yield

    @contextmanager
    def disable(self):
        yield

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
