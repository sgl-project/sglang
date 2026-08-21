"""NIXL memory-registration helpers.

A ``NixlRegistry`` instance bundles the agent, the memory type, and
(optionally) the file manager.  ``acquire_storage(...)`` performs the
entire open-register-build-descs sequence for the storage side of a
transfer and returns a live ``StorageRegistration`` (or None on failure,
with any partially acquired resources released).  ``release_storage(...)``
unwinds ``agent.deregister_memory`` plus any ``os.close(fd)``.  The
``storage(...)`` context manager wraps the pair for callers whose
registration lifetime matches one ``with`` block; an asynchronous caller
can instead hold the ``StorageRegistration`` for the lifetime of its
transfer handle and release on completion.

The host side is pre-registered up front by ``HiCacheNixl`` and is not
touched per transfer.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Any, List, Optional

import msgspec

from .nixl_utils import NixlFileManager

logger = logging.getLogger(__name__)


def _buffer_sizes(buffers) -> Optional[List[int]]:
    """Per-buffer byte sizes for ``(addr, len)`` tuple inputs."""
    if not buffers or not isinstance(buffers[0], tuple):
        return None
    return [b[1] for b in buffers]


class StorageRegistration(msgspec.Struct):
    """Live storage-side registration for one transfer.

    Returned by ``NixlRegistry.acquire_storage``; release with
    ``NixlRegistry.release_storage`` exactly once, after the last use of
    ``descs``.  For fd-mode FILE registrations the fds backing ``descs``
    stay open until release.
    """

    descs: Any
    reg: Any = None
    fds: List[int] = []


class NixlRegistry:
    """Owns the (agent, mem_type, file_manager) triple and manages the
    storage-side registration lifetime of transfers.

    A single instance is created once per HiCacheNixl in __init__ and
    reused for every transfer.
    """

    def __init__(
        self,
        agent,
        mem_type: str,
        file_manager: Optional[NixlFileManager] = None,
    ):
        self.agent = agent
        self.mem_type = mem_type
        self.file_manager = file_manager
        # OBJ devIds key a process-wide map in the NIXL OBJ plugin
        # (devIdToObjKey_) that is not protected by a lock, so concurrent
        # OBJ registrations must use disjoint devId ranges. Allocate them
        # from a single monotonic counter.
        self._obj_devid_lock = threading.Lock()
        self._obj_devid_next = 1
        self.path_mode = mem_type == "FILE" and self._probe_path_mode()
        if mem_type == "FILE" and self.path_mode:
            logger.info("HiCacheNixl: path-mode FILE registration active.")
        elif mem_type == "FILE":
            # TODO: NIXL 1.3.0 adds path-mode support; remove this fd fallback once 1.3.0 is widely installed.
            logger.info(
                "HiCacheNixl: the installed NIXL build does not "
                "support path-mode FILE registration; using legacy "
                "fd registration."
            )

    def _probe_path_mode(self) -> bool:
        """Probe whether NIXL honours path-mode metaInfo.

        Register a FILE_SEG with a valid path-mode string pointing at a
        nonexistent path (no 'create' flag). A path-mode-capable NIXL tries
        to open() the path, fails with NIXL_ERR_BACKEND, and raises. A
        pre-path-mode NIXL ignores metaInfo and returns NIXL_SUCCESS.
        Error from register_memory => path mode supported.
        """
        reg_descs = self.agent.get_reg_descs(
            [(0, 4096, 1, "rw:/nonexistent-nixl-probe")], "FILE"
        )
        if reg_descs is None:
            return False
        try:
            reg = self.agent.register_memory(reg_descs)
            if reg is not None:
                try:
                    self.agent.deregister_memory(reg)
                except Exception:
                    pass
            return False
        except Exception:
            return True

    def _register(self, items: List[tuple], mem_type: str):
        """Register ``items`` with NIXL; returns the handle or None."""
        if not items:
            return None
        reg_descs = self.agent.get_reg_descs(items, mem_type)
        if reg_descs is None:
            return None
        try:
            return self.agent.register_memory(reg_descs)
        except Exception as e:
            logger.error(f"Failed to register memory of type {mem_type}: {e}")
            return None

    def _close_fds(self, fds: List[int]) -> None:
        for fd in fds:
            self.file_manager.close_file(fd)

    def _acquire_file_path_mode(
        self, *, keys: List[str], sizes: List[int], direction: str
    ) -> Optional[StorageRegistration]:
        parts = ["rw", "create"] if direction == "WRITE" else ["ro"]
        if self.file_manager.use_direct_io:
            parts.append("direct")
        spec = ",".join(parts)
        tuples = [(0, sizes[i], i + 1, f"{spec}:{keys[i]}") for i in range(len(keys))]
        reg = self._register(tuples, "FILE")
        if reg is None:
            return None
        return StorageRegistration(descs=reg.trim(), reg=reg)

    def _acquire_file_fd_mode(
        self, *, keys: List[str], sizes: List[int], direction: str
    ) -> Optional[StorageRegistration]:
        fds: List[int] = []
        for path in keys:
            fd = self.file_manager.open_file(path, create=(direction == "WRITE"))
            if fd is None:
                self._close_fds(fds)
                return None
            fds.append(fd)
        tuples = [(0, sizes[i], fds[i], keys[i]) for i in range(len(keys))]
        reg = self._register(tuples, "FILE")
        if reg is None:
            self._close_fds(fds)
            return None
        descs = self.agent.get_xfer_descs(
            [(0, sizes[i], fds[i]) for i in range(len(fds))], "FILE"
        )
        return StorageRegistration(descs=descs, reg=reg, fds=fds)

    def _acquire_obj(
        self, *, keys: List[str], sizes: List[int]
    ) -> Optional[StorageRegistration]:
        # Reg tuple: (addr=0, size, devId, metaInfo=key).
        # Xfer tuple: (addr=0, size, devId). devId links each xfer desc
        # back to its registered object's metaInfo, so devIds must be
        # unique within the list AND globally unique across concurrent
        # acquire_storage() calls (the OBJ plugin's devIdToObjKey_ map is
        # shared and unlocked). NIXL's pybind layer requires position 3 to
        # be int, hence the key goes in metaInfo (position 4).
        n = len(keys)
        with self._obj_devid_lock:
            base = self._obj_devid_next
            self._obj_devid_next += n
        dev_ids = list(range(base, base + n))
        tuples = [(0, sizes[i], dev_ids[i], keys[i]) for i in range(n)]
        reg = self._register(tuples, "OBJ")
        if reg is None:
            return None
        descs = self.agent.get_xfer_descs(
            [(0, sizes[i], dev_ids[i]) for i in range(n)],
            self.mem_type,
        )
        return StorageRegistration(descs=descs, reg=reg)

    def acquire_storage(
        self, buffers, keys: List[str], direction: str
    ) -> Optional[StorageRegistration]:
        """Open + register the storage side of one transfer.

        For the FILE backend, files are created (O_CREAT) when
        ``direction == "WRITE"``.  Returns None on failure, with any
        partially acquired resources (fds) released.  ``descs`` on the
        returned registration may be None if descriptor construction
        failed; the registration must still be released.
        """
        sizes = _buffer_sizes(buffers)
        if sizes is None:
            return None

        if self.mem_type == "FILE":
            if self.path_mode:
                return self._acquire_file_path_mode(
                    keys=keys, sizes=sizes, direction=direction
                )
            return self._acquire_file_fd_mode(
                keys=keys, sizes=sizes, direction=direction
            )
        return self._acquire_obj(keys=keys, sizes=sizes)

    def release_storage(self, registration: StorageRegistration) -> None:
        """Deregister and close any fds. Exception-safe; call exactly once,
        after the last use of ``registration.descs`` (fds must outlive the
        registration, so deregister happens before the fds close)."""
        if registration.reg is not None:
            try:
                self.agent.deregister_memory(registration.reg)
            except Exception as e:
                logger.debug("deregister_memory skipped: %s", e)
        self._close_fds(registration.fds)

    @contextmanager
    def storage(self, buffers, keys: List[str], direction: str):
        """Acquire + release around one ``with`` block.

        Yields the storage xfer_descs, or None on failure.  Kept for
        callers whose registration lifetime matches the block; async
        callers should use acquire_storage/release_storage directly.
        """
        registration = self.acquire_storage(buffers, keys, direction)
        try:
            yield registration.descs if registration is not None else None
        finally:
            if registration is not None:
                self.release_storage(registration)
