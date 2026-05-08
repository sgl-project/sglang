"""NIXL memory-registration helpers, exposed as context managers.

A ``NixlRegistry`` instance bundles the agent, the memory type, and
(optionally) the file manager.  Its ``storage(...)`` method is a context
manager that performs the entire register-and-build-descs sequence for
the storage side of a transfer on entry, yields the ``xfer_descs`` (or
None on failure), and unwinds ``agent.deregister_memory`` plus any
``os.close(fd)`` on exit.

The host side is pre-registered up front by ``HiCacheNixl`` and is not
touched per transfer.
"""

import logging
import threading
from contextlib import contextmanager
from typing import List, Optional

from .nixl_utils import NixlFileManager

logger = logging.getLogger(__name__)


def _buffer_sizes(buffers) -> Optional[List[int]]:
    """Per-buffer byte sizes for ``(addr, len)`` tuple inputs."""
    if not buffers or not isinstance(buffers[0], tuple):
        return None
    return [b[1] for b in buffers]


class NixlRegistry:
    """Owns the (agent, mem_type, file_manager) triple and provides a
    context manager for the storage side of a transfer.

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

    @contextmanager
    def _open_files(self, paths: List[str], create: bool):
        """Open fds for ``paths``; close all of them on exit.

        Yields the list of fds, or None if any open fails (already-opened
        fds are closed before returning by the same ``finally``).
        """
        fds: List[int] = []
        try:
            for path in paths:
                fd = self.file_manager.open_file(path, create=create)
                if fd is None:
                    yield None
                    return
                fds.append(fd)
            yield fds
        finally:
            for fd in fds:
                self.file_manager.close_file(fd)

    @contextmanager
    def _registered(self, items: List[tuple], mem_type: str):
        """Register ``items`` with NIXL; deregister on exit.

        Yields the registration handle, or None if registration fails.
        """
        reg = None
        if items:
            reg_descs = self.agent.get_reg_descs(items, mem_type)
            if reg_descs is not None:
                try:
                    reg = self.agent.register_memory(reg_descs)
                except Exception as e:
                    logger.error(
                        f"Failed to register memory of type {mem_type}: {e}"
                    )
        try:
            yield reg
        finally:
            if reg is not None:
                try:
                    self.agent.deregister_memory(reg)
                except Exception as e:
                    logger.debug("deregister_memory skipped: %s", e)

    @contextmanager
    def storage(self, buffers, keys, direction):
        """Open + register the storage side; deregister and close fds on exit.

        Yields the storage xfer_descs, or None on failure.  For the FILE
        backend, files are created (O_CREAT) when ``direction == "WRITE"``.
        """
        sizes = _buffer_sizes(buffers)
        if sizes is None:
            yield None
            return

        if self.mem_type == "FILE":
            with self._open_files(keys, create=(direction == "WRITE")) as fds:
                if fds is None:
                    yield None
                    return
                tuples = [
                    (0, sizes[i], fds[i], keys[i]) for i in range(len(keys))
                ]
                with self._registered(tuples, "FILE") as reg:
                    if reg is None:
                        yield None
                        return
                    yield self.agent.get_xfer_descs(
                        [(0, sizes[i], fds[i]) for i in range(len(fds))], "FILE"
                    )
        else:  # OBJ
            # Reg tuple: (addr=0, size, devId, metaInfo=key).
            # Xfer tuple: (addr=0, size, devId). devId links each xfer desc
            # back to its registered object's metaInfo, so devIds must be
            # unique within the list AND globally unique across concurrent
            # storage() calls (the OBJ plugin's devIdToObjKey_ map is shared
            # and unlocked). NIXL's pybind layer requires position 3 to be
            # int, hence the key goes in metaInfo (position 4).
            n = len(keys)
            with self._obj_devid_lock:
                base = self._obj_devid_next
                self._obj_devid_next += n
            dev_ids = list(range(base, base + n))
            tuples = [
                (0, sizes[i], dev_ids[i], keys[i]) for i in range(n)
            ]
            with self._registered(tuples, "OBJ") as reg:
                if reg is None:
                    yield None
                    return
                yield self.agent.get_xfer_descs(
                    [(0, sizes[i], dev_ids[i]) for i in range(n)],
                    self.mem_type,
                )
