"""Mooncake store backend: one hard-pinned key per part, ``{object key}/{part name}``.

The client only uses store memory other processes provide (``global_segment_size=0``);
the deployment runs the master and the processes holding segments. Behaviors of
Mooncake 0.3.13 this relies on (probed against a live master):

- ``put_batch`` returns one status for the whole call, and a failed call can leave some
  of its keys stored.
- A put copies through the client's local buffer and fails at once (-600) when the
  buffer has no room; it never waits.
- ``get_into`` reads one part straight into memory registered with ``register_buffer``,
  without the local buffer. It returns the bytes read; -704 for a missing part; -600
  when the part is larger than the destination; or another negative code, such as -707
  for a read that outlasted its lease. It releases the GIL while it waits.
- A read leaves a lease on the key, and a remove without ``force`` fails (-706) while
  the lease lasts.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import threading
import uuid
from typing import Callable, Mapping, NoReturn, Sequence, Union

import msgspec
import numpy as np
from mooncake.store import MooncakeDistributedStore, ReplicateConfig

from radix_transfer_plugins.backend import StorageBackend

logger = logging.getLogger(__name__)

_LOCAL_BUFFER_FULL = -600
# The same code from get_into: the stored part is larger than the destination.
_BUFFER_TOO_SMALL = -600
_OBJECT_NOT_FOUND = -704
# Reads fetch one part per get_into call; this many calls in flight fill the gaps between
# calls (4 matched one batch_get_into over RDMA), and more only share the same bandwidth.
_READ_THREADS = 4

_SIZE_UNITS = {
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "k": 1024,
    "m": 1024**2,
    "g": 1024**3,
}


class MooncakeConfig(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    master_server_address: str
    local_hostname: str
    # Bytes, or a string such as "2gb"; bounds the bytes of puts in flight at once.
    # Reads do not use it, so a client that only reads can set 0.
    local_buffer_size: Union[int, str]
    key_prefix: str
    protocol: str = "rdma"
    metadata_server: str = "P2PHANDSHAKE"
    device_name: str = ""
    replica_num: int = 1


class MooncakeBackend(StorageBackend):
    name = "mooncake"
    config_type = MooncakeConfig

    def __init__(self, config: MooncakeConfig) -> None:
        self._key_prefix = config.key_prefix
        store = MooncakeDistributedStore()
        setup_error = store.setup(
            {
                "local_hostname": config.local_hostname,
                "metadata_server": config.metadata_server,
                "global_segment_size": 0,
                "local_buffer_size": parse_size(config.local_buffer_size),
                "protocol": config.protocol,
                "rdma_devices": config.device_name,
                "master_server_addr": config.master_server_address,
            }
        )
        if setup_error:
            raise RuntimeError(f"Mooncake output store setup failed: {setup_error}")
        self._store = store
        # Hard pin: Mooncake never evicts an object whose ref may still be in use.
        self._put_config = ReplicateConfig()
        self._put_config.replica_num = config.replica_num
        self._put_config.with_hard_pin = True
        self._local_buffer = _LocalBufferGate()

    def put_many(self, objects: Sequence[Mapping[str, memoryview]]) -> list[str]:
        object_keys = [f"{self._key_prefix}/{uuid.uuid4().hex}" for _ in objects]
        keys, values = [], []
        for object_key, parts in zip(object_keys, objects):
            for name, data in parts.items():
                keys.append(_part_key(object_key, name))
                values.append(data)
        if keys:
            status = self._local_buffer.run(
                put=lambda: self._store.put_batch(keys, values, self._put_config),
                undo=lambda: self._remove(keys),
            )
            if status != 0:
                leaked = self._remove(keys)
                if leaked:
                    logger.error(
                        "Keys of a failed output store put could not be removed: %s",
                        leaked[:4],
                    )
                raise RuntimeError(
                    f"Mooncake put_batch of {len(keys)} keys failed: {status}"
                )
        return object_keys

    def get_many(
        self, keys: Sequence[str], *, parts: Sequence[Sequence[tuple[str, int]]]
    ) -> list[dict[str, memoryview]]:
        part_keys = [
            _part_key(key, name) for key, named in zip(keys, parts) for name, _ in named
        ]
        sizes = [nbytes for named in parts for _, nbytes in named]
        offsets = list(itertools.accumulate(sizes, initial=0))[:-1]
        # One array for every part: one registration per read, and the caller owns it.
        memory = np.empty(sum(sizes), dtype=np.uint8)
        if part_keys:
            self._read_into(memory, keys=part_keys, offsets=offsets, sizes=sizes)
        views = (memoryview(memory[o : o + n]) for o, n in zip(offsets, sizes))
        return [{name: next(views) for name, _ in named} for named in parts]

    def _read_into(
        self,
        memory: np.ndarray,
        *,
        keys: Sequence[str],
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> None:
        base = memory.ctypes.data
        status = self._store.register_buffer(base, memory.nbytes)
        if status != 0:
            raise RuntimeError(
                f"Mooncake register_buffer of {memory.nbytes} bytes failed: {status}"
            )
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(_READ_THREADS, len(keys))
            ) as pool:
                results = list(
                    pool.map(
                        self._store.get_into, keys, [base + o for o in offsets], sizes
                    )
                )
        finally:
            status = self._store.unregister_buffer(base)
            if status != 0:
                logger.error("Mooncake unregister_buffer failed: %s", status)
        failed = [
            (key, got, nbytes)
            for key, got, nbytes in zip(keys, results, sizes)
            if got != nbytes
        ]
        if failed:
            _raise_failed_reads(failed)

    def delete_many(
        self, keys: Sequence[str], *, parts: Sequence[Sequence[str]]
    ) -> None:
        failed = self._remove(
            [_part_key(key, name) for key, names in zip(keys, parts) for name in names]
        )
        if failed:
            raise RuntimeError(
                f"Mooncake remove failed for {len(failed)} keys: {failed[:4]}"
            )

    def close(self) -> None:
        self._store.close()

    def _remove(self, keys: Sequence[str]) -> list[tuple[str, int]]:
        """Remove keys that may or may not exist; returns the ones that failed."""
        if not keys:
            return []
        # force: owners remove only after every reader is done, and a fresh read
        # would otherwise block the remove with its lease.
        statuses = self._store.batch_remove(list(keys), True)
        return [
            (key, status)
            for key, status in zip(keys, statuses)
            if status not in (0, _OBJECT_NOT_FOUND)
        ]


class _LocalBufferGate:
    """Retries a put that found the local buffer full once, with no other put running.

    A waiting put holds no buffer, so concurrent puts that all fail cannot wait on each
    other; a put that fails while running alone cannot fit, and its status is returned.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._running = 0
        self._exclusive = False

    def run(self, *, put: Callable[[], int], undo: Callable[[], object]) -> int:
        status = self._attempt(put=put, exclusive=False)
        if status != _LOCAL_BUFFER_FULL:
            return status
        # A failed batch can leave some keys stored; a retry must start clean.
        undo()
        return self._attempt(put=put, exclusive=True)

    def _attempt(self, *, put: Callable[[], int], exclusive: bool) -> int:
        with self._cond:
            self._cond.wait_for(lambda: not self._exclusive)
            if exclusive:
                self._exclusive = True
                self._cond.wait_for(lambda: self._running == 0)
            self._running += 1
        try:
            return put()
        finally:
            with self._cond:
                self._running -= 1
                if exclusive:
                    self._exclusive = False
                self._cond.notify_all()


def parse_size(value: Union[int, str]) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        for suffix, multiplier in _SIZE_UNITS.items():
            if text.endswith(suffix):
                return int(float(text[: -len(suffix)]) * multiplier)
        return int(text)
    raise ValueError(f"invalid size {value!r}; use bytes or a string such as '2gb'")


def _raise_failed_reads(failed: Sequence[tuple[str, int, int]]) -> NoReturn:
    """``failed`` holds (key, get_into result, the size the ref says) of each failed read."""
    for key, got, _ in failed:
        if got == _OBJECT_NOT_FOUND:
            raise KeyError(f"output store part not found: {key}")
    for key, got, nbytes in failed:
        if got >= 0 or got == _BUFFER_TOO_SMALL:
            stored = got if got >= 0 else f"more than {nbytes}"
            raise ValueError(
                f"output store part {key} has {stored} bytes, its ref says {nbytes}"
            )
    key, got, _ = failed[0]
    raise RuntimeError(
        f"Mooncake could not read {len(failed)} of the stored parts, such as {key}: "
        f"{got} (see the Mooncake log)"
    )


def _part_key(object_key: str, name: str) -> str:
    return f"{object_key}/{name}"
