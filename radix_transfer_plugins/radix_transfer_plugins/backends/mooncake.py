"""Mooncake store backend: one hard-pinned key per part, ``{object key}/{part name}``.

The client only uses store memory other processes provide (``global_segment_size=0``);
the deployment runs the master and the processes holding segments. Behaviors of
Mooncake 0.3.13 this relies on (probed in doc/mooncake-data-transfer/evidence/s0-20261017):

- ``put_batch`` returns one status for the whole call, and a failed call can leave some
  of its keys stored.
- A put copies through the client's local buffer and fails at once (-600) when the
  buffer has no room; it never waits. Reads come out of the same buffer and hold
  their part of it for as long as their views live.
- A read leaves a lease on the key, and a remove without ``force`` fails (-706) while
  the lease lasts.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Callable, Mapping, Sequence, Union

import msgspec
from mooncake.store import MooncakeDistributedStore, ReplicateConfig

from radix_transfer_plugins.backend import StorageBackend

logger = logging.getLogger(__name__)

_LOCAL_BUFFER_FULL = -600
_OBJECT_NOT_FOUND = -704

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
        handles = iter(self._store.batch_get_buffer(part_keys) if part_keys else [])
        objects = []
        for key, named in zip(keys, parts):
            got = {}
            for name, nbytes in named:
                handle = next(handles)
                if handle is None:
                    raise KeyError(
                        f"output store part not found: {_part_key(key, name)}"
                    )
                # Mooncake exports signed bytes; the view keeps the handle alive.
                data = memoryview(handle).cast("B")
                if data.nbytes != nbytes:
                    raise ValueError(
                        f"output store part {_part_key(key, name)} has {data.nbytes} bytes, "
                        f"its ref says {nbytes}"
                    )
                got[name] = data
            objects.append(got)
        return objects

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
    """Retries a put that found the local buffer full once another put frees room.

    A put that fails that way while no other put is in flight and none finished since
    it started cannot fit at all, so its status is returned as is.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._in_flight = 0
        self._finished = 0

    def run(self, *, put: Callable[[], int], undo: Callable[[], object]) -> int:
        with self._cond:
            self._in_flight += 1
        try:
            while True:
                with self._cond:
                    finished_before = self._finished
                status = put()
                if status != _LOCAL_BUFFER_FULL:
                    return status
                # A failed batch can leave some keys stored; a retry must start clean.
                undo()
                with self._cond:
                    if self._in_flight == 1 and self._finished == finished_before:
                        return status
                    self._cond.wait_for(lambda: self._finished != finished_before)
        finally:
            with self._cond:
                self._in_flight -= 1
                self._finished += 1
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


def _part_key(object_key: str, name: str) -> str:
    return f"{object_key}/{name}"
