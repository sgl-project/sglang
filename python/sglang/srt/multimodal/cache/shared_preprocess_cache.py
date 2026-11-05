"""Bounded, collision-checked CPU artifacts shared by local worker processes.

The directory should be on a private tmpfs mount. SQLite holds only the index;
tensor files are mapped copy-on-write so readers share pages without being able
to modify another request's input. No Python objects are unpickled from disk.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Awaitable, Callable, Sequence, TypeVar, get_args, get_type_hints

import numpy as np
import torch

T = TypeVar("T")
_TENSOR_FILE = re.compile(r"tensor-[0-9]+\.npy\Z")
_OBJECT_DIRECTORY = re.compile(r"(?:\.writing|object)-([0-9]{3})-[a-zA-Z0-9_-]+\Z")
_LOCK_SHARDS = 256
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SharedCacheRequest:
    key: str
    source: bytes
    configuration: bytes


def _type_name(value_type: type) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _lock_shard(key: str) -> int:
    return (
        int.from_bytes(hashlib.sha256(key.encode()).digest()[:2], "big") % _LOCK_SHARDS
    )


class _ArtifactCodec:
    def __init__(self, artifact_type: type):
        self.artifact_type = artifact_type
        self.types: dict[str, type] = {}
        self._register(artifact_type)

    def _register(self, value_type):
        if isinstance(value_type, type) and dataclasses.is_dataclass(value_type):
            name = _type_name(value_type)
            if name in self.types:
                return
            self.types[name] = value_type
            for child in get_type_hints(value_type).values():
                self._register(child)
        else:
            for child in get_args(value_type):
                self._register(child)

    def write(self, directory: Path, value: T) -> None:
        tensor_count = 0

        def encode(item):
            nonlocal tensor_count
            if item is None or type(item) in (str, bool, int, float):
                return item
            if isinstance(item, torch.Tensor):
                if item.device.type != "cpu" or item.layout != torch.strided:
                    raise TypeError(
                        "Shared preprocess artifacts require strided CPU tensors"
                    )
                filename = f"tensor-{tensor_count}.npy"
                tensor_count += 1
                # Serialize only this tensor's logical values, not the storage
                # of the entire batch from which an image was sliced.
                data = item.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
                np.save(directory / filename, data, allow_pickle=False)
                return {
                    "kind": "tensor",
                    "file": filename,
                    "dtype": str(item.dtype).removeprefix("torch."),
                    "shape": list(item.shape),
                }
            if isinstance(item, np.ndarray):
                if item.dtype.hasobject:
                    raise TypeError("Object arrays cannot be shared artifacts")
                filename = f"tensor-{tensor_count}.npy"
                tensor_count += 1
                np.save(directory / filename, item, allow_pickle=False)
                return {"kind": "ndarray", "file": filename}
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                name = _type_name(type(item))
                if name not in self.types:
                    raise TypeError(f"Undeclared artifact value type: {name}")
                return {
                    "kind": "dataclass",
                    "type": name,
                    "fields": {
                        field.name: encode(getattr(item, field.name))
                        for field in dataclasses.fields(item)
                    },
                }
            if isinstance(item, dict):
                return {
                    "kind": "dict",
                    "items": [
                        [encode(key), encode(value)] for key, value in item.items()
                    ],
                }
            if isinstance(item, (tuple, list)):
                return {
                    "kind": "tuple" if isinstance(item, tuple) else "list",
                    "items": [encode(child) for child in item],
                }
            if isinstance(item, bytes):
                return {"kind": "bytes", "hex": item.hex()}
            raise TypeError(f"Unsupported shared artifact value: {type(item).__name__}")

        payload = encode(value)
        (directory / "artifact.json").write_text(
            json.dumps(payload, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )

    def read(self, directory: Path) -> T:
        def load_array(filename):
            if not isinstance(filename, str) or not _TENSOR_FILE.fullmatch(filename):
                raise ValueError("Invalid artifact tensor filename")
            return np.load(directory / filename, mmap_mode="c", allow_pickle=False)

        def decode(item):
            if not isinstance(item, dict):
                return item
            kind = item["kind"]
            if kind == "tensor":
                dtype = getattr(torch, item["dtype"], None)
                if not isinstance(dtype, torch.dtype):
                    raise ValueError("Invalid artifact tensor dtype")
                data = load_array(item["file"])
                if data.dtype != np.uint8 or data.ndim != 1:
                    raise ValueError("Invalid artifact tensor storage")
                return torch.from_numpy(data).view(dtype).reshape(item["shape"])
            if kind == "ndarray":
                return load_array(item["file"])
            if kind == "dataclass":
                value_type = self.types.get(item["type"])
                if value_type is None:
                    raise ValueError("Artifact contains an undeclared value type")
                return value_type(
                    **{key: decode(value) for key, value in item["fields"].items()}
                )
            if kind == "dict":
                return {decode(key): decode(value) for key, value in item["items"]}
            if kind in ("tuple", "list"):
                values = [decode(child) for child in item["items"]]
                return tuple(values) if kind == "tuple" else values
            if kind == "bytes":
                return bytes.fromhex(item["hex"])
            raise ValueError("Invalid artifact value kind")

        result = decode(json.loads((directory / "artifact.json").read_text("utf-8")))
        if not isinstance(result, self.artifact_type):
            raise ValueError("Unexpected artifact type")
        return result


class SharedPreprocessCache:
    """An opt-in, host-local artifact store with one global LRU budget.

    Every participant in a directory must use the same budget. Retained files
    are bounded; live request mappings and in-progress preprocessing are not
    part of that retention budget.
    """

    def __init__(self, directory: str, max_size_bytes: int, max_entries: int = 8192):
        if os.name != "posix":
            raise ValueError("Shared preprocessing cache requires POSIX file locks")
        if max_size_bytes < 0 or max_entries <= 0:
            raise ValueError("Cache size must be non-negative and entry limit positive")
        self.directory = Path(directory)
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self._tasks: set[asyncio.Task] = set()
        if not self.enabled:
            return
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.directory / "locks").mkdir(mode=0o700, exist_ok=True)
        with self._index() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS settings "
                "(id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL, "
                "max_bytes INTEGER NOT NULL, max_entries INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries "
                "(id TEXT PRIMARY KEY, key TEXT NOT NULL, codec TEXT NOT NULL, "
                "source_size INTEGER NOT NULL, config_size INTEGER NOT NULL, "
                "size_bytes INTEGER NOT NULL, touched INTEGER NOT NULL)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS entry_keys ON entries(key)")
            connection.execute(
                "INSERT OR IGNORE INTO settings VALUES (1, 0, ?, ?)",
                (max_size_bytes, max_entries),
            )
            settings = connection.execute(
                "SELECT * FROM settings WHERE id=1"
            ).fetchone()
            if (settings["max_bytes"], settings["max_entries"]) != (
                max_size_bytes,
                max_entries,
            ):
                raise ValueError(
                    "Workers sharing a cache directory must use the same budget"
                )
        self._remove_abandoned_files()

    @property
    def enabled(self) -> bool:
        return self.max_size_bytes > 0

    @contextmanager
    def _file_lock(self, path: Path, *, blocking: bool = True):
        import fcntl

        with path.open("a+b") as stream:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(stream.fileno(), flags)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _index(self):
        with self._file_lock(self.directory / "index.lock"):
            connection = sqlite3.connect(self.directory / "index.sqlite3", timeout=30)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def cache_info(self) -> dict[str, int]:
        if not self.enabled:
            return {"entries": 0, "size_bytes": 0, "generation": 0}
        with self._index() as connection:
            count, size = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM entries"
            ).fetchone()
            generation = connection.execute(
                "SELECT generation FROM settings WHERE id=1"
            ).fetchone()[0]
            return {"entries": count, "size_bytes": size, "generation": generation}

    def _remove_abandoned_files(self) -> None:
        for directory in self.directory.iterdir():
            match = _OBJECT_DIRECTORY.fullmatch(directory.name)
            if match is None or not directory.is_dir():
                continue
            shard = int(match[1])
            if shard >= _LOCK_SHARDS:
                continue
            try:
                # Take locks in the same order as publishers. A live writer
                # owns its shard; never remove its partially written files.
                with self._file_lock(
                    self.directory / "locks" / f"{shard:03d}.lock", blocking=False
                ):
                    with self._index() as connection:
                        exists = connection.execute(
                            "SELECT 1 FROM entries WHERE id=?", (directory.name,)
                        ).fetchone()
                        if exists is None:
                            shutil.rmtree(directory, ignore_errors=True)
            except BlockingIOError:
                continue

    def clear(self) -> None:
        if not self.enabled:
            return
        with self._index() as connection:
            entries = connection.execute("SELECT id FROM entries").fetchall()
            connection.execute("DELETE FROM entries")
            connection.execute("UPDATE settings SET generation=generation+1 WHERE id=1")
        for entry in entries:
            shutil.rmtree(self.directory / entry["id"], ignore_errors=True)

    def _lookup(self, request: SharedCacheRequest, codec: _ArtifactCodec):
        broken = []
        value = None
        with self._index() as connection:
            entries = connection.execute(
                "SELECT * FROM entries WHERE key=? AND codec=? "
                "AND source_size=? AND config_size=?",
                (
                    request.key,
                    _type_name(codec.artifact_type),
                    len(request.source),
                    len(request.configuration),
                ),
            ).fetchall()
            for entry in entries:
                directory = self.directory / entry["id"]
                try:
                    # A digest is an index, never the final identity check.
                    if (directory / "source.bin").read_bytes() != request.source:
                        continue
                    if (
                        directory / "configuration.bin"
                    ).read_bytes() != request.configuration:
                        continue
                    value = codec.read(directory)
                except (
                    OSError,
                    EOFError,
                    ValueError,
                    TypeError,
                    KeyError,
                    RuntimeError,
                ):
                    connection.execute("DELETE FROM entries WHERE id=?", (entry["id"],))
                    broken.append(directory)
                    continue
                connection.execute(
                    "UPDATE entries SET touched=? WHERE id=?",
                    (time.time_ns(), entry["id"]),
                )
                break
        for directory in broken:
            shutil.rmtree(directory, ignore_errors=True)
        return value

    def _publish(self, request, value, codec, generation):
        shard = _lock_shard(request.key)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".writing-{shard:03d}-", dir=self.directory)
        )
        published = False
        victims = []
        try:
            (temporary / "source.bin").write_bytes(request.source)
            (temporary / "configuration.bin").write_bytes(request.configuration)
            codec.write(temporary, value)
            size = sum(path.stat().st_size for path in temporary.iterdir())
            if size > self.max_size_bytes:
                return
            identifier = f"object-{shard:03d}-" + uuid.uuid4().hex
            destination = self.directory / identifier
            with self._index() as connection:
                current_generation = connection.execute(
                    "SELECT generation FROM settings WHERE id=1"
                ).fetchone()[0]
                if generation != current_generation:
                    return
                entries = connection.execute(
                    "SELECT id, size_bytes FROM entries ORDER BY touched, id"
                ).fetchall()
                total = sum(entry["size_bytes"] for entry in entries)
                count = len(entries)
                for entry in entries:
                    if total + size <= self.max_size_bytes and count < self.max_entries:
                        break
                    connection.execute("DELETE FROM entries WHERE id=?", (entry["id"],))
                    total -= entry["size_bytes"]
                    count -= 1
                    victims.append(self.directory / entry["id"])
                os.replace(temporary, destination)
                connection.execute(
                    "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        identifier,
                        request.key,
                        _type_name(codec.artifact_type),
                        len(request.source),
                        len(request.configuration),
                        size,
                        time.time_ns(),
                    ),
                )
            published = True
        finally:
            if not published:
                shutil.rmtree(temporary, ignore_errors=True)
                if "destination" in locals():
                    shutil.rmtree(destination, ignore_errors=True)
            if published:
                for victim in victims:
                    shutil.rmtree(victim, ignore_errors=True)

    @contextmanager
    def _shard_locks(self, requests):
        from contextlib import ExitStack

        shards = sorted({_lock_shard(request.key) for request in requests})
        with ExitStack() as stack:
            for shard in shards:
                stack.enter_context(
                    self._file_lock(
                        self.directory / "locks" / f"{shard:03d}.lock", blocking=False
                    )
                )
            yield

    async def resolve_many(
        self,
        requests: Sequence[SharedCacheRequest],
        compute: Callable[[list[int]], Awaitable[list[T]]],
        *,
        artifact_type: type[T],
        generation: int | None = None,
    ) -> list[T]:
        if not requests:
            return []
        if not self.enabled:
            return await compute(list(range(len(requests))))
        if generation is None:
            generation = self.cache_info()["generation"]
        task = asyncio.create_task(
            self._resolve_many(requests, compute, artifact_type, generation)
        )
        self._tasks.add(task)

        def completed(done):
            self._tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed)
        return await asyncio.shield(task)

    async def _resolve_many(self, requests, compute, artifact_type, generation):
        codec = _ArtifactCodec(artifact_type)
        first_indices = {}
        for index, request in enumerate(requests):
            first_indices.setdefault(request, index)
        resolved = {}
        for request in first_indices:
            value = await asyncio.to_thread(self._lookup, request, codec)
            if value is not None:
                resolved[request] = value
        missing = [request for request in first_indices if request not in resolved]
        if not missing:
            return [resolved[request] for request in requests]

        # Do not fill the default executor with blocking flock waiters: the
        # owner also needs that executor to read and publish its result.
        while True:
            locks = self._shard_locks(missing)
            try:
                locks.__enter__()
                break
            except BlockingIOError:
                await asyncio.sleep(0.01)
        try:
            missing_after_lock = []
            for request in missing:
                value = await asyncio.to_thread(self._lookup, request, codec)
                if value is not None:
                    resolved[request] = value
                else:
                    missing_after_lock.append(request)
            if missing_after_lock:
                values = await compute(
                    [first_indices[request] for request in missing_after_lock]
                )
                if len(values) != len(missing_after_lock):
                    raise ValueError(
                        "Preprocessing must return one artifact per cache miss"
                    )
                for request, value in zip(missing_after_lock, values):
                    if not isinstance(value, artifact_type):
                        raise TypeError(
                            "Preprocessor returned an unexpected artifact type"
                        )
                    try:
                        await asyncio.to_thread(
                            self._publish, request, value, codec, generation
                        )
                    except (OSError, sqlite3.Error, TypeError, ValueError) as error:
                        logger.warning(
                            "Could not retain shared preprocessing artifact: %s", error
                        )
                    resolved[request] = value
            return [resolved[request] for request in requests]
        finally:
            locks.__exit__(None, None, None)
