# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verified engine-host encoded cache and rank-owned direct-DE host arenas.

Encoded publication files are shared through tmpfs. Every scheduler
decodes its local tensors into its own original CUDA HOST_NUMA allocation.
"""

import fcntl
import hashlib
import json
import mmap
import os
import re
import stat
import time
import uuid
from pathlib import Path

import orjson

from sglang.srt.weight_sync.gpu_delta_memory import HostAllocation
from sglang.srt.weight_sync.gpu_delta_payload import validate_outer_entries


def _cache_base():
    root = Path(
        os.environ.get(
            "GPU_DELTA_HOST_CACHE_DIR", f"/dev/shm/sglang-gpu-delta-{os.getuid()}"
        )
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    info = root.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("GPU-delta host cache must be private to the current user")
    mounts = []
    with open("/proc/self/mountinfo") as source:
        for line in source:
            left, right = line.rstrip().split(" - ", 1)
            mount = left.split()[4]
            for escaped, value in ((r"\040", " "), (r"\011", "\t"), (r"\134", "\\")):
                mount = mount.replace(escaped, value)
            mount = Path(mount)
            if root.is_relative_to(mount):
                mounts.append((len(mount.parts), right.split()[0]))
    if not mounts or max(mounts)[1] != "tmpfs":
        raise ValueError("GPU_DELTA_HOST_CACHE_DIR must be on host-shared tmpfs")
    return root


def _cache_root(engine_id):
    if not isinstance(engine_id, str) or not engine_id:
        raise ValueError("GPU-delta host cache requires an engine identity")
    root = _cache_base() / hashlib.sha256(engine_id.encode()).hexdigest()
    root.mkdir(mode=0o700, exist_ok=True)
    return root


def host_cache_id(engine_id):
    """An engine-scoped tmpfs root, not hostname, defines the sharing domain."""
    root = _cache_root(engine_id)
    with (root / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / ".host-id"
        if not path.exists():
            path.write_text(uuid.uuid4().hex)
            path.chmod(0o400)
        value = path.read_text()
        if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid GPU-delta host cache identity")
        return value


def _identity(info):
    # RW mapping metadata timestamps can change on attachment. The feature owns
    # immutable contents after READY; bind the exact retained inode and extent.
    return [info.st_dev, info.st_ino, info.st_size]


# Cold capacity fits the required extent; growth reserves twice the new extent.
# This is not a publication-format parameter or a tuning surface.
_CAPACITY_ALIGNMENT = 64 << 20


def _reserve(directory, previous, size, metrics):
    if previous is not None and size <= previous["capacity"]:
        return previous
    capacity = size if previous is None else 2 * size
    capacity = (
        (capacity + _CAPACITY_ALIGNMENT - 1)
        // _CAPACITY_ALIGNMENT
        * _CAPACITY_ALIGNMENT
    )
    generation = previous["generation"] + 1 if previous else 1
    path = directory / f"encoded-{generation}.bin"
    started = time.perf_counter()
    with path.open("xb+") as target:
        if capacity:
            # Reserve pages once per capacity generation; fitting updates never
            # truncate or resize the retained encoded-cache inode.
            os.posix_fallocate(target.fileno(), 0, capacity)
        identity = _identity(os.fstat(target.fileno()))
    metrics["host_encoded_cache_allocation_s"] += time.perf_counter() - started
    metrics["host_encoded_cache_allocation_calls"] += 1
    metrics["host_encoded_cache_allocation_bytes"] += capacity
    return {
        "file": path.name,
        "generation": generation,
        "capacity": capacity,
        "identity": identity,
    }


def _write_record(directory, name, record):
    temporary = directory / (name + ".pending")
    temporary.write_bytes(orjson.dumps(record))
    temporary.replace(directory / (name + ".json"))


def _read_payload(source, destination, expected):
    started = time.perf_counter()
    with source.open("rb", buffering=0) as incoming:
        before = os.fstat(incoming.fileno())
        if before.st_size != expected["nbytes"]:
            raise ValueError("delta payload size mismatch")
        position = 0
        while position < len(destination):
            count = incoming.readinto(destination[position:])
            if not count:
                raise ValueError("truncated delta payload")
            position += count
        after = os.fstat(incoming.fileno())
        if incoming.read(1) or (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("delta payload size mismatch or source changed")
    return time.perf_counter() - started


def _hash_payloads(files, definitions):
    """Hash the retained immutable copy, never reread publication files."""
    started = time.perf_counter()
    for name, payload in files.items():
        if hashlib.sha256(payload).hexdigest() != definitions[name]["sha256"]:
            raise ValueError("delta payload SHA256 mismatch")
    return time.perf_counter() - started


def _natural_key(name):
    return tuple(
        int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", name)
    )


def _tensor_layout(entries):
    layout, size = {}, 0
    for entry in entries:
        count = (
            (entry["nbytes"] if entry["changed_bytes"] else 0)
            if entry["encoding"] == "raw_bytes"
            else (entry["outer"]["decoded_bytes"] if entry["frames"] else 0)
        )
        if count:
            size = (size + 15) // 16 * 16
            layout[entry["name"]] = {"offset": size, "nbytes": count}
            size += count
    return layout, size


_DECODE_METRICS = (
    "host_rank_outer_zstd_validate_s",
    "host_rank_outer_zstd_worker_decode_sum_s",
    "host_rank_outer_zstd_encoded_bytes",
    "host_rank_outer_zstd_decoded_bytes",
    "host_rank_outer_zstd_tensors",
    "host_rank_outer_zstd_frames",
)


def _decode_group(jobs, destination, files, pool):
    metrics = {name: 0 for name in _DECODE_METRICS}
    for entry, record in jobs:
        outer = entry["outer"]
        offset, length = outer["encoded_offset"], outer["encoded_bytes"]
        start, count = record["offset"], record["nbytes"]
        validate_s, decode_s = pool.decode(
            files[outer["file"]][offset : offset + length],
            outer["frames"],
            destination[start : start + count],
        )
        metrics["host_rank_outer_zstd_validate_s"] += validate_s
        metrics["host_rank_outer_zstd_worker_decode_sum_s"] += decode_s
        metrics["host_rank_outer_zstd_encoded_bytes"] += outer["encoded_bytes"]
        metrics["host_rank_outer_zstd_decoded_bytes"] += outer["decoded_bytes"]
        metrics["host_rank_outer_zstd_tensors"] += 1
        metrics["host_rank_outer_zstd_frames"] += len(outer["frames"])
    return metrics


def _decode_arena(destination, layout, files, entries, pool, metrics):
    started = time.perf_counter()
    jobs, futures, error = [], [], None
    try:
        for entry in entries:
            record = layout.get(entry["name"])
            if record is None:
                continue
            if entry["encoding"] == "raw_bytes":
                raw = entry["raw"]
                offset, count = record["offset"], record["nbytes"]
                start = raw["encoded_offset"]
                destination[offset : offset + count] = files[raw["file"]][
                    start : start + count
                ]
            else:
                jobs.append((entry, record))
        count = min(4 * pool.workers, len(jobs))
        for index in range(count):
            futures.append(
                pool.executor.submit(
                    _decode_group, jobs[index::count], destination, files, pool
                )
            )
    except BaseException as exc:  # noqa: BLE001 - drain submitted jobs before re-raise
        error = exc
    for future in futures:
        try:
            partial = future.result()
            for key in _DECODE_METRICS:
                metrics[key] += partial[key]
        except BaseException as exc:  # noqa: BLE001 - drain peers before re-raise
            if error is None:
                error = exc
    metrics["host_rank_outer_zstd_decode_s"] = time.perf_counter() - started
    if error is not None:
        raise error


def _map_files(directory, encoded, definitions):
    with (directory / encoded["file"]).open("r+b") as source:
        if _identity(os.fstat(source.fileno())) != encoded["identity"]:
            raise ValueError("encoded cache inode or extent changed")
        mapping = mmap.mmap(source.fileno(), 0) if encoded["capacity"] else None
    files, position = {}, 0
    for name, record in definitions.items():
        end = position + record["nbytes"]
        files[name] = (
            memoryview(mapping)[position:end]
            if mapping is not None
            else memoryview(b"")
        )
        position = end
    # Each view owns its mmap. Failed decode tracebacks may retain drained views;
    # their mapping must remain valid until those last references are released.
    return files


class HostArena:
    """Persistent rank-owned DE storage with an engine-host encoded cache.

    The original engine cohort shares only verified encoded files. Successful
    all-rank apply/resume permits the next publication to overwrite that cache;
    aborted/failed preparation never grants release.
    """

    def __init__(self, engine_id, device):
        self.engine_id, self.device = engine_id, device
        self.allocation = self.mapping = self.tensor = None
        self.capacity = None
        self.directory = None
        self.tensor_order = None

    def _reserve_rank(self, size, metrics):
        import torch

        if self.capacity is not None and size <= self.capacity["capacity"]:
            metrics["host_rank_mapping_reused"] = 1
            return
        previous = self.capacity
        capacity = size if previous is None else 2 * size
        capacity = (
            (capacity + _CAPACITY_ALIGNMENT - 1)
            // _CAPACITY_ALIGNMENT
            * _CAPACITY_ALIGNMENT
        )
        started = time.perf_counter()
        allocation = HostAllocation(capacity, self.device) if capacity else None
        self.close()
        self.allocation = allocation
        self.mapping = allocation.view if allocation is not None else None
        self.tensor = (
            torch.frombuffer(self.mapping, dtype=torch.uint8)
            if self.mapping is not None
            else torch.empty(0, dtype=torch.uint8)
        )
        generation = previous["generation"] + 1 if previous else 1
        self.capacity = {
            "generation": generation,
            "capacity": allocation.capacity if allocation is not None else 0,
            "identity": uuid.uuid4().hex,
        }
        metrics["host_rank_allocation_s"] = time.perf_counter() - started
        metrics["host_rank_allocation_calls"] = int(allocation is not None)
        metrics["host_rank_allocation_bytes"] = self.capacity["capacity"]

    def prepare(
        self, manifest_path, manifest_sha256, manifest, names, pool, timings, metadata
    ):
        root = _cache_root(self.engine_id)
        publication = Path(manifest_path).resolve(strict=True)
        namespace = {
            "stream_id": metadata["stream_id"],
            "participants": sorted(
                metadata["participants"],
                key=lambda value: json.dumps(value, sort_keys=True),
            ),
        }
        key = hashlib.sha256(json.dumps(namespace, sort_keys=True).encode()).hexdigest()
        directory = root / key
        if self.directory is not None and self.directory != directory:
            raise ValueError("host cache original cohort or stream changed")
        definitions = {}
        for record in manifest["files"]:
            name, size = record["name"], record["nbytes"]
            if (
                name in definitions
                or Path(name).name != name
                or name in {".", ".."}
                or type(size) is not int
                or size < 0
            ):
                raise ValueError("invalid or duplicate delta payload path/size")
            definitions[name] = {"nbytes": size, "sha256": record["sha256"]}
        expected = {
            "manifest_path": str(publication),
            "manifest_sha256": manifest_sha256,
            "session_id": metadata["session_id"],
            "base_version": metadata["base_version"],
            "target_version": metadata["target_version"],
            "files": definitions,
        }
        token = hashlib.sha256(
            json.dumps(expected, sort_keys=True).encode()
        ).hexdigest()
        metrics = {
            name: 0
            for name in (
                "host_encoded_cache_read_s",
                "host_encoded_cache_sha256_s",
                "host_encoded_cache_hash_bytes",
                "host_encoded_cache_hash_files",
                "host_encoded_cache_created",
                "host_encoded_cache_reused",
                "host_encoded_cache_frames_validate_s",
                "host_encoded_cache_frames_validations",
                "host_encoded_cache_allocation_s",
                "host_encoded_cache_allocation_calls",
                "host_encoded_cache_allocation_bytes",
                "host_rank_allocation_s",
                "host_rank_allocation_calls",
                "host_rank_allocation_bytes",
                "host_rank_mapping_reused",
                *_DECODE_METRICS,
            )
        }
        waiting = time.perf_counter()
        directory.mkdir(mode=0o700, exist_ok=True)
        # Only encoded-cache construction is serialized. Rank-local Zstd and
        # CUDA host allocation occur afterward, without holding this mutex.
        with (directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            metrics["host_encoded_cache_wait_s"] = time.perf_counter() - waiting
            index_path = directory / "index.json"
            previous = (
                orjson.loads(index_path.read_bytes()) if index_path.exists() else None
            )
            state = (
                orjson.loads((directory / "state.json").read_bytes())
                if previous
                else None
            )
            if previous and previous["namespace"] != namespace:
                raise ValueError("encoded cache namespace differs")
            if previous and previous["publication"] == expected:
                if state != {
                    "token": token,
                    "state": "READY",
                    "generation": previous["encoded"]["generation"],
                }:
                    raise ValueError(
                        "encoded publication is failed or already released"
                    )
                index = previous
                files = _map_files(directory, index["encoded"], definitions)
                metrics["host_encoded_cache_reused"] = 1
            else:
                if previous and (
                    state["state"] != "REUSABLE"
                    or state["token"] != previous["token"]
                    or previous["publication"]["target_version"]
                    != expected["base_version"]
                ):
                    raise ValueError(
                        "encoded cache requires prior engine APPLIED release before overwrite"
                    )
                build_started = time.perf_counter()
                frames_started = time.perf_counter()
                # Validate the entire immutable publication once, including
                # foreign tensors whose compressed contents this rank skips.
                validate_outer_entries(
                    manifest["tensors"],
                    {name: record["nbytes"] for name, record in definitions.items()},
                    manifest["frame_bytes"],
                )
                metrics["host_encoded_cache_frames_validate_s"] = (
                    time.perf_counter() - frames_started
                )
                metrics["host_encoded_cache_frames_validations"] = 1
                encoded_size = sum(record["nbytes"] for record in definitions.values())
                encoded = _reserve(
                    directory,
                    previous["encoded"] if previous else None,
                    encoded_size,
                    metrics,
                )
                index = {
                    "namespace": namespace,
                    "publication": expected,
                    "token": token,
                    "encoded": encoded,
                }
                # A failed read/hash leaves BUILDING and cannot authorize reuse.
                _write_record(directory, "index", index)
                _write_record(
                    directory,
                    "state",
                    {
                        "token": token,
                        "state": "BUILDING",
                        "generation": encoded["generation"],
                    },
                )
                files = _map_files(directory, encoded, definitions)
                for name, record in definitions.items():
                    source = (publication.parent / name).resolve(strict=True)
                    if source.parent != publication.parent:
                        raise ValueError("delta payload escapes immutable publication")
                    metrics["host_encoded_cache_read_s"] += _read_payload(
                        source, files[name], record
                    )
                metrics["host_encoded_cache_sha256_s"] = _hash_payloads(
                    files, definitions
                )
                metrics["host_encoded_cache_hash_bytes"] = encoded_size
                metrics["host_encoded_cache_hash_files"] = len(definitions)
                index["build_s"] = time.perf_counter() - build_started
                _write_record(directory, "index", index)
                _write_record(
                    directory,
                    "state",
                    {
                        "token": token,
                        "state": "READY",
                        "generation": encoded["generation"],
                    },
                )
                if previous and previous["encoded"]["file"] != encoded["file"]:
                    (directory / previous["encoded"]["file"]).unlink()
                metrics["host_encoded_cache_created"] = 1
        self.directory = directory
        entries_by_name = {entry["name"]: entry for entry in manifest["tensors"]}
        if self.tensor_order is None:
            self.tensor_order = sorted(
                names,
                key=lambda name: (
                    entries_by_name[name]["encoding"] == "raw_bytes",
                    _natural_key(name),
                ),
            )
        entries = [entries_by_name[name] for name in self.tensor_order]
        layout, size = _tensor_layout(entries)
        self._reserve_rank(size, metrics)
        _decode_arena(
            self.mapping if self.mapping is not None else memoryview(b""),
            layout,
            files,
            entries,
            pool,
            metrics,
        )
        metrics.update(
            host_rank_arena_bytes=size,
            host_rank_capacity_bytes=self.capacity["capacity"],
            host_rank_capacity_generation=self.capacity["generation"],
            host_rank_cpu_workers=pool.workers,
            host_encoded_cache_capacity_bytes=index["encoded"]["capacity"],
            host_encoded_cache_capacity_generation=index["encoded"]["generation"],
            host_encoded_cache_build_s=index["build_s"],
        )
        timings.update(metrics)
        return HostDecodedSnapshot(
            self,
            index
            | {"tensors": layout, "arena_bytes": size, "rank_arena": self.capacity},
        )

    def close(self):
        self.tensor = self.mapping = None
        if self.allocation is not None:
            self.allocation.close()
            self.allocation = None
        self.capacity = None


class HostDecodedSnapshot:
    """One publication's local views, retaining backend-owned DE host storage."""

    def __init__(self, arena, index):
        self.arena, self.index = arena, index
        self.directory = arena.directory

    def get(self, name):
        record = self.index["tensors"][name]
        return self.arena.tensor[record["offset"] : record["offset"] + record["nbytes"]]

    def mark_reusable(self):
        # Miles sends resume only after every original engine rank applied. A
        # stale callback cannot release a newer encoded-cache publication.
        with (self.directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = orjson.loads((self.directory / "state.json").read_bytes())
            if state == {
                "token": self.index["token"],
                "state": "READY",
                "generation": self.index["encoded"]["generation"],
            }:
                state["state"] = "REUSABLE"
                _write_record(self.directory, "state", state)

    def close(self):
        # Backend retains this rank's host pages after all DE readers drain.
        self.arena = None
