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

"""Engine-local verified publication bytes in shared DE-compatible host memory.

Ranks of one engine share a CUDA HOST_NUMA arena. Tmpfs holds encoded-file
staging and publication records. Separate engines have separate lifetimes.
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

from sglang.srt.weight_sync.gpu_delta_memory import SharedHostAllocation
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


def _reserve(directory, prefix, previous, size, metrics):
    if previous is not None and size <= previous["capacity"]:
        return previous
    capacity = size if previous is None else 2 * size
    capacity = (
        (capacity + _CAPACITY_ALIGNMENT - 1)
        // _CAPACITY_ALIGNMENT
        * _CAPACITY_ALIGNMENT
    )
    generation = previous["generation"] + 1 if previous else 1
    path = directory / f"{prefix}-{generation}.bin"
    started = time.perf_counter()
    with path.open("xb+") as target:
        if capacity:
            # Reserve pages once per capacity generation. Reusing a registered
            # mmap never truncates/resizes its inode or frees its backing pages.
            os.posix_fallocate(target.fileno(), 0, capacity)
        identity = _identity(os.fstat(target.fileno()))
    metrics[f"host_{prefix}_allocation_s"] += time.perf_counter() - started
    metrics[f"host_{prefix}_allocation_calls"] += 1
    metrics[f"host_{prefix}_allocation_bytes"] += capacity
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
    "host_outer_zstd_validate_s",
    "host_outer_zstd_worker_decode_sum_s",
    "host_outer_zstd_encoded_bytes",
    "host_outer_zstd_decoded_bytes",
    "host_outer_zstd_tensors",
    "host_outer_zstd_frames",
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
        metrics["host_outer_zstd_validate_s"] += validate_s
        metrics["host_outer_zstd_worker_decode_sum_s"] += decode_s
        metrics["host_outer_zstd_encoded_bytes"] += outer["encoded_bytes"]
        metrics["host_outer_zstd_decoded_bytes"] += outer["decoded_bytes"]
        metrics["host_outer_zstd_tensors"] += 1
        metrics["host_outer_zstd_frames"] += len(outer["frames"])
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
    metrics["host_outer_zstd_decode_s"] = time.perf_counter() - started
    if error is not None:
        raise error


class HostArena:
    """Backend-owned shared host mapping, retained across updates.

    A namespace binds the original engine ranks, delta stream and host tensor union.
    Miles sends resume after all engine ranks apply; it releases the generation.
    Abort/failure retains its bytes and cannot recycle the slot automatically.
    """

    def __init__(self, engine_id, device):
        self.engine_id, self.device = engine_id, device
        self.allocation = None
        self.mapping = self.tensor = None
        self.identity = None
        self.directory = None
        self.tensor_order = None

    def _attach(self, shared, allocation=None):
        import torch

        if self.identity == shared["identity"]:
            return True
        if allocation is None and shared["capacity"]:
            allocation = SharedHostAllocation(
                shared["capacity"], self.device, shared["handle"]
            )
        self.close()
        self.allocation = allocation
        self.mapping = allocation.view if allocation is not None else None
        self.tensor = (
            torch.frombuffer(self.mapping, dtype=torch.uint8)
            if self.mapping is not None
            else torch.empty(0, dtype=torch.uint8)
        )
        self.identity = shared["identity"]
        return False

    def _reserve_shared(self, previous, size, metrics):
        if previous is not None and size <= previous["capacity"]:
            self._attach(previous)
            return previous
        capacity = size if previous is None else 2 * size
        capacity = (
            (capacity + _CAPACITY_ALIGNMENT - 1)
            // _CAPACITY_ALIGNMENT
            * _CAPACITY_ALIGNMENT
        )
        started = time.perf_counter()
        allocation = SharedHostAllocation(capacity, self.device) if capacity else None
        capacity = allocation.capacity if allocation is not None else 0
        generation = previous["generation"] + 1 if previous else 1
        shared = {
            "generation": generation,
            "capacity": capacity,
            "identity": [uuid.uuid4().hex, generation, capacity],
            "handle": allocation.shareable if allocation is not None else None,
        }
        self._attach(shared, allocation)
        metrics["host_shared_allocation_s"] += time.perf_counter() - started
        metrics["host_shared_allocation_calls"] += 1
        metrics["host_shared_allocation_bytes"] += capacity
        return shared

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
            "host_tensor_names": names,
        }
        key = hashlib.sha256(json.dumps(namespace, sort_keys=True).encode()).hexdigest()
        directory = root / key
        if self.directory is not None and self.directory != directory:
            raise ValueError(
                "host arena original cohort, stream, or tensor union changed"
            )
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
                "host_payload_read_s",
                "host_payload_sha256_s",
                "host_payload_hash_wait_s",
                "host_payload_decode_hash_s",
                "host_payload_hash_bytes",
                "host_payload_hash_files",
                "host_payload_cache_created",
                "host_payload_cache_reused",
                "host_frames_validate_s",
                "host_frames_validations",
                "host_outer_zstd_validate_s",
                "host_outer_zstd_decode_s",
                "host_outer_zstd_worker_decode_sum_s",
                "host_outer_zstd_encoded_bytes",
                "host_outer_zstd_decoded_bytes",
                "host_outer_zstd_tensors",
                "host_outer_zstd_frames",
                "host_shared_allocation_s",
                "host_shared_allocation_calls",
                "host_shared_allocation_bytes",
                "host_encoded_allocation_s",
                "host_encoded_allocation_calls",
                "host_encoded_allocation_bytes",
            )
        }
        waiting = time.perf_counter()
        # The mutex covers CPU construction/attachment, including host VMM setup.
        # It never encloses GPU stream work or inference collectives.
        # Original participant identities also scope the lock: unrelated engine
        # incarnations may reuse the same controller-assigned engine name.
        directory.mkdir(mode=0o700, exist_ok=True)
        previous_identity = self.identity
        with (directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            metrics["host_payload_cache_wait_s"] = time.perf_counter() - waiting
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
                raise ValueError("host arena namespace differs")
            if previous and previous["publication"] == expected:
                if state != {
                    "token": token,
                    "state": "READY",
                    "generation": previous["shared"]["generation"],
                }:
                    raise ValueError("host publication is failed or already released")
                index = previous
                metrics["host_payload_cache_reused"] = 1
            else:
                if previous and (
                    state["state"] != "REUSABLE"
                    or state["token"] != previous["token"]
                    or previous["publication"]["target_version"]
                    != expected["base_version"]
                ):
                    raise ValueError(
                        "host arena requires prior engine APPLIED release "
                        "before overwrite"
                    )
                build_started = time.perf_counter()
                frames_started = time.perf_counter()
                # READY certifies this exact immutable manifest for all local
                # consumers. Validate every tensor, including foreign EP data,
                # once before any arena sizing, allocation or payload access.
                validate_outer_entries(
                    manifest["tensors"],
                    {name: record["nbytes"] for name, record in definitions.items()},
                    manifest["frame_bytes"],
                )
                metrics["host_frames_validate_s"] = time.perf_counter() - frames_started
                metrics["host_frames_validations"] = 1
                entries_by_name = {
                    entry["name"]: entry for entry in manifest["tensors"]
                }
                if self.tensor_order is None:
                    # Keep each layer's compressed tensors adjacent, with expert
                    # numbers in natural order. Raw targets use a separate pass.
                    # The namespace fixes this inventory for the arena lifetime.
                    self.tensor_order = sorted(
                        names,
                        key=lambda name: (
                            entries_by_name[name]["encoding"] == "raw_bytes",
                            _natural_key(name),
                        ),
                    )
                entries = [entries_by_name[name] for name in self.tensor_order]
                layout, size = _tensor_layout(entries)
                encoded_size = sum(record["nbytes"] for record in definitions.values())
                shared = self._reserve_shared(
                    previous["shared"] if previous else None,
                    size,
                    metrics,
                )
                encoded = _reserve(
                    directory,
                    "encoded",
                    previous["encoded"] if previous else None,
                    encoded_size,
                    metrics,
                )
                index = {
                    "namespace": namespace,
                    "publication": expected,
                    "token": token,
                    "shared": shared,
                    "encoded": encoded,
                    "arena_bytes": size,
                    "tensors": layout,
                }
                # Publish the nonreusable state before the first overwrite. An
                # exception leaves this generation poisoned, including on abort.
                _write_record(directory, "index", index)
                _write_record(
                    directory,
                    "state",
                    {
                        "token": token,
                        "state": "BUILDING",
                        "generation": shared["generation"],
                    },
                )
                files = {}
                with (directory / encoded["file"]).open("r+b") as source:
                    encoded_map = (
                        mmap.mmap(source.fileno(), 0) if encoded_size else None
                    )
                position = 0
                for name, record in definitions.items():
                    source = (publication.parent / name).resolve(strict=True)
                    if source.parent != publication.parent:
                        raise ValueError("delta payload escapes immutable publication")
                    end = position + record["nbytes"]
                    view = (
                        memoryview(encoded_map)[position:end]
                        if encoded_map is not None
                        else memoryview(b"")
                    )
                    metrics["host_payload_read_s"] += _read_payload(
                        source, view, record
                    )
                    files[name] = view
                    position = end
                decode_hash_started = time.perf_counter()
                hash_future = pool.hash_executor.submit(
                    _hash_payloads, files, definitions
                )
                try:
                    _decode_arena(
                        self.mapping if self.mapping is not None else memoryview(b""),
                        layout,
                        files,
                        entries,
                        pool,
                        metrics,
                    )
                finally:
                    # Decode drains its tasks. Always join the independent hash
                    # before releasing views; hash failure takes precedence.
                    hash_wait_started = time.perf_counter()
                    metrics["host_payload_sha256_s"] = hash_future.result()
                    metrics["host_payload_hash_wait_s"] = (
                        time.perf_counter() - hash_wait_started
                    )
                    metrics["host_payload_decode_hash_s"] = (
                        time.perf_counter() - decode_hash_started
                    )
                    metrics["host_payload_hash_bytes"] = encoded_size
                    metrics["host_payload_hash_files"] = len(definitions)
                files.clear()
                index.update(
                    cpu_workers=pool.workers,
                    build_s=time.perf_counter() - build_started,
                )
                _write_record(directory, "index", index)
                _write_record(
                    directory,
                    "state",
                    {
                        "token": token,
                        "state": "READY",
                        "generation": shared["generation"],
                    },
                )
                # Each imported CUDA handle retains its old physical pages
                # until that rank attaches the new capacity generation.
                if previous:
                    if previous["encoded"]["file"] != index["encoded"]["file"]:
                        (directory / previous["encoded"]["file"]).unlink()
                metrics["host_payload_cache_created"] = 1
            self._attach(index["shared"])
        reused_mapping = previous_identity == self.identity
        self.directory = directory
        metrics.update(
            host_shared_arena_bytes=index["arena_bytes"],
            host_shared_capacity_bytes=index["shared"]["capacity"],
            host_shared_capacity_generation=index["shared"]["generation"],
            host_shared_mapping_reused=int(reused_mapping),
            host_encoded_capacity_bytes=index["encoded"]["capacity"],
            host_encoded_capacity_generation=index["encoded"]["generation"],
            host_shared_build_s=index["build_s"],
            host_outer_zstd_cpu_workers=index["cpu_workers"],
        )
        timings.update(metrics)
        return HostDecodedSnapshot(self, index)

    def close(self):
        self.tensor = self.mapping = None
        if self.allocation is not None:
            self.allocation.close()
            self.allocation = None
        self.identity = None


class HostDecodedSnapshot:
    """One immutable publication's views over a backend-owned capacity arena."""

    def __init__(self, arena, index):
        self.arena, self.index = arena, index
        self.directory = arena.directory

    def get(self, name):
        record = self.index["tensors"][name]
        return self.arena.tensor[record["offset"] : record["offset"] + record["nbytes"]]

    def mark_reusable(self):
        # Called only by queued successful engine-resume cleanup. A late release
        # from the prior generation must never release newly prepared bytes.
        with (self.directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = orjson.loads((self.directory / "state.json").read_bytes())
            if state == {
                "token": self.index["token"],
                "state": "READY",
                "generation": self.index["shared"]["generation"],
            }:
                state["state"] = "REUSABLE"
                _write_record(self.directory, "state", state)

    def close(self):
        # Backend retains host pages; caller has already fenced all DE readers.
        self.arena = None
