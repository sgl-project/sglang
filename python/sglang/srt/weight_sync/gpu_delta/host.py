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

Encoded publication files and multi-consumer inner compressed bytes are shared
through tmpfs. Each scheduler retains its original CUDA HOST_NUMA allocation.
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
from collections import Counter
from pathlib import Path

import numpy as np
import orjson

from sglang.srt.weight_sync.gpu_delta.memory import HostAllocation
from sglang.srt.weight_sync.gpu_delta.payload import validate_outer_entries


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


def _reserve_cache(directory, previous, size, metrics, kind):
    if previous is not None and size <= previous["capacity"]:
        return previous
    capacity = size if previous is None else 2 * size
    capacity = (
        (capacity + _CAPACITY_ALIGNMENT - 1)
        // _CAPACITY_ALIGNMENT
        * _CAPACITY_ALIGNMENT
    )
    generation = previous["generation"] + 1 if previous else 1
    path = directory / f"{kind}-{generation}.bin"
    started = time.perf_counter()
    with path.open("xb+") as target:
        if capacity:
            # Reserve pages once per capacity generation; fitting updates never
            # truncate or resize the retained encoded-cache inode.
            os.posix_fallocate(target.fileno(), 0, capacity)
        identity = _identity(os.fstat(target.fileno()))
    metrics[f"host_{kind}_cache_allocation_s"] += time.perf_counter() - started
    metrics[f"host_{kind}_cache_allocation_calls"] += 1
    metrics[f"host_{kind}_cache_allocation_bytes"] += capacity
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


def _read_verify_payload(source, destination, expected, skip_payload_hash):
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
    read_s = time.perf_counter() - started
    if skip_payload_hash:
        return read_s, 0.0
    started = time.perf_counter()
    # Authenticate the retained copy, never reread the publication file.
    if hashlib.sha256(destination).hexdigest() != expected["sha256"]:
        raise ValueError("delta payload SHA256 mismatch")
    return read_s, time.perf_counter() - started


def _read_verify_payloads(
    publication, files, definitions, manifest, pool, metrics, skip_payload_hash
):
    def validate_frames():
        started = time.perf_counter()
        validate_outer_entries(
            manifest["tensors"],
            {name: record["nbytes"] for name, record in definitions.items()},
            manifest["frame_bytes"],
        )
        return time.perf_counter() - started

    started = time.perf_counter()
    futures, validation, error = [], None, None
    try:
        # Independent CPU work; neither worker waits for another pool task.
        validation = pool.executor.submit(validate_frames)
        for name, record in definitions.items():
            source = (publication.parent / name).resolve(strict=True)
            if source.parent != publication.parent:
                raise ValueError("delta payload escapes immutable publication")
            futures.append(
                pool.executor.submit(
                    _read_verify_payload,
                    source,
                    files[name],
                    record,
                    skip_payload_hash,
                )
            )
    except BaseException as exc:  # noqa: BLE001 - drain submitted file tasks
        error = exc
    for future in futures:
        try:
            read_s, hash_s = future.result()
            metrics["host_encoded_cache_read_worker_sum_s"] += read_s
            metrics["host_encoded_cache_sha256_worker_sum_s"] += hash_s
        except BaseException as exc:  # noqa: BLE001 - retain mappings until peers drain
            if error is None:
                error = exc
    metrics["host_encoded_cache_read_hash_s"] = time.perf_counter() - started
    if validation is not None:
        try:
            metrics["host_encoded_cache_frames_validate_s"] = validation.result()
            metrics["host_encoded_cache_frames_validations"] = 1
        except BaseException as exc:  # noqa: BLE001 - both branches drain before READY
            if error is None:
                error = exc
    if error is not None:
        raise error


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
    metrics["finished_at"] = time.perf_counter()
    return metrics


def _decode_arena(destination, layout, files, entries, pool, metrics, shared):
    started = time.perf_counter()
    jobs, futures, common_futures, copies, error = [], [], [], [], None
    shared_layout = shared.index["shared"]["tensors"]
    common_metrics = {
        key.replace("host_rank_", "host_shared_"): 0 for key in _DECODE_METRICS
    }
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
            elif entry["name"] not in shared_layout:
                jobs.append((entry, record))
        common = [
            (entry, shared_layout[entry["name"]])
            for entry in shared.entries
            if entry["name"] in shared_layout
        ]
        common_count = min(pool.workers, len(common))
        private_count = min(4 * pool.workers, len(jobs))
        # Submit common and private tasks directly, interleaved. No pool task
        # submits work and waits on that same executor, even with one worker.
        for i in range(max(common_count, private_count)):
            if i < common_count:
                common_futures.append(
                    pool.executor.submit(
                        _decode_group,
                        common[i::common_count],
                        shared.data,
                        files,
                        pool,
                    )
                )
            if i < private_count:
                futures.append(
                    pool.executor.submit(
                        _decode_group,
                        jobs[i::private_count],
                        destination,
                        files,
                        pool,
                    )
                )
        for future in common_futures:
            partial = future.result()
            for key in _DECODE_METRICS:
                common_metrics[key.replace("host_rank_", "host_shared_")] += partial[
                    key
                ]
        metrics["host_shared_outer_zstd_decode_s"] = (
            time.perf_counter() - started if shared.creator else 0
        )
        waiting = time.perf_counter()
        shared.ready()
        metrics["host_shared_cache_wait_s"] = time.perf_counter() - waiting
        copying = time.perf_counter()
        ranges = []
        for entry in entries:
            source = shared_layout.get(entry["name"])
            if source is None:
                continue
            target = layout[entry["name"]]
            if source["nbytes"] != target["nbytes"]:
                raise ValueError("shared and private decoded tensor extents differ")
            row = (source["offset"], target["offset"], target["nbytes"])
            if ranges:
                before = ranges[-1]
                gap = row[0] - (before[0] + before[2])
                # Only identical inter-tensor alignment padding may be included.
                if 0 <= gap < 16 and row[1] - (before[1] + before[2]) == gap:
                    ranges[-1] = (before[0], before[1], before[2] + gap + row[2])
                    continue
            ranges.append(row)
        count = min(pool.workers, len(ranges))
        for i in range(count):
            copies.append(
                pool.executor.submit(
                    _copy_group,
                    ranges[i::count],
                    shared.data,
                    destination,
                )
            )
        for future in copies:
            future.result()
        metrics["host_rank_shared_copy_s"] = time.perf_counter() - copying
        metrics["host_rank_shared_copy_bytes"] = sum(row[2] for row in ranges)
        metrics["host_rank_shared_copy_ranges"] = len(ranges)
    except BaseException as exc:  # noqa: BLE001 - drain every submitted reader/writer
        error = exc
    for future in common_futures + copies:
        try:
            future.result()
        except BaseException as exc:  # noqa: BLE001 - preserve the first failure
            if error is None:
                error = exc
    private_finished = started
    for future in futures:
        try:
            partial = future.result()
            private_finished = max(private_finished, partial["finished_at"])
            for key in _DECODE_METRICS:
                metrics[key] += partial[key]
        except BaseException as exc:  # noqa: BLE001 - private destinations stay alive
            if error is None:
                error = exc
    metrics.update(common_metrics)
    metrics["host_rank_outer_zstd_decode_s"] = private_finished - started
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


class SharedDecodeLease:
    """CPU-only common bytes; the producer holds the lock until workers drain."""

    def __init__(self, directory, index, entries, lock=None):
        self.directory, self.index = directory, index
        self.entries, self.lock = entries, lock
        self.creator = lock is not None
        descriptor = index["shared"]
        with (directory / descriptor["file"]).open("r+b") as source:
            if _identity(os.fstat(source.fileno())) != descriptor["identity"]:
                raise ValueError("shared decoded cache inode or extent changed")
            self.data = (
                memoryview(
                    mmap.mmap(
                        source.fileno(),
                        0,
                        access=mmap.ACCESS_WRITE if self.creator else mmap.ACCESS_READ,
                    )
                )
                if descriptor["capacity"]
                else memoryview(b"")
            )

    def ready(self):
        expected = {
            "token": self.index["token"],
            "generation": self.index["shared"]["generation"],
            "state": "READY",
        }
        if self.creator:
            _write_record(self.directory, "shared-state", expected)
            self.lock.close()
            self.lock = None
        else:
            with (self.directory / ".shared-lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if (
                    orjson.loads((self.directory / "shared-state.json").read_bytes())
                    != expected
                ):
                    raise ValueError("shared decoded publication failed before READY")

    def close(self):
        # The caller has drained submitted work. Failed preparation leaves the
        # shared state BUILDING; dropping the lock wakes peers to reject it.
        if self.lock is not None:
            self.lock.close()
            self.lock = None
        self.data = self.entries = None


def _copy_group(ranges, source, destination):
    # Numeric contiguous copies release the GIL; each job owns disjoint output.
    source = np.frombuffer(source, dtype=np.uint8)
    destination = np.frombuffer(destination, dtype=np.uint8)
    for start, target, size in ranges:
        np.copyto(destination[target : target + size], source[start : start + size])


class HostArena:
    """Persistent rank-owned DE storage with an engine-host encoded cache.

    The original engine cohort shares encoded files under one hash policy. Successful
    all-rank apply/resume permits the next publication to overwrite that cache;
    aborted/failed preparation never grants release.
    """

    def __init__(self, engine_id, device):
        self.engine_id, self.device = engine_id, device
        self.skip_payload_hash = (
            os.environ.get("GPU_DELTA_SKIP_PAYLOAD_HASH", "0") == "1"
        )
        self.allocation = self.mapping = self.tensor = None
        self.capacity = None
        self.directory = None
        self.tensor_order = None
        self.rank_identity = self.shared_names = self.cohort = None

    def register_rank(self, identity, names):
        """Publish immutable local membership before the existing describe barrier."""
        root = _cache_root(self.engine_id)
        record = {"identity": identity, "names": sorted(names)}
        path = root / (identity["rank_id"] + ".json")
        with path.open("xb") as target:
            target.write(orjson.dumps(record))
        path.chmod(0o400)
        self.rank_identity = dict(identity)

    def _shared_names(self, metadata):
        participants = metadata["participants"]
        if self.cohort is not None:
            if participants != self.cohort:
                raise ValueError("shared decoded cache original cohort changed")
            return self.shared_names
        root = _cache_root(self.engine_id)
        counts = Counter()
        if self.rank_identity not in participants:
            raise ValueError("shared decoded cache rank was not admitted")
        for identity in participants:
            if identity["host_cache_id"] == self.rank_identity["host_cache_id"]:
                record = orjson.loads(
                    (root / (identity["rank_id"] + ".json")).read_bytes()
                )
                if record["identity"] != identity:
                    raise ValueError("shared decoded cache rank identity changed")
                counts.update(record["names"])
        self.shared_names = {name for name, count in counts.items() if count > 1}
        self.cohort = [dict(identity) for identity in participants]
        return self.shared_names

    def _reserve_rank_arena(self, size, metrics):
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

    def prepare_encoded(
        self,
        manifest_path,
        manifest_sha256,
        manifest,
        pool,
        timings,
        metadata,
    ):
        """Admit READY encoded bytes; returned file views own their mmap lease."""
        started = time.perf_counter()
        root = _cache_root(self.engine_id)
        shared_names = self._shared_names(metadata)
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
            "skip_payload_hash": self.skip_payload_hash,
        }
        token = hashlib.sha256(
            json.dumps(expected, sort_keys=True).encode()
        ).hexdigest()
        metrics = {
            name: 0
            for name in (
                "host_encoded_cache_read_hash_s",
                "host_encoded_cache_read_worker_sum_s",
                "host_encoded_cache_sha256_worker_sum_s",
                "host_encoded_cache_hash_bytes",
                "host_encoded_cache_hash_files",
                "host_encoded_cache_created",
                "host_encoded_cache_reused",
                "host_encoded_cache_frames_validate_s",
                "host_encoded_cache_frames_validations",
                "host_encoded_cache_allocation_s",
                "host_encoded_cache_allocation_calls",
                "host_encoded_cache_allocation_bytes",
                "host_shared_cache_allocation_s",
                "host_shared_cache_allocation_calls",
                "host_shared_cache_allocation_bytes",
            )
        }
        shared_lock, shared_entries = None, []
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
            if (
                previous
                and previous["publication"]["skip_payload_hash"]
                != self.skip_payload_hash
            ):
                raise ValueError("encoded cache payload hash policy differs")
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
                encoded_size = sum(record["nbytes"] for record in definitions.values())
                encoded = _reserve_cache(
                    directory,
                    previous["encoded"] if previous else None,
                    encoded_size,
                    metrics,
                    "encoded",
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
                _read_verify_payloads(
                    publication,
                    files,
                    definitions,
                    manifest,
                    pool,
                    metrics,
                    self.skip_payload_hash,
                )
                if not self.skip_payload_hash:
                    metrics["host_encoded_cache_hash_bytes"] = encoded_size
                    metrics["host_encoded_cache_hash_files"] = len(definitions)
                shared_entries = sorted(
                    (
                        entry
                        for entry in manifest["tensors"]
                        if entry["name"] in shared_names
                        and entry["encoding"] == "xor_bytes"
                    ),
                    key=lambda entry: _natural_key(entry["name"]),
                )
                shared_layout, shared_size = _tensor_layout(shared_entries)
                shared = dict(
                    _reserve_cache(
                        directory,
                        previous["shared"] if previous else None,
                        shared_size,
                        metrics,
                        "shared",
                    )
                )
                index["shared"] = shared | {
                    "tensors": shared_layout,
                    "arena_bytes": shared_size,
                }
                _write_record(
                    directory,
                    "shared-state",
                    {
                        "token": token,
                        "generation": shared["generation"],
                        "state": "BUILDING",
                    },
                )
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
                if previous and previous["shared"]["file"] != shared["file"]:
                    (directory / previous["shared"]["file"]).unlink()
                shared_lock = (directory / ".shared-lock").open("a+b")
                fcntl.flock(shared_lock, fcntl.LOCK_EX)
                metrics["host_encoded_cache_created"] = 1
        metrics["host_encoded_cache_access_s"] = time.perf_counter() - started
        self.directory = directory
        metrics.update(
            host_encoded_cache_capacity_bytes=index["encoded"]["capacity"],
            host_encoded_cache_capacity_generation=index["encoded"]["generation"],
            host_encoded_cache_build_s=index["build_s"],
            host_encoded_cache_skip_payload_hash=int(self.skip_payload_hash),
            host_shared_cache_capacity_bytes=index["shared"]["capacity"],
            host_shared_cache_capacity_generation=index["shared"]["generation"],
            host_shared_cache_created=metrics["host_encoded_cache_created"],
            host_shared_cache_reused=metrics["host_encoded_cache_reused"],
        )
        timings.update(metrics)
        try:
            shared_lease = SharedDecodeLease(
                directory, index, shared_entries, shared_lock
            )
        except BaseException:
            if shared_lock is not None:
                shared_lock.close()
            raise
        return index, files, shared_lease

    def decode_local(self, index, files, local_entries, pool, timings, shared_lease):
        """Decode local entries; the caller retains file views until this returns."""
        metrics = {
            name: 0
            for name in (
                "host_rank_allocation_s",
                "host_rank_allocation_calls",
                "host_rank_allocation_bytes",
                "host_rank_mapping_reused",
                *_DECODE_METRICS,
            )
        }
        layout_started = time.perf_counter()
        if self.tensor_order is None:
            # Local binding order is immutable; retain indices, not old entries.
            self.tensor_order = sorted(
                range(len(local_entries)),
                key=lambda i: (
                    local_entries[i]["encoding"] == "raw_bytes",
                    _natural_key(local_entries[i]["name"]),
                ),
            )
        entries = [local_entries[i] for i in self.tensor_order]
        layout, size = _tensor_layout(entries)
        metrics["host_rank_layout_s"] = time.perf_counter() - layout_started
        self._reserve_rank_arena(size, metrics)
        decode_started = time.perf_counter()
        _decode_arena(
            self.mapping if self.mapping is not None else memoryview(b""),
            layout,
            files,
            entries,
            pool,
            metrics,
            shared_lease,
        )
        metrics["host_rank_decode_call_s"] = time.perf_counter() - decode_started
        metrics.update(
            host_rank_arena_bytes=size,
            host_rank_capacity_bytes=self.capacity["capacity"],
            host_rank_capacity_generation=self.capacity["generation"],
            host_rank_cpu_workers=pool.workers,
        )
        snapshot = HostDecodedSnapshot(
            self,
            index
            | {"tensors": layout, "arena_bytes": size, "rank_arena": self.capacity},
        )
        timings.update(metrics)
        return snapshot

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
