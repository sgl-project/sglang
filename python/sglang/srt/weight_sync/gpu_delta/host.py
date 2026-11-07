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

"""Verified engine-host encoded cache and persistent pinned input arenas.

Each engine host shares one prepared tmpfs image and one original rank's workers.
CUDA decoders consume it directly; DE decoders stage only their local spans into
rank-private hardware-decompression-capable input storage.
"""

import fcntl
import hashlib
import json
import mmap
import os
import re
import stat
import threading
import time
import uuid
from array import array
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path

import numpy as np
import orjson

from sglang.srt.weight_sync.gpu_delta import profiling
from sglang.srt.weight_sync.gpu_delta.memory import HostAllocation, HostMapping


def validate_codec(manifest):
    """Admit the authenticated publication codec before examining any tensor payload."""
    if (
        manifest.get("codec")
        not in {"snappy-zstd", "lz4-zstd", "lz4", "bitcomp", "bitcomp-zstd"}
        or type(manifest.get("frame_bytes")) is not int
        or not 0 < manifest["frame_bytes"] <= 4 << 20
    ):
        raise ValueError(
            "GPU delta requires snappy-zstd, lz4-zstd, lz4, bitcomp or bitcomp-zstd "
            "and an integer frame size in (0, 4 MiB]"
        )


def configured_cpu_workers():
    value = int(os.environ.get("GPU_DELTA_CPU_WORKERS", "32"))
    if not 1 <= value <= 64:
        raise ValueError("GPU_DELTA_CPU_WORKERS must be between 1 and 64")
    return value


def copy_bytes(destination, source):
    """Copy between retained byte buffers without holding the GIL for large spans."""
    if len(source) < 64 << 10:
        destination[:] = source
    else:
        np.copyto(
            np.frombuffer(destination, dtype=np.uint8),
            np.frombuffer(source, dtype=np.uint8),
        )


class HostPayloadPool:
    """Reusable bounded CPU workers; no CUDA work or shared decoder contexts."""

    def __init__(self, workers):
        self.workers = workers
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="gpu-delta-payload"
        )
        self.local = threading.local()

    def decode_zstd(self, payloads, sizes, offsets, destination):
        if not hasattr(self.local, "decoder"):
            import zstandard as zstd

            self.local.decoder = zstd.ZstdDecompressor()
        decoded_sizes = np.asarray(sizes, dtype=np.uint64)
        started = time.perf_counter()
        # The existing pool supplies concurrency; native decode stays single-threaded.
        decoded = self.local.decoder.multi_decompress_to_buffer(
            payloads, decompressed_sizes=decoded_sizes, threads=0
        )
        # Native decode verifies every declared output size before copying any result.
        for segment, offset, size in zip(decoded, offsets, sizes, strict=True):
            copy_bytes(destination[offset : offset + size], memoryview(segment))
        return time.perf_counter() - started

    def close(self):
        self.executor.shutdown(wait=True)


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
            # truncate or resize the retained cache inode.
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


@profiling.annotate("gpu_delta.host.read_verify_worker")
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


@profiling.annotate("gpu_delta.host.read_verify")
def _read_verify_payloads(
    publication, backing, file_spans, definitions, pool, metrics, skip_payload_hash
):
    started = time.perf_counter()
    futures, error = [], None
    try:
        for name, record in definitions.items():
            source = (publication.parent / name).resolve(strict=True)
            if source.parent != publication.parent:
                raise ValueError("delta payload escapes immutable publication")
            start, size = file_spans[name]
            futures.append(
                pool.executor.submit(
                    _read_verify_payload,
                    source,
                    backing[start : start + size],
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
    if error is not None:
        raise error


def _natural_key(name):
    return tuple(
        int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", name)
    )


@profiling.annotate("gpu_delta.host.tensor_layout")
def _tensor_layout(entries, previous=None):
    layout, size = {}, 0
    for entry in entries:
        count = (
            (entry["nbytes"] if entry["changed_bytes"] else 0)
            if entry["encoding"] == "raw_bytes"
            else (entry["outer"]["decoded_bytes"] if entry["frames"] else 0)
        )
        if count:
            size = (size + 15) // 16 * 16
            name = entry["name"]
            record = previous.get(name) if previous is not None else None
            if record is None:
                record = {}
            record["offset"] = size
            record["nbytes"] = count
            layout[name] = record
            size += count
    return layout, size


_PREPARE_METRICS = (
    "host_rank_outer_zstd_worker_decode_sum_s",
    "host_rank_outer_zstd_encoded_bytes",
    "host_rank_outer_zstd_decoded_bytes",
    "host_rank_outer_zstd_tensors",
    "host_rank_outer_zstd_frames",
    "host_rank_encoded_copy_worker_sum_s",
    "host_rank_encoded_copy_bytes",
    "host_rank_encoded_copy_tensors",
)


@profiling.annotate("gpu_delta.host.outer_zstd_worker")
def _decode_zstd_group(jobs, destination, backing, file_spans, layout, pool):
    from zstandard import BufferWithSegments

    metrics = {name: 0 for name in _PREPARE_METRICS}
    segments, sizes, offsets = array("Q"), [], []
    for entry in jobs:
        record = layout[entry["name"]]
        outer = entry["outer"]
        file_start, file_size = file_spans[outer["file"]]
        for chunk in outer["frames"]:
            start = outer["encoded_offset"] + chunk["encoded_offset"]
            size = chunk["encoded_bytes"]
            if not 0 <= start <= start + size <= file_size:
                raise ValueError("delta payload span escapes declared file")
            segments.append(file_start + start)
            segments.append(size)
            sizes.append(chunk["decoded_bytes"])
            offsets.append(record["offset"] + chunk["decoded_offset"])
        metrics["host_rank_outer_zstd_encoded_bytes"] += outer["encoded_bytes"]
        metrics["host_rank_outer_zstd_decoded_bytes"] += outer["decoded_bytes"]
        metrics["host_rank_outer_zstd_tensors"] += 1
        metrics["host_rank_outer_zstd_frames"] += len(outer["frames"])
    metrics["host_rank_outer_zstd_worker_decode_sum_s"] = pool.decode_zstd(
        BufferWithSegments(backing, segments), sizes, offsets, destination
    )
    return metrics


@profiling.annotate("gpu_delta.host.copy_worker")
def _copy_group(jobs, destination, backing, file_spans, layout, pool):
    metrics = {name: 0 for name in _PREPARE_METRICS}
    for entry in jobs:
        record = layout[entry["name"]]
        outer = entry["outer"]
        length = outer["encoded_bytes"]
        started = time.perf_counter()
        file_start, file_size = file_spans[outer["file"]]
        start = outer["encoded_offset"]
        if not 0 <= start <= start + length <= file_size:
            raise ValueError("delta payload span escapes declared file")
        start += file_start
        offset = record["offset"]
        copy_bytes(
            destination[offset : offset + length],
            backing[start : start + length],
        )
        metrics["host_rank_encoded_copy_worker_sum_s"] += time.perf_counter() - started
        metrics["host_rank_encoded_copy_bytes"] += length
        metrics["host_rank_encoded_copy_tensors"] += 1
    return metrics


@profiling.annotate("gpu_delta.host.prepare_arena")
def _prepare_arena(
    destination, layout, backing, file_spans, entries, pool, metrics, codec
):
    started = time.perf_counter()
    jobs, futures, error = [], [], None
    if not codec.endswith("-zstd"):
        worker, phase = _copy_group, "host_rank_encoded_copy_s"
    else:
        worker, phase = _decode_zstd_group, "host_rank_outer_zstd_decode_s"
    try:
        for entry in entries:
            record = layout.get(entry["name"])
            if record is None:
                continue
            if entry["encoding"] == "raw_bytes":
                raw = entry["raw"]
                offset, count = record["offset"], record["nbytes"]
                file_start, file_size = file_spans[raw["file"]]
                start = raw["encoded_offset"]
                if not 0 <= start <= start + count <= file_size:
                    raise ValueError("delta payload span escapes declared file")
                start += file_start
                copy_bytes(
                    destination[offset : offset + count],
                    backing[start : start + count],
                )
            else:
                jobs.append(entry)
        count = min(4 * pool.workers, len(jobs))
        for index in range(count):
            futures.append(
                pool.executor.submit(
                    worker,
                    jobs[index::count],
                    destination,
                    backing,
                    file_spans,
                    layout,
                    pool,
                )
            )
    except BaseException as exc:  # noqa: BLE001 - drain submitted jobs before re-raise
        error = exc
    for future in futures:
        try:
            partial = future.result()
            for key in _PREPARE_METRICS:
                metrics[key] += partial[key]
        except BaseException as exc:  # noqa: BLE001 - drain peers before re-raise
            if error is None:
                error = exc
    metrics["host_rank_outer_zstd_decode_s"] = 0.0
    metrics["host_rank_encoded_copy_s"] = 0.0
    metrics[phase] = time.perf_counter() - started
    if error is not None:
        raise error


def _map_encoded(directory, encoded, definitions):
    with (directory / encoded["file"]).open("r+b") as source:
        if _identity(os.fstat(source.fileno())) != encoded["identity"]:
            raise ValueError("encoded cache inode or extent changed")
        mapping = mmap.mmap(source.fileno(), 0) if encoded["capacity"] else None
    file_spans, position = {}, 0
    for name, record in definitions.items():
        file_spans[name] = (position, record["nbytes"])
        position += record["nbytes"]
    # The view owns its mmap, including while drained failure tracebacks retain it.
    backing = memoryview(mapping) if mapping is not None else memoryview(b"")
    return backing, file_spans


class HostArena:
    """Persistent input storage with an engine-host encoded cache.

    The original engine cohort shares encoded files under each publication's hash
    policy. Successful all-rank apply/resume permits the next publication to
    overwrite encoded and prepared bytes; failed preparation never grants release.
    """

    def __init__(self, engine_id, device, identity):
        self.engine_id, self.device = engine_id, device
        self.identity = identity
        self.allocation = self.mapping = self.tensor = None
        self.capacity = None
        self.hardware = None
        self.shared_allocation = self.shared_capacity = None
        self.directory = None
        self.tensor_order = None
        self.shared_tensor_order = None
        self._pool = None
        self._index = None

    @property
    def payload_pool(self):
        if self._pool is None:
            self._pool = HostPayloadPool(configured_cpu_workers())
        return self._pool

    @cached_property
    def root(self):
        return _cache_root(self.engine_id)

    def _reserve_rank_arena(self, size, metrics):
        import torch

        if (
            self.capacity is not None
            and size <= self.capacity["capacity"]
            and self.hardware
        ):
            metrics["host_rank_mapping_reused"] = 1
            return
        previous = self.capacity if self.hardware else None
        capacity = size if previous is None else 2 * size
        capacity = (
            (capacity + _CAPACITY_ALIGNMENT - 1)
            // _CAPACITY_ALIGNMENT
            * _CAPACITY_ALIGNMENT
        )
        started = time.perf_counter()
        allocation = None
        if capacity:
            allocation = HostAllocation(capacity, self.device)
        self._close_private()
        self.allocation = allocation
        self.hardware = True
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

    def _attach_shared_arena(self, prepared, metrics, hardware):
        if self.shared_capacity == prepared and (
            not prepared["capacity"]
            or self.shared_allocation.registered == (not hardware)
        ):
            if not hardware:
                metrics["host_rank_mapping_reused"] = int(
                    not metrics["host_rank_allocation_calls"]
                )
            return
        started = time.perf_counter()
        self._close_shared()
        if prepared["capacity"]:
            with (self.directory / prepared["file"]).open("r+b") as source:
                if _identity(os.fstat(source.fileno())) != prepared["identity"]:
                    raise ValueError("prepared cache inode or extent changed")
                mapping = mmap.mmap(
                    source.fileno(),
                    prepared["capacity"],
                    flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                )
            if not hardware:
                import torch

                # A fresh preparation thread may have no current CUDA context.
                torch.cuda.set_device(self.device)
            self.shared_allocation = HostMapping(mapping, pinned=not hardware)
        self.shared_capacity = prepared
        if not hardware:
            metrics["host_rank_allocation_s"] = time.perf_counter() - started
            metrics["host_rank_allocation_calls"] = int(
                self.shared_allocation is not None
            )
            metrics["host_rank_allocation_bytes"] = prepared["capacity"]

    @profiling.annotate("gpu_delta.host.prepare_shared")
    def _prepare_shared_arena(
        self, index, previous, backing, file_spans, entries, metrics
    ):
        layout_started = time.perf_counter()
        if self.shared_tensor_order is None:
            self.shared_tensor_order = sorted(
                range(len(entries)),
                key=lambda i: (
                    entries[i]["encoding"] == "raw_bytes",
                    _natural_key(entries[i]["name"]),
                ),
            )
        ordered = [entries[i] for i in self.shared_tensor_order]
        # Prior readers have released before BUILDING; retain small host records,
        # but recompute offsets and membership for this publication's payloads.
        layout, size = _tensor_layout(
            ordered, previous["prepared_records"] if previous else None
        )
        metrics["host_prepared_cache_layout_s"] = time.perf_counter() - layout_started
        prepared = _reserve_cache(
            self.directory,
            previous["prepared"] if previous else None,
            size,
            metrics,
            "prepared",
        )
        self._attach_shared_arena(prepared, metrics, index["publication"]["hardware"])
        _prepare_arena(
            self.shared_allocation.view
            if self.shared_allocation is not None
            else memoryview(b""),
            layout,
            backing,
            file_spans,
            ordered,
            self.payload_pool,
            metrics,
            index["publication"]["codec"],
        )
        index.update(
            prepared=prepared,
            prepared_records=layout,
            prepared_size=size,
            prepared_build_s=time.perf_counter() - layout_started,
        )

    @profiling.annotate("gpu_delta.host.prepare_encoded")
    def prepare_encoded(
        self,
        manifest_path,
        manifest_sha256,
        manifest,
        timings,
        metadata,
        hardware=True,
        cancelled=None,
    ):
        """Admit the complete verified and prepared engine-host image."""
        started = time.perf_counter()
        skip_payload_hash = not metadata["validate_payload_hash"]
        publication = Path(manifest_path).resolve(strict=True)
        namespace = {
            "stream_id": metadata["stream_id"],
            "participants": sorted(
                metadata["participants"],
                key=lambda value: json.dumps(value, sort_keys=True),
            ),
        }
        key = hashlib.sha256(json.dumps(namespace, sort_keys=True).encode()).hexdigest()
        directory = self.root / key
        if self.directory is not None and self.directory != directory:
            raise ValueError("host cache original cohort or stream changed")
        definitions = {}
        for record in manifest["files"]:
            name, size = record["name"], record["nbytes"]
            if size < 0:
                raise ValueError("negative delta payload size")
            if record["sha256"] is None and not skip_payload_hash:
                raise ValueError(
                    "payload checksum omitted; request validate_payload_hash=False "
                    "to accept an unchecked publication"
                )
            definitions[name] = {"nbytes": size, "sha256": record["sha256"]}
        expected = {
            "manifest_path": str(publication),
            "manifest_sha256": manifest_sha256,
            "codec": manifest["codec"],
            "session_id": metadata["session_id"],
            "base_version": metadata["base_version"],
            "target_version": metadata["target_version"],
            "files": definitions,
            "skip_payload_hash": skip_payload_hash,
            "hardware": hardware,
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
                "host_encoded_cache_allocation_s",
                "host_encoded_cache_allocation_calls",
                "host_encoded_cache_allocation_bytes",
                "host_prepared_cache_created",
                "host_prepared_cache_reused",
                "host_prepared_cache_wait_s",
                "host_prepared_cache_layout_s",
                "host_prepared_cache_allocation_s",
                "host_prepared_cache_allocation_calls",
                "host_prepared_cache_allocation_bytes",
                "host_rank_allocation_s",
                "host_rank_allocation_calls",
                "host_rank_allocation_bytes",
                "host_rank_mapping_reused",
            )
        }
        metrics.update((name, 0) for name in _PREPARE_METRICS)
        metrics["host_rank_outer_zstd_decode_s"] = 0.0
        metrics["host_rank_encoded_copy_s"] = 0.0
        owner_identity = min(
            (
                participant
                for participant in namespace["participants"]
                if participant["engine_id"] == self.engine_id
                and participant["host_cache_id"] == self.identity["host_cache_id"]
            ),
            key=lambda participant: (
                participant["tp_rank"],
                participant["dp_rank"],
                participant["pid"],
                participant["rank_id"],
            ),
        )
        owner = self.identity == owner_identity
        waiting = time.perf_counter()
        directory.mkdir(mode=0o700, exist_ok=True)
        self.directory = directory
        # The original owner publishes READY only after preparing the shared
        # image. Followers inspect the small state before reading the full index,
        # then share the read lock while attaching the same immutable publication.
        # The control-plane abort event also ends a wait when the owner failed
        # before reaching host preparation while its process is still alive.
        with (directory / ".lock").open("a+b") as lock:
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError("GPU-delta host preparation aborted")
                try:
                    mode = fcntl.LOCK_EX if owner else fcntl.LOCK_SH
                    fcntl.flock(lock, mode | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    state_path = directory / "state.json"
                    state = (
                        orjson.loads(state_path.read_bytes())
                        if state_path.exists()
                        else None
                    )
                    if owner or (state is not None and state["token"] == token):
                        if not owner and state["state"] != "READY":
                            raise ValueError(
                                "encoded publication is failed or already released"
                            )
                        index_path = directory / "index.json"
                        if owner:
                            if self._index is None and index_path.exists():
                                self._index = orjson.loads(index_path.read_bytes())
                            previous = self._index
                        else:
                            previous = orjson.loads(index_path.read_bytes())
                        break
                    fcntl.flock(lock, fcntl.LOCK_UN)
                if cancelled is None:
                    time.sleep(0.01)
                else:
                    cancelled.wait(0.01)
            metrics["host_encoded_cache_wait_s"] = time.perf_counter() - waiting
            metrics["host_prepared_cache_wait_s"] = metrics["host_encoded_cache_wait_s"]
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
                metrics["host_encoded_cache_reused"] = 1
                metrics["host_prepared_cache_reused"] = 1
            else:
                if not owner:
                    raise ValueError("encoded cache publication differs")
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
                    "prepared": previous["prepared"] if previous else None,
                }
                # Failed read/hash or shared preparation leaves BUILDING and
                # cannot authorize release or reuse of either cache.
                _write_record(directory, "index", index)
                self._index = index
                _write_record(
                    directory,
                    "state",
                    {
                        "token": token,
                        "state": "BUILDING",
                        "generation": encoded["generation"],
                    },
                )
                backing, file_spans = _map_encoded(directory, encoded, definitions)
                _read_verify_payloads(
                    publication,
                    backing,
                    file_spans,
                    definitions,
                    self.payload_pool,
                    metrics,
                    skip_payload_hash,
                )
                if not skip_payload_hash:
                    metrics["host_encoded_cache_hash_bytes"] = encoded_size
                    metrics["host_encoded_cache_hash_files"] = len(definitions)
                index["build_s"] = time.perf_counter() - build_started
                self._prepare_shared_arena(
                    index,
                    previous,
                    backing,
                    file_spans,
                    manifest["tensors"],
                    metrics,
                )
                del backing  # All readers have drained; READY retains prepared bytes.
                metrics["host_prepared_cache_created"] = 1
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
                if (
                    previous
                    and previous["prepared"] is not None
                    and previous["prepared"]["file"] != index["prepared"]["file"]
                ):
                    (directory / previous["prepared"]["file"]).unlink()
                metrics["host_encoded_cache_created"] = 1
        metrics["host_encoded_cache_access_s"] = time.perf_counter() - started
        metrics.update(
            host_encoded_cache_capacity_bytes=index["encoded"]["capacity"],
            host_encoded_cache_capacity_generation=index["encoded"]["generation"],
            host_encoded_cache_build_s=index["build_s"],
            host_encoded_cache_skip_payload_hash=int(skip_payload_hash),
            host_prepared_cache_capacity_bytes=index["prepared"]["capacity"],
            host_prepared_cache_capacity_generation=index["prepared"]["generation"],
            host_prepared_cache_arena_bytes=index["prepared_size"],
            host_prepared_cache_build_s=index["prepared_build_s"],
        )
        timings.update(metrics)
        return index

    @profiling.annotate("gpu_delta.host.prepare_local")
    def prepare_local(self, index, local_entries, timings, hardware=True):
        """Stage local DE spans or directly bind the shared CUDA prepared image."""
        metrics = {
            name: timings.get(name, 0)
            for name in (
                "host_rank_allocation_s",
                "host_rank_allocation_calls",
                "host_rank_allocation_bytes",
                "host_rank_mapping_reused",
                "host_rank_outer_zstd_decode_s",
                "host_rank_encoded_copy_s",
                *_PREPARE_METRICS,
            )
        }
        self._attach_shared_arena(index["prepared"], metrics, hardware)
        layout_started = time.perf_counter()
        if hardware:
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
        else:
            # GPU plans already select this rank's bindings and H2D spans; keep
            # the shared records directly rather than rebuilding a local map.
            layout = index["prepared_records"]
            size = sum(
                layout[entry["name"]]["nbytes"]
                for entry in local_entries
                if entry["name"] in layout
            )
        metrics["host_rank_layout_s"] = time.perf_counter() - layout_started
        staging_bytes = 0
        if hardware:
            self._reserve_rank_arena(size, metrics)
            prepare_started = time.perf_counter()
            records = index["prepared_records"]
            for name, record in layout.items():
                source = records[name]["offset"]
                target, count = record["offset"], record["nbytes"]
                copy_bytes(
                    self.mapping[target : target + count],
                    self.shared_allocation.view[source : source + count],
                )
                staging_bytes += count
            staging_s = time.perf_counter() - prepare_started
        else:
            import torch

            prepare_started = time.perf_counter()
            if self.hardware is not False or self.capacity != index["prepared"]:
                self._close_private()
                self.mapping = (
                    self.shared_allocation.view
                    if self.shared_allocation is not None
                    else None
                )
                self.tensor = (
                    torch.frombuffer(self.mapping, dtype=torch.uint8)
                    if self.mapping is not None
                    else torch.empty(0, dtype=torch.uint8)
                )
                self.hardware, self.capacity = False, index["prepared"]
            staging_s = 0.0
        metrics["host_rank_prepare_call_s"] = time.perf_counter() - prepare_started
        metrics.update(
            host_rank_arena_bytes=size,
            host_rank_capacity_bytes=self.capacity["capacity"],
            host_rank_capacity_generation=self.capacity["generation"],
            host_rank_cpu_workers=self._pool.workers if self._pool is not None else 0,
            host_rank_private_capacity_bytes=self.capacity["capacity"]
            if hardware
            else 0,
            host_rank_staging_copy_s=staging_s,
            host_rank_staging_copy_bytes=staging_bytes,
        )
        timings.update(metrics)
        return HostPreparedSnapshot(self, index, layout)

    def _close_private(self):
        if self.hardware:
            self.tensor = self.mapping = self.capacity = None
        if self.allocation is not None:
            self.allocation.close()
            self.allocation = None

    def _close_shared(self):
        if self.hardware is False:
            self.tensor = self.mapping = self.capacity = None
        if self.shared_allocation is not None:
            self.shared_allocation.close()
            self.shared_allocation = None
        self.shared_capacity = None

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        self._close_private()
        self._close_shared()
        self._index = None


class HostPreparedSnapshot:
    """One publication's local views into backend-retained pinned host storage."""

    def __init__(self, arena, index, records):
        self.arena, self.records = arena, records
        self.directory = arena.directory
        self.ready_state = {
            "token": index["token"],
            "state": "READY",
            "generation": index["encoded"]["generation"],
        }

    def get(self, name):
        record = self.records[name]
        return self.arena.mapping[
            record["offset"] : record["offset"] + record["nbytes"]
        ]

    def mark_reusable(self):
        # The tokenizer resumes only after every original engine rank applied. A
        # stale callback cannot release a newer encoded/prepared publication.
        with (self.directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = orjson.loads((self.directory / "state.json").read_bytes())
            if state == self.ready_state:
                state["state"] = "REUSABLE"
                _write_record(self.directory, "state", state)

    def close(self):
        # Backend retains its mapping after all DE or H2D readers drain.
        self.arena = None
