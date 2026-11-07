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

"""CPU-only builder for one engine host; shared files carry all bulk data.

Launched by absolute file path so the child never imports SGLang or Torch.
The child owns publication locks, shared allocations and READY/release transitions.
"""

import argparse
import fcntl
import hashlib
import json
import mmap
import os
import re
import select
import threading
import time
from array import array
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from multiprocessing.connection import Connection
from operator import itemgetter
from pathlib import Path

import numpy as np
import orjson

if os.environ.get("SGLANG_ENABLE_NVTX_OPERATIONS", "").lower() in {
    "1",
    "true",
    "yes",
    "y",
}:
    from nvtx import annotate
else:

    def annotate(name):
        return lambda function: function


TENSOR_DTYPE = np.dtype(
    [
        (name, "<i8")
        for name in (
            "prepared_offset",
            "prepared_nbytes",
            "frame_start",
            "frame_count",
            "gap_start",
            "gap_count",
            "changed_bytes",
        )
    ]
)


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


def _identity(info):
    # RW mapping metadata timestamps can change on attachment. The feature owns
    # immutable contents after READY; bind the exact retained inode and extent.
    return [info.st_dev, info.st_ino, info.st_size]


# Cold capacity fits the required extent; growth reserves twice the new extent.
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


@annotate("gpu_delta.host.read_verify_worker")
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


@annotate("gpu_delta.host.read_verify")
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


def _gap_values(entries, counts):
    for ordinal, entry in enumerate(entries):
        if entry["encoding"] == "raw_bytes":
            continue
        cursor = 0
        count = 0
        for frame in entry["frames"]:
            start = frame["decoded_offset"]
            if cursor < start:
                yield cursor
                yield start - cursor
                count += 1
            cursor = start + frame["decoded_bytes"]
        if cursor < entry["nbytes"]:
            yield cursor
            yield entry["nbytes"] - cursor
            count += 1
        counts[ordinal] = count


def _metadata_views(mapping, counts):
    tensor_count, frame_count, gap_count = counts
    frames_offset = (tensor_count * TENSOR_DTYPE.itemsize + 15) // 16 * 16
    gaps_offset = frames_offset + 4 * frame_count * 8
    return (
        np.ndarray(tensor_count, dtype=TENSOR_DTYPE, buffer=mapping),
        np.ndarray((4, frame_count), dtype="<i8", buffer=mapping, offset=frames_offset),
        np.ndarray((gap_count, 2), dtype="<i8", buffer=mapping, offset=gaps_offset),
    )


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


@annotate("gpu_delta.host.outer_zstd_worker")
def _decode_zstd_group(jobs, destination, backing, file_spans, ordinals, tensors, pool):
    from zstandard import BufferWithSegments

    metrics = {name: 0 for name in _PREPARE_METRICS}
    segments, sizes, offsets = array("Q"), [], []
    prepared_offsets = tensors["prepared_offset"]
    for entry in jobs:
        output_offset = int(prepared_offsets[ordinals[entry["name"]]])
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
            offsets.append(output_offset + chunk["decoded_offset"])
        metrics["host_rank_outer_zstd_encoded_bytes"] += outer["encoded_bytes"]
        metrics["host_rank_outer_zstd_decoded_bytes"] += outer["decoded_bytes"]
        metrics["host_rank_outer_zstd_tensors"] += 1
        metrics["host_rank_outer_zstd_frames"] += len(outer["frames"])
    metrics["host_rank_outer_zstd_worker_decode_sum_s"] = pool.decode_zstd(
        BufferWithSegments(backing, segments), sizes, offsets, destination
    )
    return metrics


@annotate("gpu_delta.host.copy_worker")
def _copy_group(jobs, destination, backing, file_spans, ordinals, tensors, pool):
    metrics = {name: 0 for name in _PREPARE_METRICS}
    prepared_offsets = tensors["prepared_offset"]
    for entry in jobs:
        outer = entry["outer"]
        length = outer["encoded_bytes"]
        started = time.perf_counter()
        file_start, file_size = file_spans[outer["file"]]
        start = outer["encoded_offset"]
        if not 0 <= start <= start + length <= file_size:
            raise ValueError("delta payload span escapes declared file")
        start += file_start
        offset = int(prepared_offsets[ordinals[entry["name"]]])
        copy_bytes(
            destination[offset : offset + length],
            backing[start : start + length],
        )
        metrics["host_rank_encoded_copy_worker_sum_s"] += time.perf_counter() - started
        metrics["host_rank_encoded_copy_bytes"] += length
        metrics["host_rank_encoded_copy_tensors"] += 1
    return metrics


@annotate("gpu_delta.host.prepare_arena")
def _prepare_arena(
    destination, tensors, ordinals, backing, file_spans, entries, pool, metrics, codec
):
    started = time.perf_counter()
    jobs, futures, error = [], [], None
    prepared_offsets = tensors["prepared_offset"]
    prepared_sizes = tensors["prepared_nbytes"]
    if not codec.endswith("-zstd"):
        worker, phase = _copy_group, "host_rank_encoded_copy_s"
    else:
        worker, phase = _decode_zstd_group, "host_rank_outer_zstd_decode_s"
    try:
        for entry in entries:
            ordinal = ordinals[entry["name"]]
            count = int(prepared_sizes[ordinal])
            if not count:
                continue
            if entry["encoding"] == "raw_bytes":
                raw = entry["raw"]
                offset = int(prepared_offsets[ordinal])
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
                    ordinals,
                    tensors,
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


def new_metrics():
    return {
        name: 0
        for name in (
            "host_manifest_read_parse_s",
            "host_rank_metadata_release_s",
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
            "host_prepared_cache_layout_s",
            "host_prepared_cache_allocation_s",
            "host_prepared_cache_allocation_calls",
            "host_prepared_cache_allocation_bytes",
            "host_metadata_cache_allocation_s",
            "host_metadata_cache_allocation_calls",
            "host_metadata_cache_allocation_bytes",
            "host_rank_allocation_s",
            "host_rank_allocation_calls",
            "host_rank_allocation_bytes",
            "host_rank_mapping_reused",
            "host_rank_outer_zstd_decode_s",
            "host_rank_encoded_copy_s",
            "host_cpu_worker_spawn_s",
            "host_cpu_worker_wait_s",
            *_PREPARE_METRICS,
        )
    }


def publication_token(expected):
    return hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()


def _write_record(directory, name, record):
    temporary = directory / (name + ".pending")
    temporary.write_bytes(orjson.dumps(record))
    temporary.replace(directory / (name + ".json"))


class HostBuilder:
    """Persistent CPU state; no device, model, CUDA registration or rank objects."""

    def __init__(self, directory, workers, backends):
        self.directory = Path(directory)
        self.backends = backends
        self._ordinals = self._shared_order = None
        self._index = None
        self._metadata_mapping = self._metadata_capacity = None
        self._prepared_capacity = None
        self._prepared_view = memoryview(b"")
        self.pool = HostPayloadPool(workers)
        stat = Path("/proc/self/stat").read_text()
        self.identity = {
            "pid": os.getpid(),
            "start_ticks": int(stat[stat.rfind(")") + 2 :].split()[19]),
            "workers": workers,
        }

    def prepare(self, expected, namespace):
        token = publication_token(expected)
        waiting = time.perf_counter()
        with (self.directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            wait_s = time.perf_counter() - waiting
            state_path = self.directory / "state.json"
            state = (
                orjson.loads(state_path.read_bytes()) if state_path.exists() else None
            )
            if self._index is None and (self.directory / "index.json").exists():
                self._index = orjson.loads((self.directory / "index.json").read_bytes())
            previous = self._index
            if previous and previous["namespace"] != namespace:
                raise ValueError("encoded cache namespace differs")
            if previous and previous["token"] == token:
                if (
                    state is None
                    or state["state"] != "READY"
                    or state["token"] != token
                ):
                    raise ValueError(
                        "encoded publication is failed or already released"
                    )
                result = {"index": previous, "metrics": new_metrics()}
                result["metrics"].update(
                    host_encoded_cache_reused=1, host_prepared_cache_reused=1
                )
            else:
                if previous and (
                    state is None
                    or state["state"] != "REUSABLE"
                    or state["token"] != previous["token"]
                    or previous["publication"]["target_version"]
                    != expected["base_version"]
                    or previous["publication"]["plan_digest"] != expected["plan_digest"]
                ):
                    raise ValueError(
                        "encoded cache requires prior engine APPLIED release before overwrite"
                    )
                self._index = {
                    "namespace": namespace,
                    "publication": expected,
                    "token": token,
                }
                _write_record(self.directory, "index", self._index)
                _write_record(
                    self.directory, "state", {"token": token, "state": "BUILDING"}
                )
                result = self.build(self._index, previous)
                self._index = index = result["index"]
                _write_record(self.directory, "index", index)
                _write_record(
                    self.directory,
                    "state",
                    {
                        "token": token,
                        "state": "READY",
                        "generation": index["encoded"]["generation"],
                    },
                )
                if previous:
                    for kind in ("encoded", "prepared", "metadata"):
                        if previous[kind]["file"] != index[kind]["file"]:
                            (self.directory / previous[kind]["file"]).unlink()
            result["metrics"].update(
                host_encoded_cache_wait_s=wait_s, host_prepared_cache_wait_s=wait_s
            )
            return result

    def release(self, ready_state):
        with (self.directory / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = orjson.loads((self.directory / "state.json").read_bytes())
            if state == ready_state:
                _write_record(self.directory, "state", state | {"state": "REUSABLE"})

    def _attach_prepared(self, allocation):
        if self._prepared_capacity != allocation:
            self._prepared_view.release()
            if allocation["capacity"]:
                with (self.directory / allocation["file"]).open("r+b") as source:
                    if _identity(os.fstat(source.fileno())) != allocation["identity"]:
                        raise ValueError("prepared cache inode or extent changed")
                    mapping = mmap.mmap(source.fileno(), allocation["capacity"])
                self._prepared_view = memoryview(mapping)
            else:
                self._prepared_view = memoryview(b"")
            self._prepared_capacity = allocation

    def _attach_metadata(self, allocation):
        if self._metadata_capacity != allocation:
            with (self.directory / allocation["file"]).open("r+b") as source:
                if _identity(os.fstat(source.fileno())) != allocation["identity"]:
                    raise ValueError("metadata cache inode or extent changed")
                self._metadata_mapping = mmap.mmap(
                    source.fileno(), allocation["capacity"]
                )
            self._metadata_capacity = allocation

    def _catalog(self, manifest, previous):
        if previous is None:
            fields = (
                "name",
                "dtype",
                "shape",
                "nbytes",
                "encoding",
                "byte_order",
                "views",
            )
            catalog = {
                "plan_digest": manifest["plan_digest"],
                "tensors": sorted(
                    [
                        {key: entry[key] for key in fields}
                        for entry in manifest["tensors"]
                    ],
                    key=lambda entry: entry["name"],
                ),
            }
            raw = orjson.dumps(catalog)
            with (self.directory / "catalog.json").open("xb") as target:
                target.write(raw)
            descriptor = {
                "file": "catalog.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        else:
            descriptor = previous["catalog"]
            catalog = None
        if self._ordinals is None:
            if catalog is None:
                catalog = orjson.loads(
                    (self.directory / descriptor["file"]).read_bytes()
                )
            tensors = catalog["tensors"]
            self._ordinals = {entry["name"]: i for i, entry in enumerate(tensors)}
            self._shared_order = np.array(
                sorted(
                    range(len(tensors)),
                    key=lambda i: (
                        tensors[i]["encoding"] == "raw_bytes",
                        _natural_key(tensors[i]["name"]),
                    ),
                ),
                dtype=np.int64,
            )
        return descriptor

    @annotate("gpu_delta.host.tensor_layout")
    def _prepare_metadata(self, manifest, previous, metrics):
        catalog = self._catalog(manifest, previous)
        entries = [None] * len(self._ordinals)
        for entry in manifest["tensors"]:
            entries[self._ordinals[entry["name"]]] = entry
        if len(manifest["tensors"]) != len(entries) or any(
            entry is None for entry in entries
        ):
            raise ValueError("publication canonical tensor set changed")
        frame_counts = np.fromiter((len(e["frames"]) for e in entries), dtype=np.int64)
        gap_counts = np.zeros(len(entries), dtype=np.int64)
        # Exhaust once: a final yield can precede the tensor's count assignment.
        gap_values = np.fromiter(_gap_values(entries, gap_counts), dtype=np.int64)
        counts = (len(entries), int(frame_counts.sum()), len(gap_values) // 2)
        size = (
            (counts[0] * TENSOR_DTYPE.itemsize + 15) // 16 * 16
            + counts[1] * 32
            + counts[2] * 16
        )
        allocation = _reserve_cache(
            self.directory,
            previous["metadata"] if previous else None,
            size,
            metrics,
            "metadata",
        )
        self._attach_metadata(allocation)
        tensors, frames, gaps = _metadata_views(self._metadata_mapping, counts)
        tensors["changed_bytes"] = np.fromiter(
            (e["changed_bytes"] for e in entries), dtype=np.int64
        )
        tensors["prepared_nbytes"] = np.fromiter(
            (
                (e["nbytes"] if e["changed_bytes"] else 0)
                if e["encoding"] == "raw_bytes"
                else (e["outer"]["decoded_bytes"] if e["frames"] else 0)
                for e in entries
            ),
            dtype=np.int64,
        )
        sizes = tensors["prepared_nbytes"][self._shared_order]
        aligned = (sizes + 15) // 16 * 16
        offsets = np.cumsum(aligned) - aligned
        tensors["prepared_offset"][self._shared_order] = offsets
        tensors["prepared_offset"][tensors["prepared_nbytes"] == 0] = -1
        prepared_size = int(np.max((offsets + sizes)[sizes != 0], initial=0))
        tensors["frame_count"], tensors["gap_count"] = frame_counts, gap_counts
        tensors["frame_start"] = np.cumsum(frame_counts) - frame_counts
        tensors["gap_start"] = np.cumsum(gap_counts) - gap_counts
        frames[:] = (
            np.fromiter(
                chain.from_iterable(
                    map(
                        itemgetter(
                            "encoded_offset",
                            "encoded_bytes",
                            "decoded_bytes",
                            "decoded_offset",
                        ),
                        chain.from_iterable(entry["frames"] for entry in entries),
                    )
                ),
                dtype=np.int64,
                count=counts[1] * 4,
            )
            .reshape(-1, 4)
            .T
        )
        gaps[:] = gap_values.reshape(-1, 2)
        return catalog, allocation, counts, entries, tensors, prepared_size, size

    @annotate("gpu_delta.host.prepare_shared")
    def _prepare_shared_arena(
        self, index, previous, backing, file_spans, manifest, metrics
    ):
        layout_started = time.perf_counter()
        catalog, metadata, counts, entries, tensors, size, metadata_size = (
            self._prepare_metadata(manifest, previous, metrics)
        )
        metrics["host_prepared_cache_layout_s"] = time.perf_counter() - layout_started
        prepared = _reserve_cache(
            self.directory,
            previous["prepared"] if previous else None,
            size,
            metrics,
            "prepared",
        )
        self._attach_prepared(prepared)
        _prepare_arena(
            self._prepared_view,
            tensors,
            self._ordinals,
            backing,
            file_spans,
            (entries[ordinal] for ordinal in self._shared_order),
            self.pool,
            metrics,
            index["publication"]["codec"],
        )
        index.update(
            catalog=catalog,
            metadata=metadata,
            metadata_counts=counts,
            metadata_size=metadata_size,
            prepared=prepared,
            prepared_size=size,
            prepared_build_s=time.perf_counter() - layout_started,
        )

    @annotate("gpu_delta.host.build_publication")
    def build(self, index, previous):
        expected = index["publication"]
        publication = Path(expected["manifest_path"])
        metrics = new_metrics()
        parse_started = time.perf_counter()
        content = publication.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected["manifest_sha256"]:
            raise ValueError("immutable delta manifest SHA256 mismatch")
        manifest = orjson.loads(content)
        metrics["host_manifest_read_parse_s"] = time.perf_counter() - parse_started
        validate_codec(manifest)
        if any(
            manifest[key] != expected[key]
            for key in ("base_version", "target_version", "plan_digest")
        ):
            raise ValueError("publication header differs from prepare request")
        index["publication"] = expected | {
            "codec": manifest["codec"],
            "backend": self.backends[manifest["codec"].removesuffix("-zstd")],
        }
        definitions = {}
        for record in manifest["files"]:
            if record["nbytes"] < 0:
                raise ValueError("negative delta payload size")
            if record["sha256"] is None and not expected["skip_payload_hash"]:
                raise ValueError(
                    "payload checksum omitted; request validate_payload_hash=False "
                    "to accept an unchecked publication"
                )
            definitions[record["name"]] = {
                "nbytes": record["nbytes"],
                "sha256": record["sha256"],
            }
        build_started = time.perf_counter()
        encoded = _reserve_cache(
            self.directory,
            previous["encoded"] if previous else None,
            sum(record["nbytes"] for record in definitions.values()),
            metrics,
            "encoded",
        )
        index["encoded"] = encoded
        backing, file_spans = _map_encoded(self.directory, encoded, definitions)
        _read_verify_payloads(
            publication,
            backing,
            file_spans,
            definitions,
            self.pool,
            metrics,
            expected["skip_payload_hash"],
        )
        if not expected["skip_payload_hash"]:
            metrics["host_encoded_cache_hash_bytes"] = sum(
                record["nbytes"] for record in definitions.values()
            )
            metrics["host_encoded_cache_hash_files"] = len(definitions)
        index["build_s"] = time.perf_counter() - build_started
        self._prepare_shared_arena(
            index, previous, backing, file_spans, manifest, metrics
        )
        release_started = time.perf_counter()
        del manifest, content, backing
        metrics["host_rank_metadata_release_s"] = time.perf_counter() - release_started
        metrics["host_prepared_cache_created"] = metrics[
            "host_encoded_cache_created"
        ] = 1
        index["cpu_worker"] = self.identity
        return {"index": index, "metrics": metrics}

    def close(self):
        self.pool.close()
        self._prepared_view.release()
        self._prepared_view = memoryview(b"")
        self._metadata_mapping = None


def _parent_watch(parent_fd):
    select.select([parent_fd], [], [])
    # The exact parent process died, even if another process inherited an IPC fd.
    os._exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-fd", type=int, required=True)
    parser.add_argument("--parent-fd", type=int, required=True)
    args = parser.parse_args()
    threading.Thread(target=_parent_watch, args=(args.parent_fd,), daemon=True).start()
    connection = Connection(args.socket_fd)
    builder = None
    try:
        while True:
            request = orjson.loads(connection.recv_bytes(1 << 20))
            if request is None:
                break
            try:
                if builder is None:
                    builder = HostBuilder(**request["configuration"])
                if request["command"] == "prepare":
                    result = builder.prepare(request["expected"], request["namespace"])
                else:
                    builder.release(request["ready_state"])
                    result = {}
                result.update(token=request["token"], status="OK")
            except BaseException as error:
                connection.send_bytes(
                    orjson.dumps(
                        {
                            "token": request["token"],
                            "status": "FAILED",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                )
                raise
            connection.send_bytes(orjson.dumps(result))
    except EOFError:
        pass
    finally:
        if builder is not None:
            builder.close()
        connection.close()
        os.close(args.parent_fd)


if __name__ == "__main__":
    main()
