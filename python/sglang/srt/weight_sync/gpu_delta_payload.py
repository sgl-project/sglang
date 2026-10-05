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

"""Immutable delta transport validation; never reads or mutates weights."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def configured_codec():
    """Freeze the launch constraint when the scheduler admits a delta backend."""
    codec = os.environ.get("GPU_DELTA_CODEC", "snappy-zstd")
    if codec not in {"snappy-zstd", "lz4-zstd"}:
        raise ValueError("GPU_DELTA_CODEC must be snappy-zstd or lz4-zstd")
    return codec


def validate_codec(manifest, expected):
    """Authenticate the negotiated wire codec before examining any tensor payload."""
    if (
        type(manifest.get("protocol_version")) is not int
        or manifest["protocol_version"] != 4
        or manifest.get("codec") != expected
        or expected not in {"snappy-zstd", "lz4-zstd"}
        or "codec_profile" in manifest
        or type(manifest.get("frame_bytes")) is not int
        or manifest["frame_bytes"] not in {1 << 16, 1 << 20}
    ):
        raise ValueError(
            "GPU delta requires protocol 4 with the negotiated snappy-zstd or lz4-zstd codec"
        )


def raw_payload_range(entry, files):
    """Validate one complete, uncompressed scalar/vector target."""
    raw = entry.get("raw")
    size, changed = entry.get("nbytes"), entry.get("changed_bytes")
    if (
        entry.get("encoding") != "raw_bytes"
        or entry.get("frames") != []
        or "outer" in entry
        or type(size) is not int
        or type(changed) is not int
        or not 0 <= changed <= size
    ):
        raise ValueError("invalid direct tensor payload")
    if changed == 0:
        if "raw" in entry:
            raise ValueError("unchanged direct tensor must omit its payload")
        return None
    if not isinstance(raw, dict) or set(raw) != {
        "file",
        "encoded_offset",
        "encoded_bytes",
    }:
        raise ValueError("invalid direct tensor payload")
    name, start, count = raw["file"], raw["encoded_offset"], raw["encoded_bytes"]
    if (
        name not in files
        or any(type(v) is not int for v in (start, count))
        or start < 0
        or count != size
        or start + count > files[name]
    ):
        raise ValueError("direct tensor exceeds immutable payload or is incomplete")
    return name, start, start + count


def _reject_overlapping_ranges(ranges):
    for intervals in ranges.values():
        end = 0
        for start, stop in sorted(intervals):
            if start < end:
                raise ValueError("overlapping immutable tensor payloads")
            end = stop


def validate_outer_entries(entries, files, frame_bytes):
    """Bound reconstructed spans before allocation, including foreign EP tensors."""
    ranges = {name: [] for name in files}
    for entry in entries:
        if entry["encoding"] == "raw_bytes":
            payload_range = raw_payload_range(entry, files)
            if payload_range is not None:
                name, start, stop = payload_range
                ranges[name].append((start, stop))
            continue
        if entry["encoding"] != "xor_bytes" or "raw" in entry:
            raise ValueError("compressed tensor has an unexpected raw descriptor")
        frames, outer = entry["frames"], entry.get("outer")
        if not frames:
            if "outer" in entry:
                raise ValueError("empty tensor must omit the outer envelope")
            continue
        name = outer["file"]
        start = outer["encoded_offset"]
        count = outer["encoded_bytes"]
        size = outer["decoded_bytes"]
        if (
            name not in files
            or type(start) is not int
            or type(count) is not int
            or type(size) is not int
            or start < 0
            or count <= 0
            or size <= 0
            or start + count > files[name]
        ):
            raise ValueError("outer Zstd descriptor exceeds immutable payload")
        _validate_outer_frames(outer)
        end = decoded_end = 0
        for frame in frames:
            offset = frame["encoded_offset"]
            encoded = frame["encoded_bytes"]
            decoded_offset = frame["decoded_offset"]
            decoded = frame["decoded_bytes"]
            if (
                type(offset) is not int
                or type(encoded) is not int
                or type(decoded_offset) is not int
                or type(decoded) is not int
                or offset != (end + 15) // 16 * 16
                or not 0 < decoded <= 1 << 20
                or not 0 < encoded <= 32 + decoded + decoded // 6
                or decoded_offset % frame_bytes
                or decoded != min(frame_bytes, entry["nbytes"] - decoded_offset)
                or decoded_offset < decoded_end
                or decoded_offset + decoded > entry["nbytes"]
            ):
                raise ValueError("invalid relative inner compressed frame")
            end, decoded_end = offset + encoded, decoded_offset + decoded
        if end != size:
            raise ValueError("outer decoded length differs from the inner tensor span")
        ranges[name].append((start, start + count))
    _reject_overlapping_ranges(ranges)


def _validate_outer_frames(outer):
    """GPU Zstd chunks exactly cover one natural tensor's inner compressed arena."""
    chunks = outer["frames"]
    if not chunks:
        raise ValueError("GPU outer Zstd requires independent chunks")
    encoded_end = decoded_end = 0
    for chunk in chunks:
        start = chunk["encoded_offset"]
        count = chunk["encoded_bytes"]
        offset = chunk["decoded_offset"]
        size = chunk["decoded_bytes"]
        if (
            type(start) is not int
            or type(count) is not int
            or type(offset) is not int
            or type(size) is not int
            or start != (encoded_end + 15) // 16 * 16
            or count <= 0
            or offset != decoded_end
            or offset % (1 << 20)
            or size != min(1 << 20, outer["decoded_bytes"] - offset)
            or size <= 0
        ):
            raise ValueError("invalid GPU outer Zstd chunk range")
        encoded_end, decoded_end = start + count, offset + size
    if (encoded_end, decoded_end) != (outer["encoded_bytes"], outer["decoded_bytes"]):
        raise ValueError("GPU outer Zstd chunks do not exactly cover their tensor")


def validate_zstd_frame(payload, expected_size):
    """Require one complete, bounded frame before streaming into pinned memory.

    Stream readers may return EOF on a truncated frame. The standard Zstd block
    envelope (3-byte header, raw/compressed size or one RLE byte) independently
    checks its exact encoded extent, including the optional 4-byte checksum.
    The decoder validates block contents; no full decoded temporary is created.
    """
    import zstandard as zstd

    if len(payload) < 4 or bytes(payload[:4]) != b"\x28\xb5\x2f\xfd":
        raise ValueError("outer payload must contain one standard Zstd frame")
    parameters = zstd.get_frame_parameters(payload)
    if (
        parameters.content_size not in {expected_size, zstd.CONTENTSIZE_UNKNOWN}
        or parameters.dict_id != 0
        or parameters.window_size > (1 << 20)
    ):
        raise ValueError("outer Zstd content size/window/dictionary mismatch")
    position = zstd.frame_header_size(payload)
    while True:
        if position + 3 > len(payload):
            raise ValueError("truncated outer Zstd block header")
        header = int.from_bytes(payload[position : position + 3], "little")
        position += 3
        last, kind, size = header & 1, (header >> 1) & 3, header >> 3
        if kind == 3:
            raise ValueError("reserved outer Zstd block type")
        position += 1 if kind == 1 else size
        if position > len(payload):
            raise ValueError("truncated outer Zstd block")
        if last:
            break
    position += 4 if parameters.has_checksum else 0
    if position != len(payload):
        raise ValueError("outer Zstd frame has trailing or truncated bytes")


def configured_cpu_workers():
    value = int(os.environ.get("GPU_DELTA_CPU_WORKERS", "32"))
    if not 1 <= value <= 32:
        raise ValueError("GPU_DELTA_CPU_WORKERS must be between 1 and 32")
    return value


class OuterZstdPool:
    """Reusable bounded CPU workers; no CUDA work or shared decoder contexts."""

    def __init__(self, workers):
        self.workers = workers
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="gpu-delta-zstd"
        )
        self.hash_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gpu-delta-sha256"
        )
        self.local = threading.local()

    def decode(self, payload, chunks, destination):
        import zstandard as zstd

        if not hasattr(self.local, "decoder"):
            self.local.decoder = zstd.ZstdDecompressor()
        started = time.perf_counter()
        for chunk in chunks:
            offset, length = chunk["encoded_offset"], chunk["encoded_bytes"]
            validate_zstd_frame(
                payload[offset : offset + length], chunk["decoded_bytes"]
            )
        validation_s = time.perf_counter() - started
        started = time.perf_counter()
        for chunk in chunks:
            offset, length = chunk["encoded_offset"], chunk["encoded_bytes"]
            position, stop = (
                chunk["decoded_offset"],
                chunk["decoded_offset"] + chunk["decoded_bytes"],
            )
            with self.local.decoder.stream_reader(
                payload[offset : offset + length], read_across_frames=False
            ) as reader:
                while position < stop:
                    count = reader.readinto(destination[position:stop])
                    if not count:
                        raise ValueError("truncated outer Zstd decode")
                    position += count
                if reader.read(1):
                    raise ValueError(
                        "outer Zstd output exceeds its declared tensor span"
                    )
        return validation_s, time.perf_counter() - started

    def close(self):
        self.executor.shutdown(wait=True)
        self.hash_executor.shutdown(wait=True)
