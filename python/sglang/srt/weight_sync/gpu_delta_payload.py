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

import re
import time


def outer_zstd_profile(manifest):
    """Select the authenticated envelope contract, independent of local env vars."""
    version, profile = (
        manifest.get("protocol_version"),
        manifest.get("codec_profile", ""),
    )
    if (
        type(version) is int
        and version == 2
        and isinstance(profile, str)
        and re.fullmatch(r"zstd-independent-(64kib|1mib|2mib)-v1", profile)
    ):
        return None
    if (
        type(version) is int
        and version == 3
        and isinstance(profile, str)
        and re.fullmatch(r"snappy-independent-(64kib|1mib|2mib)-zstd-v1", profile)
    ):
        return "cpu"
    if (
        type(version) is int
        and version == 4
        and isinstance(profile, str)
        and re.fullmatch(r"snappy-independent-(64kib|1mib|2mib)-gpu-zstd-v1", profile)
    ):
        return "gpu"
    raise ValueError(
        "unsupported direct-delta protocol/profile: Snappy requires protocol 3 "
        "CPU or protocol 4 GPU Zstd envelopes; plain Zstd uses protocol 2"
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


def validate_plain_entries(entries, files):
    """Check raw ranges against the absolute frame ranges of plain Zstd."""
    ranges = {name: [] for name in files}
    for entry in entries:
        if entry["encoding"] == "raw_bytes":
            payload_range = raw_payload_range(entry, files)
            if payload_range is not None:
                name, start, stop = payload_range
                ranges[name].append((start, stop))
            continue
        if entry["encoding"] != "xor_bytes" or "raw" in entry or "outer" in entry:
            raise ValueError("compressed tensor has an unexpected payload descriptor")
        for frame in entry["frames"]:
            name, start, size = (
                frame[k] for k in ("file", "encoded_offset", "encoded_bytes")
            )
            if (
                name not in files
                or any(type(v) is not int for v in (start, size))
                or start < 0
                or size <= 0
                or start + size > files[name]
            ):
                raise ValueError("encoded frame exceeds immutable payload")
            ranges[name].append((start, start + size))
    _reject_overlapping_ranges(ranges)


def validate_outer_entries(entries, files, *, gpu=False):
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
        fields = {"codec", "file", "encoded_offset", "encoded_bytes", "decoded_bytes"}
        if gpu:
            fields.add("frames")
        if not isinstance(outer, dict) or set(outer) != fields:
            raise ValueError("invalid outer Zstd descriptor")
        name = outer["file"]
        start, count, size = (
            outer[k] for k in ("encoded_offset", "encoded_bytes", "decoded_bytes")
        )
        if (
            outer["codec"] != "zstd"
            or name not in files
            or any(type(v) is not int for v in (start, count, size))
            or start < 0
            or count <= 0
            or size <= 0
            or start + count > files[name]
        ):
            raise ValueError("outer Zstd descriptor exceeds immutable payload")
        if gpu:
            _validate_gpu_outer_frames(outer)
        end = decoded_end = 0
        for frame in frames:
            offset, encoded, decoded_offset, decoded = (
                frame[k]
                for k in (
                    "encoded_offset",
                    "encoded_bytes",
                    "decoded_offset",
                    "decoded_bytes",
                )
            )
            if (
                frame["file"] != name
                or any(
                    type(v) is not int
                    for v in (offset, encoded, decoded_offset, decoded)
                )
                or offset != (end + 15) // 16 * 16
                or not 0 < encoded <= decoded <= 1 << 20
                or decoded_offset < decoded_end
                or decoded_offset + decoded > entry["nbytes"]
                or frame["codec"] not in {"snappy", "none"}
                or (frame["codec"] == "none" and encoded != decoded)
            ):
                raise ValueError("invalid relative inner Snappy frame")
            end, decoded_end = offset + encoded, decoded_offset + decoded
        if end != size:
            raise ValueError("outer decoded length differs from the inner tensor span")
        ranges[name].append((start, start + count))
    _reject_overlapping_ranges(ranges)


def _validate_gpu_outer_frames(outer):
    """GPU Zstd chunks exactly cover one natural tensor's inner Snappy arena."""
    chunks = outer["frames"]
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("GPU outer Zstd requires independent chunks")
    encoded_end = decoded_end = 0
    for chunk in chunks:
        if not isinstance(chunk, dict) or set(chunk) != {
            "encoded_offset",
            "encoded_bytes",
            "decoded_offset",
            "decoded_bytes",
        }:
            raise ValueError("invalid GPU outer Zstd chunk")
        start, count, offset, size = (
            chunk[key]
            for key in (
                "encoded_offset",
                "encoded_bytes",
                "decoded_offset",
                "decoded_bytes",
            )
        )
        if (
            any(type(value) is not int for value in (start, count, offset, size))
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


def validate_zstd_frame(payload, expected_size, *, gpu_chunk=False):
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
        parameters.content_size
        not in (
            {expected_size, zstd.CONTENTSIZE_UNKNOWN} if gpu_chunk else {expected_size}
        )
        or parameters.dict_id != 0
        or parameters.window_size > ((1 << 20) if gpu_chunk else expected_size)
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


class OuterZstdReader:
    """One preparation worker, one context, one final pinned copy per tensor."""

    def __init__(self, files, allocate, timings):
        import zstandard as zstd

        self.files, self.allocate, self.timings = files, allocate, timings
        self.decoder = zstd.ZstdDecompressor()
        self.cache = {}
        for key in (
            "validate_s",
            "pin_allocate_s",
            "decode_s",
            "encoded_bytes",
            "decoded_bytes",
            "tensors",
            "frames",
        ):
            timings["host_outer_zstd_" + key] = 0

    def get(self, entry):
        name = entry["name"]
        if name in self.cache:
            return self.cache[name]
        outer = entry["outer"]
        start, count, size = (
            outer[k] for k in ("encoded_offset", "encoded_bytes", "decoded_bytes")
        )
        payload = memoryview(self.files[outer["file"]])[start : start + count]
        chunks = outer.get("frames")
        if chunks is None:
            chunks = [
                dict(
                    encoded_offset=0,
                    encoded_bytes=count,
                    decoded_offset=0,
                    decoded_bytes=size,
                )
            ]
        started = time.perf_counter()
        for chunk in chunks:
            offset, length = chunk["encoded_offset"], chunk["encoded_bytes"]
            validate_zstd_frame(
                payload[offset : offset + length],
                chunk["decoded_bytes"],
                gpu_chunk="frames" in outer,
            )
        self.timings["host_outer_zstd_validate_s"] += time.perf_counter() - started
        started = time.perf_counter()
        owner, destination = self.allocate(size)
        self.timings["host_outer_zstd_pin_allocate_s"] += time.perf_counter() - started
        started = time.perf_counter()
        for chunk in chunks:
            offset, length = chunk["encoded_offset"], chunk["encoded_bytes"]
            position, stop = (
                chunk["decoded_offset"],
                chunk["decoded_offset"] + chunk["decoded_bytes"],
            )
            with self.decoder.stream_reader(
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
        self.timings["host_outer_zstd_decode_s"] += time.perf_counter() - started
        self.timings["host_outer_zstd_encoded_bytes"] += len(payload)
        self.timings["host_outer_zstd_decoded_bytes"] += size
        self.timings["host_outer_zstd_tensors"] += 1
        self.timings["host_outer_zstd_frames"] += len(chunks)
        self.cache[name] = owner
        return owner
