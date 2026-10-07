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

"""Miles payload bounds and parallel CPU decoding into rank-owned host arenas."""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def validate_codec(manifest):
    """Admit the authenticated publication codec before examining any tensor payload."""
    if (
        manifest.get("codec") not in {"snappy-zstd", "lz4-zstd", "lz4"}
        or type(manifest.get("frame_bytes")) is not int
        or not 0 < manifest["frame_bytes"] <= 4 << 20
    ):
        raise ValueError(
            "GPU delta requires a snappy-zstd, lz4-zstd or lz4 codec "
            "and an integer frame size in (0, 4 MiB]"
        )


def validate_payload_ranges(entries, files, frame_bytes, codec):
    """Bound Miles payload spans before allocation and native decoding.

    Miles owns the schema, packing and frame construction. Check the memory
    extents here; nvCOMP validates device geometry and decode status later.
    Immutable source ranges may overlap because they are only read.
    """
    for entry in entries:
        if entry["encoding"] == "raw_bytes":
            if entry["changed_bytes"]:
                raw = entry["raw"]
                start, count = raw["encoded_offset"], raw["encoded_bytes"]
                if (
                    count != entry["nbytes"]
                    or not 0 <= start <= files[raw["file"]] - count
                ):
                    raise ValueError(
                        "direct tensor exceeds immutable payload or is incomplete"
                    )
            continue
        if not entry["frames"]:
            continue
        outer = entry["outer"]
        start, count, size = (
            outer["encoded_offset"],
            outer["encoded_bytes"],
            outer["decoded_bytes"],
        )
        if count <= 0 or size <= 0 or not 0 <= start <= files[outer["file"]] - count:
            raise ValueError("outer descriptor exceeds immutable payload")
        if codec == "lz4":
            if count != size:
                raise ValueError("plain LZ4 requires an exact unwrapped inner arena")
        else:
            _validate_outer_frames(outer)
        decoded_end = 0
        for frame in entry["frames"]:
            offset, encoded = frame["encoded_offset"], frame["encoded_bytes"]
            decoded_offset, decoded = frame["decoded_offset"], frame["decoded_bytes"]
            if (
                offset < 0
                or offset % 16
                or encoded <= 0
                or offset + encoded > size
                or not 0 < decoded <= frame_bytes
                or decoded_offset < decoded_end
                or decoded_offset + decoded > entry["nbytes"]
            ):
                raise ValueError("invalid relative inner compressed frame")
            decoded_end = decoded_offset + decoded


def _validate_outer_frames(outer):
    """Bound source reads and fully initialize the inner arena before DE reads it."""
    decoded_end = 0
    for chunk in outer["frames"]:
        start, count = chunk["encoded_offset"], chunk["encoded_bytes"]
        offset, size = chunk["decoded_offset"], chunk["decoded_bytes"]
        if (
            count <= 0
            or not 0 <= start <= outer["encoded_bytes"] - count
            or offset != decoded_end
            or size <= 0
            or offset + size > outer["decoded_bytes"]
        ):
            raise ValueError("invalid GPU outer Zstd chunk range")
        decoded_end = offset + size
    if decoded_end != outer["decoded_bytes"]:
        raise ValueError("GPU outer Zstd chunks do not exactly cover their tensor")


def configured_cpu_workers():
    value = int(os.environ.get("GPU_DELTA_CPU_WORKERS", "32"))
    if not 1 <= value <= 32:
        raise ValueError("GPU_DELTA_CPU_WORKERS must be between 1 and 32")
    return value


class HostPayloadPool:
    """Reusable bounded CPU workers; no CUDA work or shared decoder contexts."""

    def __init__(self, workers):
        self.workers = workers
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="gpu-delta-payload"
        )
        self.local = threading.local()

    def decode_zstd(self, payload, chunks, destination):
        import zstandard as zstd

        if not hasattr(self.local, "decoder"):
            self.local.decoder = zstd.ZstdDecompressor()
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
        return time.perf_counter() - started

    def close(self):
        self.executor.shutdown(wait=True)
