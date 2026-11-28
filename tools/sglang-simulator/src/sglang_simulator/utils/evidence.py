"""Optional lossless evidence codecs, independent of prediction and progress."""

import gzip
import io
import os
from contextlib import contextmanager


def evidence_level():
    """Explicitly separate serving metrics from verbose per-step evidence."""
    value = os.getenv("SGLANG_SIMULATOR_EVIDENCE_LEVEL", "full")
    if value not in {"full", "metrics"}:
        raise ValueError("SGLANG_SIMULATOR_EVIDENCE_LEVEL must be full or metrics")
    return value


def metrics_cpu_detail_enabled():
    """Keep legacy evidence work unless an explicit metrics-only opt-in disables it."""
    value = os.getenv("SGLANG_SIMULATOR_METRICS_CPU_DETAIL", "on")
    if value not in {"on", "off"}:
        raise ValueError("SGLANG_SIMULATOR_METRICS_CPU_DETAIL must be on or off")
    if value == "off" and evidence_level() != "metrics":
        raise ValueError("metrics CPU detail off requires metrics evidence retention")
    return value == "on"


class ZstdCodec:
    """Small reusable contexts; each encoded row is a complete checksummed frame."""

    def __init__(self):
        try:
            import zstandard
        except ImportError as error:
            raise ImportError(
                "zstd evidence requires the sglang-simulator[zstd] extra"
            ) from error
        self._module = zstandard
        self.compressor = zstandard.ZstdCompressor(level=3, write_checksum=True)
        self._decompressor = zstandard.ZstdDecompressor()

    def compress(self, raw):
        try:
            return self.compressor.compress(raw)
        except self._module.ZstdError as error:
            raise ValueError("zstd evidence compression failed") from error

    def decompress(self, encoded):
        try:
            return self._decompressor.decompress(encoded, allow_extra_data=False)
        except self._module.ZstdError as error:
            raise ValueError("invalid or incomplete zstd evidence frame") from error


@contextmanager
def open_evidence_jsonl(path, compression):
    """Write original JSONL text, completing the output frame on successful close."""
    if compression == "none":
        with open(path, "w", encoding="utf-8", newline="\n") as output:
            yield output
    elif compression == "gzip":
        with gzip.open(
            path, "wt", encoding="utf-8", compresslevel=1, newline="\n"
        ) as output:
            yield output
    elif compression == "zstd":
        codec = ZstdCodec()
        with open(path, "wb") as raw:
            writer = codec.compressor.stream_writer(raw, closefd=False)
            output = io.TextIOWrapper(writer, encoding="utf-8", newline="\n")
            try:
                yield output
            finally:
                output.close()
    else:
        raise ValueError("evidence compression must be none or gzip or zstd")
