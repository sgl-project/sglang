"""Keep verbose native iteration evidence off the scheduler's Python heap."""

import json
import os
import struct
import tempfile
import zlib
from pathlib import Path


class IterationStats:
    """Disk-backed JSON rows with small, ordered summaries for metric reduction.

    The exclusive spool lives in the configured output directory, rather than
    a possibly memory-backed temporary directory. It survives write/export
    errors and is removed only by an explicit successful profile reset.
    Export preserves the existing ``json.dumps(row) + "\n"`` representation.
    """

    def __init__(self, output_directory):
        self._output_directory = output_directory
        self._stream = None
        self.path = None
        self._offsets = []
        self._metrics = []
        self._failed = False

    def __len__(self):
        return len(self._metrics)

    def append(self, row):
        if self._failed:
            raise RuntimeError("iteration evidence spool previously failed")
        try:
            raw = (json.dumps(row) + "\n").encode("utf-8")
            data = zlib.compress(raw, level=1)
            if self._stream is None:
                directory = Path(self._output_directory())
                directory.mkdir(parents=True, exist_ok=True)
                descriptor, name = tempfile.mkstemp(
                    prefix=".iteration-", suffix=".spool", dir=directory
                )
                self.path = Path(name)
                self._stream = os.fdopen(descriptor, "w+b")
            offset = self._stream.seek(0, os.SEEK_END)
            record = struct.pack("!Q", len(data)) + data
            if self._stream.write(record) != len(record):
                raise OSError("short iteration evidence spool write")
            self._offsets.append(offset)
            self._metrics.append(
                {
                    key: row.get(key, 0)
                    for key in ("forward_latency", "l2_load_latency", "cpu_overhead")
                }
            )
        except BaseException:
            self._failed = True
            raise

    def metric_rows(self):
        return self._metrics

    def _read(self, index):
        if self._failed:
            raise RuntimeError("iteration evidence spool previously failed")
        self._stream.seek(self._offsets[index])
        header = self._stream.read(8)
        if len(header) != 8:
            raise OSError("truncated iteration evidence spool header")
        (size,) = struct.unpack("!Q", header)
        data = self._stream.read(size)
        if len(data) != size:
            raise OSError("truncated iteration evidence spool record")
        return zlib.decompress(data).decode("utf-8")

    def __getitem__(self, index):
        return json.loads(self._read(index))

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def write_jsonl(self, destination):
        """Stream complete original rows; retain the spool if export fails."""
        if self._failed:
            raise RuntimeError("iteration evidence spool previously failed")
        for index in range(len(self)):
            raw = self._read(index)
            if destination.write(raw) != len(raw):
                raise OSError("short iteration evidence export write")

    def clear(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self.path is not None:
            self.path.unlink()
        self.path = None
        self._offsets.clear()
        self._metrics.clear()
        self._failed = False
