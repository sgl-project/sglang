"""NVMe-backed Qwen4 PLE embeddings.

Qwen4's PLE table is a large, row-sharded FP8 matrix. This module parses the
safetensors headers without loading the tensors, reads only selected rows, and
overlaps those reads with the decoder layer immediately before PLE.
"""

from __future__ import annotations

import ctypes
import errno
import json
import logging
import math
import mmap
import os
import re
import struct
import time
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import msgspec
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)

logger = logging.getLogger(__name__)

_PLE_SHARD_PATTERN = re.compile(
    r"^(?P<prefix>.+\.ngram_embedding)\.shard_(?P<index>\d+)\.weight$"
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_MAX_HEADER_BYTES = 128 * 1024 * 1024


class TensorRecord(msgspec.Struct, frozen=True):
    path: Path
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    @property
    def itemsize(self) -> int:
        return _DTYPE_BYTES[self.dtype]

    @property
    def row_bytes(self) -> int:
        if len(self.shape) != 2:
            raise ValueError(f"{self.name} is not a matrix: {self.shape}")
        return self.shape[1] * self.itemsize


class RowLocation(msgspec.Struct, frozen=True):
    path: Path
    offset: int
    nbytes: int


class PLEShard(msgspec.Struct, frozen=True):
    index: int
    row_start: int
    row_end: int
    tensor: TensorRecord


def _read_safetensors_header(path: Path) -> dict[str, TensorRecord]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"truncated safetensors length in {path}")
        (header_length,) = struct.unpack("<Q", length_bytes)
        if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
            raise ValueError(
                f"invalid safetensors header length {header_length} in {path}"
            )
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"truncated safetensors header in {path}")

    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid safetensors JSON in {path}") from error

    data_start = 8 + header_length
    records = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = str(metadata["dtype"])
            shape = tuple(int(value) for value in metadata["shape"])
            relative_start, relative_end = (
                int(value) for value in metadata["data_offsets"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid metadata for tensor {name!r} in {path}"
            ) from error
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {dtype!r} for {name!r}")
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"negative tensor shape for {name!r}: {shape}")
        if relative_start < 0 or relative_end < relative_start:
            raise ValueError(f"invalid data offsets for {name!r}")
        expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        actual_bytes = relative_end - relative_start
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"size mismatch for {name!r}: {actual_bytes} != {expected_bytes}"
            )
        records[name] = TensorRecord(
            path=path,
            name=name,
            dtype=dtype,
            shape=shape,
            offset=data_start + relative_start,
            nbytes=actual_bytes,
        )
    return records


class PLEManifest(msgspec.Struct, frozen=True):
    prefix: str
    dtype: str
    embedding_dim: int
    total_rows: int
    shard_size: int
    shards: tuple[PLEShard, ...]

    @classmethod
    def from_snapshot(
        cls, snapshot: str | Path, *, expected_shards: int | None = None
    ) -> PLEManifest:
        snapshot_path = Path(snapshot)
        index_path = snapshot_path / "model.safetensors.index.json"
        with index_path.open() as handle:
            weight_map = json.load(handle)["weight_map"]

        matched = []
        for name, filename in weight_map.items():
            match = _PLE_SHARD_PATTERN.match(name)
            if match:
                matched.append(
                    (name, match.group("prefix"), int(match.group("index")), filename)
                )
        if not matched:
            raise ValueError(f"no sharded PLE embedding found in {index_path}")

        prefixes = {prefix for _, prefix, _, _ in matched}
        if len(prefixes) != 1:
            raise ValueError(f"expected one PLE table, found {sorted(prefixes)}")
        prefix = prefixes.pop()
        matched.sort(key=lambda item: item[2])
        indices = [index for _, _, index, _ in matched]
        if indices != list(range(len(indices))):
            raise ValueError(f"PLE shard indices are not contiguous: {indices}")
        if expected_shards is not None and len(matched) != expected_shards:
            raise ValueError(
                f"expected {expected_shards} PLE shards, found {len(matched)}"
            )

        header_cache: dict[Path, dict[str, TensorRecord]] = {}
        records = []
        for name, _, shard_index, filename in matched:
            path = snapshot_path / filename
            header = header_cache.setdefault(path, _read_safetensors_header(path))
            try:
                record = header[name]
            except KeyError as error:
                raise ValueError(f"{name!r} is absent from {path}") from error
            if len(record.shape) != 2:
                raise ValueError(f"PLE shard {name!r} is not a matrix")
            records.append((shard_index, record))

        dtypes = {record.dtype for _, record in records}
        dimensions = {record.shape[1] for _, record in records}
        if len(dtypes) != 1 or len(dimensions) != 1:
            raise ValueError(
                f"inconsistent PLE shards: dtypes={dtypes}, dimensions={dimensions}"
            )
        shard_size = max(record.shape[0] for _, record in records)
        shards = tuple(
            PLEShard(
                index=shard_index,
                row_start=shard_index * shard_size,
                row_end=shard_index * shard_size + record.shape[0],
                tensor=record,
            )
            for shard_index, record in records
        )
        for previous, current in pairwise(shards):
            if previous.row_end != current.row_start:
                raise ValueError(
                    "only the final PLE shard may be short; "
                    f"shards {previous.index} and {current.index} are discontinuous"
                )
        return cls(
            prefix=prefix,
            dtype=records[0][1].dtype,
            embedding_dim=records[0][1].shape[1],
            total_rows=shards[-1].row_end,
            shard_size=shard_size,
            shards=shards,
        )

    @property
    def row_bytes(self) -> int:
        return self.shards[0].tensor.row_bytes

    def locate(self, row_id: int) -> RowLocation:
        if row_id < 0 or row_id >= self.total_rows:
            raise IndexError(f"PLE row {row_id} is outside [0, {self.total_rows})")
        shard = self.shards[row_id // self.shard_size]
        if row_id >= shard.row_end:
            raise IndexError(f"PLE row {row_id} is not materialized")
        local_row = row_id - shard.row_start
        return RowLocation(
            path=shard.tensor.path,
            offset=shard.tensor.offset + local_row * shard.tensor.row_bytes,
            nbytes=shard.tensor.row_bytes,
        )

    def summary(self) -> dict[str, int | str]:
        return {
            "prefix": self.prefix,
            "dtype": self.dtype,
            "embedding_dim": self.embedding_dim,
            "row_bytes": self.row_bytes,
            "total_rows": self.total_rows,
            "shards": len(self.shards),
            "tensor_bytes": sum(shard.tensor.nbytes for shard in self.shards),
            "files": len({shard.tensor.path for shard in self.shards}),
        }


class RowReader(Protocol):
    def read_rows(self, row_ids: Sequence[int]) -> list[bytes]: ...

    def close(self) -> None: ...


class MMapRowReader:
    """Portable correctness/debug backend backed by the page cache."""

    def __init__(self, manifest: PLEManifest) -> None:
        self.manifest = manifest
        self._files: dict[Path, Any] = {}
        self._maps: dict[Path, mmap.mmap] = {}

    def _mapping(self, path: Path) -> mmap.mmap:
        mapping = self._maps.get(path)
        if mapping is None:
            handle = path.open("rb")
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._files[path] = handle
            self._maps[path] = mapping
        return mapping

    def read_rows(self, row_ids: Sequence[int]) -> list[bytes]:
        output = []
        for row_id in row_ids:
            location = self.manifest.locate(row_id)
            mapping = self._mapping(location.path)
            output.append(mapping[location.offset : location.offset + location.nbytes])
        return output

    def close(self) -> None:
        for mapping in self._maps.values():
            mapping.close()
        for handle in self._files.values():
            handle.close()
        self._maps.clear()
        self._files.clear()


class IoUringPageRowReader:
    """Read aligned pages through SGLang's persistent native io_uring reader."""

    def __init__(
        self,
        manifest: PLEManifest,
        *,
        queue_depth: int,
        max_batch: int,
        cache_pages: int,
        page_size: int = 4096,
    ) -> None:
        if not hasattr(os, "O_DIRECT"):
            raise OSError("O_DIRECT is unavailable on this platform")
        if cache_pages < 0:
            raise ValueError("cache_pages cannot be negative")
        from sglang.srt.rust_extensions import load_rust_extension

        IoUringReader = load_rust_extension(
            "sglang.srt.rust_extensions._storage"
        ).IoUringReader

        self.manifest = manifest
        self.page_size = page_size
        self.max_batch = max_batch
        self.cache_pages = cache_pages
        try:
            self._ring = IoUringReader(queue_depth, max_batch, page_size)
        except OSError as error:
            if error.errno == errno.EPERM:
                raise OSError(
                    errno.EPERM,
                    "io_uring is blocked; allow io_uring_setup, io_uring_enter, "
                    "and io_uring_register in the container seccomp profile",
                ) from error
            raise
        self._fds: dict[Path, int] = {}
        self._cache: OrderedDict[tuple[Path, int], bytes] = OrderedDict()

    def _fd(self, path: Path) -> int:
        descriptor = self._fds.get(path)
        if descriptor is None:
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECT)
            self._fds[path] = descriptor
        return descriptor

    def _page_keys(self, location: RowLocation) -> tuple[tuple[Path, int], ...]:
        first = location.offset // self.page_size * self.page_size
        last_byte = location.offset + location.nbytes - 1
        last = last_byte // self.page_size * self.page_size
        return tuple(
            (location.path, offset)
            for offset in range(first, last + self.page_size, self.page_size)
        )

    def _load_pages(
        self, keys: Sequence[tuple[Path, int]]
    ) -> dict[tuple[Path, int], bytes]:
        pages = {}
        misses = []
        for key in dict.fromkeys(keys):
            page = self._cache.get(key)
            if page is None:
                misses.append(key)
            else:
                self._cache.move_to_end(key)
                pages[key] = page
        for start in range(0, len(misses), self.max_batch):
            chunk = misses[start : start + self.max_batch]
            loaded = self._ring.read_pages(
                [self._fd(path) for path, _ in chunk],
                [offset for _, offset in chunk],
            )
            for key, page in zip(chunk, loaded, strict=True):
                pages[key] = page
                if self.cache_pages:
                    self._cache[key] = page
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.cache_pages:
                        self._cache.popitem(last=False)
        return pages

    def read_rows(self, row_ids: Sequence[int]) -> list[bytes]:
        locations = [self.manifest.locate(row_id) for row_id in row_ids]
        location_keys = [self._page_keys(location) for location in locations]
        pages = self._load_pages([key for keys in location_keys for key in keys])

        output = []
        for location, keys in zip(locations, location_keys, strict=True):
            remaining = location.nbytes
            cursor = location.offset
            parts = []
            for key in keys:
                page_offset = key[1]
                within_page = cursor - page_offset
                available = min(remaining, len(pages[key]) - within_page)
                if available <= 0:
                    raise OSError(
                        f"io_uring page does not cover {location.path}:{cursor}"
                    )
                parts.append(pages[key][within_page : within_page + available])
                cursor += available
                remaining -= available
            if remaining:
                raise OSError(f"incomplete row read: {remaining} bytes remain")
            output.append(b"".join(parts))
        return output

    def close(self) -> None:
        for descriptor in self._fds.values():
            os.close(descriptor)
        self._fds.clear()
        self._cache.clear()


class PendingGather:
    """Mutable bridge updated by breakable CUDA-graph replay."""

    def __init__(
        self, future: Future[list[bytes]], input_shape: tuple[int, ...]
    ) -> None:
        self.future = future
        self.input_shape = input_shape


def _capture_start_gather(embedding: Any, input_ids: torch.Tensor) -> PendingGather:
    future: Future[list[bytes]] = Future()
    future.set_result([])
    return PendingGather(future, tuple(input_ids.shape))


def _capture_finish_gather(
    embedding: Any,
    pending: PendingGather,
    device: torch.device,
    out: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    expected_shape = (*pending.input_shape, embedding.embedding_dim)
    output = (
        out if out is not None else embedding.allocate_output(expected_shape, device)
    )
    output.zero_()
    return output


class NVMePLEEmbedding(nn.Module):
    """TP1 Qwen4 PLE embedding backed by sparse reads from a local snapshot."""

    def __init__(
        self,
        snapshot: str | Path,
        *,
        num_embeddings: int,
        embedding_dim: int,
        expected_shards: int | None = None,
    ) -> None:
        super().__init__()
        self.manifest = PLEManifest.from_snapshot(
            snapshot, expected_shards=expected_shards
        )
        if self.manifest.total_rows != num_embeddings:
            raise ValueError(
                "PLE snapshot row count does not match the model config: "
                f"{self.manifest.total_rows} != {num_embeddings}"
            )
        if self.manifest.embedding_dim != embedding_dim:
            raise ValueError(
                "PLE snapshot dimension does not match the model config: "
                f"{self.manifest.embedding_dim} != {embedding_dim}"
            )
        if self.manifest.dtype != "F8_E4M3":
            raise ValueError(
                f"the NVMe PLE path requires FP8 E4M3 rows, got {self.manifest.dtype}"
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = 1
        self.register_buffer(
            "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )
        self._reader = self._create_reader()
        self._io_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ple-prefetch"
        )
        self._stage: torch.Tensor | None = None
        self._stage_event: torch.cuda.Event | None = None
        self._calls = 0
        self._rows = 0
        self._read_seconds = 0.0
        summary = self.manifest.summary()
        logger.info(
            "Qwen4 PLE NVMe table: %.2f GiB across %d files (%d rows)",
            int(summary["tensor_bytes"]) / (1024**3),
            summary["files"],
            self.manifest.total_rows,
        )

    def _create_reader(self) -> RowReader:
        backend = envs.SGLANG_QWEN4_PLE_NVME_BACKEND.get()
        if backend == "mmap":
            return MMapRowReader(self.manifest)
        if backend == "io_uring":
            return IoUringPageRowReader(
                self.manifest,
                queue_depth=envs.SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH.get(),
                max_batch=envs.SGLANG_QWEN4_PLE_NVME_MAX_BATCH_PAGES.get(),
                cache_pages=envs.SGLANG_QWEN4_PLE_NVME_CACHE_PAGES.get(),
            )
        raise ValueError(f"unsupported SGLANG_QWEN4_PLE_NVME_BACKEND={backend!r}")

    def _stage_buffer(self, nbytes: int) -> torch.Tensor:
        if self._stage_event is not None:
            self._stage_event.synchronize()
        if self._stage is None or self._stage.numel() < nbytes:
            self._stage = torch.empty(
                nbytes, dtype=torch.uint8, device="cpu", pin_memory=True
            )
        return self._stage[:nbytes]

    def allocate_output(
        self, shape: Sequence[int], device: torch.device
    ) -> torch.Tensor:
        return torch.empty(tuple(shape), dtype=torch.bfloat16, device=device)

    def gather(
        self, input_ids: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.finish_gather(
            self.start_gather(input_ids), input_ids.device, out=out
        )

    @eager_on_graph(True, capture_stub=_capture_start_gather)
    def start_gather(self, input_ids: torch.Tensor) -> PendingGather:
        row_ids = (
            input_ids.detach().reshape(-1).to(device="cpu", dtype=torch.int64).tolist()
        )
        return PendingGather(
            self._io_executor.submit(self._timed_read_rows, row_ids),
            tuple(input_ids.shape),
        )

    def _timed_read_rows(self, row_ids: list[int]) -> list[bytes]:
        started = time.perf_counter()
        rows = self._reader.read_rows(row_ids)
        self._calls += 1
        self._rows += len(row_ids)
        self._read_seconds += time.perf_counter() - started
        interval = envs.SGLANG_QWEN4_PLE_NVME_LOG_INTERVAL.get()
        if interval > 0 and self._calls % interval == 0:
            logger.info(
                "Qwen4 PLE NVMe: calls=%d rows=%d mean_read_ms=%.3f",
                self._calls,
                self._rows,
                self._read_seconds / self._calls * 1000,
            )
        return rows

    @eager_on_graph(True, capture_stub=_capture_finish_gather)
    def finish_gather(
        self,
        pending: PendingGather,
        device: torch.device,
        out: torch.Tensor | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        rows = pending.future.result()
        expected_shape = (*pending.input_shape, self.embedding_dim)
        output = (
            out if out is not None else self.allocate_output(expected_shape, device)
        )
        if (
            tuple(output.shape) != expected_shape
            or output.dtype != torch.bfloat16
            or output.device != device
        ):
            raise ValueError("invalid NVMe PLE output buffer")

        raw = b"".join(rows)
        expected_bytes = math.prod(pending.input_shape) * self.embedding_dim
        if len(raw) != expected_bytes:
            raise OSError(
                f"NVMe PLE read returned {len(raw)} bytes; expected {expected_bytes}"
            )
        if raw:
            stage = self._stage_buffer(len(raw))
            ctypes.memmove(stage.data_ptr(), raw, len(raw))
            stream_context = (
                torch.cuda.stream(stream) if stream is not None else nullcontext()
            )
            with stream_context:
                device_bytes = stage.to(device=device, non_blocking=True)
                decoded = device_bytes.view(torch.float8_e4m3fn).to(torch.bfloat16)
                output.copy_(decoded.view(expected_shape))
                if self._stage_event is None:
                    self._stage_event = torch.cuda.Event()
                self._stage_event.record()
            if is_in_breakable_cuda_graph():
                self._stage_event.synchronize()
        return output

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.gather(input_ids)

    def close(self) -> None:
        self._io_executor.shutdown()
        self._reader.close()

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, backend={type(self._reader).__name__}"
        )


def is_nvme_ple_embedding(module: Any) -> bool:
    return isinstance(module, NVMePLEEmbedding)
