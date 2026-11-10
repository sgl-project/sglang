#!/usr/bin/env python3
"""
dump_to_mooncake.py - Upload a safetensors checkpoint into Mooncake Store.

Packs tensors into buckets (target size max(256 MiB, largest tensor))
plus a JSON manifest under the key prefix ``ckpt:<checkpoint-name>`` so mooncake_loader.py can stream them
into a running SGLang server. Upload once, load many times.

  python -m sglang.srt.checkpoint_engine.dump_to_mooncake \
      --checkpoint-path /models/Qwen3-0.6B --checkpoint-name Qwen3-0.6B \
      --master-addr <mooncake-master-host>:50051 \
      --metadata-server P2PHANDSHAKE --protocol rdma --rdma-devices <nic>

Store contents are volatile (lost when the store service restarts);
verify with mooncake_store_health_check.py before relying on them.

Adapted from the kvcache-ai/checkpoint_engine dump tool.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import sys
import struct
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import torch
from loguru import logger
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

if TYPE_CHECKING:
    from mooncake.store import MooncakeDistributedStore



ALIGN_SIZE = 256
PAGE_ALIGNMENT = 4096


@dataclass
class TensorRecord:
    name: str
    dtype: torch.dtype
    shape: List[int]
    size_aligned: int
    path: str
    file_offset: int
    nbytes: int


@dataclass
class _BufferWrapper:
    view: torch.Tensor
    backing: torch.Tensor
    ptr: int
    registered_size: int


@dataclass
class _StoreTask:
    index: int
    size: int
    items: List[Dict[str, Any]]
    buffer_view: torch.Tensor
    buffer_wrapper: _BufferWrapper
    fill_elapsed: float


_DTYPE_MAP: Dict[str, torch.dtype] = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "BF16": torch.bfloat16,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "U16": torch.uint16,
    "U32": torch.uint32,
    "U64": torch.uint64,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _align_size(dtype: torch.dtype, shape: torch.Size) -> int:
    return (dtype.itemsize * shape.numel() + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE


def _list_safetensors_files(checkpoint_path: str) -> List[str]:
    """List checkpoint shards, proving completeness against the index.

    A silently missing shard would upload a structurally valid but partial
    checkpoint; the loader would then leave those tensors at their dummy
    values with every protocol step still reporting success.
    """
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".safetensors"):
        return [checkpoint_path]
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} is not a directory or safetensors file")

    entries = os.listdir(checkpoint_path)
    files = [
        os.path.join(checkpoint_path, entry)
        for entry in entries
        if entry.endswith(".safetensors")
    ]

    index_files = sorted(
        entry for entry in entries if entry.endswith(".safetensors.index.json")
    )
    if len(index_files) > 1:
        raise RuntimeError(f"Multiple safetensors indexes found: {index_files}")
    if not index_files:
        if not files:
            raise FileNotFoundError(
                f"No .safetensors files found under {checkpoint_path}"
            )
        if len(files) > 1:
            raise RuntimeError(
                f"{checkpoint_path} has {len(files)} safetensors shards but no "
                ".safetensors.index.json; refusing to dump because shard "
                "completeness cannot be proven"
            )
        return sorted(files)

    # With an index present it is the single source of truth: exactly the
    # shards it references get dumped. Verifying one set but uploading
    # another lets a stale unreferenced shard smuggle in old tensor bytes,
    # or a nested referenced shard pass validation without being uploaded.
    index_path = os.path.join(checkpoint_path, index_files[0])
    with open(index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    if not isinstance(index, dict):
        raise RuntimeError(f"Invalid index root in {index_path}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Invalid or empty weight_map in {index_path}")

    root = os.path.realpath(checkpoint_path)
    referenced_by_name: Dict[str, str] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise RuntimeError(f"Invalid tensor/shard entry in {index_path}")
        shard_path = os.path.realpath(os.path.join(root, shard))
        if os.path.commonpath([root, shard_path]) != root:
            raise RuntimeError(
                f"Indexed shard escapes the checkpoint directory: {shard!r}"
            )
        referenced_by_name[name] = shard_path

    referenced = set(referenced_by_name.values())
    missing = sorted(path for path in referenced if not os.path.isfile(path))
    if missing:
        raise FileNotFoundError(
            f"Checkpoint is missing {len(missing)} indexed shard(s); "
            f"first: {missing[0]}"
        )

    shard_names: Dict[str, set] = {}
    for shard_path in sorted(referenced):
        header, _ = _read_header(shard_path)
        shard_names[shard_path] = {
            name for name in header if not name.startswith("__")
        }

    misplaced = sorted(
        name
        for name, shard_path in referenced_by_name.items()
        if name not in shard_names[shard_path]
    )
    if misplaced:
        raise RuntimeError(
            f"Index maps {len(misplaced)} tensor(s) to shards that do not "
            f"contain them; first: {misplaced[0]}"
        )

    unreferenced = sorted(
        {os.path.realpath(path) for path in files} - referenced
    )
    if unreferenced:
        raise RuntimeError(
            f"Checkpoint directory contains {len(unreferenced)} safetensors "
            f"file(s) not referenced by the index (stale shard?); "
            f"first: {unreferenced[0]}. Remove them or dump a clean directory"
        )

    return sorted(referenced)


def _dtype_from_header(value: str) -> torch.dtype:
    dtype = _DTYPE_MAP.get(value)
    if dtype is None:
        raise ValueError(f"Unsupported safetensors dtype: {value}")
    return dtype


def _read_header(path: str) -> Tuple[Dict[str, Any], int]:
    with open(path, "rb") as fh:
        header_size_bytes = fh.read(8)
        if len(header_size_bytes) != 8:
            raise RuntimeError(f"{path} is not a valid safetensors file")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = fh.read(header_size)
        if len(header_bytes) != header_size:
            raise RuntimeError(f"{path} truncated while reading header")
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise RuntimeError(f"Invalid safetensors header in {path}")
    return header, 8 + header_size


def _collect_tensor_records(files: Iterable[str]) -> List[TensorRecord]:
    records: List[TensorRecord] = []
    seen: set[str] = set()
    for file_path in files:
        header, data_start = _read_header(file_path)
        for name, entry in header.items():
            if name.startswith("__"):
                continue
            if name in seen:
                raise ValueError(f"Duplicate tensor name {name} in {file_path}")
            dtype = _dtype_from_header(entry["dtype"])
            shape = entry["shape"]
            start, end = entry["data_offsets"]
            nbytes = end - start
            expected = dtype.itemsize * int(torch.Size(shape).numel())
            if nbytes != expected:
                raise RuntimeError(f"Tensor {name} in {file_path} reports {nbytes}B but expected {expected}B")
            records.append(
                TensorRecord(
                    name=name,
                    dtype=dtype,
                    shape=list(shape),
                    size_aligned=_align_size(dtype, torch.Size(shape)),
                    path=file_path,
                    file_offset=data_start + start,
                    nbytes=nbytes,
                )
            )
            seen.add(name)
    if not records:
        raise RuntimeError("No tensors loaded from checkpoint files")
    records.sort(key=lambda record: record.name)
    return records


def _allocate_aligned_cpu(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = torch.empty(size + PAGE_ALIGNMENT, dtype=torch.uint8)
    offset = (PAGE_ALIGNMENT - (raw.data_ptr() % PAGE_ALIGNMENT)) % PAGE_ALIGNMENT
    view = raw.narrow(0, offset, size)
    return view, raw


def _partition_records(records: List[TensorRecord]) -> List[Tuple[List[TensorRecord], int]]:
    if not records:
        return []
    max_tensor = max(record.size_aligned for record in records)
    bucket_size = max(1 << 28, max_tensor)

    partitions: List[Tuple[List[TensorRecord], int]] = []
    current: List[TensorRecord] = []
    current_size = 0
    for record in records:
        if current and current_size + record.size_aligned > bucket_size:
            partitions.append((current, current_size))
            current = []
            current_size = 0
        current.append(record)
        current_size += record.size_aligned
    if current:
        partitions.append((current, current_size))
    return partitions


def _fill_bucket_from_records(
    buffer: torch.Tensor,
    records: List[TensorRecord],
) -> List[Dict[str, Any]]:
    offset = 0
    buffer_array = buffer.numpy()
    items: List[Dict[str, Any]] = []
    for record in records:
        target_slice = buffer_array[offset : offset + record.nbytes]
        with open(record.path, "rb", buffering=0) as fh:
            fh.seek(record.file_offset)
            total_read = 0
            while total_read < record.nbytes:
                chunk = target_slice[total_read:]
                n = fh.readinto(chunk)
                if n == 0:
                    break
                total_read += n
            if total_read != record.nbytes:
                raise RuntimeError(
                    f"Short read loading {record.name} from {record.path}: "
                    f"expected {record.nbytes}B, got {total_read}B"
                )
        pad = record.size_aligned - record.nbytes
        if pad > 0:
            buffer[offset + record.nbytes : offset + record.size_aligned].zero_()
        items.append({
            "name": record.name,
            "dtype": str(record.dtype),
            "shape": list(record.shape),
            "offset": offset,
        })
        offset += record.size_aligned
    return items


def _wait_store_accepts_puts(
    store: MooncakeDistributedStore, prefix: str, timeout_s: float = 120.0
) -> None:
    """Probe with a canary put until the store actually accepts writes.

    A freshly started store service mounts its segment and answers RPCs tens
    of seconds before large pinned segments are fully registered for writes;
    during that window every put fails with rc=-200. Probing here turns that
    into a bounded warm-up wait instead of failing the first real bucket.
    """
    canary_key = f"{prefix}:dump-canary"
    view, _backing = _allocate_aligned_cpu(4096)
    ptr = view.data_ptr()
    ret = store.register_buffer(ptr, view.numel())
    if ret != 0:
        raise RuntimeError(f"register_buffer failed for canary with code {ret}")
    try:
        deadline = time.monotonic() + timeout_s
        attempt = 0
        while True:
            ret = store.put_from(canary_key, ptr, view.numel())
            if ret == 0:
                remove = getattr(store, "remove", None)
                if callable(remove):
                    try:
                        remove(canary_key)
                    except Exception:
                        pass
                if attempt:
                    logger.info(f"Store accepted puts after {attempt} retry(ies)")
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Store did not accept puts within {timeout_s:.0f}s "
                    f"(last put_from rc={ret}); is the store service fully "
                    "started and its segment registered?"
                )
            attempt += 1
            logger.warning(
                f"Store not accepting puts yet (put_from rc={ret}); retrying..."
            )
            time.sleep(2)
    finally:
        store.unregister_buffer(ptr)


def _store_bucket(
    store: MooncakeDistributedStore,
    prefix: str,
    owner_rank: int,
    index: int,
    buffer: torch.Tensor,
    size: int,
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    bucket_key = f"{prefix}:bucket:{owner_rank}:{index}"
    ptr = buffer.data_ptr()
    ret = store.put_from(bucket_key, ptr, size)
    if ret != 0:
        raise RuntimeError(f"put_from failed for {bucket_key} with code {ret}")
    logger.debug(f"Stored {bucket_key} size={size / 1024 / 1024:.2f}MiB tensors={len(items)}")
    return {
        "bucket_key": bucket_key,
        "bucket_index": index,
        "owner_rank": owner_rank,
        "size": size,
        "items": items,
    }


def _store_manifest(store: MooncakeDistributedStore, key: str, manifest: Dict[str, Any]) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    view, _backing = _allocate_aligned_cpu(len(payload))
    view.copy_(torch.tensor(list(payload), dtype=torch.uint8))
    ptr = view.data_ptr()
    ret = store.register_buffer(ptr, view.numel())
    if ret != 0:
        raise RuntimeError(f"register_buffer failed for manifest {key} with code {ret}")
    ret = store.put_from(key, ptr, view.numel())
    store.unregister_buffer(ptr)
    if ret != 0:
        raise RuntimeError(f"put_from failed for manifest {key} with code {ret}")
    logger.info(f"Manifest stored at {key} (buckets={len(manifest.get('buckets', []))})")


def dump_to_store(
    checkpoint_path: str,
    checkpoint_name: str,
    *,
    hostname: str,
    metadata_server: str,
    master_addr: str,
    segment_size: int,
    buffer_size: int,
    protocol: str,
    rdma_devices: str,
    prefix: str | None = None,
) -> Dict[str, Any]:
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as e:
        raise ImportError(
            "mooncake-transfer-engine is required for Mooncake store access. "
            "Install it with: pip install mooncake-transfer-engine"
        ) from e

    files = _list_safetensors_files(checkpoint_path)
    logger.info(f"Discovered {len(files)} safetensors shard(s)")
    records = _collect_tensor_records(files)
    partitions = _partition_records(records)
    if not partitions:
        raise RuntimeError("No tensor partitions were generated for dumping")

    store = MooncakeDistributedStore()
    ret = store.setup(
        hostname,
        metadata_server,
        segment_size,
        buffer_size,
        protocol,
        rdma_devices,
        master_addr,
    )
    if ret != 0:
        raise RuntimeError(
            f"Failed to setup MooncakeDistributedStore with error code {ret}. "
            f"Parameters: hostname={hostname}, metadata_server={metadata_server}, "
            f"protocol={protocol}, rdma_devices={rdma_devices}, master_addr={master_addr}"
        )

    prefix = prefix or f"ckpt:{checkpoint_name}"
    _wait_store_accepts_puts(store, prefix)
    manifest_entries: List[Optional[Dict[str, Any]]] = []
    buffer_wrappers: List[_BufferWrapper] = []
    store_queue: "queue.Queue[Any]" = queue.Queue()
    worker_thread: Optional[threading.Thread] = None
    stop_worker = threading.Event()
    sentinel = object()
    try:
        total_buckets = len(partitions)
        max_bucket_size = max(total_size for _, total_size in partitions)
        pool_size = min(2, total_buckets)
        buffer_pool: "queue.Queue[_BufferWrapper]" = queue.Queue()
        for _ in range(pool_size):
            pool_view, pool_backing = _allocate_aligned_cpu(max_bucket_size)
            ptr = pool_view.data_ptr()
            ret = store.register_buffer(ptr, max_bucket_size)
            if ret != 0:
                raise RuntimeError(
                    f"register_buffer failed for dump buffer with code {ret}"
                )
            wrapper = _BufferWrapper(
                view=pool_view,
                backing=pool_backing,
                ptr=ptr,
                registered_size=max_bucket_size,
            )
            buffer_pool.put(wrapper)
            buffer_wrappers.append(wrapper)

        store_queue = queue.Queue(maxsize=pool_size)
        result_queue: "queue.Queue[Any]" = queue.Queue()
        manifest_entries = [None] * total_buckets
        worker_errors: List[BaseException] = []

        def worker() -> None:
            while True:
                task = store_queue.get()
                try:
                    if task is sentinel or stop_worker.is_set():
                        return
                    assert isinstance(task, _StoreTask)
                    store_start = time.perf_counter()
                    entry = _store_bucket(
                        store,
                        prefix=prefix,
                        owner_rank=0,
                        index=task.index,
                        buffer=task.buffer_view,
                        size=task.size,
                        items=task.items,
                    )
                    store_elapsed = time.perf_counter() - store_start
                    result_queue.put(
                        (
                            task.index,
                            entry,
                            task.buffer_wrapper,
                            task.fill_elapsed,
                            store_elapsed,
                            task.size,
                        )
                    )
                except BaseException as exc:
                    # Surface the failure to the main thread; a silently dead
                    # worker leaves it blocked on the buffer pool forever.
                    worker_errors.append(exc)
                    return
                finally:
                    store_queue.task_done()

        def check_worker_health() -> None:
            if worker_errors:
                raise RuntimeError("Mooncake store worker failed") from worker_errors[0]

        worker_thread = threading.Thread(target=worker, name="dump-store-worker", daemon=True)
        worker_thread.start()

        bytes_written = 0
        results_received = 0
        progress = None
        if tqdm is not None and sys.stderr.isatty():
            progress = tqdm(total=total_buckets, unit="bucket", desc="Dump buckets")

        def drain_results(block: bool) -> None:
            nonlocal bytes_written, results_received
            while results_received < total_buckets:
                try:
                    if block:
                        result = result_queue.get(timeout=0.1)
                    else:
                        result = result_queue.get_nowait()
                except queue.Empty:
                    break
                (
                    result_index,
                    manifest_entry,
                    buffer_wrapper,
                    fill_elapsed,
                    store_elapsed,
                    size,
                ) = result
                manifest_entries[result_index] = manifest_entry
                bytes_written += size
                results_received += 1
                buffer_pool.put(buffer_wrapper)
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        fill=f"{fill_elapsed:.2f}s",
                        store=f"{store_elapsed:.2f}s",
                        written=f"{bytes_written / (1024 ** 3):.2f}GiB",
                        pending=store_queue.qsize(),
                    )
                else:
                    logger.info(
                        f"[bucket {result_index + 1}/{total_buckets}] fill={fill_elapsed:.2f}s "
                        f"store={store_elapsed:.2f}s (cumulative={bytes_written / (1024 ** 3):.2f} GiB "
                        f"inflight={store_queue.qsize()})"
                    )

        for index, (bucket_records, total_size) in enumerate(partitions):
            if progress is not None:
                progress.set_description(
                    f"Bucket {index + 1}/{total_buckets} ({len(bucket_records)} tensors, "
                    f"{total_size / (1024 ** 2):.1f} MiB)"
                )
            else:
                logger.info(
                    f"[bucket {index + 1}/{total_buckets}] preparing {len(bucket_records)} tensor(s) "
                    f"total={total_size / (1024 ** 2):.1f} MiB"
                )
            # Wait for a free buffer while draining results: wrappers are
            # only returned to the pool by drain_results, so a plain blocking
            # get() deadlocks whenever the store worker is slower than the
            # file reads (both wrappers in flight, nobody draining).
            while True:
                drain_results(block=False)
                check_worker_health()
                try:
                    buffer_wrapper = buffer_pool.get(timeout=0.1)
                    break
                except queue.Empty:
                    if not worker_thread.is_alive():
                        check_worker_health()
                        raise RuntimeError(
                            "Mooncake store worker exited before returning a buffer"
                        )
            bucket_buffer = buffer_wrapper.view.narrow(0, 0, total_size)
            fill_start = time.perf_counter()
            items = _fill_bucket_from_records(bucket_buffer, bucket_records)
            fill_elapsed = time.perf_counter() - fill_start
            store_queue.put(
                _StoreTask(
                    index=index,
                    size=total_size,
                    items=items,
                    buffer_view=bucket_buffer,
                    buffer_wrapper=buffer_wrapper,
                    fill_elapsed=fill_elapsed,
                )
            )
            drain_results(block=False)

        while results_received < total_buckets:
            check_worker_health()
            if not worker_thread.is_alive():
                check_worker_health()
                raise RuntimeError(
                    "Mooncake store worker exited before all results arrived"
                )
            drain_results(block=True)
        store_queue.put(sentinel)
        worker_thread.join(timeout=60)
        if worker_thread.is_alive():
            raise RuntimeError(
                "Mooncake store worker did not stop after receiving sentinel"
            )

        if progress is not None:
            progress.close()

        if any(entry is None for entry in manifest_entries):
            raise RuntimeError("Dump completed with missing manifest entries")

        manifest = {
            "checkpoint": checkpoint_name,
            "align_size": ALIGN_SIZE,
            "buckets": manifest_entries,
        }
        _store_manifest(store, f"{prefix}:manifest", manifest)
    finally:
        # A main-thread failure can land here while the worker is still
        # inside a native put_from(); freeing its registered buffer or
        # closing the store underneath it is a use-after-free. Stop it,
        # wait, and if it will not stop, leak the resources to process exit.
        if worker_thread is not None and worker_thread.is_alive():
            stop_worker.set()
            try:
                store_queue.put_nowait(sentinel)
            except queue.Full:
                # Queue full means the worker has more items to consume; it
                # checks stop_worker before touching each one.
                pass
            worker_thread.join(timeout=60)
        if worker_thread is not None and worker_thread.is_alive():
            logger.error(
                "Store worker is still active; retaining Mooncake buffers "
                "and connection until process exit"
            )
        else:
            for wrapper in buffer_wrappers:
                store.unregister_buffer(wrapper.ptr)
            store.close()
    return manifest


def parse_args() -> argparse.Namespace:
    default_hostname = os.getenv("MOONCAKE_HOSTNAME", socket.gethostname())
    default_metadata = os.getenv("MOONCAKE_TE_META_DATA_SERVER",
                                 os.getenv("MOONCAKE_METADATA_SERVER", "P2PHANDSHAKE"))
    default_master = os.getenv("MOONCAKE_MASTER", os.getenv("MOONCAKE_MASTER_ADDR", "localhost:50051"))
    default_segment = int(os.getenv("MOONCAKE_SEGMENT_SIZE", "0"))
    default_buffer = int(os.getenv("MOONCAKE_BUFFER_SIZE", "0"))
    default_protocol = os.getenv("MOONCAKE_PROTOCOL", "rdma")
    default_rdma = os.getenv("MOONCAKE_DEVICE", os.getenv("MOONCAKE_RDMA_DEVICES", ""))

    parser = argparse.ArgumentParser(
        description="Dump a safetensors checkpoint into Mooncake Store."
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to safetensors directory or a single safetensors shard.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Logical checkpoint name (defaults to basename of checkpoint_path).",
    )
    parser.add_argument(
        "--hostname",
        default=default_hostname,
        help="Local hostname for Mooncake registration (default: %(default)s).",
    )
    parser.add_argument(
        "--metadata-server",
        default=default_metadata,
        help="Mooncake metadata server URL (default: %(default)s).",
    )
    parser.add_argument(
        "--master-addr",
        default=default_master,
        help="Mooncake master address host:port (default: %(default)s).",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=default_segment,
        help="Global segment size in bytes (default from $MOONCAKE_SEGMENT_SIZE or 0).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=default_buffer,
        help="Local buffer size in bytes (default from $MOONCAKE_BUFFER_SIZE or 0).",
    )
    parser.add_argument(
        "--protocol",
        default=default_protocol,
        choices=["tcp", "rdma"],
        help="Mooncake transport protocol.",
    )
    parser.add_argument(
        "--rdma-devices",
        default=default_rdma,
        help="Comma-separated RDMA device list (default from $MOONCAKE_RDMA_DEVICES).",
    )
    parser.add_argument(
        "--store-prefix",
        default=None,
        help="Key prefix when writing to store (defaults to ckpt:<checkpoint-name>).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for loguru (default INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    checkpoint_path = os.path.abspath(args.checkpoint_path)
    checkpoint_name = args.checkpoint_name or (
        os.path.basename(checkpoint_path.rstrip("/")) if os.path.isdir(checkpoint_path)
        else os.path.splitext(os.path.basename(checkpoint_path))[0]
    )

    prefix = args.store_prefix or f"ckpt:{checkpoint_name}"
    logger.info(f"Dumping checkpoint {checkpoint_name} from {checkpoint_path}")

    start_time = time.perf_counter()
    manifest = dump_to_store(
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
        hostname=args.hostname,
        metadata_server=args.metadata_server,
        master_addr=args.master_addr,
        segment_size=args.segment_size,
        buffer_size=args.buffer_size,
        protocol=args.protocol,
        rdma_devices=args.rdma_devices,
        prefix=args.store_prefix,
    )
    elapsed = time.perf_counter() - start_time

    total_size_gb = sum(bucket["size"] for bucket in manifest["buckets"]) / (1024 ** 3)
    throughput = total_size_gb / elapsed if elapsed > 0 else 0

    logger.info("=" * 60)
    logger.info(f"Checkpoint dump completed: {checkpoint_name}")
    logger.info(f"  Buckets:    {len(manifest['buckets'])}")
    logger.info(f"  Total size: {total_size_gb:.2f} GiB")
    logger.info(f"  Duration:   {elapsed:.1f}s")
    logger.info(f"  Throughput: {throughput:.2f} GiB/s")
    logger.info(f"  Manifest:   {prefix}:manifest")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
