#!/usr/bin/env python3
"""
mooncake_loader.py - One-shot checkpoint loader: Mooncake Store -> SGLang.

Loads model weights from a Mooncake distributed store into a running SGLang
server that was started with `--load-format dummy
--checkpoint-engine-wait-weights-before-ready`, using the checkpoint-engine
IPC protocol (ZMQ metadata + CUDA-IPC shared GPU buffer).

Pipeline architecture (per rank):
1. Fetch: read bucket from the Mooncake store (bucket owner rank only)
2. Broadcast: NCCL broadcast to all ranks
3. Apply: hand tensor metadata to the SGLang worker via ZMQ IPC; the worker
   copies views out of the shared CUDA-IPC buffer

Usage (TP size N, one loader process per GPU):

  torchrun --nproc-per-node N -m sglang.srt.checkpoint_engine.mooncake_loader \
      --checkpoint-name <name-used-at-dump-time> \
      --endpoint http://localhost:30000 --parallel-size N \
      --master-addr <mooncake-master-host>:50051 \
      --metadata-server P2PHANDSHAKE --protocol rdma \
      --rdma-devices <rdma-nic> --flush-cache

Weights must first be uploaded once with dump_to_mooncake.py (same
--checkpoint-name). See README_MOONCAKE.md in this directory for the
end-to-end walkthrough.

GPU memory per rank = max_bucket_size x pipeline_depth (about 2 GiB x 4 with
the defaults); the staging buffer is freed when the loader exits.

This is the one-shot (first load / cold start) variant of the loader from the
JD-AI-Infra fast-startup project, itself evolved from the
kvcache-ai/checkpoint_engine example loader. Daemon mode, the HTTP control
plane and multi-loader merged IPC were removed. The fetch/broadcast/apply
data plane follows the production loader, hardened for unattended one-shot
use: worker acks are content-checked (a checkpoint-engine worker reports a
failed bucket with a traceback reply, not a closed socket), manifests are
validated before any GPU work, and every failure path aims to end with the
server reporting the update failed rather than hanging or serving partially
loaded weights.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import httpx
import torch
import torch.distributed as dist
import zmq
from loguru import logger
from torch.multiprocessing.reductions import reduce_tensor
from tqdm import tqdm

if TYPE_CHECKING:
    from mooncake.store import MooncakeDistributedStore

ALIGN_SIZE = 256

# checkpoint-engine 0.4.0 finalizes over TWO None rounds (worker.py state
# machine): None #1 releases the IPC buffer, None #2 runs post_hook and acks.
# The post_hook ack must be awaited long enough for quant re-processing of
# large models (tens of seconds); abandoning it tears down our bound ZMQ
# endpoint while the worker's reply is in flight, which strands that reply
# and deadlocks the worker's scheduler thread in zmq ctx.term() at GC time.
POST_HOOK_SEND_TIMEOUT_MS = int(
    os.getenv("SGLANG_IPC_POST_HOOK_SEND_TIMEOUT_MS", "2000")
)
# Must stay below --update-timeout, otherwise the HTTP update request aborts
# while loader and workers are still finalizing legitimately.
POST_HOOK_ACK_TIMEOUT_MS = int(
    os.getenv("SGLANG_IPC_POST_HOOK_TIMEOUT_MS", "120000")
)
# Per-bucket apply acknowledgements normally arrive within seconds; bound the
# wait so a dead worker fails the load instead of hanging it forever.
APPLY_ACK_TIMEOUT_MS = int(os.getenv("SGLANG_IPC_APPLY_ACK_TIMEOUT_MS", "300000"))
# How long socket close may wait to flush a queued abort message; sending is
# only queueing in ZMQ, so closing with linger=0 right after would drop it.
ABORT_LINGER_MS = 2000


def _str_to_dtype(value: str) -> torch.dtype:
    """Convert string representation to torch dtype."""
    if value.startswith("torch."):
        value = value.split(".", 1)[1]
    dtype = getattr(torch, value, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype: {value}")
    return dtype


def _align_size(dtype: torch.dtype, shape: torch.Size) -> int:
    """Calculate 256-byte aligned size for a tensor."""
    return (dtype.itemsize * shape.numel() + ALIGN_SIZE - 1) // ALIGN_SIZE * ALIGN_SIZE


def _get_physical_gpu_id(device_index: int | None = None) -> str:
    """Get physical GPU UUID for IPC binding."""
    props = torch.cuda.get_device_properties(device_index)
    return f"GPU-{props.uuid!s}"


def _load_manifest(store: MooncakeDistributedStore, manifest_key: str) -> dict[str, Any]:
    """Load and validate a manifest from the Mooncake store."""
    payload = store.get(manifest_key)
    # A missing key surfaces as an empty payload, not None.
    if not payload:
        raise RuntimeError(
            f"Manifest key {manifest_key} not found in the store "
            "(wrong --checkpoint-name, or the checkpoint was never uploaded / "
            "was lost on a store restart)"
        )
    manifest = json.loads(payload.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object")
    align_size = manifest.get("align_size")
    if align_size not in (None, ALIGN_SIZE):
        raise ValueError(f"Manifest align_size {align_size} != {ALIGN_SIZE}")
    _validate_manifest_layout(manifest)
    return manifest


def _validate_manifest_layout(manifest: dict[str, Any]) -> None:
    """Reject structurally broken manifests before any GPU work starts.

    A tolerant reader here is a correctness hazard: a bucket whose items are
    missing or whose offsets disagree with the packed bytes would inject
    garbage (or nothing) while every protocol step still acks success.
    """
    buckets = manifest.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("Manifest must contain a non-empty buckets list")

    seen_keys: set[str] = set()
    seen_names: set[str] = set()
    for bucket_idx, bucket in enumerate(buckets):
        if not isinstance(bucket, dict):
            raise ValueError(f"Bucket {bucket_idx} is not an object")
        key = bucket.get("bucket_key")
        if not isinstance(key, str) or not key or key in seen_keys:
            raise ValueError(f"Invalid or duplicate bucket_key: {key!r}")
        seen_keys.add(key)

        size = bucket.get("size")
        items = bucket.get("items")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Bucket {key} has invalid size: {size!r}")
        if not isinstance(items, list) or not items:
            raise ValueError(f"Bucket {key} has no tensor items")

        expected_offset = 0
        for item_idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Bucket {key} item {item_idx} is not an object")
            name = item.get("name")
            if not isinstance(name, str) or not name or name in seen_names:
                raise ValueError(f"Invalid or duplicate tensor name: {name!r}")
            seen_names.add(name)
            if item.get("offset") != expected_offset:
                raise ValueError(
                    f"Tensor {name} offset {item.get('offset')!r} != "
                    f"expected {expected_offset}"
                )
            dtype = _str_to_dtype(item["dtype"])
            shape = torch.Size(item["shape"])
            expected_offset += _align_size(dtype, shape)

        if expected_offset != size:
            raise ValueError(
                f"Bucket {key} item extent {expected_offset} != size {size}"
            )


def verify_checkpoint_integrity(
    store: MooncakeDistributedStore,
    manifest: dict[str, Any],
    rank: int = 0,
) -> tuple[bool, list[str]]:
    """Verify all buckets in the manifest exist in the Mooncake Store.

    Args:
        store: Mooncake store instance
        manifest: Parsed manifest dict containing "buckets" list
        rank: Current rank (logging only on rank 0)

    Returns:
        (all_complete, missing_keys) tuple
    """
    buckets = manifest.get("buckets", [])
    if not buckets:
        return False, ["no buckets in manifest"]

    bucket_keys = [b["bucket_key"] for b in buckets]

    # Use batch_is_exist if available, otherwise check one by one
    if hasattr(store, "batch_is_exist"):
        existence = store.batch_is_exist(bucket_keys)
        missing = [k for k, e in zip(bucket_keys, existence) if e != 1]
    else:
        missing = []
        for key in bucket_keys:
            if store.is_exist(key) != 1:
                missing.append(key)

    if rank == 0 and missing:
        logger.warning(
            f"Integrity check: {len(missing)}/{len(bucket_keys)} buckets missing"
        )

    return len(missing) == 0, missing


def _rebalance_manifest(manifest: dict[str, Any], world_size: int, strategy: str) -> dict[str, Any]:
    """Rebalance bucket ownership across ranks."""
    if world_size <= 1 or strategy == "none":
        return manifest
    assert strategy == "round_robin", f"Unsupported rebalance strategy: {strategy}"
    updated = json.loads(json.dumps(manifest))
    buckets = updated.get("buckets", [])
    for idx, entry in enumerate(buckets):
        entry["owner_rank"] = idx % world_size
    return updated


def _init_process_group(timeout_seconds: int = 300, use_gloo: bool = False, device_id: int | None = None) -> None:
    """Initialize process group with timeout.

    Args:
        timeout_seconds: Timeout for process group initialization
        use_gloo: If True, use gloo backend (TCP-based, avoids NCCL conflicts)
                 If False, use nccl backend with cuda-checkpoint compatible settings
        device_id: GPU device ID for this rank (required for NCCL to avoid hangs)
    """
    if not dist.is_initialized():
        rank = int(os.getenv("RANK", "0"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        if use_gloo:
            # Use gloo backend to avoid NCCL conflicts with sglang
            # Gloo uses TCP/IP and doesn't interfere with existing NCCL resources
            # Note: gloo is slower than nccl, so use longer timeout
            if rank == 0:
                logger.info("Using gloo backend (TCP-based) to avoid NCCL conflicts")
            dist.init_process_group(backend="gloo", timeout=timedelta(seconds=300))
        else:
            dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))


def _gather_objects(obj: Any) -> list[Any]:
    """Gather objects from all ranks."""
    world_size = dist.get_world_size()
    gathered: list[Any | None] = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered  # type: ignore[return-value]


def _make_payload(items: list[dict[str, Any]], base_offset: int) -> list[dict[str, Any]]:
    """Create payload for SGLang worker (name, dtype, shape, offset).

    Uses the manifest's explicit per-item offsets (validated against the
    packing rule at manifest load) rather than recomputing them, so the
    dumper's layout is the single source of truth.
    """
    payload: list[dict[str, Any]] = []
    for item in items:
        dtype = _str_to_dtype(item["dtype"])
        shape = torch.Size(item["shape"])
        payload.append({
            "name": item["name"],
            "dtype": dtype,
            "shape": shape,
            "offset": base_offset + item["offset"],
        })
    return payload


def _check_sglang_ready(
    endpoint: str,
    inference_parallel_size: int,
    uds: str | None = None,
    rank: int = 0,
    timeout: float = 300.0,
) -> None:
    """Wait for SGLang server to be ready before starting weight loading."""
    # Only the first rank in each inference group checks
    if rank != rank // inference_parallel_size * inference_parallel_size:
        return

    logger.info(f"[Rank {rank}] Checking if SGLang server is ready at {endpoint}...")
    retry_num = 0
    start_time = time.time()
    deadline = time.monotonic() + timeout
    transport = httpx.HTTPTransport(uds=uds) if uds else None

    with httpx.Client(transport=transport) as client:
        while True:
            try:
                response = client.get(f"{endpoint}/ping", timeout=10)
                response.raise_for_status()
                elapsed = time.time() - start_time
                logger.info(
                    f"[Rank {rank}] ✓ SGLang server is ready at {endpoint} "
                    f"(waited {elapsed:.2f}s, {retry_num} retries)"
                )
                break
            except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException) as e:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"SGLang did not answer {endpoint}/ping within {timeout:.0f}s"
                    ) from e
                if retry_num % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.warning(
                        f"[Rank {rank}] Waiting for SGLang server... "
                        f"(elapsed: {elapsed:.1f}s, retries: {retry_num}, error: {type(e).__name__})"
                    )
                retry_num += 1
                time.sleep(0.5)


def _request_inference_to_update(
    url: str,
    socket_paths: dict[str, str],
    timeout: float = 300.0,
    uds: str | None = None,
    flush_cache: bool = True,
) -> None:
    """Request SGLang inference server to update weights from IPC."""
    logger.info(
        f"Requesting SGLang IPC weight update via {url} "
        f"(timeout={timeout:.0f}s, flush_cache={flush_cache})"
    )
    with httpx.Client(transport=httpx.HTTPTransport(uds=uds)) as client:
        response = client.post(
            url,
            json={
                "zmq_handles": socket_paths,
                "flush_cache": flush_cache,
            },
            timeout=timeout,
        )
        response.raise_for_status()
    logger.info("SGLang IPC weight update request completed")


class _Profiler:
    """Performance profiler for tracking fetch/broadcast/apply times."""

    def __init__(self, total_buckets: int):
        self.total_buckets = total_buckets
        self.fetch_times: list[float] = [0.0] * total_buckets
        self.bcast_times: list[float] = [0.0] * total_buckets
        self.apply_times: list[float] = [0.0] * total_buckets
        self.bucket_sizes: list[int] = [0] * total_buckets
        self.bucket_owners: list[int] = [0] * total_buckets
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def record(
        self,
        idx: int,
        *,
        fetch_time: float,
        bcast_time: float,
        apply_time: float,
        size: int,
        owner: int,
    ) -> None:
        """Record timing for a single bucket."""
        self.fetch_times[idx] = fetch_time
        self.bcast_times[idx] = bcast_time
        self.apply_times[idx] = apply_time
        self.bucket_sizes[idx] = size
        self.bucket_owners[idx] = owner

    def summary(self, rank: int, world_size: int) -> dict[str, Any]:
        """Generate performance summary."""
        total_time = self.end_time - self.start_time
        total_size = sum(self.bucket_sizes)

        def _stats(times: list[float]) -> dict[str, float]:
            valid = [t for t in times if t > 0]
            if not valid:
                return {"avg": 0.0, "p99": 0.0, "total": 0.0}
            sorted_times = sorted(valid)
            p99_idx = min(len(sorted_times) - 1, int(len(sorted_times) * 0.99))
            return {"avg": sum(valid) / len(valid), "p99": sorted_times[p99_idx], "total": sum(times)}

        fetch_stats = _stats(self.fetch_times)
        bcast_stats = _stats(self.bcast_times)
        apply_stats = _stats(self.apply_times)

        # Rank-wise distribution
        rank_buckets: dict[int, dict[str, Any]] = {}
        for i in range(world_size):
            rank_buckets[i] = {"count": 0, "size": 0, "time": 0.0}

        for idx in range(self.total_buckets):
            owner = self.bucket_owners[idx]
            rank_buckets[owner]["count"] += 1
            rank_buckets[owner]["size"] += self.bucket_sizes[idx]
            rank_buckets[owner]["time"] += self.fetch_times[idx]

        return {
            "total_time": total_time,
            "total_size": total_size,
            "throughput_gib_s": total_size / total_time / (1024**3) if total_time > 0 else 0.0,
            "fetch": fetch_stats,
            "bcast": bcast_stats,
            "apply": apply_stats,
            "rank_distribution": rank_buckets,
        }


def _log_summary(rank: int, world_size: int, summaries: list[dict[str, Any] | None]) -> None:
    """Log performance summary (rank 0 only)."""
    if rank != 0 or not summaries[0]:
        return

    summary = summaries[0]
    logger.info(
        f"Load completed in {summary['total_time']:.2f}s, "
        f"throughput={summary['throughput_gib_s']:.2f} GiB/s"
    )

    fetch = summary["fetch"]
    bcast = summary["bcast"]
    apply = summary["apply"]
    logger.info(
        f"fetch: avg={fetch['avg']*1000:.1f}ms p99={fetch['p99']*1000:.1f}ms total={fetch['total']:.2f}s | "
        f"bcast: avg={bcast['avg']*1000:.1f}ms p99={bcast['p99']*1000:.1f}ms total={bcast['total']:.2f}s | "
        f"apply: avg={apply['avg']*1000:.1f}ms p99={apply['p99']*1000:.1f}ms total={apply['total']:.2f}s"
    )

    # Calculate pipeline efficiency
    stage_sum = fetch['total'] + bcast['total'] + apply['total']
    efficiency = (stage_sum / summary['total_time'] - 1) * 100 if summary['total_time'] > 0 else 0
    logger.info(
        f"Pipeline efficiency: {efficiency:.1f}% "
        f"(stages sum {stage_sum:.2f}s / wall time {summary['total_time']:.2f}s = {stage_sum/summary['total_time']:.2f}x speedup)"
    )

    for r in range(world_size):
        if r < len(summaries) and summaries[r]:
            dist_info = summaries[r]["rank_distribution"][r]
            size_gib = dist_info["size"] / (1024**3)
            logger.info(
                f"Rank {r}: {dist_info['count']} buckets, {size_gib:.2f} GiB, {dist_info['time']:.2f}s"
            )


class _PipelineStage:
    """Pipeline stage with queue-based task distribution."""

    def __init__(
        self,
        rank: int,
        device: torch.device,
        store: MooncakeDistributedStore,
        buffer: torch.Tensor,
        max_bucket_size: int,
        socket: zmq.Socket,
        profiler: _Profiler,
        num_slots: int = 4,
        progress: Any = None,
    ):
        self.rank = rank
        self.device = device
        self.store = store
        self.buffer = buffer
        self.max_bucket_size = max_bucket_size
        self.socket = socket
        self.profiler = profiler
        self.num_slots = num_slots
        self.progress = progress

        # Pipeline queues (unbounded - slot management controls concurrency)
        self.fetch_queue: queue.Queue = queue.Queue()
        self.bcast_queue: queue.Queue = queue.Queue()
        self.apply_queue: queue.Queue = queue.Queue()

        # Slot management
        self.slot_available = [threading.Event() for _ in range(num_slots)]
        for event in self.slot_available:
            event.set()

        # Sequential apply: ensure in-order processing
        self.next_apply_idx = 0
        self.apply_lock = threading.Lock()
        self.apply_ready: dict[int, dict[str, Any]] = {}

        # Worker threads
        self.fetch_thread: threading.Thread | None = None
        self.bcast_thread: threading.Thread | None = None
        self.apply_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        # Failures raised inside worker threads, re-raised by wait(). Without
        # this a dead stage would leave its neighbours blocked on queues or
        # slot events and hang the load (and the SGLang worker) forever.
        self.errors: list[BaseException] = []

    def start(self) -> None:
        """Start pipeline worker threads."""
        self.fetch_thread = threading.Thread(target=self._fetch_worker, daemon=True)
        self.bcast_thread = threading.Thread(target=self._bcast_worker, daemon=True)
        self.apply_thread = threading.Thread(target=self._apply_worker, daemon=True)
        self.fetch_thread.start()
        self.bcast_thread.start()
        self.apply_thread.start()

    def submit(self, task: dict[str, Any]) -> None:
        """Submit a bucket task to the pipeline."""
        self.fetch_queue.put(task)

    def wait(self) -> None:
        """Wait for all pipeline stages to complete; raise on stage failure."""
        self.fetch_queue.put(None)
        if self.fetch_thread:
            self.fetch_thread.join()
        if self.bcast_thread:
            self.bcast_thread.join()
        if self.apply_thread:
            self.apply_thread.join()
        if self.errors:
            raise RuntimeError(f"pipeline failed: {self.errors[0]}") from self.errors[0]

    def cancel_and_wait(self) -> None:
        """Stop the pipeline and join all started threads.

        Must be called before tearing down resources the workers may still be
        touching (the ZMQ socket, the registered GPU buffer, the process
        group); freeing those under a live thread is a use-after-free. A
        worker parked in an uninterruptible call (store fetch, NCCL
        collective, socket recv) delays the join until that call's own
        timeout fires.
        """
        self.stop_event.set()
        self.fetch_queue.put(None)
        self.bcast_queue.put(None)
        self.apply_queue.put(None)
        for thread in (self.fetch_thread, self.bcast_thread, self.apply_thread):
            if thread is not None and thread.is_alive():
                thread.join()

    def _fetch_worker(self) -> None:
        """Fetch worker: get bucket from store."""
        try:
            self._fetch_loop()
        except BaseException as e:  # noqa: BLE001 - surfaced via wait()
            logger.exception(f"[Rank {self.rank}] fetch stage failed")
            self.errors.append(e)
            self.stop_event.set()
        finally:
            if self.stop_event.is_set():
                # Unblock the downstream stage even on abnormal exit.
                self.bcast_queue.put(None)

    def _fetch_loop(self) -> None:
        # Check if we should skip broadcast (gloo is slow, better to have all ranks fetch)
        skip_broadcast = dist.get_backend() == "gloo"

        while not self.stop_event.is_set():
            task = self.fetch_queue.get()
            if task is None:
                self.bcast_queue.put(None)
                break

            slot_id = task["slot_id"]
            # A slot frees when apply acks it; poll the stop flag so a dead
            # apply stage cannot park us here forever. Re-check after waking:
            # a failing apply stage force-sets every slot event to unblock us,
            # and reusing such a slot could overwrite bytes the worker is
            # still reading.
            while True:
                if self.stop_event.is_set():
                    raise RuntimeError("pipeline stopped while waiting for a free buffer slot")
                if self.slot_available[slot_id].wait(timeout=1.0):
                    if self.stop_event.is_set():
                        raise RuntimeError("pipeline stopped before reusing a buffer slot")
                    self.slot_available[slot_id].clear()
                    break

            owner_rank = task["owner_rank"]
            size = task["size"]
            bucket_key = task["bucket_key"]
            slot_offset = slot_id * self.max_bucket_size

            # For gloo backend: ALL ranks fetch from store directly (replicated fetch)
            # This is faster than gloo's slow TCP-based broadcast
            # For NCCL backend: only owner rank fetches, then broadcasts via NCCL
            fetch_elapsed = 0.0
            should_fetch = skip_broadcast or (self.rank == owner_rank)

            if should_fetch:
                slot_buffer = self.buffer.narrow(0, slot_offset, size)
                ptr = slot_buffer.data_ptr()
                fetch_start = time.perf_counter()
                bytes_read = self.store.get_into(bucket_key, ptr, size)
                fetch_elapsed = time.perf_counter() - fetch_start
                if bytes_read != size:
                    logger.error(
                        f"[Rank {self.rank}] get_into failed for {bucket_key}: "
                        f"expected {size} bytes, got {bytes_read} bytes"
                    )
                    raise RuntimeError(f"get_into returned {bytes_read}, expected {size}")

            task["fetch_time"] = fetch_elapsed
            task["skip_broadcast"] = skip_broadcast
            self.bcast_queue.put(task)

    def _bcast_worker(self) -> None:
        """Broadcast worker: distribute bucket to all ranks."""
        try:
            self._bcast_loop()
        except BaseException as e:  # noqa: BLE001 - surfaced via wait()
            # A peer rank dying mid-collective surfaces here after the NCCL
            # timeout (process-group timeout, 300s) rather than instantly.
            logger.exception(f"[Rank {self.rank}] broadcast stage failed")
            self.errors.append(e)
            self.stop_event.set()
        finally:
            if self.stop_event.is_set():
                self.apply_queue.put(None)

    def _bcast_loop(self) -> None:
        # Check if using gloo backend (needs CPU-based broadcast)
        use_cpu_bcast = dist.get_backend() == "gloo"

        while not self.stop_event.is_set():
            task = self.bcast_queue.get()
            if task is None:
                self.apply_queue.put(None)
                break

            idx = task["idx"]
            owner_rank = task["owner_rank"]
            size = task["size"]
            slot_id = task["slot_id"]
            slot_offset = slot_id * self.max_bucket_size
            slot_buffer = self.buffer.narrow(0, slot_offset, size)
            skip_broadcast = task.get("skip_broadcast", False)

            bcast_start = time.perf_counter()

            if skip_broadcast:
                # All ranks already fetched from store, just sync with barrier
                dist.barrier()
            elif use_cpu_bcast:
                # Gloo doesn't support CUDA tensor broadcast, use CPU intermediate
                cpu_buffer = slot_buffer.cpu()
                dist.broadcast(cpu_buffer, src=owner_rank, async_op=False)
                slot_buffer.copy_(cpu_buffer)
                del cpu_buffer
            else:
                # NCCL supports direct CUDA tensor broadcast
                dist.broadcast(slot_buffer, src=owner_rank, async_op=False)

            torch.cuda.synchronize(self.device)  # Required: ensure GPU kernel completes before apply worker reads
            bcast_elapsed = time.perf_counter() - bcast_start

            task["bcast_time"] = bcast_elapsed
            self.apply_queue.put(task)

    def _apply_worker(self) -> None:
        """Apply worker: send to SGLang via ZMQ IPC in strict order."""
        try:
            self._apply_loop()
        except BaseException as e:  # noqa: BLE001 - surfaced via wait()
            logger.exception(f"[Rank {self.rank}] apply stage failed")
            self.errors.append(e)
            self.stop_event.set()
        finally:
            if self.stop_event.is_set():
                # Unblock a fetch worker parked on a busy slot.
                for event in self.slot_available:
                    event.set()

    def _apply_loop(self) -> None:
        while not self.stop_event.is_set():
            task = self.apply_queue.get()
            if task is None:
                break

            idx = task["idx"]

            # Store task and wait for our turn
            with self.apply_lock:
                self.apply_ready[idx] = task

            # Process tasks in order
            while True:
                with self.apply_lock:
                    if self.next_apply_idx not in self.apply_ready:
                        break
                    current_task = self.apply_ready.pop(self.next_apply_idx)
                    self.next_apply_idx += 1

                items = current_task["items"]
                slot_id = current_task["slot_id"]
                slot_offset = slot_id * self.max_bucket_size

                apply_start = time.perf_counter()
                payload = _make_payload(items, slot_offset)
                self.socket.send_pyobj(payload)
                _recv_worker_ack(
                    self.socket, f"bucket {current_task['idx']} weight load"
                )
                apply_elapsed = time.perf_counter() - apply_start

                # Release slot after SGLang confirms data read
                self.slot_available[slot_id].set()

                self.profiler.record(
                    current_task["idx"],
                    fetch_time=current_task["fetch_time"],
                    bcast_time=current_task["bcast_time"],
                    apply_time=apply_elapsed,
                    size=current_task["size"],
                    owner=current_task["owner_rank"],
                )

                # Update progress bar after task completion
                if self.progress is not None:
                    self.progress.update(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_hostname = os.getenv("MOONCAKE_HOSTNAME", socket.gethostname())
    # Support both MOONCAKE_TE_META_DATA_SERVER (new) and MOONCAKE_METADATA_SERVER (legacy).
    # P2PHANDSHAKE needs no central metadata service; every component of the
    # toolchain (dump/health-check/loader) defaults to it so they never
    # disagree silently.
    default_metadata = os.getenv("MOONCAKE_TE_META_DATA_SERVER",
                                 os.getenv("MOONCAKE_METADATA_SERVER", "P2PHANDSHAKE"))
    default_master = os.getenv("MOONCAKE_MASTER", os.getenv("MOONCAKE_MASTER_ADDR", "localhost:50051"))
    default_parallel = int(os.getenv("MOONCAKE_PARALLEL_SIZE", "1"))
    default_protocol = os.getenv("MOONCAKE_PROTOCOL", "rdma")
    default_rdma_devices = os.getenv("MOONCAKE_DEVICE", "")
    # Note: segment_size=0 and buffer_size=0 mean "use library defaults" for sglang
    # but we need explicit values for store.get() to work.
    # The local buffer must hold the manifest in one get(): large-model
    # manifests (per-tensor metadata for hundreds of buckets) run to tens of
    # MB, and an undersized buffer surfaces as "manifest not found".
    default_segment_size = int(os.getenv("MOONCAKE_SEGMENT_SIZE", "0"))
    env_buffer_size = int(os.getenv("MOONCAKE_BUFFER_SIZE", str(256 * 1024 * 1024)))
    default_buffer_size = env_buffer_size if env_buffer_size > 0 else 256 * 1024 * 1024  # Ensure non-zero
    default_update_timeout = float(os.getenv("SGLANG_IPC_UPDATE_TIMEOUT", "1800"))

    parser = argparse.ArgumentParser(
        description="One-shot loader: Mooncake Store checkpoint -> running SGLang server"
    )
    parser.add_argument("--checkpoint-name", required=True, help="Checkpoint name used at dump time")
    parser.add_argument("--hostname", default=default_hostname)
    parser.add_argument("--metadata-server", default=default_metadata)
    parser.add_argument("--master-addr", default=default_master)
    parser.add_argument("--segment-size", type=int, default=default_segment_size)
    parser.add_argument("--buffer-size", type=int, default=default_buffer_size)
    parser.add_argument("--protocol", default=default_protocol, choices=["tcp", "rdma"])
    parser.add_argument("--rdma-devices", default=default_rdma_devices)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--endpoint", default="http://localhost:30000")
    parser.add_argument("--parallel-size", type=int, default=default_parallel, dest="inference_parallel_size")
    parser.add_argument("--uds", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--rebalance-buckets", default="round_robin", choices=["none", "round_robin"])
    parser.add_argument("--pipeline-depth", type=int, default=4, help="Number of buffer slots (default: 4)")
    parser.add_argument("--flush-cache", action="store_true", help="Flush cache after weight update")
    parser.add_argument(
        "--update-timeout",
        type=float,
        default=default_update_timeout,
        help="Timeout in seconds for the /update_weights_from_ipc request",
    )
    args = parser.parse_args()
    if args.pipeline_depth <= 0:
        parser.error("--pipeline-depth must be positive")
    if args.inference_parallel_size <= 0:
        parser.error("--parallel-size must be positive")
    if args.update_timeout <= 0:
        parser.error("--update-timeout must be positive")
    return args


def init_gpu() -> tuple[int, int, int, torch.device]:
    """Initialize CUDA environment (without NCCL).

    Returns:
        (rank, world_size, gpu_index, device)
    """
    assert torch.cuda.is_available(), "CUDA required for GDR"

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank_env = os.getenv("LOCAL_RANK")
    gpu_index = int(local_rank_env) if local_rank_env is not None else rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda", gpu_index)
    torch.cuda.empty_cache()

    return rank, world_size, gpu_index, device


def init_store(
    args: argparse.Namespace,
    rank: int,
) -> MooncakeDistributedStore:
    """Initialize Mooncake store connection only (no manifest load).

    Args:
        args: Parsed arguments
        rank: Current rank

    Returns:
        Initialized MooncakeDistributedStore
    """
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as e:
        raise ImportError(
            "mooncake-transfer-engine is required for the Mooncake loader. "
            "Install it with: pip install mooncake-transfer-engine"
        ) from e

    t0 = time.perf_counter()
    store = MooncakeDistributedStore()
    ret = store.setup(
        args.hostname,
        args.metadata_server,
        args.segment_size,
        args.buffer_size,
        args.protocol,
        args.rdma_devices,
        args.master_addr,
    )
    if ret != 0:
        logger.error(
            f"[Rank {rank}] Failed to setup MooncakeDistributedStore with error code {ret}. "
            f"Parameters: hostname={args.hostname}, metadata_server={args.metadata_server}, "
            f"protocol={args.protocol}, rdma_devices={args.rdma_devices}"
        )
        raise RuntimeError(f"Store setup failed with error code {ret}")
    if rank == 0:
        logger.info(f"Store setup completed in {time.perf_counter() - t0:.2f}s")
    return store


def init_resources(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> tuple[MooncakeDistributedStore, list[dict[str, Any]]]:
    """Initialize Mooncake store and load manifest.

    Args:
        args: Parsed arguments
        rank: Current rank
        world_size: Total number of ranks

    Returns:
        (store, buckets)
    """
    ckpt_name = args.checkpoint_name

    # Setup store
    store = init_store(args, rank)

    # Load and rebalance manifest
    t0 = time.perf_counter()
    prefix = args.prefix or f"ckpt:{ckpt_name}"
    manifest_key = f"{prefix}:manifest"
    manifest = _load_manifest(store, manifest_key)
    manifest = _rebalance_manifest(manifest, world_size, args.rebalance_buckets)
    buckets = manifest.get("buckets", [])
    # With --rebalance-buckets none the manifest's owner_rank is used as-is;
    # an owner outside the group would make dist.broadcast fail obscurely.
    bad_owner = [
        b["bucket_key"] for b in buckets
        if not 0 <= b.get("owner_rank", 0) < world_size
    ]
    if bad_owner:
        raise ValueError(
            f"{len(bad_owner)} bucket(s) have owner_rank outside "
            f"world_size={world_size} (e.g. {bad_owner[0]}); "
            "use --rebalance-buckets round_robin or fix the manifest"
        )

    if rank == 0:
        owner_counts = {}
        for entry in buckets:
            owner = entry.get("owner_rank", 0)
            owner_counts[owner] = owner_counts.get(owner, 0) + 1
        dist_str = ", ".join(f"rank{k}={v}" for k, v in sorted(owner_counts.items()))
        logger.info(f"Manifest loaded in {time.perf_counter() - t0:.2f}s, bucket distribution: {dist_str}")

    return store, buckets


def _alloc_buffer(
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    store: MooncakeDistributedStore,
    buckets: list[dict[str, Any]],
) -> tuple[torch.Tensor, int]:
    """Allocate and register GPU buffer for weight loading.

    Args:
        args: Parsed arguments
        rank: Current rank
        device: CUDA device
        store: Mooncake store
        buckets: List of bucket metadata

    Returns:
        (buffer, max_bucket_size)
    """
    max_bucket_size = max(entry["size"] for entry in buckets)

    # Note: Cannot use torch_memory_saver here because buffer needs CUDA IPC support
    # (reduce_tensor requires _share_cuda_() which is incompatible with torch_memory_saver)
    buffer = torch.empty(max_bucket_size * args.pipeline_depth, dtype=torch.uint8, device=device)

    ret = store.register_buffer(buffer.data_ptr(), buffer.nbytes)
    if ret != 0:
        logger.error(
            f"[Rank {rank}] Failed to register buffer with error code {ret}. "
            f"Buffer size: {buffer.nbytes / (1024**3):.2f} GiB, "
            f"max_bucket_size: {max_bucket_size / (1024**2):.2f} MiB, pipeline_depth: {args.pipeline_depth}"
        )
        raise RuntimeError(f"Buffer registration failed with error code {ret}")
    return buffer, max_bucket_size


def _recv_worker_ack(
    sock: zmq.Socket,
    phase: str,
    *,
    acknowledge_initial_error: bool = False,
) -> None:
    """Receive a checkpoint-engine reply, accepting only the empty success ack.

    The worker signals success with exactly ``b""``. On a bucket-load failure
    it replies with the traceback string and *keeps serving* (it expects the
    peer to send an Exception payload to shut it down), so treating any reply
    as success would silently leave dummy weights in place.
    """
    reply = sock.recv()
    if reply == b"":
        return

    detail = reply.decode("utf-8", errors="replace")
    if acknowledge_initial_error:
        # The worker's initial-handshake error branch sends its traceback and
        # then blocks in recv for one final message before re-raising; answer
        # it so the worker can exit instead of waiting forever.
        try:
            sock.send(b"")
        except zmq.ZMQError:
            logger.exception("Failed to acknowledge initial IPC error")

    raise RuntimeError(
        f"SGLang checkpoint-engine worker failed during {phase}:\n{detail}"
    )


def _send_abort_to_worker(sock: zmq.Socket | None, exc: BaseException) -> bool:
    """Best-effort: tell the SGLang-side worker to abort a failed load.

    The checkpoint-engine worker raises any Exception payload it receives, so
    on success the HTTP update returns failure instead of the worker staying
    blocked in recv forever. Only protocol-legal when the REQ socket is in the
    send state — true for fetch/broadcast/apply failures (the worker's error
    reply completes the recv). After a recv timeout (apply ack, Phase 5/6) the
    socket is in the recv state, the send fails with EFSM, and a worker that
    is stuck rather than dead can only be recovered by restarting the server.

    Returns True if the abort was queued; the caller must then close the
    socket with a bounded positive linger so the message can actually flush.
    """
    if sock is None:
        return False
    try:
        sock.setsockopt(zmq.SNDTIMEO, 2000)
        sock.send_pyobj(RuntimeError(f"mooncake loader aborted: {exc}"))
        logger.info("Abort signal sent to SGLang worker")
        return True
    except zmq.ZMQError as ze:
        logger.warning(
            f"Could not send abort to SGLang worker ({ze}); if the server "
            "stays in a loading state it must be restarted"
        )
        return False


def do_load(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    gpu_index: int,
    device: torch.device,
    store: MooncakeDistributedStore,
    buckets: list[dict[str, Any]],
    buffer: torch.Tensor,
    max_bucket_size: int,
) -> tuple[bool, str]:
    """Execute a single weight loading operation.

    NCCL is initialized at the start and destroyed at the end.
    Other resources (GPU, store, buffer) are managed externally.

    Args:
        args: Parsed arguments
        rank: Current rank
        world_size: Total number of ranks
        gpu_index: GPU index
        device: CUDA device
        store: Mooncake store (pre-initialized)
        buckets: List of bucket metadata
        buffer: Pre-allocated buffer
        max_bucket_size: Max bucket size

    Returns:
        (success, message)
    """
    ep = args.endpoint
    do_flush = args.flush_cache

    zmq_ctx: zmq.Context | None = None
    socket: zmq.Socket | None = None
    req_thread: threading.Thread | None = None
    req_thread_error: list[Exception] = []
    profiler: _Profiler | None = None
    pipeline: _PipelineStage | None = None
    load_failed = False

    try:
        # ========== Phase 1: Initialize process group ==========
        _init_process_group(device_id=gpu_index)
        if rank == 0:
            logger.info(f"Loading checkpoint -> {ep}")

        # ========== Phase 2: Wait for SGLang ready ==========
        _check_sglang_ready(
            ep,
            args.inference_parallel_size,
            args.uds,
            rank,
            timeout=args.update_timeout,
        )
        dist.barrier()
        if rank == 0:
            logger.info("All ranks confirmed SGLang server is ready, starting weight loading...")

        # ========== Phase 3: Setup ZMQ IPC ==========
        logger.info(f"[Rank {rank}] Phase 3: Setting up ZMQ IPC...")
        device_uuid = _get_physical_gpu_id(gpu_index)
        logger.info(f"[Rank {rank}] Device UUID: {device_uuid}, gpu_index={gpu_index}")
        zmq_ctx = zmq.Context()
        # Per-load nonce: the abstract-namespace name must not repeat across
        # load rounds, otherwise a stale reply left over from a previous round
        # can reconnect into the next round's socket and corrupt the protocol.
        zmq_handle = (
            f"ipc://@load-{device_uuid}-{rank}-{os.getpid()}-{time.time_ns()}.sock"
        )
        socket = zmq_ctx.socket(zmq.REQ)
        socket.bind(zmq_handle)
        logger.info(f"[Rank {rank}] ZMQ socket bound at {zmq_handle}")
        # Handshake window: the worker only connects once the server has
        # processed the HTTP update request, which on large models can trail
        # process start by minutes. Bound it by the overall update timeout so
        # a server that never picks up the request fails the load instead of
        # hanging it.
        handshake_timeout_ms = int(args.update_timeout * 1000)
        socket.setsockopt(zmq.SNDTIMEO, handshake_timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, handshake_timeout_ms)

        socket_paths = _gather_objects((device_uuid, zmq_handle))
        logger.info(f"[Rank {rank}] Socket paths gathered: {socket_paths}")

        def req_func(paths: list[tuple[str, str]]) -> None:
            try:
                src = rank // args.inference_parallel_size * args.inference_parallel_size
                logger.info(f"[Rank {rank}] req_func: src={src}, inference_parallel_size={args.inference_parallel_size}, paths length={len(paths)}")
                if rank == src:
                    paths_slice = paths[src : src + args.inference_parallel_size]
                    zmq_dict = dict(paths_slice)
                    logger.info(f"[Rank {rank}] Sending zmq_handles to SGLang: {zmq_dict}")
                    _request_inference_to_update(
                        f"{ep}/update_weights_from_ipc",
                        zmq_dict,
                        timeout=args.update_timeout,
                        uds=args.uds,
                        flush_cache=do_flush,
                    )
            except Exception as e:
                req_thread_error.append(e)
                logger.exception(f"[Rank {rank}] Failed to request SGLang IPC weight update")

        # daemon=True: the thread only carries an HTTP POST; on early failure
        # paths the process must be able to exit without waiting out the full
        # --update-timeout on an abandoned request.
        req_thread = threading.Thread(target=req_func, args=(socket_paths,), daemon=True)
        req_thread.start()

        logger.info(f"[Rank {rank}] Sending CUDA IPC handle...")

        # Send CUDA IPC handle
        handle = reduce_tensor(buffer)
        socket.send_pyobj(handle)
        logger.info(f"[Rank {rank}] Waiting for CUDA IPC handle response...")
        _recv_worker_ack(
            socket,
            "CUDA IPC handle reconstruction",
            acknowledge_initial_error=True,
        )
        logger.info(f"[Rank {rank}] CUDA IPC handle response received")
        # Steady state: per-bucket apply acks arrive within seconds; a much
        # tighter bound than the handshake so a worker that dies mid-load
        # fails the pipeline instead of hanging it. Phase 6 manages its own
        # timeouts on top of these.
        socket.setsockopt(zmq.RCVTIMEO, APPLY_ACK_TIMEOUT_MS)

        # ========== Phase 4: Execute pipeline ==========
        logger.info(f"[Rank {rank}] Phase 4: Creating pipeline (num_buckets={len(buckets)})...")
        profiler = _Profiler(len(buckets))

        # Create progress bar before pipeline
        progress = None
        if sys.stderr.isatty() and rank == 0:
            progress = tqdm(total=len(buckets), unit="bucket", desc="Loading")

        if args.pipeline_depth <= 0:
            raise ValueError("--pipeline-depth must be positive")
        # Build every task before starting the pipeline: an exception past
        # start() would tear down resources the worker threads still use.
        tasks = [
            {
                "idx": idx,
                "owner_rank": entry["owner_rank"],
                "size": entry["size"],
                "bucket_key": entry["bucket_key"],
                "items": entry["items"],
                "slot_id": idx % args.pipeline_depth,
            }
            for idx, entry in enumerate(buckets)
        ]

        pipeline = _PipelineStage(
            rank, device, store, buffer, max_bucket_size, socket, profiler, args.pipeline_depth, progress
        )
        pipeline.start()
        logger.info(f"[Rank {rank}] Pipeline started, submitting {len(buckets)} buckets...")

        profiler.start_time = time.perf_counter()

        for task in tasks:
            pipeline.submit(task)

        pipeline.wait()
        if progress is not None:
            progress.close()

        profiler.end_time = time.perf_counter()

        # Finalize the checkpoint-engine IPC protocol (two None rounds on
        # checkpoint-engine 0.4.0, mirroring upstream ps.py):
        #   None #1 -> worker releases the IPC buffer and acks.
        #   None #2 -> worker runs post_hook (quant re-processing; seconds to
        #              minutes on large models) and acks.
        logger.info(f"[Rank {rank}] Phase 5: Finalizing IPC update (release)...")
        socket.send_pyobj(None)
        _recv_worker_ack(socket, "buffer release finalize")
        logger.info(f"[Rank {rank}] Phase 5: Release ack received")

        # The post_hook round is mandatory on the two-stage protocol this
        # loader requires (checkpoint-engine >= 0.4.0) and its ack MUST be
        # awaited (see POST_HOOK_ACK_TIMEOUT_MS above). A send failure here
        # means the peer is gone or speaks a different protocol version —
        # both are load failures, never a compatibility case to skip.
        old_sndtimeo = socket.getsockopt(zmq.SNDTIMEO)
        old_rcvtimeo = socket.getsockopt(zmq.RCVTIMEO)
        post_hook_error = None
        phase6_step = "send"
        try:
            socket.setsockopt(zmq.SNDTIMEO, POST_HOOK_SEND_TIMEOUT_MS)
            socket.setsockopt(zmq.RCVTIMEO, POST_HOOK_ACK_TIMEOUT_MS)
            logger.info(f"[Rank {rank}] Phase 6: Requesting post-hook finalize...")
            socket.send_pyobj(None)
            phase6_step = "receive"
            _recv_worker_ack(socket, "post-hook finalize")
            logger.info(f"[Rank {rank}] Phase 6: Post-hook ack received")
        except Exception as exc:
            post_hook_error = (
                f"post-hook finalize {phase6_step} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            logger.error(f"[Rank {rank}] Phase 6: {post_hook_error}")
        finally:
            try:
                socket.setsockopt(zmq.SNDTIMEO, old_sndtimeo)
                socket.setsockopt(zmq.RCVTIMEO, old_rcvtimeo)
            except Exception as exc:
                post_hook_error = post_hook_error or (
                    f"failed to restore Phase 6 socket options: {exc}"
                )

        # All-rank consensus on Phase 6 before any further collective: if a
        # rank raised alone it would skip the final dist.barrier() below and
        # deadlock the surviving ranks inside it. Either every rank continues
        # or every rank fails this load.
        phase6_errors = _gather_objects(post_hook_error)
        failed_ranks = [(i, e) for i, e in enumerate(phase6_errors) if e]
        if failed_ranks:
            summary = "; ".join(f"rank{i}: {e}" for i, e in failed_ranks[:4])
            if len(failed_ranks) > 4:
                summary += f"; ... ({len(failed_ranks)} ranks total)"
            raise TimeoutError(
                f"post-hook finalize failed on {len(failed_ranks)}/{world_size} "
                f"rank(s): {summary} — failing the load on all ranks instead of "
                "abandoning workers mid-finalize (a stranded final ack deadlocks "
                "the worker's scheduler thread in zmq ctx.term())"
            )

        if req_thread is not None:
            req_thread.join()
        # Share request outcomes across ranks: only the group-source rank
        # POSTs, and without this the other ranks would sail into the final
        # barrier and stall on a failure only the source knows about.
        local_error = (
            f"{type(req_thread_error[0]).__name__}: {req_thread_error[0]}"
            if req_thread_error
            else None
        )
        request_errors = _gather_objects(local_error)
        failed_requests = [(i, err) for i, err in enumerate(request_errors) if err]
        if failed_requests:
            detail = "; ".join(f"rank{i}: {err}" for i, err in failed_requests)
            raise RuntimeError(f"SGLang IPC weight update request failed: {detail}")

        # Final synchronization
        dist.barrier()

        # Log summary
        if profiler is not None:
            summary = profiler.summary(rank, world_size)
            summaries = _gather_objects(summary)
            _log_summary(rank, world_size, summaries)

        if rank == 0:
            logger.info("Weight loading completed successfully")

        return True, "Success"

    except Exception as e:
        logger.error(f"[Rank {rank}] Error during weight loading: {e}")
        import traceback
        traceback.print_exc()
        # Join pipeline threads before touching the socket: wait() joins on
        # the normal path, but an exception between start() and wait() would
        # otherwise leave workers alive while we abort and free resources.
        if pipeline is not None:
            pipeline.cancel_and_wait()
        load_failed = True
        _send_abort_to_worker(socket, e)
        return False, str(e)

    finally:
        # Cleanup ZMQ resources. On failure paths close with a bounded
        # positive linger: a queued abort — or the ack that unblocks the
        # worker's initial-error recv — still needs to flush, and linger=0
        # would drop it. On success nothing is queued and close is instant.
        if socket is not None:
            socket.close(linger=ABORT_LINGER_MS if load_failed else 0)
        if zmq_ctx is not None:
            zmq_ctx.term()

        # Cleanup NCCL process group
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
                logger.info(f"Rank {rank}: Process group destroyed")
            except Exception as e:
                logger.warning(f"Rank {rank}: Failed to destroy process group: {e}")



def run_once(args: argparse.Namespace) -> None:
    """Run a single load operation."""
    # Initialize GPU (once)
    rank, world_size, gpu_index, device = init_gpu()

    if rank == 0:
        logger.info(
            f"Loading {args.checkpoint_name} (pipeline_depth={args.pipeline_depth})"
        )

    store: MooncakeDistributedStore | None = None
    buffer: torch.Tensor | None = None

    try:
        # Initialize store and buffer
        store, buckets = init_resources(args, rank, world_size)

        # Fail fast on an incomplete checkpoint (e.g. the store restarted and
        # lost its data) before allocating GPU buffers and involving the
        # server. Store contents are volatile; this check is cheap.
        complete, missing = verify_checkpoint_integrity(
            store, {"buckets": buckets}, rank=rank
        )
        if not complete:
            raise RuntimeError(
                f"Checkpoint '{args.checkpoint_name}' is incomplete in the store: "
                f"{len(missing)} bucket(s) missing (e.g. {missing[0]}). "
                "Re-upload it with dump_to_mooncake.py"
            )

        buffer, max_bucket_size = _alloc_buffer(args, rank, device, store, buckets)

        # Execute load (NCCL init/destroy inside)
        success, message = do_load(
            args, rank, world_size, gpu_index, device,
            store, buckets, buffer, max_bucket_size
        )

        if not success:
            logger.error(f"Load failed: {message}")
            sys.exit(1)

        if rank == 0:
            logger.info(f"Completed loading {args.checkpoint_name}")

    finally:
        # Cleanup buffer
        if buffer is not None and store is not None:
            try:
                store.unregister_buffer(buffer.data_ptr())
            except Exception:
                pass
            del buffer
            torch.cuda.empty_cache()

        # Cleanup store
        if store is not None:
            try:
                store.close()
            except Exception:
                pass



def main() -> None:
    """Main entry point."""
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())
    run_once(args)


if __name__ == "__main__":
    main()
