"""KVTransferEngine backed by Mooncake's native TENT Python module (`import tent`)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.distributed.device_communicators.kv_transfer_engine import (
    KVTransferEngine,
    TransferIntent,
)
from sglang.srt.environ import envs
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

# Same budget as mooncake.engine's batch_transfer_sync: MC_TRANSFER_TIMEOUT
# seconds (default 30, floor 5) plus 1 ns per byte.
_DEFAULT_TRANSFER_TIMEOUT_S = 30
_MIN_TRANSFER_TIMEOUT_S = 5

# TENT's own default metrics port; each process adds its GPU id. That id is
# unique among processes that see the same devices (a TP rank is not: DP
# replicas repeat it), but not across per-process CUDA_VISIBLE_DEVICES. On a
# clash TENT logs a warning and the losing process keeps log-only metrics.
_METRICS_BASE_PORT = 9100

# The only data paths this backend enables (see _build_tent_config).
_SUPPORTED_PROTOCOLS = ("", "rdma", "tcp")

# Requests per TENT batch and submit threads. A batch status poll scans every task,
# and submit_transfer releases the GIL, so small chunks on a pool overlap building
# requests with submitting them. More threads contend with the scheduler thread
# for the GIL and slow scheduling. Building one tent.Request per block in Python
# is the per-request cost, so small blocks are bound by CPU, not the NIC.
_SUBMIT_CHUNK = 128
_SUBMIT_THREADS = 2
# Busy-poll this many times per batch before yielding the CPU between polls.
_SPIN_POLLS = 64
_POLL_SLEEP_S = 20e-6

# tent.Request arguments after `length`: (priority, transport_hint, policy_name,
# deadline_ns, intent_type). Empty means TENT's defaults.
_RequestTail = Tuple[Any, ...]
# A submitted TENT batch and how many tasks it holds (task ids are 0..n-1).
_Batch = Tuple[int, int]


def _build_tent_config(
    hostname: str, ib_device: Optional[str], gpu_id: Optional[int]
) -> dict:
    protocol = envs.MOONCAKE_PROTOCOL.get().strip().lower()
    if protocol not in _SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"SGLANG_MOONCAKE_TRANSFER_ENGINE=tent supports MOONCAKE_PROTOCOL "
            f"rdma or tcp, got {protocol!r}; use the classic engine (te) for it."
        )
    force_tcp = protocol == "tcp" or os.environ.get("MC_FORCE_TCP") == "1"
    config = {
        "metadata_type": "p2p",
        "rpc_server_hostname": hostname,
        "rpc_server_port": 0,
        "log_level": "warning",
        # TENT reads this file before its TENT_METRICS_* env vars, so the port
        # set here wins. Without a GPU id there is no port known to be free on
        # the host, so leave the endpoint off rather than collide. Only a TENT
        # built with -DTENT_METRICS_ENABLED=ON serves it at all.
        "metrics": (
            {"enabled": True, "http_port": _METRICS_BASE_PORT + gpu_id}
            if gpu_id is not None
            else {"enabled": False}
        ),
        # Same-host GPU peers use NVLink; TENT skips it across machines and
        # uses RDMA. SHM stays off. TCP mode disables both device transports.
        "transports": {
            "rdma": {"enable": not force_tcp},
            "tcp": {"enable": force_tcp},
            "hp_tcp": {"enable": False},
            "shm": {"enable": False},
            "nvlink": {"enable": not force_tcp},
            "mnnvl": {"enable": False},
            "gds": {"enable": False},
            "io_uring": {"enable": False},
        },
    }
    if ib_device and not force_tcp:
        config["topology"] = {
            "rdma_whitelist": [d.strip() for d in ib_device.split(",") if d.strip()]
        }
    override = envs.SGLANG_TENT_CONFIG_OVERRIDE.get()
    if override:
        overrides = json.loads(override)
        if not isinstance(overrides, dict):
            raise ValueError("SGLANG_TENT_CONFIG_OVERRIDE must be a JSON object")
        # Every process on the host reads the same override, so a fixed
        # metrics.http_port here makes all ranks but one fail to bind it.
        _deep_merge(config, overrides)
    return config


def _deep_merge(base: dict, extra: dict) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _transfer_timeout_ns() -> int:
    timeout_env = os.environ.get("MC_TRANSFER_TIMEOUT")
    if not timeout_env:
        return _DEFAULT_TRANSFER_TIMEOUT_S * 1_000_000_000
    try:
        timeout_s = int(timeout_env)
    except ValueError as e:
        raise ValueError(
            f"MC_TRANSFER_TIMEOUT must be an integer number of seconds, got {timeout_env!r}"
        ) from e
    return max(_MIN_TRANSFER_TIMEOUT_S, timeout_s) * 1_000_000_000


@dataclass
class _Segment:
    """An open peer segment, shared by every in-flight transfer to that peer."""

    handle: int
    # batch_transfer_sync / send_probe calls currently using `handle`.
    users: int = 0
    # A transfer failed: close once the last user is done, then re-resolve.
    stale: bool = False


class TentTransferEngine(KVTransferEngine):
    """Shared TENT engine for RDMA transfers; one per process, never torn down.

    Like the classic engine it is a process singleton, so its submit pool lives
    until the process exits; KVManager teardown on a P<->D role switch reuses it.
    """

    def __init__(
        self, hostname: str, ib_device: Optional[str], gpu_id: Optional[int] = None
    ):
        try:
            import tent
        except ImportError as e:
            raise ImportError(
                "SGLANG_MOONCAKE_TRANSFER_ENGINE=tent needs the native `tent` "
                "module. Please install mooncake-transfer-engine>=0.3.14, or "
                "build Mooncake with -DUSE_TENT=ON and put tent.*.so on "
                "PYTHONPATH."
            ) from e
        self._tent = tent
        # The intent only tags requests with a priority and intent type. Traffic
        # is isolated only if SGLANG_TENT_CONFIG_OVERRIDE configures separate
        # foreground/background QP pools; otherwise both share the same QPs.
        # Passed positionally: pybind11's keyword path is slower per request.
        self._request_tail: Dict[Optional[TransferIntent], _RequestTail] = {
            None: (),
            TransferIntent.FOREGROUND: (
                tent.PRIO_HIGH,
                tent.TransportType.UNSPEC,
                None,
                0,
                tent.IntentType.FOREGROUND_GET,
            ),
            TransferIntent.BACKGROUND: (
                tent.PRIO_LOW,
                tent.TransportType.UNSPEC,
                None,
                0,
                tent.IntentType.BACKGROUND_PREFETCH,
            ),
        }
        self._failed_states = (
            tent.TransferStatusEnum.FAILED,
            tent.TransferStatusEnum.TIMEOUT,
            tent.TransferStatusEnum.CANCELED,
            tent.TransferStatusEnum.INVALID,
        )
        self.ib_device = ib_device

        config = _build_tent_config(hostname, ib_device, gpu_id)
        with tempfile.NamedTemporaryFile(
            "w", prefix="sglang-tent-", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            config_path = f.name
        try:
            self.engine = tent.TransferEngine(config_path)
        finally:
            os.unlink(config_path)
        if not self.engine.available():
            raise RuntimeError(
                f"TENT TransferEngine failed to start (hostname={hostname!r}, "
                f"ib_device={ib_device!r})"
            )

        # p2p segment names are "host:rpc_port", the same format as a classic
        # session id, so the PD bootstrap exchange needs no change.
        self.session_id = NetworkAddress(
            hostname, self.engine.get_rpc_server_port()
        ).to_host_port_str()
        self._segments: Dict[str, _Segment] = {}
        self._segment_lock = threading.Lock()
        # Registered region sizes, since TENT's unregister needs the length.
        self._registered: Dict[int, int] = {}
        self._submit_pool = ThreadPoolExecutor(
            max_workers=_SUBMIT_THREADS, thread_name_prefix="tent-submit"
        )

        self._timeout_ns = _transfer_timeout_ns()
        logger.info(
            "TENT transfer engine ready: session_id=%s ib_device=%s metrics=%s",
            self.session_id,
            ib_device,
            config["metrics"],
        )

    def register(self, ptr: int, length: int) -> None:
        try:
            self.engine.register_local_memory(ptr, length)
        except self._tent.TentException as exc:
            raise RuntimeError("TENT memory registration failed") from exc
        self._registered[ptr] = length

    def deregister(self, ptr: int) -> None:
        try:
            self.engine.unregister_local_memory(ptr, self._registered.pop(ptr, 0))
        except self._tent.TentException as exc:
            raise RuntimeError("TENT memory deregistration failed") from exc

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        try:
            self.engine.register_local_memory_batch(list(ptrs), list(lengths))
        except self._tent.TentException as exc:
            logger.debug("TENT batch memory registration failed: %s", exc)
            return -1
        self._registered.update(zip(ptrs, lengths))
        return 0

    def batch_deregister(self, ptrs: List[int]) -> int:
        lengths = [self._registered.pop(ptr, 0) for ptr in ptrs]
        try:
            self.engine.unregister_local_memory_batch(list(ptrs), lengths)
        except self._tent.TentException as exc:
            logger.debug("TENT batch memory deregistration failed: %s", exc)
            return -1
        return 0

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
        intent: Optional[TransferIntent] = None,
    ) -> int:
        if not (len(buffers) == len(peer_buffer_addresses) == len(lengths)):
            logger.error("buffers, peer_buffer_addresses and lengths differ in size")
            return -1
        if not buffers:
            return 0
        rc = -1
        handle = self._acquire_segment(session_id)
        try:
            rc = self._write_and_wait(
                handle=handle,
                buffers=buffers,
                peer_buffer_addresses=peer_buffer_addresses,
                lengths=lengths,
                request_tail=self._request_tail[intent],
            )
        except self._tent.TentException as exc:
            logger.error(
                "TENT batch transfer to %s failed (%d requests): %s",
                session_id,
                len(buffers),
                exc,
            )
        finally:
            # On failure, re-resolve the peer once the other layers' transfers
            # to it are done, in case it restarted on a new segment.
            self._release_segment(session_id, failed=rc != 0)
        return rc

    def send_probe(self, peer_session_id: str) -> int:
        # TENT has no liveness RPC. open_segment is local bookkeeping only, but
        # closing first bumps TENT's segment version, which invalidates the
        # cached descriptor, so get_segment_info must fetch it from the peer's
        # RPC server: success means the peer answers.
        with self._segment_lock:
            segment = self._segments.get(peer_session_id)
            if segment is not None:
                if segment.users:
                    # Closing would pull the handle from under those transfers,
                    # and without closing the descriptor may come from cache.
                    # Report not-alive-yet and let the next probe pass retry.
                    return -1
                self._close_segment_locked(peer_session_id)
        handle = self._acquire_segment(peer_session_id)
        failed = True
        try:
            self.engine.get_segment_info(handle)
            failed = False
        except self._tent.TentException:
            pass
        finally:
            self._release_segment(peer_session_id, failed=failed)
        return -1 if failed else 0

    def _acquire_segment(self, session_id: str) -> int:
        with self._segment_lock:
            segment = self._segments.get(session_id)
            if segment is None:
                segment = _Segment(handle=self.engine.open_segment(session_id))
                self._segments[session_id] = segment
            # A stale segment is still reused: TENT returns the same handle for
            # the same name until it is closed, which waits for the last user.
            segment.users += 1
            return segment.handle

    def _release_segment(self, session_id: str, *, failed: bool) -> None:
        with self._segment_lock:
            segment = self._segments[session_id]
            segment.users -= 1
            segment.stale = segment.stale or failed
            if segment.stale and not segment.users:
                # Under the lock, so no acquire can get this handle mid-close.
                self._close_segment_locked(session_id)

    def _close_segment_locked(self, session_id: str) -> None:
        handle = self._segments.pop(session_id).handle
        try:
            self.engine.close_segment(handle)
        except self._tent.TentException as exc:
            logger.debug("TENT close_segment(%s) failed: %s", session_id, exc)

    def _write_and_wait(
        self,
        *,
        handle: int,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
        request_tail: _RequestTail,
    ) -> int:
        deadline_ns = time.monotonic_ns() + self._timeout_ns + sum(lengths)
        batches = self._submit_all(
            handle=handle,
            buffers=buffers,
            peer_buffer_addresses=peer_buffer_addresses,
            lengths=lengths,
            request_tail=request_tail,
        )
        done = False
        try:
            for batch_id, _ in batches:
                if self._wait_batch(batch_id=batch_id, deadline_ns=deadline_ns) != 0:
                    return -1
            done = True
            return 0
        finally:
            if not done:
                self._cancel_batches(batches)
            self._free_batches(batches)

    def _submit_all(
        self,
        *,
        handle: int,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
        request_tail: _RequestTail,
    ) -> List[_Batch]:
        """Submit every chunk; on any failure abort what was submitted and re-raise."""

        def submit(start: int) -> _Batch:
            end = start + _SUBMIT_CHUNK
            return self._submit_chunk(
                handle=handle,
                buffers=buffers[start:end],
                peer_buffer_addresses=peer_buffer_addresses[start:end],
                lengths=lengths[start:end],
                request_tail=request_tail,
            )

        starts = range(0, len(buffers), _SUBMIT_CHUNK)
        if len(starts) == 1:
            return [submit(0)]
        futures = [self._submit_pool.submit(submit, start) for start in starts]
        batches: List[_Batch] = []
        first_error: Optional[BaseException] = None
        for future in futures:
            try:
                batches.append(future.result())
            except BaseException as exc:
                first_error = first_error or exc
        if first_error is not None:
            self._cancel_batches(batches)
            self._free_batches(batches)
            raise first_error
        return batches

    def _cancel_batches(self, batches: List[_Batch]) -> None:
        # Best effort: TENT skips tasks that already finished, and an RDMA write
        # already posted to the NIC may still land after the cancel.
        for batch_id, num_tasks in batches:
            for task_id in range(num_tasks):
                try:
                    self.engine.cancel_transfer(batch_id, task_id)
                except self._tent.TentException as exc:
                    logger.debug("TENT cancel_transfer failed: %s", exc)

    def _free_batches(self, batches: List[_Batch]) -> None:
        for batch_id, _ in batches:
            try:
                self.engine.free_transfer_batch(batch_id)
            except self._tent.TentException as exc:
                logger.debug("TENT free_transfer_batch failed: %s", exc)

    def _submit_chunk(
        self,
        *,
        handle: int,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
        request_tail: _RequestTail,
    ) -> _Batch:
        request = self._tent.Request
        write = self._tent.OpCode.WRITE
        requests = [
            request(write, src, handle, dst, length, *request_tail)
            for src, dst, length in zip(buffers, peer_buffer_addresses, lengths)
        ]
        batch_id = self.engine.allocate_transfer_batch(len(requests))
        if not batch_id:
            raise self._tent.InternalError("allocate_transfer_batch returned 0")
        batch = (batch_id, len(requests))
        try:
            self.engine.submit_transfer(batch_id, requests)
        except BaseException:
            # Most submit errors happen before anything is posted, but the
            # batch release at the end of submitTransfer can fail afterwards.
            self._cancel_batches([batch])
            self._free_batches([batch])
            raise
        return batch

    def _wait_batch(self, *, batch_id: int, deadline_ns: int) -> int:
        completed = self._tent.TransferStatusEnum.COMPLETED
        polls = 0
        while True:
            state = self.engine.get_transfer_status_overall(batch_id).state
            if state == completed:
                return 0
            if state in self._failed_states:
                logger.error("TENT transfer ended in state %s", state)
                return -1
            if time.monotonic_ns() > deadline_ns:
                logger.error("TENT transfer timed out")
                return -1
            polls += 1
            if polls > _SPIN_POLLS:
                time.sleep(_POLL_SLEEP_S)
