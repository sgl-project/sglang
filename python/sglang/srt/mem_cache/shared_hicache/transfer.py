from __future__ import annotations

import base64
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np

from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.environ import (
    default_shared_hicache_transfer_parallelism,
    envs,
)
from sglang.srt.mem_cache.shared_hicache.config import (
    shared_hicache_transfer_backend_name,
)

logger = logging.getLogger(__name__)

SHARED_HICACHE_NIXL_NOTIFICATION_PREFIX = "shared_hicache:"

if TYPE_CHECKING:
    import torch


class SharedHiCacheTransferBackend(ABC):
    name: str

    def __init__(
        self,
        *,
        target_session_id: str,
        target_kv_ptrs,
        target_kv_item_lens,
        parallel_metadata: Optional[Mapping[str, int]] = None,
    ):
        if not getattr(self, "name", None):
            raise ValueError("SharedHiCache transfer backend must define a name")
        self.target_session_id = str(target_session_id)
        self.target_kv_ptrs = [int(ptr) for ptr in target_kv_ptrs]
        self.target_kv_item_lens = [int(length) for length in target_kv_item_lens]
        self.parallel_metadata = {
            key: int(value) for key, value in (parallel_metadata or {}).items()
        }

    @property
    @abstractmethod
    def enabled(self) -> bool: ...

    def target_descriptor(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "session_id": self.target_session_id,
            **self.parallel_metadata,
        }

    @abstractmethod
    def transfer_pages(
        self,
        *,
        target_session_id: str,
        source_page_indices: np.ndarray,
        target_page_indices: np.ndarray,
        target_kv_ptrs: list[int],
        target_kv_item_lens: list[int],
        target_metadata: Optional[Mapping[str, Any]] = None,
        notification: Optional[str] = None,
    ) -> None: ...

    def shutdown(self) -> None:
        pass


def _target_kv_pool_from_scheduler(scheduler):
    target_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if hasattr(target_pool, "full_kv_pool") and hasattr(target_pool, "full_layer_nums"):
        raise RuntimeError(
            "SharedHiCache direct transfer does not support hybrid linear-attention KV pools"
        )
    return target_pool


def _server_arg(scheduler, name: str, default: int) -> int:
    server_args = getattr(scheduler, "server_args", None)
    value = getattr(server_args, name, default)
    return int(value if value is not None else default)


def _parallel_value(scheduler, name: str, default: int) -> int:
    ps = getattr(scheduler, "ps", None)
    if ps is not None and hasattr(ps, name):
        value = getattr(ps, name)
    else:
        value = getattr(scheduler, name, default)
    return int(value if value is not None else default)


def scheduler_parallel_metadata(scheduler) -> dict[str, int]:
    """Return the TP/PP/CP rank metadata needed for same-shape direct reuse."""

    return {
        "tp_rank": _parallel_value(scheduler, "tp_rank", 0),
        "tp_size": _parallel_value(
            scheduler, "tp_size", _server_arg(scheduler, "tp_size", 1)
        ),
        "pp_rank": _parallel_value(scheduler, "pp_rank", 0),
        "pp_size": _parallel_value(
            scheduler, "pp_size", _server_arg(scheduler, "pp_size", 1)
        ),
        "attn_cp_rank": _parallel_value(scheduler, "attn_cp_rank", 0),
        "attn_cp_size": _parallel_value(
            scheduler, "attn_cp_size", _server_arg(scheduler, "attn_cp_size", 1)
        ),
    }


def shared_hicache_parallel_rejection(
    *, pp_size: int, attn_cp_size: int
) -> Optional[str]:
    unsupported = []
    if pp_size != 1:
        unsupported.append(f"pp_size={pp_size}")
    if attn_cp_size != 1:
        unsupported.append(f"attn_cp_size={attn_cp_size}")
    if unsupported:
        return (
            "SharedHiCache direct transfer supports same-shape TP, but PP/CP "
            f"are deferred; got {', '.join(unsupported)}"
        )
    return None


def _direct_topology_rejection(scheduler) -> Optional[str]:
    return shared_hicache_parallel_rejection(
        pp_size=_server_arg(scheduler, "pp_size", 1),
        attn_cp_size=_server_arg(scheduler, "attn_cp_size", 1),
    )


def _scheduler_gpu_id(scheduler) -> int:
    gpu_id = getattr(scheduler, "gpu_id", None)
    if gpu_id is None:
        gpu_id = getattr(getattr(scheduler, "ps", None), "gpu_id", None)
    return int(gpu_id if gpu_id is not None else 0)


def _nixl_backend_params(backend: str, transfer_parallelism: int) -> dict[str, str]:
    backend_params = json.loads(envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS.get())
    if not isinstance(backend_params, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in backend_params.items()
    ):
        raise ValueError(
            "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS must be a JSON object "
            "with string keys and string values"
        )
    if transfer_parallelism > 0:
        if backend in {"UCX", "OBJ"}:
            backend_params.setdefault("num_threads", str(transfer_parallelism))
        elif backend == "GDS_MT":
            backend_params.setdefault("thread_count", str(transfer_parallelism))
        elif backend == "UCCL":
            backend_params.setdefault("num_cpus", str(transfer_parallelism))
    return backend_params


def _create_nixl_agent(*, transfer_parallelism: int):
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as err:
        raise ImportError(
            "Please install NIXL by following the instructions at "
            "https://github.com/ai-dynamo/nixl/blob/main/README.md "
            "to use SharedHiCache NIXL direct transfer."
        ) from err

    backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
    backend_params = _nixl_backend_params(backend, transfer_parallelism)
    agent_config = nixl_agent_config(
        backends=[],
        capture_telemetry=envs.SGLANG_SHARED_HICACHE_NIXL_TELEMETRY.get(),
        num_threads=max(0, int(transfer_parallelism)),
    )
    agent_name = f"shared_hicache_nixl_{uuid.uuid4()}"
    agent = nixl_agent(agent_name, agent_config)
    agent.create_backend(backend, backend_params)
    available_plugins = agent.get_plugin_list()
    if backend not in available_plugins:
        raise RuntimeError(
            f"NIXL backend {backend!r} not found. Available: {available_plugins}"
        )
    return agent, agent_name, backend


def build_shared_hicache_transfer_notification(
    *,
    transfer_id: str,
    transferred_blocks: int,
    reason: str,
) -> str:
    payload = {
        "transfer_id": str(transfer_id),
        "transferred_blocks": int(transferred_blocks),
        "reason": str(reason),
    }
    return SHARED_HICACHE_NIXL_NOTIFICATION_PREFIX + json.dumps(
        payload, separators=(",", ":")
    )


def parse_shared_hicache_transfer_notification(
    message: bytes | str,
) -> Optional[tuple[str, int, str]]:
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(message, str) or not message.startswith(
        SHARED_HICACHE_NIXL_NOTIFICATION_PREFIX
    ):
        return None
    try:
        payload = json.loads(message[len(SHARED_HICACHE_NIXL_NOTIFICATION_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    transfer_id = str(payload.get("transfer_id") or "")
    if not transfer_id:
        return None
    try:
        transferred_blocks = int(payload.get("transferred_blocks", 0))
    except (TypeError, ValueError):
        return None
    if transferred_blocks < 0:
        return None
    return transfer_id, transferred_blocks, str(payload.get("reason", "ok"))


def _source_host_buf_infos(tree_cache) -> tuple[list[int], list[int]]:
    refs = _source_host_tensors(tree_cache)
    page_size = tree_cache.page_size
    ptrs = [int(ref.data_ptr()) for ref in refs]
    item_lens = [int(ref[0].nbytes) * page_size for ref in refs]
    return ptrs, item_lens


def _source_host_tensors(tree_cache) -> list[torch.Tensor]:
    host_pool = tree_cache.cache_controller.mem_pool_host
    if getattr(host_pool, "layout", None) != "layer_first":
        raise RuntimeError(
            "SharedHiCache direct transfer requires layer_first host layout, "
            f"got {getattr(host_pool, 'layout', None)!r}"
        )

    if hasattr(host_pool, "k_data_refs") and hasattr(host_pool, "v_data_refs"):
        refs = host_pool.k_data_refs + host_pool.v_data_refs
    elif hasattr(host_pool, "data_refs"):
        refs = host_pool.data_refs
    else:
        raise RuntimeError("Unsupported HiCache host pool for SharedHiCache direct transfer")
    return list(refs)


def _validate_kv_item_lens_match(
    source_kv_item_lens: list[int], target_kv_item_lens: list[int]
) -> None:
    if len(source_kv_item_lens) != len(target_kv_item_lens):
        raise RuntimeError(
            "KV item length count mismatch: "
            f"source={len(source_kv_item_lens)} target={len(target_kv_item_lens)}"
        )

    src_item_lens = np.asarray(source_kv_item_lens, dtype=np.uint64)
    dst_item_lens = np.asarray(target_kv_item_lens, dtype=np.uint64)
    mismatched_items = np.nonzero(src_item_lens != dst_item_lens)[0]
    if len(mismatched_items) > 0:
        idx = int(mismatched_items[0])
        raise RuntimeError(
            "KV item length mismatch: "
            f"source={int(src_item_lens[idx])} target={int(dst_item_lens[idx])}"
        )


def _build_grouped_transfer_arrays(
    *,
    source_page_indices: np.ndarray,
    target_page_indices: np.ndarray,
    source_kv_ptrs: list[int],
    target_kv_ptrs: list[int],
    source_kv_item_lens: list[int],
    target_kv_item_lens: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if len(source_kv_ptrs) != len(target_kv_ptrs):
        raise RuntimeError(
            "KV pointer count mismatch: "
            f"source={len(source_kv_ptrs)} target={len(target_kv_ptrs)}"
        )
    _validate_kv_item_lens_match(source_kv_item_lens, target_kv_item_lens)

    src_item_lens = np.asarray(source_kv_item_lens, dtype=np.uint64)
    dst_item_lens = np.asarray(target_kv_item_lens, dtype=np.uint64)
    src_blocks, dst_blocks = group_concurrent_contiguous(
        source_page_indices.astype(np.int32, copy=False),
        target_page_indices.astype(np.int32, copy=False),
    )
    if not src_blocks:
        empty = np.empty((0,), dtype=np.uint64)
        return empty, empty, empty, 0

    # Keep address arithmetic in uint64, matching existing disagg transfer code.
    # Some platforms can expose addresses beyond signed int64.
    src_ptrs = np.asarray(source_kv_ptrs, dtype=np.uint64)
    dst_ptrs = np.asarray(target_kv_ptrs, dtype=np.uint64)
    src_starts = np.fromiter((block[0] for block in src_blocks), dtype=np.uint64)
    dst_starts = np.fromiter((block[0] for block in dst_blocks), dtype=np.uint64)
    block_lens = np.fromiter((len(block) for block in src_blocks), dtype=np.uint64)

    src_addrs = (
        src_ptrs[:, None] + src_starts[None, :] * src_item_lens[:, None]
    ).reshape(-1)
    dst_addrs = (
        dst_ptrs[:, None] + dst_starts[None, :] * dst_item_lens[:, None]
    ).reshape(-1)
    lengths = (src_item_lens[:, None] * block_lens[None, :]).reshape(-1)
    return src_addrs, dst_addrs, lengths, len(src_blocks)


def _nixl_req_array(addrs: np.ndarray, lengths: np.ndarray, gpu_id: int) -> np.ndarray:
    if addrs.size == 0:
        return np.empty((0, 3), dtype=np.uint64)
    return np.column_stack(
        (
            addrs.astype(np.uint64, copy=False),
            lengths.astype(np.uint64, copy=False),
            np.full(addrs.shape, int(gpu_id), dtype=np.uint64),
        )
    )


@dataclass
class _NixlSourceWorkerState:
    agent: Any
    agent_name: str
    backend_name: str
    remote_agents: set[str] = field(default_factory=set)


class NixlSharedHiCacheTransferBackend(SharedHiCacheTransferBackend):
    """NIXL-backed source-HiCache-host to target-GPU-device transfer helper."""

    name = "nixl"

    def __init__(
        self,
        *,
        agent,
        agent_name: str,
        backend_name: str,
        tree_cache,
        target_kv_ptrs,
        target_kv_item_lens,
        target_registered: bool,
        gpu_id: int,
        parallel_metadata: Optional[Mapping[str, int]] = None,
        transfer_parallelism: Optional[int] = None,
    ):
        super().__init__(
            target_session_id=agent_name,
            target_kv_ptrs=target_kv_ptrs,
            target_kv_item_lens=target_kv_item_lens,
            parallel_metadata=parallel_metadata,
        )
        self.agent = agent
        self.agent_name = agent_name
        self.backend_name = backend_name
        self.tree_cache = tree_cache
        self._target_registered = bool(target_registered)
        self._source_pool_ready = False
        self._source_registered_ptrs: list[int] = []
        self._source_worker_states: dict[int, _NixlSourceWorkerState] = {}
        self._source_worker_lock = threading.Lock()
        self._target_notification_lock = threading.Lock()
        self._target_notifications: dict[str, tuple[int, str]] = {}
        self._gpu_id = int(gpu_id)
        if transfer_parallelism is None:
            transfer_parallelism = default_shared_hicache_transfer_parallelism()
        self._transfer_parallelism = max(1, int(transfer_parallelism))
        self._capture_telemetry = envs.SGLANG_SHARED_HICACHE_NIXL_TELEMETRY.get()
        self._shutdown = False

    @classmethod
    def from_scheduler(cls, scheduler) -> Optional["NixlSharedHiCacheTransferBackend"]:
        server_args = scheduler.server_args
        backend = shared_hicache_transfer_backend_name(server_args)
        if backend not in {"auto", "nixl"}:
            return None
        topology_rejection = _direct_topology_rejection(scheduler)
        if topology_rejection is not None:
            if backend == "nixl":
                logger.warning(
                    "SharedHiCache NIXL direct transfer disabled: %s",
                    topology_rejection,
                )
            else:
                logger.debug(
                    "SharedHiCache NIXL direct transfer disabled: %s",
                    topology_rejection,
                )
            return None

        try:
            transfer_parallelism = default_shared_hicache_transfer_parallelism()
            agent, agent_name, backend_name = _create_nixl_agent(
                transfer_parallelism=transfer_parallelism
            )
            target_pool = _target_kv_pool_from_scheduler(scheduler)
            target_kv_ptrs, target_kv_lens, target_kv_item_lens = (
                target_pool.get_contiguous_buf_infos()
            )
            gpu_id = _scheduler_gpu_id(scheduler)
            target_descs = agent.register_memory(
                [
                    (int(ptr), int(length), gpu_id, "")
                    for ptr, length in zip(target_kv_ptrs, target_kv_lens)
                ],
                "VRAM",
            )
            if not target_descs:
                logger.warning(
                    "SharedHiCache NIXL disabled: target KV registration failed"
                )
                return None
            transfer = cls(
                agent=agent,
                agent_name=agent_name,
                backend_name=backend_name,
                tree_cache=scheduler.tree_cache,
                target_kv_ptrs=target_kv_ptrs,
                target_kv_item_lens=target_kv_item_lens,
                target_registered=True,
                gpu_id=gpu_id,
                parallel_metadata=scheduler_parallel_metadata(scheduler),
                transfer_parallelism=transfer_parallelism,
            )
            transfer._register_source_host_pool()
            if transfer.enabled:
                transfer._log_ready()
                return transfer
            transfer.shutdown()
            return None
        except Exception:
            if backend == "nixl":
                logger.exception(
                    "SharedHiCache NIXL direct transfer initialization failed"
                )
            else:
                logger.debug(
                    "SharedHiCache NIXL direct transfer unavailable; using fallback",
                    exc_info=True,
                )
            return None

    @property
    def enabled(self) -> bool:
        return self._target_registered and self._source_pool_ready

    def target_descriptor(self) -> dict[str, Any]:
        descriptor = super().target_descriptor()
        descriptor.update(
            {
                "agent_name": self.agent_name,
                "agent_metadata": base64.b64encode(
                    self.agent.get_agent_metadata()
                ).decode("ascii"),
                "gpu_id": self._gpu_id,
                "transport": {
                    "backend": self.backend_name,
                    "transfer_parallelism": self._transfer_parallelism,
                },
            }
        )
        return descriptor

    def _log_ready(self) -> None:
        logger.info(
            "SharedHiCache NIXL direct transfer enabled agent=%s backend=%s "
            "gpu_id=%d parallelism=%d",
            self.agent_name,
            self.backend_name,
            self._gpu_id,
            self._transfer_parallelism,
        )

    def _register_source_host_pool(self) -> None:
        host_pool = self.tree_cache.cache_controller.mem_pool_host
        if getattr(host_pool, "layout", None) != "layer_first":
            logger.info(
                "SharedHiCache NIXL direct transfer disabled for HiCache layout=%s; "
                "source host pool must be layer_first",
                getattr(host_pool, "layout", None),
            )
            return
        if not hasattr(host_pool, "kv_buffer"):
            logger.info("SharedHiCache NIXL direct transfer disabled: no host kv_buffer")
            return
        try:
            _, source_kv_item_lens = _source_host_buf_infos(self.tree_cache)
            _validate_kv_item_lens_match(
                source_kv_item_lens, self.target_kv_item_lens
            )
        except RuntimeError as err:
            logger.info("SharedHiCache NIXL direct transfer disabled: %s", err)
            return
        self._source_pool_ready = True
        self._source_registered_ptrs = [int(host_pool.kv_buffer.data_ptr())]

    def _create_source_worker_state(self) -> _NixlSourceWorkerState:
        host_pool = self.tree_cache.cache_controller.mem_pool_host
        agent, agent_name, backend_name = _create_nixl_agent(
            transfer_parallelism=self._transfer_parallelism
        )
        source_descs = agent.register_memory(
            [
                (
                    int(host_pool.kv_buffer.data_ptr()),
                    int(host_pool.kv_buffer.nbytes),
                    0,
                    "",
                )
            ],
            "DRAM",
        )
        if not source_descs:
            raise RuntimeError(
                "SharedHiCache NIXL source host registration failed"
            )
        logger.info(
            "SharedHiCache NIXL source transfer worker enabled agent=%s backend=%s "
            "thread=%d parallelism=%d",
            agent_name,
            backend_name,
            threading.get_ident(),
            self._transfer_parallelism,
        )
        return _NixlSourceWorkerState(
            agent=agent,
            agent_name=agent_name,
            backend_name=backend_name,
        )

    def _source_worker_state(self) -> _NixlSourceWorkerState:
        thread_id = threading.get_ident()
        with self._source_worker_lock:
            state = self._source_worker_states.get(thread_id)
        if state is not None:
            return state

        state = self._create_source_worker_state()
        with self._source_worker_lock:
            existing = self._source_worker_states.get(thread_id)
            if existing is not None:
                return existing
            self._source_worker_states[thread_id] = state
            return state

    def prepare_source_worker(self) -> None:
        if self._shutdown or not self.enabled:
            raise RuntimeError("NIXL direct KV transfer backend is not enabled")
        self._source_worker_state()

    def _add_remote_target(
        self,
        state: _NixlSourceWorkerState,
        target_metadata: Optional[Mapping[str, Any]],
    ) -> tuple[str, int]:
        if not isinstance(target_metadata, Mapping):
            raise RuntimeError("NIXL target metadata must be an object")
        target_agent_name = str(target_metadata.get("agent_name") or "")
        encoded_metadata = target_metadata.get("agent_metadata")
        if not target_agent_name:
            raise RuntimeError("NIXL target metadata missing agent_name")
        if not isinstance(encoded_metadata, str) or not encoded_metadata:
            raise RuntimeError("NIXL target metadata missing agent_metadata")
        if target_agent_name not in state.remote_agents:
            state.agent.add_remote_agent(base64.b64decode(encoded_metadata))
            state.remote_agents.add(target_agent_name)
        return target_agent_name, int(target_metadata.get("gpu_id", 0))

    def _wait_for_transfer(
        self, agent, handle
    ) -> Optional[tuple[int, int, int, int]]:
        transfer_state = agent.transfer(handle)
        while transfer_state != "DONE":
            if transfer_state == "ERR":
                raise RuntimeError("NIXL direct KV transfer failed")
            transfer_state = agent.check_xfer_state(handle)
            time.sleep(0.0001)
        telemetry = self._get_xfer_telemetry(agent, handle)
        release = getattr(agent, "release_xfer_handle", None)
        if callable(release):
            release(handle)
        return telemetry

    def _drain_target_notifications_locked(self) -> None:
        get_new_notifs = getattr(self.agent, "get_new_notifs", None)
        if not callable(get_new_notifs):
            return
        try:
            notif_map = get_new_notifs()
        except Exception:
            logger.debug(
                "SharedHiCache NIXL target notification poll failed", exc_info=True
            )
            return
        if not isinstance(notif_map, Mapping):
            return
        for messages in notif_map.values():
            for message in messages or ():
                parsed = parse_shared_hicache_transfer_notification(message)
                if parsed is None:
                    continue
                transfer_id, transferred_blocks, reason = parsed
                self._target_notifications[transfer_id] = (
                    transferred_blocks,
                    reason,
                )

    def pop_target_transfer_notification(
        self, transfer_id: str
    ) -> Optional[tuple[int, str]]:
        with self._target_notification_lock:
            self._drain_target_notifications_locked()
            return self._target_notifications.pop(str(transfer_id), None)

    def _get_xfer_telemetry(
        self, agent, handle
    ) -> Optional[tuple[int, int, int, int]]:
        if not self._capture_telemetry:
            return None
        get_telemetry = getattr(agent, "get_xfer_telemetry", None)
        if not callable(get_telemetry):
            return None
        try:
            telemetry = get_telemetry(handle)
            return (
                int(getattr(telemetry, "postDuration")),
                int(getattr(telemetry, "xferDuration")),
                int(getattr(telemetry, "totalBytes")),
                int(getattr(telemetry, "descCount")),
            )
        except Exception:
            logger.debug("SharedHiCache NIXL transfer telemetry unavailable", exc_info=True)
            return None

    def _source_host_buf_infos(self) -> tuple[list[int], list[int]]:
        return _source_host_buf_infos(self.tree_cache)

    def transfer_pages(
        self,
        *,
        target_session_id: str,
        source_page_indices: np.ndarray,
        target_page_indices: np.ndarray,
        target_kv_ptrs: list[int],
        target_kv_item_lens: list[int],
        target_metadata: Optional[Mapping[str, Any]] = None,
        notification: Optional[str] = None,
    ) -> None:
        if self._shutdown or not self.enabled:
            raise RuntimeError("NIXL direct KV transfer backend is not enabled")
        setup_start = time.perf_counter()
        source_state = self._source_worker_state()
        target_agent_name, target_gpu_id = self._add_remote_target(
            source_state, target_metadata
        )
        source_kv_ptrs, source_kv_item_lens = self._source_host_buf_infos()
        src_addrs, dst_addrs, lengths, num_blocks = _build_grouped_transfer_arrays(
            source_page_indices=source_page_indices,
            target_page_indices=target_page_indices,
            source_kv_ptrs=source_kv_ptrs,
            target_kv_ptrs=target_kv_ptrs,
            source_kv_item_lens=source_kv_item_lens,
            target_kv_item_lens=target_kv_item_lens,
        )
        if src_addrs.size == 0:
            return

        src_descs = source_state.agent.get_xfer_descs(
            _nixl_req_array(src_addrs, lengths, 0), "DRAM"
        )
        dst_descs = source_state.agent.get_xfer_descs(
            _nixl_req_array(dst_addrs, lengths, target_gpu_id), "VRAM"
        )
        if src_descs is None or dst_descs is None:
            raise RuntimeError("NIXL direct KV transfer descriptor creation failed")
        xfer_args = [
            "WRITE",
            src_descs,
            dst_descs,
            target_agent_name,
        ]
        if notification:
            xfer_args.append(str(notification).encode("utf-8"))
        handle = source_state.agent.initialize_xfer(*xfer_args)
        if not handle:
            raise RuntimeError("NIXL direct KV transfer initialization failed")
        setup_ms = (time.perf_counter() - setup_start) * 1000
        start = time.perf_counter()
        telemetry = self._wait_for_transfer(source_state.agent, handle)
        transfer_ms = (time.perf_counter() - start) * 1000
        if telemetry is None:
            logger.info(
                "SharedHiCache NIXL transferred blocks=%d slices=%d bytes=%d "
                "ms=%.3f setup_ms=%.3f source_agent=%s",
                num_blocks,
                len(src_addrs),
                int(lengths.sum()),
                transfer_ms,
                setup_ms,
                source_state.agent_name,
            )
        else:
            post_us, xfer_us, nixl_bytes, desc_count = telemetry
            logger.info(
                "SharedHiCache NIXL transferred blocks=%d slices=%d bytes=%d "
                "ms=%.3f setup_ms=%.3f source_agent=%s "
                "nixl_post_us=%d nixl_xfer_us=%d nixl_bytes=%d nixl_descs=%d",
                num_blocks,
                len(src_addrs),
                int(lengths.sum()),
                transfer_ms,
                setup_ms,
                source_state.agent_name,
                post_us,
                xfer_us,
                nixl_bytes,
                desc_count,
            )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._source_registered_ptrs = []
        self._source_pool_ready = False
        self._source_worker_states = {}
        self._target_registered = False


def make_shared_hicache_transfer_backend(scheduler) -> Optional[SharedHiCacheTransferBackend]:
    backend = shared_hicache_transfer_backend_name(scheduler.server_args)
    topology_rejection = _direct_topology_rejection(scheduler)
    if topology_rejection is not None:
        if backend == "nixl":
            raise RuntimeError(topology_rejection)
        logger.warning("SharedHiCache direct transfer unavailable: %s", topology_rejection)
        return None

    if backend == "nixl":
        transfer = NixlSharedHiCacheTransferBackend.from_scheduler(scheduler)
        if transfer is None:
            raise RuntimeError(
                "SharedHiCache NIXL transfer backend was requested but unavailable"
            )
        return transfer

    if backend != "auto":
        raise RuntimeError(
            f"SharedHiCache transfer backend {backend!r} is not supported; "
            "this path supports only 'nixl'"
        )

    transfer = NixlSharedHiCacheTransferBackend.from_scheduler(scheduler)
    if transfer is not None:
        return transfer
    return None
