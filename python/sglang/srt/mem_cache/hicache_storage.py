import hashlib
import json
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


def get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            # EAGLE bigram mode: hash both elements to uniquely identify the bigram
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            # Regular mode: single integer token
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


def hash_str_to_int64(hash_str: str) -> int:
    """Convert SHA256 hex string to signed 64-bit integer for events.

    Takes first 16 hex characters (64 bits) and converts to signed int64 range.
    """
    # Take first 16 hex chars to get 64-bit value
    uint64_val = int(hash_str[:16], 16)
    # Convert to signed int64 range [-2^63, 2^63-1]
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    is_mla_model: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = (None,)
    extra_info: Optional[dict] = None


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host
        self._fault_reporter: Optional["HiCacheStorageFaultManager"] = None

    def set_fault_reporter(self, reporter: Optional["HiCacheStorageFaultManager"]):
        self._fault_reporter = reporter

    def _report_storage_op(self, op: str, fatal: bool, detail: str = ""):
        if self._fault_reporter is None:
            return
        try:
            self._fault_reporter.report_op(op=op, fatal=fatal, detail=detail)
        except Exception:
            logger.exception("HiCacheStorage fault reporter failed.")

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


@dataclass
class HiCacheStorageFaultConfig:
    enabled: bool = True
    auto_detach: bool = True
    auto_reconnect: bool = False
    consecutive_fatal_threshold: int = 3
    ratio_window_s: float = 60.0
    ratio_threshold: float = 0.5
    ratio_min_events: int = 10
    reconnect_backoff_initial_s: float = 10.0
    reconnect_backoff_max_s: float = 300.0


class HiCacheStorageFaultManager:
    def __init__(
        self,
        config: HiCacheStorageFaultConfig,
        labels: Optional[dict] = None,
        block_io_cb: Optional[Callable[[bool, str], Tuple[bool, str]]] = None,
        detach_cb: Optional[Callable[[str], Tuple[bool, str]]] = None,
        attach_cb: Optional[Callable[[], Tuple[bool, str]]] = None,
    ):
        self.config = config
        self.labels = labels or {}
        self.block_io_cb = block_io_cb
        self.detach_cb = detach_cb
        self.attach_cb = attach_cb

        self._lock = threading.Lock()
        self._events: Deque[Tuple[float, bool]] = deque()
        self._consecutive_fatal = 0
        self._state = "HEALTHY"
        self._last_fault_reason = ""
        self._detach_pending = False
        self._backoff_s = config.reconnect_backoff_initial_s
        self._next_reconnect_time = 0.0

        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._init_metrics()

    def _init_metrics(self):
        try:
            from prometheus_client import Counter, Gauge

            self._fault_events_total = Counter(
                name="sglang:hicache_storage_fault_events_total",
                documentation="Total HiCache storage fault events.",
                labelnames=self.labels.keys(),
            )
            self._fault_state = Gauge(
                name="sglang:hicache_storage_fault_state",
                documentation="HiCache storage fault state (0 healthy, 1 blocked, 2 detached, 3 reconnecting).",
                labelnames=self.labels.keys(),
            )
            self._detach_total = Counter(
                name="sglang:hicache_storage_auto_detach_total",
                documentation="Total auto detaches triggered by fault manager.",
                labelnames=self.labels.keys(),
            )
            self._reconnect_total = Counter(
                name="sglang:hicache_storage_auto_reconnect_total",
                documentation="Total auto reconnect attempts triggered by fault manager.",
                labelnames=self.labels.keys(),
            )
        except Exception:
            self._fault_events_total = None
            self._fault_state = None
            self._detach_total = None
            self._reconnect_total = None

    def report_op(self, op: str, fatal: bool, detail: str = ""):
        if not self.config.enabled:
            return
        now = time.monotonic()
        with self._lock:
            self._events.append((now, fatal))
            if fatal:
                self._consecutive_fatal += 1
                self._last_fault_reason = detail or op
            else:
                self._consecutive_fatal = 0
            self._prune_events_locked(now)

            if fatal and self._fault_events_total is not None:
                self._fault_events_total.labels(**self.labels).inc(1)

            if self._should_trigger_fault_locked():
                self._trigger_fault_locked(reason=detail or op)

    def _prune_events_locked(self, now: float):
        window_s = self.config.ratio_window_s
        while self._events and now - self._events[0][0] > window_s:
            self._events.popleft()

    def _should_trigger_fault_locked(self) -> bool:
        if self._state in ("BLOCKED", "DETACHED_BY_FAULT", "RECONNECTING"):
            return False
        if self._consecutive_fatal >= self.config.consecutive_fatal_threshold:
            return True
        total = len(self._events)
        if total < self.config.ratio_min_events:
            return False
        fatal_count = sum(1 for _, is_fatal in self._events if is_fatal)
        ratio = fatal_count / max(total, 1)
        return ratio >= self.config.ratio_threshold

    def _trigger_fault_locked(self, reason: str):
        self._state = "BLOCKED"
        self._last_fault_reason = reason
        if self._fault_state is not None:
            self._fault_state.labels(**self.labels).set(1)
        if self.block_io_cb is not None:
            ok, msg = self.block_io_cb(True, reason)
            if not ok:
                logger.error("Failed to block HiCache storage IO: %s", msg)
        if self.config.auto_detach and not self._detach_pending:
            self._detach_pending = True
            self._queue.put("detach")

    def notify_manual_detach(self):
        with self._lock:
            self._state = "DISABLED"
            self._detach_pending = False
            self._consecutive_fatal = 0
            self._events.clear()
            if self._fault_state is not None:
                self._fault_state.labels(**self.labels).set(0)

    def notify_attach_success(self):
        with self._lock:
            self._state = "HEALTHY"
            self._detach_pending = False
            self._consecutive_fatal = 0
            self._events.clear()
            self._backoff_s = self.config.reconnect_backoff_initial_s
            if self._fault_state is not None:
                self._fault_state.labels(**self.labels).set(0)

    def update_config(self, config: HiCacheStorageFaultConfig):
        with self._lock:
            self.config = config
            self._backoff_s = config.reconnect_backoff_initial_s

    def _worker_loop(self):
        while True:
            try:
                task = self._queue.get(timeout=1)
            except Exception:
                task = None

            if task == "detach":
                self._handle_detach()
            elif task == "attach":
                self._handle_attach()

            self._maybe_schedule_reconnect()

    def _handle_detach(self):
        if self.detach_cb is None:
            return
        ok, msg = self.detach_cb(self._last_fault_reason)
        if ok:
            if self._detach_total is not None:
                self._detach_total.labels(**self.labels).inc(1)
            with self._lock:
                self._state = "DETACHED_BY_FAULT"
                self._detach_pending = False
                self._next_reconnect_time = time.monotonic() + self._backoff_s
                if self._fault_state is not None:
                    self._fault_state.labels(**self.labels).set(2)
        else:
            logger.error("Auto detach failed: %s", msg)
            with self._lock:
                self._detach_pending = False

    def _handle_attach(self):
        if self.attach_cb is None:
            return
        ok, msg = self.attach_cb()
        if ok:
            if self._reconnect_total is not None:
                self._reconnect_total.labels(**self.labels).inc(1)
            if self.block_io_cb is not None:
                _ok, _msg = self.block_io_cb(False, "")
                if not _ok:
                    logger.error("Failed to unblock HiCache storage IO: %s", _msg)
            with self._lock:
                self._state = "HEALTHY"
                self._consecutive_fatal = 0
                self._events.clear()
                self._backoff_s = self.config.reconnect_backoff_initial_s
                if self._fault_state is not None:
                    self._fault_state.labels(**self.labels).set(0)
        else:
            logger.error("Auto reconnect failed: %s", msg)
            with self._lock:
                self._state = "DETACHED_BY_FAULT"
                self._backoff_s = min(
                    self._backoff_s * 2, self.config.reconnect_backoff_max_s
                )
                self._next_reconnect_time = time.monotonic() + self._backoff_s
                if self._fault_state is not None:
                    self._fault_state.labels(**self.labels).set(2)

    def _maybe_schedule_reconnect(self):
        if not self.config.auto_reconnect:
            return
        with self._lock:
            if self._state != "DETACHED_BY_FAULT":
                return
            now = time.monotonic()
            if now < self._next_reconnect_time:
                return
            self._state = "RECONNECTING"
            if self._fault_state is not None:
                self._fault_state.labels(**self.labels).set(3)
        self._queue.put("attach")


class HiCacheFile(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", file_path)

        tp_rank, tp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        model_name = "-".join(model_name.split("/")) if model_name else ""
        if is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

        self._fault_inject_path = os.getenv("SGLANG_HICACHE_FAULT_INJECT_PATH")
        self._fault_stats_path = os.getenv("SGLANG_HICACHE_FAULT_STATS_PATH")

    def _read_fault_inject_config(self) -> dict:
        if not self._fault_inject_path:
            return {}
        try:
            with open(self._fault_inject_path, "r") as fin:
                return json.load(fin)
        except Exception:
            return {}

    def _update_fault_stats(self, op: str, result: str):
        if not self._fault_stats_path:
            return
        try:
            data = {}
            if os.path.exists(self._fault_stats_path):
                with open(self._fault_stats_path, "r") as fin:
                    data = json.load(fin)
            data.setdefault(op, {})
            data[op][result] = data[op].get(result, 0) + 1
            tmp_path = f"{self._fault_stats_path}.tmp"
            with open(tmp_path, "w") as fout:
                json.dump(data, fout)
            os.replace(tmp_path, self._fault_stats_path)
        except Exception:
            logger.exception("Failed to update fault stats.")

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            cfg = self._read_fault_inject_config()
            if cfg.get("fail_get"):
                raise RuntimeError("HiCacheFile injected get failure")
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {key}")
            self._report_storage_op("get", fatal=False)
            self._update_fault_stats("get", "success")
            return target_location
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            self._report_storage_op("get", fatal=False)
            self._update_fault_stats("get", "miss")
            return None
        except Exception as e:
            self._report_storage_op("get", fatal=True, detail=str(e))
            self._update_fault_stats("get", "fatal")
            raise

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True

        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            cfg = self._read_fault_inject_config()
            if cfg.get("fail_set"):
                raise RuntimeError("HiCacheFile injected set failure")
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            self._report_storage_op("set", fatal=False)
            self._update_fault_stats("set", "success")
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            self._report_storage_op("set", fatal=True, detail=str(e))
            self._update_fault_stats("set", "fatal")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False
