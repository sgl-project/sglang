from __future__ import annotations

import logging
import pickle
import queue
import threading
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

_Key = Union[int, str]

from torch.distributed import TCPStore

from sglang.srt.utils.network import get_free_port, get_local_ip_auto

logger = logging.getLogger(__name__)

_FLUSH_DONE_PREFIX = "pp_consensus_flush_done"


class _OpKind(Enum):
    PUT = auto()
    DELETE = auto()
    FLUSH = auto()
    SHUTDOWN = auto()


class PPConsensusStore:
    """Cross-PP-rank key/value store backed by TCPStore for bootstrap consensus."""

    def __init__(self, pp_size: int, pp_rank: int, pp_group) -> None:
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.pp_group = pp_group
        self._local_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self._pending_queue: queue.Queue = queue.Queue()
        self._flush_seq = 0
        self._store, host, port = self._init_tcp_store()
        logger.info(
            "PPConsensusStore rank=%d connected to %s:%d",
            pp_rank,
            host,
            port,
        )
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _init_tcp_store(self) -> Tuple[TCPStore, str, int]:
        if self.pp_rank == 0:
            host = get_local_ip_auto()
            port = get_free_port()
            store = TCPStore(
                host_name=host,
                port=port,
                world_size=self.pp_size,
                is_master=True,
                wait_for_workers=False,
            )
            store_info = (host, port)
        else:
            store_info = None
        store_info = self.pp_group.broadcast_object(store_info, src=0)
        host, port = store_info
        if self.pp_rank > 0:
            store = TCPStore(
                host_name=host,
                port=port,
                world_size=self.pp_size,
                is_master=False,
            )
        return store, host, port

    @staticmethod
    def _normalize_key(key: _Key) -> str:
        return str(key)

    def _store_key(self, pp_rank: int, key: _Key) -> str:
        return f"pp_{pp_rank}/{self._normalize_key(key)}"

    def _worker_loop(self) -> None:
        logger.debug("worker loop")
        while True:
            op = self._pending_queue.get()
            try:
                if op[0] == _OpKind.SHUTDOWN:
                    return
                if op[0] == _OpKind.PUT:
                    _, key, value = op
                    logger.debug(f"put {key} = {value}")
                    self._store.set(self._store_key(self.pp_rank, key), pickle.dumps(value))
                elif op[0] == _OpKind.DELETE:
                    _, key = op
                    store_key = self._store_key(self.pp_rank, key)
                    try:
                        self._store.delete_key(store_key)
                    except Exception:
                        # Key may already be absent; treat as success.
                        pass
                elif op[0] == _OpKind.FLUSH:
                    _, done_event = op
                    self._run_flush_barrier()
                    done_event.set()
            finally:
                self._pending_queue.task_done()

    def _run_flush_barrier(self) -> None:
        self._flush_seq += 1
        seq = self._flush_seq
        self._store.set(f"{_FLUSH_DONE_PREFIX}/{self.pp_rank}/{seq}", b"1")
        wait_keys = [f"{_FLUSH_DONE_PREFIX}/{rank}/{seq}" for rank in range(self.pp_size)]
        self._store.wait(wait_keys)

    def put(self, key: _Key, value: Any) -> None:
        key = self._normalize_key(key)
        with self._cache_lock:
            self._local_cache[key] = value
            self._pending_queue.put((_OpKind.PUT, key, value))

    def delete(self, key: _Key) -> None:
        key = self._normalize_key(key)
        with self._cache_lock:
            self._local_cache.pop(key, None)
            self._pending_queue.put((_OpKind.DELETE, key))

    def pop(self, key: _Key, default: Any = None) -> Any:
        key = self._normalize_key(key)
        with self._cache_lock:
            if key not in self._local_cache:
                return default
            value = self._local_cache.pop(key)
            self._pending_queue.put((_OpKind.DELETE, key))
        return value

    def get(self, key: _Key, default: Any = None) -> Any:
        key = self._normalize_key(key)
        with self._cache_lock:
            return self._local_cache.get(key, default)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, (int, str)):
            return False
        key = self._normalize_key(key)
        with self._cache_lock:
            return key in self._local_cache

    def flush(self) -> None:
        done_event = threading.Event()
        self._pending_queue.put((_OpKind.FLUSH, done_event))
        done_event.wait()

    def get_local_rank(self, key: _Key) -> Any:
        key = self._normalize_key(key)
        with self._cache_lock:
            return self._local_cache.get(key)

    def get_all_ranks(self, key: _Key) -> List[Any]:
        values: List[Any] = []
        for rank in range(self.pp_size):
            store_key = self._store_key(rank, key)
            if not self._store.check([store_key]):
                values.append(None)
                continue
            logger.debug(f"get {store_key}")
            raw = self._store.get(store_key)
            if not raw:
                values.append(None)
                continue
            values.append(pickle.loads(raw))
        return values

    def multiget_all_ranks(self, keys: List[_Key]) -> Dict[str, List[Any]]:
        return {self._normalize_key(key): self.get_all_ranks(key) for key in keys}

    def __getitem__(self, key: _Key) -> Any:
        key = self._normalize_key(key)
        with self._cache_lock:
            if key not in self._local_cache:
                raise KeyError(key)
            return self._local_cache[key]

    def __setitem__(self, key: _Key, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: _Key) -> None:
        key = self._normalize_key(key)
        if key not in self:
            raise KeyError(key)
        self.delete(key)

    def close(self) -> None:
        self._pending_queue.put((_OpKind.SHUTDOWN,))
        self._worker.join(timeout=5)
