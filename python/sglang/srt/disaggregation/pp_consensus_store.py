from __future__ import annotations

import logging
import os
import pickle
import queue
import threading
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import zmq

from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)


class _OpKind(Enum):
    PUT = auto()
    DELETE = auto()


class PPConsensusStore:
    """A map that replicates metadata to PP rank 0.

    All ranks may use this as a normal Dict, e.g. ``store["key"] = value`` or ``del store["key"]``.
    All modifications done by all ranks are replicated to rank 0 in background.  Rank 0 calls
    ``collect`` to retrieve values of all ranks for the specified key.
    """

    def __init__(self, pp_size: int, pp_rank: int, pp_group) -> None:
        self._pp_size = pp_size
        self._pp_rank = pp_rank
        self._pp_group = pp_group
        self._local_map: Dict[Any, Any] = {}
        self._peer_map: Dict[int, Dict[Any, Any]] = {}
        self._cache_lock = threading.Lock()
        self._replication_queue: Optional[queue.Queue] = None
        self._zmq_ctx: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._init_zmq()
        if self._pp_rank > 0:
            self._replication_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _init_zmq(self) -> None:
        self._zmq_ctx = zmq.Context()
        # PP0 listens.
        if self._pp_rank == 0:
            host = get_local_ip_auto()
            port, self._socket = get_zmq_socket_on_host(
                self._zmq_ctx, zmq.PULL, host=host
            )
            store_info = (host, port)
            logger.info("PPConsensusStore is listening at %s:%d", host, port)
        else:
            store_info = None

        # Broadcast our zmq host and port to other ranks.
        host, port = self._pp_group.broadcast_object(store_info, src=0)

        # PP>0 connects to PP0.
        if self._pp_rank > 0:
            self._socket = get_zmq_socket(
                self._zmq_ctx,
                zmq.PUSH,
                NetworkAddress(host, port).to_tcp(),
                bind=False,
            )
            logger.info("PPConsensusStore connected to %s:%d", host, port)

    def _maybe_replicate_to_rank0(self, op: tuple) -> None:
        if self._pp_rank > 0:
            self._replication_queue.put(op)

    def pop(self, key: Any, default: Any = None) -> Any:
        with self._cache_lock:
            if key not in self._local_map:
                return default
            value = self._local_map.pop(key)
            self._maybe_replicate_to_rank0((_OpKind.DELETE, self._pp_rank, key))
        return value

    def get(self, key: Any, default: Any = None) -> Any:
        with self._cache_lock:
            return self._local_map.get(key, default)

    def __contains__(self, key: Any) -> bool:
        with self._cache_lock:
            return key in self._local_map

    def __getitem__(self, key: Any) -> Any:
        with self._cache_lock:
            return self._local_map[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._cache_lock:
            self._local_map[key] = value
            self._maybe_replicate_to_rank0((_OpKind.PUT, self._pp_rank, key, value))

    def __delitem__(self, key: Any) -> None:
        with self._cache_lock:
            if key not in self._local_map:
                raise KeyError(key)
            self._local_map.pop(key)
            self._maybe_replicate_to_rank0((_OpKind.DELETE, self._pp_rank, key))

    def close(self) -> None:
        if self._pp_rank > 0:
            self._replication_queue.put(None)
        self._zmq_ctx.term()  # This blocks until all zmq socket closed.
        self._worker_thread.join()

    def _worker_loop(self) -> None:
        try:
            if self._pp_rank == 0:
                self._recv_loop()
            else:
                self._send_loop()
        except zmq.error.ContextTerminated:
            pass  # Raised by close().
        except Exception as e:
            logger.critical(
                "PPConsensusStore background thread crashed: %s",
                e,
            )
            for handler in logger.handlers:
                handler.flush()
            os._exit(1)
        finally:
            self._socket.close(linger=0)

    def _recv_once(self) -> None:
        op = pickle.loads(self._socket.recv())
        if op[0] == _OpKind.PUT:
            _, rank, key, value = op
            logger.debug("recv put rank=%s %s = %s", rank, key, value)
            with self._cache_lock:
                self._peer_map.setdefault(rank, {})[key] = value
        elif op[0] == _OpKind.DELETE:
            _, rank, key = op
            logger.debug("recv delete rank=%s %s", rank, key)
            with self._cache_lock:
                self._peer_map.get(rank, {}).pop(key, None)

    def _recv_loop(self) -> None:
        assert self._pp_rank == 0
        while True:
            self._recv_once()

    def _send_loop(self) -> None:
        assert self._pp_rank > 0
        while True:
            op = self._replication_queue.get()
            if op is None:  # A signal for shutdown.
                return
            logger.debug("send %s", op)
            self._socket.send(pickle.dumps(op))

    def collect(self, key: Any) -> List[Any]:
        assert self._pp_rank == 0, "collect can only be used on PP rank 0"
        with self._cache_lock:
            values: List[Any] = [self._local_map.get(key)]
            for rank in range(1, self._pp_size):
                values.append(self._peer_map.get(rank, {}).get(key))
            return values
