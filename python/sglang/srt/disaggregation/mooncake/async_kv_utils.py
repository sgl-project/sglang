from __future__ import annotations

import dataclasses
import logging
import os
import queue
import threading
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TransferKVChunkSet:
    rooms: Tuple[int, ...] = dataclasses.field(default_factory=tuple)
    prefill_kv_indices: Tuple[npt.NDArray[np.int64], ...] = dataclasses.field(
        default_factory=tuple
    )
    index_slices: Tuple[slice, ...] = dataclasses.field(default_factory=tuple)
    prefill_state_indices: Tuple[int, ...] = dataclasses.field(default_factory=tuple)


@dataclasses.dataclass
class AsyncInfo:
    layer_ids: Tuple[int, ...] = dataclasses.field(default_factory=tuple)
    kv_chunk_info: TransferKVChunkSet = dataclasses.field(
        default_factory=TransferKVChunkSet
    )


class StreamAsyncSubmitter:
    """Single-worker async submitter with counters.

    The worker thread runs as a daemon and never exits. We use monotonically
    increasing counters to let the caller wait until submitted work has finished.
    """

    def __init__(self, submit_func: Callable[[], None]):
        self._submit_func = submit_func
        self._queue: queue.SimpleQueue[None] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._submitted = 0
        self._finished = 0
        self._exc: Optional[BaseException] = None
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            self._queue.get()
            try:
                self._submit_func()
            except BaseException as e:
                # Persist the exception so waiters can fail fast.
                with self._cond:
                    self._exc = e
                    self._cond.notify_all()
                logger.exception("Unhandled exception in StreamAsyncSubmitter worker.")
            finally:
                with self._cond:
                    self._finished += 1
                    self._cond.notify_all()

    def step_async(self) -> int:
        with self._cond:
            if self._exc is not None:
                raise RuntimeError("StreamAsyncSubmitter worker has failed") from self._exc
            self._submitted += 1
            self._queue.put(None)
            return self._submitted

    def get_step_count(self) -> int:
        with self._cond:
            return self._submitted

    def wait_sent_finish(self, target_count: int) -> None:
        with self._cond:
            if self._exc is not None:
                raise RuntimeError("StreamAsyncSubmitter worker has failed") from self._exc
            while self._finished < target_count:
                self._cond.wait()
                if self._exc is not None:
                    raise RuntimeError("StreamAsyncSubmitter worker has failed") from self._exc


def cached_group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]
):
    # NOTE: despite the name, this function is not memoized; it only normalizes
    # dtypes before calling the grouping helper.
    src = np.asarray(src_indices, dtype=np.int32)
    dst = np.asarray(dst_indices, dtype=np.int32)
    return group_concurrent_contiguous(src, dst)


def env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)
