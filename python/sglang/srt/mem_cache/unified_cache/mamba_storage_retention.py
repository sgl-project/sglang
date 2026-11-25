"""Keep only the newest Mamba state of each radix path in external storage."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter, OrderedDict, deque
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import PoolName

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

# Mooncake ErrorCode values returned per object by batch_remove.
_REMOVE_OUTCOMES = {
    0: "mamba_removed",
    -703: "mamba_remove_not_ready",
    -704: "mamba_remove_not_found",
    -706: "mamba_remove_leased",
}
_RETRIABLE = (-703, -706)


class MambaStorageRetention:
    """Mamba state keys this rank stored in or loaded from storage, oldest first.

    Storing a newer state on a path supersedes the known states above it,
    except at a fork of this rank's tree. A superseded state is removed once
    the newer one persists.
    """

    def __init__(self, cache: UnifiedRadixCache, capacity: int = 1 << 16):
        self.cache = cache
        self.capacity = capacity
        self.keys: OrderedDict[str, None] = OrderedDict()
        self.stats: Counter[str] = Counter()
        self.remover = MambaStateRemover(self.stats)

    def remember(self, key: str) -> None:
        self.keys[key] = None
        self.keys.move_to_end(key)
        if len(self.keys) > self.capacity:
            self.keys.popitem(last=False)

    def supersede(self, node_id: NodeId, new_key: str) -> list[str]:
        """Forget and return the known states above ``node_id``, which now
        stores the state ``new_key``."""
        page = self.cache.page_size
        forks = {
            info.path_end // page
            for info in self.cache.tree_core.collect_path_info([node_id])[0]
            if info.num_children > 1 and info.path_end % page == 0
        }
        # A node inserted whole, e.g. after a flush, holds earlier states inside it.
        path = [
            *self.cache.get_prefix_hash_values(node_id),
            *self.cache.tree_core.get_hash_values(node_id),
        ]
        superseded = [
            key
            for pages, key in enumerate(path, start=1)
            if key != new_key and key in self.keys and pages not in forks
        ]
        for key in superseded:
            del self.keys[key]
        self.remember(new_key)
        return superseded


class MambaStateRemover:
    """Removes superseded Mamba states on its own thread, counting each object.

    A state refused while leased is retried once after ``retry_delay_s``, which
    should exceed the master's lease TTL: the rank's own lookup of the next turn
    leases the state that turn supersedes. Then it is left to storage eviction.
    """

    def __init__(
        self,
        stats: Counter[str] | dict[str, int],
        retry_delay_s: float | None = None,
    ):
        self.stats = stats
        self.retry_delay_s = (
            envs.SGLANG_HICACHE_MAMBA_TAIL_ONLY_STORAGE_RETRY_DELAY_S.get()
            if retry_delay_s is None
            else retry_delay_s
        )
        self._tasks: queue.Queue[tuple[HiCacheStorage, list[str]]] = queue.Queue()
        self._thread: threading.Thread | None = None

    def submit(self, storage: HiCacheStorage, keys: list[str]) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="mamba-state-remover", daemon=True
            )
            self._thread.start()
        self._tasks.put((storage, keys))

    def _run(self) -> None:
        deferred: deque[tuple[float, HiCacheStorage, list[str]]] = deque()
        while True:
            while deferred and deferred[0][0] <= time.monotonic():
                _, storage, keys = deferred.popleft()
                self._remove(storage, keys, None)
            timeout = max(0.0, deferred[0][0] - time.monotonic()) if deferred else None
            try:
                storage, keys = self._tasks.get(timeout=timeout)
            except queue.Empty:
                continue
            self._remove(storage, keys, deferred)

    def _remove(self, storage, keys, deferred) -> None:
        try:
            codes = list(storage.batch_remove_pool_keys(PoolName.MAMBA, keys) or ())
        except Exception:
            logger.warning("Removing superseded Mamba states failed", exc_info=True)
            codes = [None] * len(keys)
        if not codes:
            return
        # Each page key expands to the same number of objects, page by page.
        per_key = max(1, len(codes) // len(keys))
        retry = []
        for index, key in enumerate(keys):
            page_codes = codes[index * per_key : (index + 1) * per_key]
            deferring = deferred is not None and any(
                code in _RETRIABLE for code in page_codes
            )
            if deferring:
                retry.append(key)
            for code in page_codes:
                outcome = (
                    "mamba_remove_deferred"
                    if deferring and code in _RETRIABLE
                    else _REMOVE_OUTCOMES.get(code, "mamba_remove_failed")
                )
                self.stats[outcome] = self.stats.get(outcome, 0) + 1
        if retry:
            deferred.append((time.monotonic() + self.retry_delay_s, storage, retry))
