from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class TLRUStrategy(EvictionStrategy):
    """Tail-Optimized LRU (Zhang et al., arXiv:2510.15152).

    A conversation with history length L whose next prompt is expected to add
    Q_hat tokens only has to keep L + Q_hat - threshold tokens cached to hold its
    next prefill under the TTFT budget; tokens past that budget cannot improve
    tail latency and are "TEL-safe", i.e. free to evict. Such nodes are reported
    as infinitely old, which is the implementation the paper suggests: the
    existing eviction driver then drains them before anything else (the paper's
    phase 1) and continues in plain recency order once they run out (phase 2),
    so neither eviction loop needs to know about T-LRU.

    threshold and next_prompt_estimate are token counts, whereas the paper states
    both in blocks; multiply the paper's values by page_size to convert.
    """

    def __init__(self, threshold: int = 0, next_prompt_estimate: int = 0):
        self.threshold = threshold
        self.next_prompt_estimate = next_prompt_estimate

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # node._tlru_history_len is the branch's high-water depth, i.e. the
        # paper's L, and deliberately does not shrink when the tail is trimmed.
        # Deriving L from what is still resident instead would leave the
        # shortened conversation over budget on the next pass too, and T-LRU
        # would walk it down to nothing rather than stopping after
        # (threshold - Q_hat) tokens.
        budget = max(
            node._tlru_history_len + self.next_prompt_estimate - self.threshold, 0
        )
        cached_without_this_node = node._tlru_cached_prefix_len - len(node.key)
        tel_safe = cached_without_this_node >= budget
        return (-1 if tel_safe else 0, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


def proactive_write_back_budget(
    capacity: int,
    available: int,
    pending: int,
    threshold: float,
    page_size: int,
) -> int:
    """Return a page-aligned soft budget for this scheduler step.

    Pending copies still occupy source slots, but already count toward the
    target. Bound the in-flight window by the reserve above the watermark and
    the cold prepared window by the occupancy above the watermark. Walkers
    count completed resident copies toward this target and submit at most
    4096 new tokens per step (radix nodes are atomic). Proactive write-back
    never releases source slots; allocation pressure performs reclamation.
    """
    if threshold >= 1:
        return 0
    watermark_target = int(capacity * threshold) // page_size * page_size
    excess = capacity - available - watermark_target - pending
    window = capacity - watermark_target - pending
    budget = min(excess, window)
    return max(0, (budget + page_size - 1) // page_size * page_size)


class StorageWriteBack:
    """Small L2->L3 adapter; I/O, locks and cold-node ordering stay with the cache.

    Confirmed page hashes survive radix splits. Only rank-agreed successful
    pages enter this bounded cache; an evicted belief costs a redundant write.
    """

    def __init__(
        self, threshold, page_size, capacity=None, *, describe=None, pools=None
    ):
        from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
            StorageExistenceCache,
        )

        self.describe = describe or self._describe_kv
        self.pools = pools
        self.threshold = threshold
        self.page_size = page_size
        # A whole source node must fit, otherwise a large successful write
        # could evict its own confirmations and never become releasable.
        self.confirmed = (
            StorageExistenceCache(max_entries=max(1, capacity // page_size))
            if capacity is not None
            else StorageExistenceCache()
        )
        self.pending = {}
        self.deferred_counts = []
        self.deferred_pages = []
        self.backend = None

    def bind(self, backend):
        if backend is not self.backend:
            self.clear_confirmed()
            self.backend = backend

    def reset(self):
        self.pending.clear()
        self.clear_confirmed()

    def clear_confirmed(self):
        self.confirmed.clear()
        self.deferred_counts.clear()
        self.deferred_pages.clear()

    def sync_poll(self, cache, counters):
        """Confirm the previous agreed ACK batch in the existing poll collective."""
        import torch

        size = counters.numel()
        combined = torch.cat((counters, counters.new_tensor(self.deferred_counts)))
        cache._all_reduce(combined, torch.distributed.ReduceOp.MIN)
        counters.copy_(combined[:size])
        for (pool, hashes), count in zip(self.deferred_pages, combined[size:].tolist()):
            self.confirmed.add(pool, hashes[:count])
        self.deferred_counts.clear()
        self.deferred_pages.clear()

    def _describe_kv(self, node):
        hashes = tuple(node.hash_value or ())
        return {"kv": (hashes, len(hashes) * self.page_size)}

    @staticmethod
    def describe_transfers(transfers):
        """Snapshot keys and source slots, resolving sidecars through their owner."""
        by_name = {transfer.name: transfer for transfer in transfers}
        result = {}
        for transfer in transfers:
            source = transfer
            while source.indices_from_pool is not None:
                source = by_name[source.indices_from_pool]
            keys = tuple(source.keys or ())
            if keys:
                result[transfer.name] = (keys, len(source.host_indices))
        return result

    def _ensure_hashes(self, node):
        if node.hash_value is None:
            from sglang.srt.mem_cache.utils import compute_node_hash_values

            missing = []
            current = node
            while current.parent is not None and current.hash_value is None:
                missing.append(current)
                current = current.parent
            for entry in reversed(missing):
                entry.hash_value = compute_node_hash_values(entry, self.page_size)

    @staticmethod
    def _prefix_nodes(node):
        # The root has no payload. Test adapters may omit parent entirely.
        nodes = [node]
        parent = getattr(node, "parent", None)
        while parent is not None and getattr(parent, "parent", None) is not None:
            nodes.append(parent)
            parent = parent.parent
        return reversed(nodes)

    def _pools_ready(self, pools):
        return all(
            self.confirmed.contains_all(pool, hashes)
            for pool, (hashes, _) in pools.items()
        )

    def ready(self, node):
        self._ensure_hashes(node)
        # L3 lookup requires a contiguous prefix. Releasing a persisted leaf
        # while an ancestor is still only in L2 creates a transient L3 miss.
        return all(
            self._pools_ready(self.describe(n)) for n in self._prefix_nodes(node)
        )

    def finish(self, operation_id, completed_tokens):
        pools = self.pending.pop(operation_id, {})
        hashes, _ = pools.get("kv", ((), 0))
        self.confirmed.add("kv", hashes[: completed_tokens // self.page_size])

    def record_prefetch(self, cache, operation, completed_tokens):
        from sglang.srt.mem_cache.hicache_storage import PoolName

        self.bind(cache.cache_controller.storage_backend)
        hashes = operation.hash_value[: completed_tokens // self.page_size]
        self.confirmed.add(PoolName.KV, hashes)
        if not getattr(operation, "pool_transfers_done", False):
            return
        hits = operation.pool_storage_result.extra_pool_hit_pages
        for transfer in operation.pool_transfers or []:
            if transfer.indices_from_pool == PoolName.KV:
                self.confirmed.add(transfer.name, hashes)
            else:
                self.confirmed.add(
                    transfer.name, (transfer.keys or [])[: hits.get(transfer.name, 0)]
                )

    def budget(self, capacity, available, pool="kv", page_size=None):
        pending = sum(pools.get(pool, ((), 0))[1] for pools in self.pending.values())
        return proactive_write_back_budget(
            capacity, available, pending, self.threshold, page_size or self.page_size
        )

    def _pools(self, cache):
        return self.pools or [
            ("kv", cache.cache_controller.mem_pool_host, cache.evict_host)
        ]

    def budgets(self, cache):
        if (
            not cache.enable_storage
            or cache.disable
            or cache.cache_controller.write_policy != "write_back"
        ):
            return []
        return [
            self.budget(
                pool.logical_size,
                pool.available_size(),
                name,
                getattr(pool, "page_size", self.page_size),
            )
            for name, pool, _ in self._pools(cache)
        ]

    def poll(self, cache, budgets):
        # Budgets were MIN-reduced with the existing event-poll counters.
        for (_, _, evict), count in zip(self._pools(cache), budgets):
            if count > 0:
                evict(count, blocking=False, write_back_only=True)

    def finish_acks(self, cache, operations, *, synchronized, defer=False):
        import torch

        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

        completed = [self.pending.pop(operation.id, {}) for operation in operations]
        # Detach/shutdown drains are local: never publish unagreed success.
        if not synchronized:
            return
        cc = cache.cache_controller
        counts = []
        pages = []
        for operation, pools in zip(operations, completed):
            result = getattr(operation, "pool_storage_result", None)
            hits = result.extra_pool_hit_pages if result is not None else {}
            for pool, (hashes, _) in pools.items():
                if pool == PoolName.KV:
                    count = (
                        len(hashes)
                        if cc.backup_skip
                        else operation.completed_tokens // self.page_size
                    )
                elif not cc.should_backup(PoolTransfer(name=pool)):
                    # Another rank writes replicated pools; sharded pools need
                    # successful local results on every rank.
                    count = len(hashes)
                else:
                    count = hits.get(pool, 0)
                counts.append(min(len(hashes), max(0, count)))
                pages.append((pool, hashes))
        if defer:
            self.deferred_counts.extend(counts)
            self.deferred_pages.extend(pages)
            return
        if not counts and not self.deferred_counts:
            return
        # The queue-count reduction has selected a common ACK prefix. Reduce
        # all its per-pool results together before any source lock is released.
        self.deferred_counts.extend(counts)
        self.deferred_pages.extend(pages)
        self.sync_poll(cache, torch.empty(0, dtype=torch.int64))

    def evict(
        self,
        num_tokens,
        walk,
        write,
        drain,
        *,
        blocking,
        pool="kv",
        write_back_only=False,
    ):
        """Walk cold nodes, stage dirty ones, and release confirmed ones only.

        write_back_only prepares a cold window without releasing source slots
        or waiting. Allocation pressure drains submitted writes and retries
        reclamation. A failed write ends that attempt.
        """
        freed = 0
        while freed < num_tokens:
            spent = 0
            submitted = 0
            pending_pages = {
                (name, key)
                for pools in self.pending.values()
                for name, (keys, _) in pools.items()
                for key in keys
            }
            pending_pages.update(
                (name, key) for name, keys in self.deferred_pages for key in keys
            )

            def prepare(node):
                # Return (release_source, stop_walk). Staging consumes the
                # budget too, even though it does not free source slots yet.
                nonlocal spent, submitted
                self._ensure_hashes(node)
                pools = self.describe(node)
                cost = pools.get(pool, ((), 0))[1]
                ready = self.ready(node)
                if not ready:
                    # Stage the entire dependency path parent-first. Shared
                    # ancestors already in flight must not be submitted again.
                    for ancestor in self._prefix_nodes(node):
                        ancestor_pools = self.describe(ancestor)
                        pages = {
                            (name, key)
                            for name, (keys, _) in ancestor_pools.items()
                            for key in keys
                        }
                        if self._pools_ready(ancestor_pools) or pages <= pending_pages:
                            continue
                        operation_id = write(ancestor)
                        if operation_id is None:
                            return False, True
                        self.pending[operation_id] = ancestor_pools
                        pending_pages.update(pages)
                        submitted += ancestor_pools.get(pool, ((), 0))[1]
                spent += cost
                stop = spent >= num_tokens - freed or (
                    write_back_only and submitted >= max(4096, self.page_size)
                )
                return ready and not write_back_only, stop

            freed += walk(num_tokens - freed, prepare)
            if (
                write_back_only
                or not blocking
                or freed >= num_tokens
                or not self.pending
            ):
                break
            drain(len(self.pending))
            reclaimed = walk(num_tokens - freed, lambda node: (self.ready(node), False))
            freed += reclaimed
            if reclaimed == 0:
                break
        return freed
