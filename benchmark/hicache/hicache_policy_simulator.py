"""Standalone L1/L2 HiCache policy simulator.

This module intentionally models cache residency at the page/hash-id level. It
does not import SGLang runtime cache classes and does not model GPU execution,
async transfers, batching, or L3 storage.
"""

from __future__ import annotations

import heapq
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

HashId = int | str
HeapEntry = tuple[int, tuple[str, ...], int, "SimNode"]


SUPPORTED_POLICIES = (
    "main_write_through",
    "boundary_l1_l2",
)


@dataclass(eq=False)
class SimNode:
    block_id: Optional[HashId]
    parent: Optional[SimNode] = None
    children: dict[HashId, SimNode] = field(default_factory=dict)
    path_key: tuple[str, ...] = field(default_factory=tuple)
    has_d: bool = False
    has_h: bool = False
    hit_count: int = 0
    last_access: int = 0

    @property
    def is_root(self) -> bool:
        return self.parent is None


@dataclass(frozen=True)
class TraceRecord:
    timestamp: float
    hash_ids: tuple[HashId, ...]
    output_length: Optional[int] = None


@dataclass
class PolicyMetrics:
    policy: str
    requests: int = 0
    total_input_pages: int = 0
    l1_hit_pages: int = 0
    l2_hit_pages: int = 0
    miss_pages: int = 0
    l1_evictions: int = 0
    l2_evictions: int = 0
    d_to_h_demotions: int = 0
    dh_to_d_l2_dedup_evictions: int = 0
    h_to_deleted_evictions: int = 0
    failed_h_allocations: int = 0
    d_pages: int = 0
    h_pages: int = 0
    dh_pages: int = 0
    unique_cached_pages: int = 0
    duplicate_ratio: float = 0.0

    @property
    def l1_hit_rate(self) -> float:
        if self.total_input_pages == 0:
            return 0.0
        return self.l1_hit_pages / self.total_input_pages

    @property
    def l1_l2_hit_rate(self) -> float:
        if self.total_input_pages == 0:
            return 0.0
        return (self.l1_hit_pages + self.l2_hit_pages) / self.total_input_pages

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["l1_hit_rate"] = self.l1_hit_rate
        data["l1_l2_hit_rate"] = self.l1_l2_hit_rate
        return data


class HiCachePolicySimulator:
    """Simulate one L1/L2 cache policy on page-level request traces."""

    def __init__(
        self,
        policy: str,
        l1_pages: int,
        l2_pages: int,
        write_through_threshold: int = 1,
    ):
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Unsupported policy {policy!r}. Supported: {SUPPORTED_POLICIES}"
            )
        if l1_pages < 0 or l2_pages < 0:
            raise ValueError("l1_pages and l2_pages must be non-negative")
        if write_through_threshold <= 0:
            raise ValueError("write_through_threshold must be positive")

        self.policy = policy
        self.l1_pages = l1_pages
        self.l2_pages = l2_pages
        self.write_through_threshold = write_through_threshold
        self.root = SimNode(block_id=None)
        self.d_pages = 0
        self.h_pages = 0
        self.clock = 0
        self.metrics = PolicyMetrics(policy=policy)
        self._heap_counter = count()
        self._l1_leaf_heap: list[HeapEntry] = []
        self._h_only_leaf_heap: list[HeapEntry] = []
        self._duplicate_h_leaf_heap: list[HeapEntry] = []

    @property
    def is_boundary_policy(self) -> bool:
        return self.policy == "boundary_l1_l2"

    @property
    def is_write_through_policy(self) -> bool:
        return self.policy == "main_write_through"

    def simulate_records(
        self,
        records: Iterable[TraceRecord],
        show_progress: bool = False,
        progress_interval: int = 1000,
        progress_seconds: float = 0.5,
        sanity_check_interval: int = 0,
    ) -> PolicyMetrics:
        sorted_records = sorted(records, key=lambda r: r.timestamp)
        progress = LivePolicyProgress(
            policy=self.policy,
            total=len(sorted_records),
            l1_pages=self.l1_pages,
            l2_pages=self.l2_pages,
            enabled=show_progress,
            interval=progress_interval,
            update_seconds=progress_seconds,
        )
        progress.start()
        for index, record in enumerate(sorted_records, 1):
            self.process_request(
                record.hash_ids,
                record.timestamp,
                validate=False,
            )
            if sanity_check_interval > 0 and (
                index % sanity_check_interval == 0 or index == len(sorted_records)
            ):
                self.sanity_check()
            progress.update(index, self)
        progress.finish(self)
        self._refresh_final_residency_metrics()
        return self.metrics

    def process_request(
        self,
        blocks: Iterable[HashId],
        timestamp: float = 0.0,
        validate: bool = True,
    ) -> None:
        block_tuple = tuple(blocks)
        self.clock += 1
        l1_hits, l2_hits = self._match_request(block_tuple)

        self.metrics.requests += 1
        self.metrics.total_input_pages += len(block_tuple)
        self.metrics.l1_hit_pages += l1_hits
        self.metrics.l2_hit_pages += l2_hits
        self.metrics.miss_pages += len(block_tuple) - l1_hits - l2_hits

        path = self._get_or_create_path(block_tuple)
        for node in path:
            node.last_access = self.clock
            self._ensure_d(node)
            self._queue_candidate_and_parent(node)

        if self.is_write_through_policy:
            self._write_through_touch(path)
        elif self.is_boundary_policy:
            self._boundary_write_through_touch(path)

        self._evict_l1_until_capacity()
        if validate:
            self.sanity_check()

    def sanity_check(self) -> None:
        d_count = 0
        h_count = 0
        for node in self._iter_nodes():
            if node.has_d:
                d_count += 1
            if node.has_h:
                h_count += 1
            if self.is_boundary_policy:
                self._check_boundary_node(node)
            elif self.is_write_through_policy:
                self._check_write_through_node(node)

        if d_count != self.d_pages:
            raise AssertionError(f"d_pages mismatch: {self.d_pages=} {d_count=}")
        if h_count != self.h_pages:
            raise AssertionError(f"h_pages mismatch: {self.h_pages=} {h_count=}")
        if self.d_pages > self.l1_pages:
            raise AssertionError(f"L1 over capacity: {self.d_pages}>{self.l1_pages}")
        if self.h_pages > self.l2_pages:
            raise AssertionError(f"L2 over capacity: {self.h_pages}>{self.l2_pages}")

    def _check_write_through_node(self, node: SimNode) -> None:
        if node.is_root or not node.has_h:
            return
        if not node.parent.is_root and not node.parent.has_h:
            raise AssertionError(
                "write-through H residency is not prefix-closed: "
                f"{self._path_tuple(node.parent)} -> {self._path_tuple(node)}"
            )

    def _check_boundary_node(self, node: SimNode) -> None:
        if node.is_root or not (node.has_h and not node.has_d):
            return
        parent = node.parent
        if parent.is_root:
            return
        if not parent.has_h:
            raise AssertionError(
                "boundary invariant violation: D-only parent has H-only child "
                f"{self._path_tuple(parent)} -> {self._path_tuple(node)}"
            )

    def _match_request(self, blocks: tuple[HashId, ...]) -> tuple[int, int]:
        node = self.root
        l1_hits = 0
        l2_hits = 0
        for block in blocks:
            child = node.children.get(block)
            if child is None:
                break
            if child.has_d:
                l1_hits += 1
                child.last_access = self.clock
            elif child.has_h:
                l2_hits += 1
                child.last_access = self.clock
            else:
                break
            self._queue_candidate_and_parent(child)
            node = child
        return l1_hits, l2_hits

    def _get_or_create_path(self, blocks: tuple[HashId, ...]) -> list[SimNode]:
        node = self.root
        path: list[SimNode] = []
        for block in blocks:
            child = node.children.get(block)
            if child is None:
                child = SimNode(
                    block_id=block,
                    parent=node,
                    path_key=node.path_key + (repr(block),),
                )
                node.children[block] = child
            path.append(child)
            node = child
        return path

    def _write_through_touch(self, path: list[SimNode]) -> None:
        for node in path:
            node.hit_count += 1
            if node.has_h or node.hit_count < self.write_through_threshold:
                continue
            if node.parent is not self.root and not node.parent.has_h:
                continue
            self._try_ensure_h(node)

    def _boundary_write_through_touch(self, path: list[SimNode]) -> None:
        self._try_reserve_h_slots(path, protected_nodes=set(path))

    def _ensure_d(self, node: SimNode) -> None:
        if not node.has_d:
            node.has_d = True
            self.d_pages += 1
            self._queue_candidate_and_parent(node)

    def _try_ensure_h(self, node: SimNode) -> bool:
        return self._try_reserve_h_slots([node])

    def _try_reserve_h_slots(
        self,
        nodes: Iterable[SimNode],
        protected_nodes: Optional[set[SimNode]] = None,
    ) -> bool:
        protected = set(protected_nodes or ())
        missing: list[SimNode] = []
        for node in nodes:
            protected.add(node)
            if not node.has_h:
                missing.append(node)

        if not missing:
            return True
        if len(missing) > self.l2_pages:
            self.metrics.failed_h_allocations += 1
            return False

        while self.h_pages + len(missing) > self.l2_pages:
            if not self._evict_l2_one(protected_nodes=protected):
                self.metrics.failed_h_allocations += 1
                return False

        for node in missing:
            node.has_h = True
            self.h_pages += 1
            self._queue_candidate_and_parent(node)
        return True

    def _evict_l1_until_capacity(self) -> None:
        while self.d_pages > self.l1_pages:
            victim = self._select_l1_victim()
            if victim is None:
                raise AssertionError("L1 over capacity but no D leaf victim exists")
            if self.is_boundary_policy:
                self._boundary_evict_l1_node(victim)
            else:
                self._write_through_evict_l1_node(victim)

    def _write_through_evict_l1_node(self, node: SimNode) -> None:
        if node.has_h:
            self._clear_d(node)
        else:
            self._delete_d_only_leaf(node)
        self.metrics.l1_evictions += 1

    def _boundary_evict_l1_node(self, node: SimNode) -> None:
        if node.has_h:
            if self._try_ensure_boundary_parent_before_h_only(node):
                self._clear_d(node)
            elif not self._has_h_descendant(node):
                self._clear_h(node)
                self.metrics.l2_evictions += 1
                self.metrics.h_to_deleted_evictions += 1
                self._delete_d_only_leaf(node)
            else:
                raise AssertionError("Cannot evict D+H node without breaking boundary")
            self.metrics.l1_evictions += 1
            return

        if self._try_ensure_h_for_demote(node):
            self._clear_d(node)
            self.metrics.d_to_h_demotions += 1
        else:
            self._delete_d_only_leaf(node)
        self.metrics.l1_evictions += 1

    def _try_ensure_h_for_demote(self, node: SimNode) -> bool:
        parent = node.parent
        required_nodes = [node]
        if (
            parent is not None
            and not parent.is_root
            and parent.has_d
            and not parent.has_h
        ):
            required_nodes.insert(0, parent)
        return self._try_reserve_h_slots(
            required_nodes,
            protected_nodes=set(required_nodes),
        )

    def _try_ensure_boundary_parent_before_h_only(self, node: SimNode) -> bool:
        parent = node.parent
        if parent is None or parent.is_root:
            return True
        if parent.has_d and not parent.has_h:
            return self._try_reserve_h_slots(
                [parent],
                protected_nodes={parent, node},
            )
        return True

    def _clear_d(self, node: SimNode) -> None:
        if node.has_d:
            node.has_d = False
            self.d_pages -= 1
            self._queue_candidate_and_parent(node)

    def _clear_h(self, node: SimNode) -> None:
        if node.has_h:
            node.has_h = False
            self.h_pages -= 1
            self._queue_candidate_and_parent(node)

    def _delete_d_only_leaf(self, node: SimNode) -> None:
        if node.has_h or self._has_d_child(node) or self._has_h_descendant(node):
            raise AssertionError("Refusing to delete a non-D-only leaf")
        self._clear_d(node)
        self._detach_if_empty(node)

    def _evict_l2_one(self, protected_nodes: Optional[set[SimNode]] = None) -> bool:
        if self.is_boundary_policy:
            return self._boundary_evict_l2_one(protected_nodes=protected_nodes)
        return self._write_through_evict_l2_one(protected_nodes=protected_nodes)

    def _write_through_evict_l2_one(
        self, protected_nodes: Optional[set[SimNode]] = None
    ) -> bool:
        victim = self._select_h_only_leaf(protected_nodes=protected_nodes)
        if victim is None:
            return False
        self._delete_h_only_leaf(victim)
        return True

    def _boundary_evict_l2_one(
        self, protected_nodes: Optional[set[SimNode]] = None
    ) -> bool:
        duplicate = self._select_duplicate_h_leaf(protected_nodes=protected_nodes)
        if duplicate is not None:
            self._clear_h(duplicate)
            self.metrics.l2_evictions += 1
            self.metrics.dh_to_d_l2_dedup_evictions += 1
            return True

        host_leaf = self._select_h_only_leaf(protected_nodes=protected_nodes)
        if host_leaf is not None:
            self._delete_h_only_leaf(host_leaf)
            return True

        return False

    def _delete_h_only_leaf(self, node: SimNode) -> None:
        if node.has_d or not node.has_h or self._has_h_child(node):
            raise AssertionError("Refusing to delete a non-H-only leaf")
        self._clear_h(node)
        self.metrics.l2_evictions += 1
        self.metrics.h_to_deleted_evictions += 1
        self._detach_if_empty(node)

    def _detach_if_empty(self, node: SimNode) -> None:
        while (
            not node.is_root and not node.has_d and not node.has_h and not node.children
        ):
            parent = node.parent
            assert parent is not None
            parent.children.pop(node.block_id, None)
            self._queue_candidate(parent)
            node = parent

    def _select_l1_victim(self) -> Optional[SimNode]:
        return self._pop_valid_candidate(self._l1_leaf_heap, self._is_l1_leaf_victim)

    def _select_h_only_leaf(
        self, protected_nodes: Optional[set[SimNode]] = None
    ) -> Optional[SimNode]:
        protected = protected_nodes or set()
        return self._pop_valid_candidate(
            self._h_only_leaf_heap,
            self._is_h_only_leaf_victim,
            protected_nodes=protected,
        )

    def _select_duplicate_h_leaf(
        self, protected_nodes: Optional[set[SimNode]] = None
    ) -> Optional[SimNode]:
        protected = protected_nodes or set()
        return self._pop_valid_candidate(
            self._duplicate_h_leaf_heap,
            self._is_duplicate_h_leaf_victim,
            protected_nodes=protected,
        )

    def _queue_candidate(self, node: Optional[SimNode]) -> None:
        if node is None or node.is_root:
            return
        entry = (
            node.last_access,
            self._path_sort_key(node),
            next(self._heap_counter),
            node,
        )
        if self._is_l1_leaf_victim(node):
            heapq.heappush(self._l1_leaf_heap, entry)
        if self._is_h_only_leaf_victim(node):
            heapq.heappush(self._h_only_leaf_heap, entry)
        if self._is_duplicate_h_leaf_victim(node):
            heapq.heappush(self._duplicate_h_leaf_heap, entry)

    def _queue_candidate_and_parent(self, node: SimNode) -> None:
        self._queue_candidate(node)
        self._queue_candidate(node.parent)

    def _pop_valid_candidate(
        self,
        heap,
        is_valid,
        protected_nodes: Optional[set[SimNode]] = None,
    ) -> Optional[SimNode]:
        protected = protected_nodes or set()
        skipped: list[HeapEntry] = []
        while heap:
            entry = heapq.heappop(heap)
            last_access, path_key, _, node = entry
            if (
                node.last_access == last_access
                and self._path_sort_key(node) == path_key
                and is_valid(node)
            ):
                if node in protected:
                    skipped.append(entry)
                    continue
                for skipped_entry in skipped:
                    heapq.heappush(heap, skipped_entry)
                return node
        for skipped_entry in skipped:
            heapq.heappush(heap, skipped_entry)
        return None

    def _is_l1_leaf_victim(self, node: SimNode) -> bool:
        return node.has_d and not self._has_d_child(node)

    def _is_h_only_leaf_victim(self, node: SimNode) -> bool:
        return node.has_h and not node.has_d and not self._has_h_child(node)

    def _is_duplicate_h_leaf_victim(self, node: SimNode) -> bool:
        return node.has_d and node.has_h and not self._has_h_child(node)

    def _has_d_child(self, node: SimNode) -> bool:
        return any(child.has_d for child in node.children.values())

    def _has_h_child(self, node: SimNode) -> bool:
        return any(child.has_h for child in node.children.values())

    def _has_h_descendant(self, node: SimNode) -> bool:
        stack = list(node.children.values())
        while stack:
            child = stack.pop()
            if child.has_h:
                return True
            stack.extend(child.children.values())
        return False

    def _iter_nodes(self) -> Iterable[SimNode]:
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def _path_tuple(self, node: SimNode) -> tuple[HashId, ...]:
        blocks: list[HashId] = []
        while not node.is_root:
            assert node.block_id is not None
            blocks.append(node.block_id)
            assert node.parent is not None
            node = node.parent
        return tuple(reversed(blocks))

    def _path_sort_key(self, node: SimNode) -> tuple[str, ...]:
        if node.path_key or node.is_root:
            return node.path_key
        return tuple(repr(block) for block in self._path_tuple(node))

    def _refresh_final_residency_metrics(self) -> None:
        d_pages = 0
        h_pages = 0
        dh_pages = 0
        for node in self._iter_nodes():
            if node.has_d:
                d_pages += 1
            if node.has_h:
                h_pages += 1
            if node.has_d and node.has_h:
                dh_pages += 1
        self.metrics.d_pages = d_pages
        self.metrics.h_pages = h_pages
        self.metrics.dh_pages = dh_pages
        self.metrics.unique_cached_pages = d_pages + h_pages - dh_pages
        self.metrics.duplicate_ratio = 0.0 if h_pages == 0 else dh_pages / h_pages


def load_trace_records(
    path: str | Path, max_requests: Optional[int] = None
) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    with Path(path).open("r") as f:
        for line_no, line in enumerate(f, 1):
            if max_requests is not None and len(records) >= max_requests:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            if "hash_ids" not in obj:
                raise ValueError(f"Missing hash_ids at line {line_no}")
            timestamp = float(obj.get("timestamp", len(records)))
            output_length = obj.get("output_length")
            records.append(
                TraceRecord(
                    timestamp=timestamp,
                    hash_ids=tuple(obj["hash_ids"]),
                    output_length=output_length,
                )
            )
    return records


class LivePolicyProgress:
    """TTY-only single-line progress for long policy simulations."""

    def __init__(
        self,
        policy: str,
        total: int,
        l1_pages: int,
        l2_pages: int,
        enabled: bool,
        interval: int,
        stream: TextIO = sys.stderr,
        update_seconds: float = 0.5,
    ):
        self.policy = policy
        self.total = total
        self.l1_pages = l1_pages
        self.l2_pages = l2_pages
        self.interval = max(interval, 1)
        self.stream = stream
        self.update_seconds = max(update_seconds, 0.0)
        self.enabled = enabled and stream.isatty()
        self.started_at = 0.0
        self.last_update_at = 0.0
        self.last_render_index = 0
        self.last_render_at = 0.0

    def start(self) -> None:
        if not self.enabled:
            return
        self.started_at = time.perf_counter()
        self.last_render_at = self.started_at

    def update(self, index: int, simulator: HiCachePolicySimulator) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        due_by_count = index % self.interval == 0
        due_by_time = now - self.last_update_at >= self.update_seconds
        if index != self.total and not due_by_count and not due_by_time:
            return
        self.last_update_at = now
        self._render(index, simulator, final=False, now=now)

    def finish(self, simulator: HiCachePolicySimulator) -> None:
        if not self.enabled:
            return
        self._render(
            simulator.metrics.requests,
            simulator,
            final=True,
            now=time.perf_counter(),
        )
        self.stream.write("\n")
        self.stream.flush()

    def _render(
        self,
        index: int,
        simulator: HiCachePolicySimulator,
        final: bool,
        now: float,
    ) -> None:
        metrics = simulator.metrics
        elapsed = max(now - self.started_at, 1e-9)
        avg_req_rate = index / elapsed
        interval_elapsed = max(now - self.last_render_at, 1e-9)
        current_req_rate = (index - self.last_render_index) / interval_elapsed
        self.last_render_index = index
        self.last_render_at = now
        progress = 0.0 if self.total == 0 else index / self.total
        l1_fill = _ratio(simulator.d_pages, self.l1_pages)
        l2_fill = _ratio(simulator.h_pages, self.l2_pages)
        l1_hit = _ratio(metrics.l1_hit_pages, metrics.total_input_pages)
        l1_l2_hit = _ratio(
            metrics.l1_hit_pages + metrics.l2_hit_pages,
            metrics.total_input_pages,
        )
        state = "done" if final else "run "
        line = (
            f"{self.policy:<18} {state} "
            f"{index:>8}/{self.total:<8} {_format_pct(progress)} "
            f"cur {current_req_rate:>7.1f} req/s avg {avg_req_rate:>7.1f} | "
            f"L1 {simulator.d_pages:>8}/{self.l1_pages:<8} {_format_pct(l1_fill)} "
            f"L2 {simulator.h_pages:>8}/{self.l2_pages:<8} {_format_pct(l2_fill)} | "
            f"hit L1 {_format_pct(l1_hit)} L1+L2 {_format_pct(l1_l2_hit)} | "
            f"evict L1 {metrics.l1_evictions} L2 {metrics.l2_evictions} "
            f"failed_H {metrics.failed_h_allocations}"
        )
        self.stream.write("\r\033[2K" + line)
        self.stream.flush()


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _format_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def simulate_policies(
    records: Iterable[TraceRecord],
    policies: Iterable[str],
    l1_pages: int,
    l2_pages: int,
    write_through_threshold: int = 1,
    show_progress: bool = False,
    progress_interval: int = 1000,
    progress_seconds: float = 0.5,
    sanity_check_interval: int = 0,
) -> dict[str, PolicyMetrics]:
    record_list = list(records)
    results: dict[str, PolicyMetrics] = {}
    for policy in policies:
        simulator = HiCachePolicySimulator(
            policy=policy,
            l1_pages=l1_pages,
            l2_pages=l2_pages,
            write_through_threshold=write_through_threshold,
        )
        results[policy] = simulator.simulate_records(
            record_list,
            show_progress=show_progress,
            progress_interval=progress_interval,
            progress_seconds=progress_seconds,
            sanity_check_interval=sanity_check_interval,
        )
    return results


def metrics_to_json_dict(metrics: dict[str, PolicyMetrics]) -> dict[str, Any]:
    return {name: value.to_dict() for name, value in metrics.items()}
