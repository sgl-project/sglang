"""Overlap write-back of cold GPU leaves with decode."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch

from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.managers.utils import allocate_distinct_stream
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_capture_attention_variant,
)

_WRITE_BACK_GRAPH = "hicache_write_back"
_BACKGROUND_LEAVES_PER_FORWARD = 4


class WriteBackFences:
    capture_labels = ("normal", _WRITE_BACK_GRAPH)

    def __init__(self):
        self.events = {}
        self.active = False

    def select(self, forward_batch):
        return _WRITE_BACK_GRAPH if self.active else "normal"

    def event_for(self, layer_id):
        event = self.events.get(layer_id)
        if event is None:
            event = self.events[layer_id] = torch.cuda.Event(external=True)
            event.record()
        return event

    def wait_before_store(self, layer_id):
        if self.active or get_capture_attention_variant() == _WRITE_BACK_GRAPH:
            self.event_for(layer_id).wait()


def write_back_fences(backend):
    full_backend = getattr(backend, "full_attn_backend", backend)
    return getattr(full_backend, "write_back_fences", None)


@dataclass
class StagedIndices:
    device: torch.Tensor
    cpu: torch.Tensor

    def matches(self, other):
        return (
            self.device.data_ptr() == other.data_ptr()
            and self.device.shape == other.shape
        )


@dataclass
class StagedLeaf:
    full: StagedIndices
    state: dict
    ready: torch.cuda.Event


class WriteBackOverlap:
    """Scheduler-owned page handoff for static, paged CUDA Full KV pools."""

    def __init__(self, scheduler, cache, fences):
        self.scheduler = scheduler
        self.cache = cache
        self.allocator = cache.token_to_kv_pool_allocator
        self.controller = cache.cache_controller
        self.fences = fences
        self.engine = self.controller.l2_transfer_engine
        self.page_size = cache.page_size
        self.staged = {}
        self.background = deque()
        self.retract_first = False
        self.pending = {}
        self.previous_readback = None
        self.forward_start = None
        self.streams = None

    @classmethod
    def create(cls, scheduler):
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel

        cache = scheduler.tree_cache
        if not isinstance(cache, UnifiedRadixCache) or not cache.is_write_back:
            return None
        parallel = get_parallel()
        if (
            not scheduler.enable_overlap
            or scheduler.enable_async_d2h_copy
            or not scheduler.spec_algorithm.is_none()
            or parallel.pp_size != 1
            or parallel.attn_dp_size != 1
            or parallel.dcp_enabled
            or get_disagg().disaggregation_mode != "null"
            or get_memory().enable_unified_memory
            or type(cache.token_to_kv_pool_allocator) is not PagedTokenToKVPoolAllocator
            or cache.page_size == 1
            or cache.enable_storage
            or cache.linker is not None
            or cache.buffer_pipeline is not None
            or cache.sidecar_pool_specs
            or cache.window_slots()
        ):
            return None
        cc = cache.cache_controller
        if cc is None or cc.io_backend != "direct" or cc.is_kv_sharded:
            return None
        anchor = cc.mem_pool_host.anchor_entry
        pool = anchor.host_pool
        if anchor.device_translate is not None:
            return None
        state = getattr(cache.token_to_kv_pool_allocator, "state", None)
        if getattr(state, "ckpt_pool", None) is not None:
            return None
        if (
            pool.layout != "page_first_direct"
            or pool.shared_allocation_domain is not None
            or any(
                entry.host_pool.shared_allocation_domain is not None
                for entry in cc.mem_pool_host.entry_map.values()
            )
        ):
            return None
        if not (
            hasattr(pool, "device_k_buffers")
            and pool.k_buffer.is_pinned()
            and pool.v_buffer.is_pinned()
            and pool.k_buffer.dtype == pool.device_k_buffers[0].dtype
            and pool.v_buffer.dtype == pool.device_v_buffers[0].dtype
        ):
            return None
        runner = scheduler.tp_worker.model_runner
        fences = write_back_fences(runner.attn_backend)
        graph = runner.decode_cuda_graph_runner
        if fences is None or (
            graph is not None
            and getattr(graph, "attention_graph_variants", None) is not fences
        ):
            return None
        return cls(scheduler, cache, fences)

    def _streams(self):
        if self.streams is None:
            avoided = (
                self.scheduler.schedule_stream,
                self.scheduler.forward_stream,
                self.scheduler.copy_stream,
                self.engine.device_to_host_stream,
                self.engine.host_to_device_stream,
            )
            indices = allocate_distinct_stream(torch.cuda, avoided)
            full = allocate_distinct_stream(torch.cuda, (*avoided, indices))
            background = allocate_distinct_stream(torch.cuda, (*avoided, indices, full))
            self.streams = indices, full, background
        return self.streams

    def _is_cold_leaf(self, node):
        tree = self.cache.tree_core
        inspect = getattr(tree, "inspect_is_device_evictable_leaf", None)
        if inspect is not None:
            return inspect(node)
        return tree._is_device_leaf(tree.node_by_id(node))

    def stage(self, node):
        tree = self.cache.tree_core
        if node in self.staged or not self._is_cold_leaf(node):
            return
        value, pools = tree.build_backup_spec(node)
        if value.numel() == 0 or any(
            transfer.device_indices is None
            or transfer.indices_from_pool is not None
            or self.controller.mem_pool_host.entry_map[transfer.name].device_translate
            is not None
            for group in pools.values()
            for transfer in group
        ):
            return
        stream, _, _ = self._streams()
        stream.wait_stream(torch.cuda.current_stream())
        stream.wait_stream(self.scheduler.forward_stream)

        def stage_indices(tensor):
            cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
            cpu.copy_(tensor, non_blocking=True)
            tensor.record_stream(stream)
            return StagedIndices(tensor, cpu)

        with torch.cuda.stream(stream):
            full = stage_indices(value)
            state = {
                transfer.name: stage_indices(transfer.device_indices)
                for transfers in pools.values()
                for transfer in transfers
            }
            self.staged[node] = StagedLeaf(full, state, torch.cuda.Event())
            self.staged[node].ready.record(stream)

    def _pick_leaves(self, tokens):
        tree = self.cache.tree_core
        tracker = {component: 0 for component in self.cache.tree_components}
        leaves = []
        tree.evict_device_start(ComponentType.FULL, tokens)
        try:
            while tracker[ComponentType.FULL] < tokens:
                result = tree.evict_device_next_node(ComponentType.FULL, tracker)
                node = result.node_id
                if node is None:
                    break
                # Only preview existing cold leaves. Never retire a device
                # value, stage a parent, or update the allocator in this pass.
                if (
                    result.device_frees
                    or result.host_frees
                    or result.cache_actions
                    or not self._is_cold_leaf(node)
                ):
                    return None
                staged = self.staged.get(node)
                if staged is None or not staged.ready.query():
                    return None
                value, pools = tree.build_backup_spec(node)
                extra = [transfer for group in pools.values() for transfer in group]
                if not staged.full.matches(value) or any(
                    t.name not in staged.state
                    or not staged.state[t.name].matches(t.device_indices)
                    for t in extra
                ):
                    return None
                row = staged.full.cpu
                starts = row[:: self.page_size]
                if (
                    row.numel() % self.page_size
                    or not torch.equal(
                        row.view(-1, self.page_size),
                        starts[:, None] + torch.arange(self.page_size),
                    )
                    or bool((starts % self.page_size).any())
                ):
                    return None
                leaves.append((node, value, pools, staged, starts.tolist()))
                tracker[ComponentType.FULL] += value.numel()
        finally:
            tree.evict_device_end(ComponentType.FULL)
        return leaves

    def _can_stage(self, leaves):
        group = self.controller.mem_pool_host
        if group.available_size() < sum(value.numel() for _, value, _, _, _ in leaves):
            return False
        per_pool = {}
        for _, _, _, staged, _ in leaves:
            for name, ids in staged.state.items():
                per_pool[name] = per_pool.get(name, 0) + ids.cpu.numel()
        return all(
            group.entry_map[name].host_pool.available_size() >= count
            for name, count in per_pool.items()
        )

    def _unanimous(self, ready):
        if self.cache.tp_world_size == 1:
            return ready
        vote = torch.tensor([int(ready)], dtype=torch.int32, device="cpu")
        self.cache._all_reduce(vote, torch.distributed.ReduceOp.MIN)
        return bool(vote.item())

    def _first_retract_can_fit(self, batch, required):
        if len(batch.reqs) < 2:
            return False
        order = batch._get_decode_retraction_order(batch.reqs)
        victim = batch.reqs[order.pop()]
        kv = victim.kv
        # Private whole pages are deterministically freed by normal retract.
        p = self.page_size
        pages = (kv.kv_allocated_len + p - 1) // p - (
            kv.cache_protected_len + p - 1
        ) // p
        survivor_tokens = batch.new_tokens_required_next_decode(order)
        victim_demand = batch.new_tokens_required_next_decode() - survivor_tokens
        return pages * p >= required - victim_demand and victim.resumable_by_reprefill

    def prepare_decode(self, batch, *, allow_new_overlap=True):
        """Return None for stock, True for a page handoff, False for retract-first."""
        if (
            self.scheduler.uocr_runtime is not None
            or self.scheduler.decode_offload_manager is not None
            or any(r.beam_group is not None for r in batch.reqs)
        ):
            return None
        shortfall = (
            batch.new_tokens_required_next_decode() - self.allocator.available_size()
        )
        if shortfall <= 0:
            # Reserving one new page per row would evict cold leaves before
            # decode actually needs them, bypassing the overlap at the boundary.
            batch.write_back_overlap_pages = ()
            return None
        if (
            not allow_new_overlap
            or self.background
            or self.pending
            or self.cache.ongoing_write_through
        ):
            return None
        evictable = self.cache.full_evictable_size()
        if not evictable:
            return None
        enough = evictable >= shortfall
        leaves = self._pick_leaves(min(evictable, shortfall))
        ready = bool(leaves) and self._can_stage(leaves)
        if ready:
            selected = sum(item[1].numel() for item in leaves)
            ready = (
                selected >= shortfall
                if enough
                else (
                    selected == evictable
                    and self._first_retract_can_fit(batch, shortfall)
                )
            )
        if not self._unanimous(ready):
            return None

        indexed = self._allocate_transfers(leaves)
        if indexed is None:
            return None
        self.retract_first = not enough
        if self.retract_first:
            tree = self.cache.tree_core
            for node, _, pools, staged, op, transfers, _ in indexed:
                tree.commit_backup(node, op.host_indices, pools)
                lock = self.cache.inc_lock_ref(node).to_dec_params()
                publish = self.cache._backup_publish_node_ids(node, pools)
                self.cache._track_write_through_node(node, lock, publish)
                self.pending[node] = None
                self.staged.pop(node, None)
                self.background.append((op, transfers, staged.ready))
            batch.write_back_overlap_pages = ()
            return False

        batch.write_back_overlap_pages = self._reserve(
            indexed, shortfall // self.page_size
        )
        return True

    def _allocate_transfers(self, leaves):
        # Prepare host destinations, but do not alter tree ownership until
        # each rank has reserved all its destinations.
        cc, tree = self.controller, self.cache.tree_core
        indexed = []
        for node, value, pools, staged, src in leaves:
            if self.cache._execute_kv_backup(node, value, pools, []) is None:
                break
            op = cc.write_queue.pop()
            state = [
                replace(t, device_indices=staged.state[t.name].cpu)
                for t in op.pool_transfers or ()
            ]
            transfers = cc._l2_transfers(op.host_indices, staged.full.cpu, state)
            dst = op.host_indices[:: self.page_size].tolist()
            indexed.append(
                (
                    node,
                    value,
                    pools,
                    staged,
                    op,
                    transfers,
                    list(zip(src, dst, strict=True)),
                )
            )

        if not self._unanimous(len(indexed) == len(leaves)):
            # Nothing was detached; let the ordinary writer settle any host
            # allocations that did succeed. Cache metadata remains native.
            for node, _, pools, _, op, _, _ in indexed:
                tree.commit_backup(node, op.host_indices, pools)
                publish = self.cache._backup_publish_node_ids(node, pools)
                self.cache._track_write_through_node(node, None, publish)
                cc.write_queue.append(op)
            self.cache.writing_check(write_back=True)
            return None
        return indexed

    def _reserve(self, indexed, num_pages):
        cc, tree = self.controller, self.cache.tree_core
        _, stream, _ = self._streams()
        anchor = cc.mem_pool_host.anchor_entry
        pool = anchor.host_pool

        # Launch Full D2H before the next forward. Only the corresponding
        # layer's store waits on each event, not the whole graph.
        with torch.cuda.stream(stream):
            for _, _, _, staged, *_ in indexed:
                staged.ready.wait(stream)
            for layer_id in range(cc.transfer_layer_id_max):
                layer = anchor.layer_mapper(layer_id)
                if layer is None:
                    continue
                for *_, pairs in indexed:
                    for src, dst in pairs:
                        page = dst // self.page_size
                        end = src + self.page_size
                        pool.k_buffer[page, layer].copy_(
                            pool.device_k_buffers[layer][src:end], non_blocking=True
                        )
                        pool.v_buffer[page, layer].copy_(
                            pool.device_v_buffers[layer][src:end], non_blocking=True
                        )
                self.fences.event_for(layer_id).record(stream)
            full_done = torch.cuda.Event()
            full_done.record(stream)

        pages = []
        for node, value, pools, _, op, transfers, _ in indexed:
            take = min(num_pages, value.numel() // self.page_size)
            num_pages -= take
            pages.append(
                value[: take * self.page_size : self.page_size] // self.page_size
            )

            # Commit and detach synchronously on the scheduler. Before
            # publishing the async ACK, the old host node is pinned and the
            # reserved device indices never pass through the public free list.
            tree.commit_backup(node, op.host_indices, pools)
            host_lock = self.cache.inc_host_lock_ref(node).to_dec_params()
            result = tree.demote(node)
            delayed = dict(result.device_frees)
            del delayed[ComponentType.FULL]
            if take * self.page_size < value.numel():
                delayed[ComponentType.FULL] = [value[take * self.page_size :]]
            self.cache._drain_host_frees(result.host_frees)
            self.cache._apply_cache_actions(result.cache_actions)
            publish = self.cache._backup_publish_node_ids(node, pools)
            self.cache._track_write_through_node(node, None, publish)
            self.pending[node] = (host_lock, delayed)
            self.staged.pop(node, None)
            if len(transfers) > 1:
                self.background.append((op, transfers[1:], full_done))
            else:
                self._ack(op, full_done, full_done)
        return torch.cat(pages)

    @contextmanager
    def forward(self, batch):
        pages = batch.write_back_overlap_pages
        self.fences.active = isinstance(pages, torch.Tensor)
        self.forward_start = None
        if batch.forward_mode.is_decode() and self.background:
            self.forward_start = torch.cuda.Event()
            self.forward_start.record()
        try:
            yield
        finally:
            self.fences.active = False

    def submit_background(self, *, drain=False):
        if not self.background or (not drain and self.forward_start is None):
            return
        if drain:
            n = len(self.background)
        else:
            n = 1 if self.retract_first else _BACKGROUND_LEAVES_PER_FORWARD
        _, _, stream = self._streams()
        for _ in range(min(n, len(self.background))):
            op, transfers, ready = self.background.popleft()
            if not drain:
                self.forward_start.wait(stream)
                if self.previous_readback is not None:
                    self.previous_readback.wait(stream)
            with self.engine._submission(
                transfers, stream, "write_back_overlap", start_event=ready
            ) as (prepared, completion):
                for transfer in prepared:
                    transfer.host_pool.backup_from_device_all_layer_physical(
                        transfer.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        self.engine.io_backend,
                    )
            self._ack(
                op,
                completion.start_event,
                completion.finish_event,
                completion.timing_enabled,
            )

    def _ack(self, op, start_event, finish_event, timing_enabled=False):
        cc = self.controller
        cc.ack_write_queue.append(
            HiCacheAck(
                start_event=start_event,
                finish_event=finish_event,
                node_ids=op.node_ids,
                num_tokens=op.device_indices.numel(),
                timing_enabled=timing_enabled,
                num_tokens_by_pool=cc._num_tokens_by_pool(op),
                num_bytes=cc._transfer_num_bytes(op),
            )
        )

    def protected_size(self, component):
        return sum(
            value.numel()
            for reservation in self.pending.values()
            if reservation is not None
            for value in reservation[1].get(component, ())
        )

    def finish(self, node):
        if node not in self.pending:
            return
        reservation = self.pending.pop(node)
        if reservation is None:
            if self._is_cold_leaf(node):
                self.cache._demote(node, {c: 0 for c in self.cache.tree_components})
        else:
            host_lock, device_frees = reservation
            self.cache._drain_device_frees(device_frees)
            self.cache.dec_host_lock_ref(node, host_lock)
