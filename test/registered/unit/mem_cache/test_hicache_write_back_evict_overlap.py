"""Write-back eviction on a PD prefill server must not wait for the in-flight
forward, and must still copy the data the victim held.

The scheduler is emulated as in production: the cache runs on a non-default
schedule stream, a forward stream records the forward-start event like
Scheduler.run_batch, and the real Scheduler._apply_war_barrier runs. The
in-flight forward is a long ``torch.cuda._sleep``."""

import contextlib
import functools
import time
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.mem_cache.unified_radix_cache as urc_mod
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_device_value,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")

PREFILL = SimpleNamespace(disaggregation_mode="prefill")
FWD_S = 1.0  # Simulated in-flight forward.


@functools.lru_cache(maxsize=None)
def _sleep_cycles_per_s() -> float:
    cycles = 50_000_000
    torch.cuda._sleep(cycles)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(4 * cycles)
    end.record()
    end.synchronize()
    return 4 * cycles / (start.elapsed_time(end) / 1e3)


def _sleep_s(seconds: float) -> None:
    torch.cuda._sleep(int(seconds * _sleep_cycles_per_s()))


def _distinct_stream(avoid) -> torch.cuda.Stream:
    handles = {s.cuda_stream for s in avoid}
    for _ in range(65):
        stream = torch.cuda.Stream()
        if stream.cuda_stream not in handles:
            return stream
    raise RuntimeError("no distinct stream")


def _kv_buffers(allocator):
    kv = allocator.get_kvcache()
    if isinstance(kv, SWAKVPool):
        pools = [kv.full_kv_pool, kv.swa_kv_pool]
    elif isinstance(kv, HybridLinearKVPool):
        pools = [kv.full_kv_pool]
    else:
        pools = [kv]
    return [buf for pool in pools for buf in list(pool.k_buffer) + list(pool.v_buffer)]


def _fill(allocator, indices, value):
    for buf in _kv_buffers(allocator):
        buf[indices] = value


def _host_values(log):
    """Distinct values in every KV host page the recorded transfers wrote."""
    torch.cuda.synchronize()
    values, mamba_transfers = set(), 0
    for transfer in log:
        if "mamba" in type(transfer.host_pool).__name__.lower():
            mamba_transfers += 1
            continue
        for h in transfer.host_indices.reshape(-1).tolist():
            page = transfer.host_pool.get_data_page(int(h))
            values |= set(page.float().unique().tolist())
    return sorted(values), mamba_transfers


class _OverlapCases(UnifiedRadixCacheFixtures):
    CFG: CacheConfig

    def setUp(self):
        self.cfg = self.CFG
        p = patch.object(urc_mod, "get_disagg", return_value=PREFILL)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(torch.cuda.synchronize)

    def _fixture(self):
        cache, allocator, req_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        cc = cache.cache_controller
        eng = cc.l2_transfer_engine
        engine_streams = [
            eng.device_to_host_stream,
            eng.host_to_device_stream,
            eng.index_stream,
        ]
        self.schedule = _distinct_stream(engine_streams)
        self.forward = _distinct_stream(engine_streams + [self.schedule])
        cc.write_fence_stream = self.forward
        cc.load_fence_stream = self.forward
        self.fwd_done = None
        return cache, allocator, req_pool, cc

    def _sched(self):
        return torch.cuda.stream(self.schedule)

    def _launch_forward(self, cc, seconds=FWD_S, barrier=True):
        with torch.cuda.stream(self.forward):
            self.forward.wait_stream(self.schedule)
            cc.record_forward_start(self.forward.record_event())
            _sleep_s(seconds)
            self.fwd_done = self.forward.record_event()
        if barrier:
            runner = SimpleNamespace(shared_read_done_event=None)
            scheduler = SimpleNamespace(
                _war_barrier_enabled=True,
                model_worker=SimpleNamespace(last_shared_read_runner=runner),
                _hicache_fence_controller=cc,
                schedule_stream=self.schedule,
                forward_stream=self.forward,
            )
            with self._sched():
                Scheduler._apply_war_barrier(scheduler)
            self.assertTrue(cc.war_barrier_after_forward)

    def _forward_running(self):
        return self.fwd_done is not None and not self.fwd_done.query()

    def _insert_leaf(self, cache, allocator, req_pool, tokens, value=7):
        with self._sched():
            result = self._insert(cache, allocator, req_pool, list(tokens))
            indices = node_device_value(
                cache, result.last_device_node, ComponentType.FULL
            )
            _fill(allocator, indices, value)
        return indices.clone()

    def _recording(self, engine, log):
        submit = engine.submit_device_to_host

        def record(transfers):
            log.extend(transfers)
            return submit(transfers)

        return patch.object(engine, "submit_device_to_host", side_effect=record)

    def _warm(self, cache, allocator, req_pool):
        # One eviction with no forward in flight: no first-use cost later.
        self._insert_leaf(cache, allocator, req_pool, range(300, 316), value=3)
        with self._sched():
            evicted = cache.evict(EvictParams(num_tokens=16))
        self.assertGreaterEqual(evicted.num_tokens_evicted, 1)
        torch.cuda.synchronize()

    def _evict_leaf(self, barrier=True, kill_switch=False, overlap_stream=True):
        cache, allocator, req_pool, cc = self._fixture()
        self._warm(cache, allocator, req_pool)
        self._insert_leaf(cache, allocator, req_pool, range(200, 216), value=7)
        torch.cuda.synchronize()
        self._launch_forward(cc, barrier=barrier)
        log = []
        with contextlib.ExitStack() as stack:
            if kill_switch:
                stack.enter_context(
                    envs.SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE.override(False)
                )
            if not overlap_stream:
                stack.enter_context(
                    patch.object(cache, "_wb_evict_overlap_stream", return_value=None)
                )
            stack.enter_context(self._recording(cc.l2_transfer_engine, log))
            stack.enter_context(self._sched())
            start = time.perf_counter()
            evicted = cache.evict(EvictParams(num_tokens=16))
            seconds = time.perf_counter() - start
        forward_running = self._forward_running()
        self.forward.synchronize()
        self.assertGreaterEqual(evicted.num_tokens_evicted, 16)
        values, mamba_transfers = _host_values(log)
        result = SimpleNamespace(
            seconds=seconds,
            forward_running=forward_running,
            values=values,
            mamba_transfers=mamba_transfers,
            counts=dict(cache._wb_overlap_counts),
        )
        print(f"{self.id()} [{self.cfg.label}]: {result}")
        return result

    def test_evict_after_war_barrier_overlaps_forward(self):
        r = self._evict_leaf()
        self.assertEqual(r.values, [7.0])
        self.assertEqual(r.counts["overlap"], 1)
        self.assertTrue(r.forward_running, "eviction waited for the forward")
        self.assertLess(r.seconds, 0.25 * FWD_S)
        if self.cfg.has_mamba:
            self.assertGreaterEqual(r.mamba_transfers, 1)

    def test_evict_without_barrier_overlaps_forward(self):
        r = self._evict_leaf(barrier=False)
        self.assertEqual(r.values, [7.0])
        self.assertEqual(r.counts["overlap"], 0)
        self.assertTrue(r.forward_running, "eviction waited for the forward")

    def test_kill_switch_waits_for_forward(self):
        r = self._evict_leaf(kill_switch=True)
        self.assertEqual(r.values, [7.0])
        self.assertFalse(r.forward_running)
        self.assertGreater(r.seconds, 0.5 * FWD_S)
        self.assertEqual(sum(r.counts.values()), 0)

    def test_barrier_blocks_relaxed_fence_without_overlap_stream(self):
        # Control: after the barrier the schedule stream itself waits on the
        # forward, so the relaxed D2H fence alone does not overlap.
        r = self._evict_leaf(overlap_stream=False)
        self.assertEqual(r.values, [7.0])
        self.assertFalse(r.forward_running)

    def _evict_then_reuse_pages(self, cpu_waits_for_copy=True):
        """The D2H runs past the forward; a later forward then overwrites every
        free page, the victim's included."""
        cache, allocator, req_pool, cc = self._fixture()
        self._warm(cache, allocator, req_pool)
        victim = self._insert_leaf(cache, allocator, req_pool, range(200, 216))
        torch.cuda.synchronize()
        self._launch_forward(cc, seconds=0.5 * FWD_S)
        with self._sched():
            # Draw the private stream first: it may redraw an aliased D2H stream.
            cache._wb_overlap_get_stream(cc)
        eng = cc.l2_transfer_engine
        with torch.cuda.stream(eng.device_to_host_stream):
            _sleep_s(1.5 * FWD_S)
        log = []
        submit = eng.submit_device_to_host

        def submit_without_cpu_wait(transfers):
            log.extend(transfers)
            completion = submit(transfers)
            finish = completion.finish_event
            no_wait = SimpleNamespace(
                synchronize=lambda: None,
                query=lambda: True,
                record=finish.record,
                wait=finish.wait,
            )
            return completion._replace(finish_event=no_wait, timing_enabled=False)

        def submit_recorded(transfers):
            log.extend(transfers)
            return submit(transfers)

        side_effect = submit_recorded if cpu_waits_for_copy else submit_without_cpu_wait
        start = time.perf_counter()
        with (
            patch.object(eng, "submit_device_to_host", side_effect=side_effect),
            self._sched(),
        ):
            evicted = cache.evict(EvictParams(num_tokens=16))
            seconds = time.perf_counter() - start
            self.assertGreaterEqual(evicted.num_tokens_evicted, 16)
            reused = allocator.alloc(allocator.available_size())
            self.assertTrue(set(victim.tolist()) <= set(reused.tolist()))
        later = _distinct_stream(
            [self.schedule, self.forward, eng.device_to_host_stream, eng.index_stream]
        )
        with torch.cuda.stream(later):
            later.wait_stream(self.schedule)  # As run_batch does.
            _fill(allocator, reused, 99)
        values, _ = _host_values(log)
        print(f"{self.id()} [{self.cfg.label}]: {seconds=:.3f} {values=}")
        return seconds, values, dict(cache._wb_overlap_counts)

    def test_reused_victim_pages_are_written_after_the_copy(self):
        seconds, values, counts = self._evict_then_reuse_pages()
        self.assertEqual(counts["overlap"], 1)
        self.assertEqual(values, [7.0], "the copy read pages a later forward wrote")
        self.assertGreater(seconds, FWD_S, "eviction did not wait for the copy")


class TestWriteBackEvictOverlapFull(_OverlapCases, CustomTestCase):
    CFG = CacheConfig(
        components=(ComponentType.FULL,),
        page_size=1,
        num_layers=2,
        full_attention_layer_ids=(0, 1),
        kv_size=4096,
        max_context_len=2048,
        head_num=1,
        head_dim=64,
    )

    def test_reuse_control_without_cpu_wait_copies_overwritten_pages(self):
        _, values, _ = self._evict_then_reuse_pages(cpu_waits_for_copy=False)
        self.assertIn(99.0, values)

    def _evict_value_copied_after_barrier(self, freshness_guard=True):
        """A leaf inserted after the barrier whose value tensor is filled behind
        the forward; its caching-allocator block first holds stale page ids."""
        cache, allocator, req_pool, cc = self._fixture()
        self._warm(cache, allocator, req_pool)
        n = 640  # 5120-byte int64 tensors: no other allocation here uses the size.
        with self._sched():
            pages = allocator.alloc(n)
            stale_pages = allocator.alloc(n)
            _fill(allocator, pages, 7)
            _fill(allocator, stale_pages, 5)
            seeds = [stale_pages.clone() for _ in range(8)]
            seed_ptrs = {s.data_ptr() for s in seeds}
            del seeds
        torch.cuda.synchronize()
        self._launch_forward(cc)
        with self._sched():
            value = torch.empty(n, dtype=torch.int64, device="cuda")
            value.copy_(pages)  # Queued behind the in-flight forward.
            key = RadixKey(array("q", range(5000, 5000 + n)))
            leaf = cache.insert(InsertParams(key=key, value=value)).last_device_node
            stored = node_device_value(cache, leaf, ComponentType.FULL)
        stale_block = stored.data_ptr() in seed_ptrs and self._forward_running()
        if not freshness_guard and not stale_block:
            torch.cuda.synchronize()
            self.skipTest("the value did not reuse a stale block before the copy")
        log = []
        with contextlib.ExitStack() as stack:
            if not freshness_guard:
                stack.enter_context(
                    patch.object(
                        cache, "_wb_overlap_chain_is_fresh", return_value=False
                    )
                )
            stack.enter_context(self._recording(cc.l2_transfer_engine, log))
            stack.enter_context(self._sched())
            evicted = cache.evict(EvictParams(num_tokens=n))
        self.assertGreaterEqual(evicted.num_tokens_evicted, n)
        values, _ = _host_values(log)
        print(f"{self.id()}: {values=} {stale_block=} {cache._wb_overlap_counts}")
        return values, dict(cache._wb_overlap_counts)

    def test_value_copied_after_barrier_keeps_the_forward_fence(self):
        values, counts = self._evict_value_copied_after_barrier()
        self.assertEqual(values, [7.0])
        self.assertEqual((counts["fresh"], counts["overlap"]), (1, 0))

    def test_mixed_batch_copies_both_victims(self):
        """One walk queues a victim built on the private stream and a fresh one
        built behind the forward; the batch's copy must see both."""
        cache, allocator, req_pool, cc = self._fixture()
        self._warm(cache, allocator, req_pool)
        self._insert_leaf(cache, allocator, req_pool, range(200, 216), value=7)
        n = 640
        with self._sched():
            pages = allocator.alloc(n)
            stale_pages = allocator.alloc(n)
            _fill(allocator, pages, 8)
            _fill(allocator, stale_pages, 5)
            seeds = [stale_pages.clone() for _ in range(8)]
            seed_ptrs = {s.data_ptr() for s in seeds}
            del seeds
        torch.cuda.synchronize()
        self._launch_forward(cc)
        log = []
        with self._sched():
            value = torch.empty(n, dtype=torch.int64, device="cuda")
            value.copy_(pages)  # Queued behind the in-flight forward.
            key = RadixKey(array("q", range(5000, 5000 + n)))
            leaf = cache.insert(InsertParams(key=key, value=value)).last_device_node
            stored = node_device_value(cache, leaf, ComponentType.FULL)
            stale_block = stored.data_ptr() in seed_ptrs and self._forward_running()
            with self._recording(cc.l2_transfer_engine, log):
                evicted = cache.evict(EvictParams(num_tokens=16 + n))
        self.assertGreaterEqual(evicted.num_tokens_evicted, 16 + n)
        values, _ = _host_values(log)
        counts = cache._wb_overlap_counts
        print(f"{self.id()}: {values=} {stale_block=} {counts}")
        self.assertEqual((counts["overlap"], counts["fresh"]), (1, 1))
        self.assertEqual(values, [7.0, 8.0])

    def test_freshness_control_overlap_reads_stale_page_ids(self):
        values, _ = self._evict_value_copied_after_barrier(freshness_guard=False)
        self.assertIn(5.0, values)

    def test_split_after_barrier_keeps_the_forward_fence(self):
        cache, allocator, req_pool, cc = self._fixture()
        self._insert_leaf(cache, allocator, req_pool, range(1000, 1032))
        torch.cuda.synchronize()
        self._launch_forward(cc)
        log = []
        with self._sched():
            # Splits the leaf: the tail gets a copied value and a new parent.
            m = match_prefix(cache, list(range(1000, 1016)) + list(range(7000, 7016)))
            self.assertEqual(m.device_prefix_len, 16)
            with self._recording(cc.l2_transfer_engine, log):
                evicted = cache.evict(EvictParams(num_tokens=32))
        self.assertGreaterEqual(evicted.num_tokens_evicted, 16)
        values, _ = _host_values(log)
        self.assertEqual(values, [7.0])
        self.assertGreaterEqual(cache._wb_overlap_counts["fresh"], 1)
        self.assertEqual(cache._wb_overlap_counts["overlap"], 0)

    def test_private_stream_avoids_scheduler_streams(self):
        cache, _, _, cc = self._fixture()
        eng = cc.l2_transfer_engine
        eng.device_to_host_stream = self.schedule
        eng.index_stream = self.forward
        with self._sched(), self.assertLogs(urc_mod.logger, level="INFO") as logs:
            stream = cache._wb_overlap_get_stream(cc)
            self.assertIs(cache._wb_overlap_get_stream(cc), stream)
        scheduler_handles = {self.schedule.cuda_stream, self.forward.cuda_stream}
        l2_handles = {
            eng.device_to_host_stream.cuda_stream,
            eng.host_to_device_stream.cuda_stream,
            eng.index_stream.cuda_stream,
        }
        self.assertFalse(l2_handles & scheduler_handles)
        self.assertNotIn(stream.cuda_stream, scheduler_handles | l2_handles)
        self.assertIn("device_to_host_stream, index_stream", logs.output[0])


class TestWriteBackEvictOverlapFullMamba(_OverlapCases, CustomTestCase):
    CFG = CacheConfig(
        components=(ComponentType.FULL, ComponentType.MAMBA),
        page_size=1,
        num_layers=2,
        full_attention_layer_ids=(0,),
        kv_size=4096,
        max_context_len=2048,
        head_num=1,
        head_dim=64,
    )


class TestWriteBackEvictInternalSwa(UnifiedRadixCacheFixtures, CustomTestCase):
    """Internal SWA state backups keep the forward-stream fence: an older
    window of the in-flight checkpoint is evictable."""

    def setUp(self):
        self.cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.SWA),
            page_size=1,
            sliding_window_size=4,
            num_layers=2,
            full_attention_layer_ids=(0,),
            kv_size=64,
            max_context_len=64,
            head_num=1,
            head_dim=64,
        )
        p = patch.object(urc_mod, "get_disagg", return_value=PREFILL)
        p.start()
        self.addCleanup(p.stop)

    def _request(self, cache, allocator, req_pool, start, length):
        req = self._make_req(req_pool)
        req.origin_input_ids = array("q", range(start, start + 2 * length))
        req.full_untruncated_fill_ids = array("q", range(start, start + length))
        req.output_ids = array("q")
        req.pending_bootstrap = False
        req.extend_end = length
        ids = allocator.alloc(length)
        self.assertIsNotNone(ids)
        req_pool.write((req.kv.req_pool_idx, slice(0, length)), ids)
        req.kv.kv_committed_len = length
        req.last_node = cache.root_node_handle()
        req.lock_receipt = DecLockRefParams()
        return req

    def _evict_checkpoint_window(self, relax_every_write_back=False):
        cache, allocator, req_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        controller = cache.cache_controller
        eng = controller.l2_transfer_engine
        forward = _distinct_stream(
            [eng.device_to_host_stream, eng.host_to_device_stream, eng.index_stream]
        )
        controller.write_fence_stream = forward
        (window,) = cache._window_sub_pools_by_slot().values()
        evict_window = EvictParams(window_num_tokens={window: 1})
        warm = self._request(cache, allocator, req_pool, 100, length=12)
        cache.checkpoint(warm, up_to=12)
        warm_evicted = cache.evict(evict_window)
        self.assertGreaterEqual(warm_evicted.window_num_tokens_evicted[window], 1)
        req = self._request(cache, allocator, req_pool, 0, length=12)
        buffers = _kv_buffers(allocator)
        for buf in buffers:
            buf.fill_(-1)
        read_done = torch.cuda.Event()
        read_done.record()
        torch.cuda.synchronize()
        with torch.cuda.stream(forward):
            controller.record_forward_start(forward.record_event())
            read_done.record()
            _sleep_s(FWD_S)
            for buf in buffers:
                buf.fill_(42)
        torch.cuda.current_stream().wait_event(read_done)
        snapshots = []
        submit = eng.submit_device_to_host

        def record_copy(transfers):
            completion = submit(transfers)
            completion.finish_event.synchronize()
            for transfer in transfers:
                if transfer.host_indices.numel():
                    page = transfer.host_pool.get_data_page(
                        int(transfer.host_indices[0])
                    )
                    snapshots.append(page.clone())
            return completion

        writing_check = UnifiedRadixCache.writing_check

        def relaxed_writing_check(self_, write_back=False, finish_count=None, **_):
            return writing_check(
                self_,
                write_back=write_back,
                finish_count=finish_count,
                relaxed_fence=write_back,
            )

        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.object(eng, "submit_device_to_host", side_effect=record_copy)
                )
                if relax_every_write_back:
                    stack.enter_context(
                        patch.object(
                            UnifiedRadixCache, "writing_check", relaxed_writing_check
                        )
                    )
                cache.checkpoint(req, up_to=12)
                evicted = cache.evict(evict_window)
            self.assertGreaterEqual(evicted.window_num_tokens_evicted[window], 1)
            self.assertTrue(snapshots)
            return sorted({v for s in snapshots for v in s.float().unique().tolist()})
        finally:
            forward.synchronize()

    def test_internal_swa_backup_waits_for_forward(self):
        self.assertEqual(self._evict_checkpoint_window(), [42.0])

    def test_control_relaxing_internal_backups_copies_stale_pages(self):
        self.assertIn(-1.0, self._evict_checkpoint_window(relax_every_write_back=True))


if __name__ == "__main__":
    unittest.main()
